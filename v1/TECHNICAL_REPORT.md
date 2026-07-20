# FedNeMo — Full Technical Report (as-implemented)

> **Scope of this document.** This describes the FedNeMo system *exactly as it
> exists in the code under `e:\FedNeMo\v1\fednemo\` right now*. Every claim is
> traceable to a specific module/function. Where something is partial, unverified,
> or a deliberate deviation, it is called out explicitly (see Section I). Measured
> VRAM and loss numbers come from real runs performed during development on the
> target machine (RTX 4050 Laptop, 6141 MiB total; Windows; Python 3.12; torch
> 2.5.1+cu121; transformers 4.57.3; peft 0.19.1; bitsandbytes 0.49.2).

---

## 0. System overview

FedNeMo is a fully local, in-process federated fine-tuning framework for
`nvidia/Nemotron-Mini-4B-Instruct`, applied to a cardiac-risk / ECG anomaly
classification task. It runs 5 simulated clients that each hold a non-IID CSV
shard, fine-tune a shared LoRA adapter locally, and contribute privacy-protected,
randomized, quantized partial updates to a local aggregator that combines them
with entropy-aware and trust-based weighting.

**No servers, no MCP, no cloud orchestration.** Everything is Python functions in
one process. The only outbound network call anywhere is a single request to
NVIDIA's hosted `nemotron-parse` model at inference time (Section H).

Code map:

```
fednemo/
  config.py                     # all tunables + device/VRAM policy
  data/
    ptbxl_loader.py             # PTB-XL CSV -> ClinicalRecord + superclass label
    uci_loader.py               # UCI Heart Disease CSV -> ClinicalRecord
    partition.py                # Dirichlet non-IID partition + logging
  model/
    nemotron_local.py           # VRAM-bounded loader + manual forward + generate
    serialization.py            # row -> prompt, target, output parsing
  federated/
    dp.py                       # Laplace DP mechanism
    quantization.py             # 2-bit adaptive affine quant
    fedrand.py                  # split-adapter mechanism + payload packaging
    trust_agent.py              # trust scoring (adapter-disabled model role)
    aggregator.py               # entropy + trust weighted aggregation
    client.py                   # ClientNode: local training
    orchestrator.py             # round loop, sequential clients, save
  inference/
    nemotron_parse_client.py    # the single external API call
    field_mapping.py            # parsed text -> training schema (deterministic)
    infer.py                    # end-to-end inference entrypoint
  scripts/
    run_training.py             # CLI
    run_inference.py            # CLI
```

---

## A. Datasets

### A.1 PTB-XL (primary) — `data/ptbxl_loader.py`

**Files actually read** (paths from `config.py`):
- `Datasets/ptb-xl-.../ptbxl_database.csv` — 21,837 rows, one per ECG record.
- `Datasets/ptb-xl-.../scp_statements.csv` — the SCP-ECG code dictionary.

**No waveform files are read.** The `records100/` and `records500/` `.dat`/`.hea`
signal files exist in the dataset but are never opened. This is the CSV-only hard
constraint, and its consequence is stated in the loader docstring: there are **no
engineered ECG signal features** — the model sees only tabular metadata + text.

**Columns actually used** from `ptbxl_database.csv`:
- `ecg_id` → `record_id` (`ptbxl_<id>`)
- `age` → `features["age"]` (int, or None if missing)
- `sex` → `features["sex"]` via `SEX_MAP = {0:"male", 1:"female"}`
- `height` → `features["height_cm"]` (float or None)
- `weight` → `features["weight_kg"]` (float or None)
- `device` → `features["recording_device"]` (string or None)
- `report` → `free_text` (the raw clinical report string, **German**, left as-is)
- `scp_codes` → used to derive the label (below); not itself a feature.

All other columns (nurse, site, validation flags, noise flags, filename columns,
etc.) are ignored.

**Label derivation (the `scp_codes` → superclass mapping)** — this is the
standard PTB-XL benchmarking approach, implemented in
`_build_code_to_superclass` + `_superclass_from_scp_codes`:

1. Load `scp_statements.csv` indexed by SCP code.
2. Keep only rows where `diagnostic == 1.0` (diagnostic statements).
3. Build `code → diagnostic_class` where `diagnostic_class ∈ {NORM, MI, STTC, CD, HYP}`.
4. For each record, `scp_codes` is a dict-string like `{'NORM': 100.0, 'SR': 0.0}`,
   parsed with `ast.literal_eval`.
5. For each code present, look up its superclass and **sum the likelihood values
   within each superclass**; take the `argmax` superclass as the label.
6. Records whose codes map to no superclass are **dropped** (returns `None`).

**Verified label distribution (real run, full dataset, from a dev probe):**
- Total labelled: 21,430 (407 unmappable dropped)
- `NORM`: 9257, `MI`: 4059, `CD`: 3435, `STTC`: 3370, `HYP`: 1309

**`SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]`** (5-class problem).

`limit` argument keeps only the first N labelled records for fast local runs.

### A.2 UCI Heart Disease (secondary) — `data/uci_loader.py`

**File:** `Datasets/heart_disease_uci.csv`. Purpose: validate the pipeline
generalizes to a *different feature schema*.

**Label:** column `num` (0–4 angiographic severity) mapped to text via
`UCI_LABELS`:
`0→no_heart_disease, 1→mild_heart_disease, 2→moderate_heart_disease,
3→severe_heart_disease, 4→critical_heart_disease`. Rows with missing `num` dropped.

**Features extracted** (13): `age, sex, chest_pain_type(cp), resting_bp_mmHg(trestbps),
cholesterol_mg_dl(chol), fasting_blood_sugar_gt_120(fbs), resting_ecg(restecg),
max_heart_rate(thalch), exercise_induced_angina(exang), st_depression_oldpeak(oldpeak),
st_slope(slope), num_major_vessels(ca), thalassemia(thal)`. `free_text` is empty for UCI.

### A.3 Common record type

Both loaders emit `ClinicalRecord(record_id, features: dict, label: str,
free_text: str, source: "ptbxl"|"uci")`. `source` decides the label space and
target formatting downstream.

---

## B. Non-IID data partitioning — `data/partition.py`

**Method:** per-class Dirichlet partition (`dirichlet_partition`).

**Algorithm, step by step:**
1. Group all sample indices by class.
2. For each class, shuffle its indices, draw `proportions ~ Dir(alpha · 1_K)`
   over the `K = num_clients` clients, convert to cumulative cut points, and
   `np.split` the class's indices among clients.
3. This alone produces **label skew** (each client gets a random uneven fraction
   of each class) and **volume skew** (client totals differ) simultaneously.
4. **Minimum-shard enforcement:** compute the smallest client shard; if it is
   below `min_records_per_client`, **resample** the entire Dirichlet draw (up to
   `max_resample=200` attempts), keeping the best (largest-min) partition seen.
   The first partition that satisfies the threshold is returned immediately.
5. If none satisfies it after 200 tries, the best-so-far is returned with a
   `WARNING` (so behavior is defined, never a crash).

**Actual configured values (`config.py`):** `DIRICHLET_ALPHA = 0.4`
(inside the required 0.3–0.5 band), `MIN_RECORDS_PER_CLIENT = 40`, `SEED = 42`,
`NUM_CLIENTS = 5`.

**Real resulting distribution** — from a dev run (`load_ptbxl_records(limit=300)`,
5 clients, alpha=0.4, min=20, seed=42; logged by `log_partition_stats`):

```
client |  total |    CD |   HYP |    MI |  NORM |  STTC
    0   |    150 |     0 |     0 |     0 |   143 |     7
    1   |     35 |     5 |     0 |    13 |    15 |     2
    2   |     26 |     3 |     0 |     0 |    22 |     1
    3   |     27 |     4 |    10 |     8 |     0 |     5
    4   |     62 |    22 |     6 |     6 |     7 |    21
global |    300 |    34 |    16 |    27 |   187 |    36
```

This is genuine, verifiable skew: client 0 is almost pure NORM (143/150, 0 MI/CD/HYP);
client 3 has 0 NORM but the run's only concentration of HYP; volume ranges 26→150.
(“satisfied min-size on attempt 2” was logged, i.e. one resample was needed.)

`log_partition_stats` prints this table every run, so the skew is always observable.

**Dependency:** partitioning consumes only the list of label strings; it is
model- and framework-independent. `run_training.py` calls it before building
`ClientNode`s.

---

## C. Model — `model/nemotron_local.py`

### C.1 Checkpoint and architecture

- **Checkpoint:** `nvidia/Nemotron-Mini-4B-Instruct` (a.k.a. Minitron-4B-Instruct),
  loaded from the local path `D:\Code\models\Nemotron-Mini-4B-Instruct` (override
  via `FEDNEMO_MODEL_PATH`).
- **Architecture (from the checkpoint's `config.json`):** `NemotronForCausalLM` —
  a **standard dense transformer** (NOT Mamba/MoE). `hidden_size=3072`,
  `num_hidden_layers=32`, `num_attention_heads=24`, `num_key_value_heads=8` (GQA),
  `intermediate_size=9216`, `hidden_act="relu2"` (squared ReLU, so the MLP has
  `up_proj`/`down_proj` and **no `gate_proj`**), `vocab_size=256000`,
  `torch_dtype=bfloat16`, `tie_word_embeddings=false`.
- **~4.2B parameters total** (`get_peft_model` reported `all params: 4,202,043,392`).

**Why vocab size mattered.** With `vocab_size=256000` and `tie_word_embeddings=false`,
`embed_tokens` and `lm_head` are each a `256000 × 3072` matrix ≈ **1.5 GB in
bf16**, stored separately (that is why the on-disk checkpoint is ~8.3 GB). These
two tensors dominate memory and drove the entire VRAM debugging effort (C.3).

### C.2 One-time format conversion (why it exists)

The downloaded checkpoint ships as `pytorch_model.bin`. transformers 4.57 refuses
to load `.bin` under torch 2.5.1 because of CVE-2025-32434 (unsafe `torch.load`).
A one-time local conversion to `model.safetensors` was performed (safetensors is
exempt from that restriction). This is a prerequisite, not part of the runtime
code path. **If `model.safetensors` is absent, loading will fail** until it is
regenerated (see Section I).

### C.3 Quantization / device layout (the winning configuration)

Final layout, driven by `CONFIG.lm_head_device` (default `"cuda"`):

| Component | Placement | Precision |
|---|---|---|
| `embed_tokens` (256k×3072) | **CPU** (always) | float32 (cast at load) |
| 32 decoder layers + `norm` | **GPU** | 4-bit NF4 (bitsandbytes) |
| `lm_head` (256k×3072) | **GPU** (default) or CPU | 4-bit NF4 (GPU) / float32 (CPU) |

bitsandbytes config (`_bnb_config`): `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`,
`bnb_4bit_compute_dtype=bfloat16`, `bnb_4bit_use_double_quant=True`,
`llm_int8_enable_fp32_cpu_offload=True`. When `lm_head` is on GPU,
`llm_int8_skip_modules=["embed_tokens"]` is set — this overrides transformers'
default (which keeps the output head unquantized) and **forces `lm_head` to 4-bit**;
`embed_tokens` is an `nn.Embedding` that bnb cannot quantize anyway, so listing it
in the skip set is a no-op for it but conveniently redirects the "don't quantize"
slot away from `lm_head`.

**Loading mechanics:** a `device_map` is built placing `embed_tokens` on CPU,
`lm_head` on GPU-or-CPU per config, and every `model.layers.{0..31}` + `model.norm`
on GPU 0. After load, accelerate's offload hooks are removed from the CPU-resident
modules (`remove_hook_from_module`) so they execute purely on CPU instead of the
hook shuttling their 1.5 GB weight to GPU on every call. CPU-resident weights are
cast to float32 (`embed` always; `lm_head` when on CPU) because **bf16 matmul on
CPU is pathologically slow** whereas float32 uses fast MKL.

**The debugging journey (why this exact layout).** Six configurations were built
and measured before converging (all real measurements):

1. **Everything on GPU, 4-bit** — `q/k/v/o/up/down_proj` correctly became
   `Linear4bit`, but `lm_head` + `embed_tokens` stayed bf16 (~3 GB together).
   Result: load alone ≈ 4299 MiB allocated. Rejected — over any tight budget.
2. **CPU-offload embed + lm_head via accelerate (hooks intact)** — load dropped to
   1291 MiB allocated, but a realistic-length training step peaked at **4269 MiB
   allocated / 4312 MiB reserved**, because the offload hook shuttles the 1.5 GB
   `lm_head` weight onto GPU during forward. Gradient checkpointing changed nothing.
3. **Force `lm_head` to 4-bit *and keep it on GPU*** — peaked at **5793 MiB
   allocated / 7248 MiB reserved**: quantizing `lm_head` forced the giant 256k
   logit projection + its backward onto GPU. Rejected (worst case).
4. **Manual forward split, embed + lm_head on CPU (hooks removed), bf16** —
   memory was fine but the step **hung > 180 s**: bf16 CPU matmul over the 256k
   head is unusably slow. Rejected.
5. **CPU embed + GPU 4-bit body + CPU float32 lm_head** — **2221 MiB allocated /
   2314 MiB reserved**, ~2.4 s/step, loss decreasing, 384/384 LoRA grads flowing.
   This was the first fully-working config and is still selectable via
   `lm_head_device="cpu"`.
6. **CPU embed (float32 gather) + GPU 4-bit body + GPU 4-bit lm_head** (current
   default) — see C.4 numbers. ~0.5–0.8 s/step; chosen after the user raised the
   budget to 4.8 GB to trade a little VRAM for a ~3–5× speedup.

The decisive insight: the transformer body in 4-bit is only ~1.3 GB and always
fits; the 256k-vocab `embed`/`lm_head` are the whole problem. `embed` is a cheap
gather (fine on CPU); only `lm_head` is a heavy matmul, so its placement is the
one real speed/VRAM lever — hence it's the single config knob.

### C.4 Measured VRAM (real logged numbers)

**Default layout, `lm_head_device="cuda"`:**
- After load: **allocated 1721 MiB, reserved 2072 MiB** (logged by `_log_vram`).
- Peak during a full round (3 clients, training + trust-gen + aggregation, seq up
  to 256): **allocated 4277 MiB, reserved 5894–6022 MiB**.
- The allocated (true live) peak sits under the 4.8 GB (4915 MiB) budget; the
  higher "reserved" is the caching allocator's fragmented high-water mark (see I.4).

**Alternative layout, `lm_head_device="cpu"`:** full-round peak **allocated
~2221 MiB / reserved ~2382 MiB** (a full 3-client integration run measured 2382
MiB reserved), at ~2.4 s/step.

### C.5 LoRA / PEFT configuration

`LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj",
"up_proj","down_proj"])`.

- **Trainable params: 11,534,336** (peft's `print_trainable_parameters`).
- **Total params: 4,202,043,392.**
- **Trainable %: 0.2745%.**
- 6 target modules × 32 layers → **384 LoRA tensors** (A+B per module), confirmed
  by `extract_adapter_state` returning 384 tensors and the orchestrator log
  "Initialized global adapter with 384 LoRA tensors."

**Dependency:** everything downstream (FedRand, aggregation, client training,
inference) operates on these 384 LoRA tensors identified by the peft naming
convention `<path>.lora_A.default.weight` / `.lora_B.default.weight`.

---

## D. Tabular-to-text serialization — `model/serialization.py`

CSV rows become instruction-style text prompts. There is no numeric feature
vector; the LLM reasons over text.

**`build_prompt(record, label_space)`** produces:
```
You are a cardiology decision-support assistant. Based only on the patient record
below, state the single most likely diagnostic class, list any other clinically
relevant findings, and briefly explain your reasoning grounded in the specific
values given.
Allowed diagnostic classes: NORM, MI, STTC, CD, HYP

[PATIENT_RECORD]
Structured fields:
  - age: 56
  - sex: male
  - weight kg: 63.0
  - recording device: CS-12   E
Clinical report:
  sinusrhythmus periphere niederspannung
[/PATIENT_RECORD]

Answer:
```
- Fields with `None`/empty values are skipped by `_format_features`.
- The free-text block is only added if `free_text` is non-empty (so UCI records
  have no report block).

**`build_target(record)`** — supervised completion:
- PTB-XL: `Diagnostic class: NORM (normal ECG).` (uses `PTBXL_LABEL_DESCRIPTIONS`).
- UCI: `Diagnostic class: no heart disease.`

**Real example pair (PTB-XL `ecg_id=1`):** the prompt above → target
`Diagnostic class: NORM (normal ECG).`

**`parse_model_output(text, label_space)`** (used at inference): lower-cases the
text, first tries a regex `diagnostic class[:\s]+([a-z_]+)` and matches the token
against the allowed labels; failing that, returns the first allowed label that
appears anywhere. Returns `{predicted_label, raw_output}`.

**`label_space_for(source)`** returns the 5 PTB-XL superclasses or the 5 UCI text
labels.

---

## E. Federated training mechanism

### E.1 FedRand split-adapter — `federated/fedrand.py`

**Purpose:** the aggregator never sees both LoRA matrices (A and B) of the same
layer from the same client in the same round, so any gradient-inversion system of
equations stays underdetermined.

**`split_and_protect(client_id, updated_state, share_prob, clip_norm, epsilon,
quant_bits, rng)` — step by step per LoRA layer:**
1. Enumerate layers via `layer_keys` (strips the A/B suffixes to get 192 base
   layer paths → but each has both A and B, so 384 tensors / 2 = 192 layer slots;
   with 6 modules × 32 layers = 192 layers).
2. Fetch `A = <layer>.lora_A.default.weight`, `B = <layer>.lora_B.default.weight`.
3. Draw `share_a ~ Bernoulli(share_prob)` using the passed `torch.Generator`
   (`FEDRAND_SHARE_PROB = 0.5`).
4. If `share_a`: **public = A** (`which="A"`), private = B. Else public = B, private = A.
5. Store the **private** tensor in `private_state` (kept locally, transmitted to
   nobody).
6. Apply Laplace DP to the public tensor (E.2), then 2-bit quantize the noised
   result (E.3), and append a `SharedMatrix(layer, which, quantized, clipped_norm,
   noise_scale)` to the payload.
7. Accumulate `total_sq += sum(public²)`; after the loop `payload.update_l2_norm =
   sqrt(total_sq)` (used for trust scoring).

Returns `(ClientPayload, private_state)`. The invariant "never both A and B of a
layer shared" was asserted in a unit test and holds.

**Persistence across rounds:** in `orchestrator.run_training`, after each client's
`split_and_protect`, `client.private_state = private_state`. On the next round,
`ClientNode.local_train` does `start_state = global ⊕ private_state` before
loading into the model — so each client always re-injects its retained private
half on top of the distributed global adapter.

**`extract_adapter_state` / `load_adapter_state`** are the read/write bridge to
the live peft model (CPU float32 snapshots; written back with device/dtype cast).

### E.2 Differential privacy (Laplace) — `federated/dp.py`

**`clip_and_add_laplace(update, clip_norm, epsilon)`:**
1. Move update to CPU float32.
2. Compute L2 norm; `scale_factor = min(1, C / (‖u‖₂ + 1e-12))`; `clipped = u * scale_factor`
   (L2 clip to bound sensitivity to `C`).
3. `noise_scale = C / epsilon`; sample elementwise `η ~ Laplace(0, C/epsilon)`
   via `torch.distributions.Laplace`; `noised = clipped + η`.
4. Return `DPResult(tensor=noised, clipped_norm=min(‖u‖₂, C), noise_scale)`.

**Parameters (`config.py`):** `DP_CLIP_NORM = 1.0` (C), `DP_EPSILON = 4.0`. So
`noise_scale = 0.25` (verified in a unit test: clip→1.0, scale→0.25).

**Where applied:** inside `split_and_protect`, to the *public* matrix only, before
quantization — i.e. per shared matrix, per round.

**Honesty note (also in the module docstring):** an L2 clip is used; the tightest
Laplace `(ε,0)`-DP form uses an L1 clip. This gives a per-round guarantee for the
released matrix; there is **no cross-round privacy composition / RDP accounting**
(Section I).

### E.3 Transmission quantization — `federated/quantization.py`

**Fixed 2-bit** (`QUANT_BITS = 2`) asymmetric uniform affine quantization with
adaptive per-tensor scale/zero-point:
- `levels = 2^bits − 1 = 3`; `scale = (max−min)/levels`; `zero_point = round(−min/scale)`
  clamped to `[0, levels]`.
- `q = clamp(round(x/scale + zero_point), 0, levels)` as uint8.
- Dequant: `x̂ = (q − zero_point) · scale`.
- **Constant-tensor edge case:** if `max == min`, codes are all zero and the
  constant is stored in `constant_value`; dequant reconstructs `full(shape,
  constant_value)`. (This prevents a divide-by-zero and was unit-tested.)

Applied to the noised public matrix in `split_and_protect`; reversed by
`reconstruct_shared` (dequantize) at aggregation time. Unit test confirmed a
random tensor round-trips with 4 code levels {0,1,2,3} and preserved mean/std.

### E.4 Sequential training design — `federated/orchestrator.py`

Clients are trained **strictly one at a time** in a Python `for client in clients`
loop; there is no parallelism/threading. The single `LoadedModel` is shared and
reused for every client and for the trust agent. `torch.cuda.empty_cache()` is
called after each client (and after each training step inside `client.local_train`)
to keep the reserved pool tight.

**Why:** a single 4-bit Nemotron instance already occupies ~1.7 GB at rest and
peaks ~4.3 GB during a training step (C.4). Running 5 instances concurrently, or
even 2, would multiply that and exceed the GPU. Sequential reuse of one instance
is what keeps the whole federation inside the measured budget on a 6 GB card.

---

## F. Aggregation — `federated/aggregator.py`

### F.1 Entropy-aware importance — `entropy_importance`

For each client `i`:
```
nu_i = lambda_h · (H(D_i)/H_max) + (1 − lambda_h) · (|D_i| / N_max)
```
- `lambda_h = 0.5` (module constant `LAMBDA_H`).
- `H(D_i)` = Shannon entropy (base-2) of the client's label distribution,
  computed in `trust_agent.label_entropy`; `H_max = log2(num_classes)`;
  `entropy_ratio = H/H_max` is precomputed in `build_summary`.
- `|D_i|` = client sample count; `N_max` = max sample count across clients.

So `nu_i` rewards both label diversity and dataset volume equally. Example
(from a real round report): `entropy_importance = {0: 0.634, 1: 0.293, 2: 0.619}`.

### F.2 Weighted aggregation — `aggregate`

Step by step:
1. Compute `nu` for all clients.
2. For each client payload, weight `w_i = max(0, trust_i) · max(1e-6, nu_i)`,
   dequantize its shared matrices, and bucket each contribution by `(layer, which)`
   **slot** (recall FedRand means a client contributes A *or* B per layer, not both).
3. Start `new_state` as a clone of the previous `global_state` (so any slot with
   no contributor this round **keeps its previous value** — never undefined).
4. For each slot with contributors, compute `total_w`; if `> 0`, set the slot to
   the weight-normalized sum `Σ (w_i/total_w) · tensor_i`.
5. Return `(new_state, contributor_counts)`.

**Consequence of trust=0:** a client with `trust_i = 0` gets `w_i = 0`, so it is
effectively excluded from every slot it contributed to (its mass is zero in the
normalization). This is exactly how bad updates are dropped (F.3).

**Dependencies:** `aggregate` needs, per client: the `ClientPayload` (quantized
shared matrices), the `UpdateSummary` (for `nu`), and the `TrustResult` (for the
score). These come from `split_and_protect`, `build_summary`, and `score_update`
respectively.

### F.3 Trust-scoring agent — `federated/trust_agent.py`

**The "second logical Nemotron".** Rather than a second loaded model (impossible
in budget — Section I.1), `score_update` runs the **same** `LoadedModel` with the
LoRA adapter disabled via `with lm.model.disable_adapter():` and a distinct
auditor prompt. It receives a **text summary only** — never raw tensors.

**Input (`_summary_text` over `UpdateSummary`):** client id, number of training
samples, class distribution, label entropy + entropy ratio, the full local loss
trajectory and its net drop, and the update L2 norm. Example text:
```
Client 0 federated update summary:
- training samples: 45
- class distribution: NORM:43, STTC:2
- label entropy: 0.258 (ratio to max: 0.11)
- local loss trajectory: 3.413, 3.201, 2.989 (net drop: 0.424)
- update L2 norm: 0.3750
```

**How the score is produced:**
1. Build the auditor prompt asking for `Score: <0.0–1.0>` + one-sentence reason.
2. Generate up to 64 tokens (greedy, adapter disabled).
3. `_parse_score` extracts a number via regex (`score/trust: <n>`, else first
   float); if the model answered on a 0–100 scale (>1.0) it is divided by 100 and
   clamped to `[0,1]`.
4. If generation fails or is unparseable, fall back to `_heuristic_score`
   (deterministic: base 0.5, + loss-drop bonus in `[−0.25, +0.25]`, − 0.25 if the
   update norm is exploding `>50` or vanishing `<1e-6`, + `0.15 · entropy_ratio`).
   This guarantees aggregation never breaks.

**The real Client-0 nan-exclusion example.** On the *first* integration run
(before the loss-masking fix, Section G), Client 0's first training step produced
a `nan` loss (`loss nan -> 3.124`). The resulting corrupted update summary, when
scored by the model, received **Trust[client 0] = 0.000**, while clients 1 and 2
received 0.001 and 0.900. With `w_0 = 0 · nu_0 = 0`, Client 0's contribution was
weighted to zero in `aggregate` — the corrupted update was excluded from the
global adapter. (After the nan fix, the same client trains cleanly, e.g.
`loss 3.413 -> 2.989`, and later runs score it 0.79–0.95.)

**Honesty note:** the base 4B model is a noisy auditor — observed scores across
runs ranged widely (0.000, 0.001, 0.9, 0.95). The *mechanism* (summary → score →
zero-weighting) works and is deterministic in how it affects aggregation, but the
raw scores are not calibrated (Section I).

---

## G. Loss function and training dynamics

### G.1 Loss — `model/compute_causal_loss` + `client._encode`

Standard next-token cross-entropy with completion-only masking:
- `_encode` tokenizes prompt and target separately; labels are `[-100]*len(prompt)
  + target_ids` so **loss is computed only on the target completion**, not the
  prompt.
- `compute_causal_loss` shifts logits/labels by one, flattens, upcasts only the
  2D logits view to float32, and calls `F.cross_entropy(..., ignore_index=-100)`.

### G.2 The nan bug — diagnosis and fix (with evidence)

**Symptom (real, first integration run):**
```
Client 0 trained 2 steps | loss nan -> 3.124 | n=45
Client 1 trained 2 steps | loss 2.650 -> 2.404 | n=21
Client 2 trained 2 steps | loss 1.933 -> 2.701 | n=24
```
Only Client 0's *first* step was `nan`.

**Diagnosis.** The original `_encode` tokenized the prompt with
`truncation=True, max_length=MAX_SEQ_LEN` and appended the target. For a long
PTB-XL prompt, truncation could consume the entire budget so that after
`labels = [-100]*len(prompt) + target` and clipping to `max_seq_len`, **every
label was `-100`**. Cross-entropy over an all-ignored sequence is `nan`.

**Fix (current `_encode`).** The target completion is now reserved first: cap
`target_ids` to `max_len-1`, compute `room_for_prompt = max_len - len(target_ids)`,
and **truncate the prompt from the left** (keeping the trailing `Answer:` cue) to
fit. A final guard forces at least one supervised token if somehow all are `-100`.

**Evidence after fix (real run, same seed/data):**
```
Client 0 trained 2 steps | loss 3.413 -> 2.989 | n=45
Client 1 trained 2 steps | loss 2.650 -> 2.424 | n=21
Client 2 trained 2 steps | loss 3.010 -> 2.645 | n=24
```
Client 0 now trains normally; all losses decrease. A later 3-step run showed even
stronger drops (e.g. `3.054 -> 1.590`).

### G.3 Per-step training loop (`client.local_train`), in order

1. `start_state = clone(global_state)` then `.update(self.private_state)`;
   `load_adapter_state(model, start_state)` writes all 384 (or private-overridden)
   LoRA tensors into the model.
2. `model.train()`; fresh `AdamW(trainable, lr=1e-4)` over the ~11.5M LoRA params.
3. For each record until `LOCAL_MAX_STEPS` (30) / `LOCAL_EPOCHS` (1):
   a. `_encode` → CPU `input_ids`, `labels` (batch size 1).
   b. **Forward** (`forward_logits`): CPU `embed` gather → move `[1,s,3072]` bf16
      to GPU → 32-layer 4-bit body on GPU → `lm_head` on GPU (default) producing
      logits; only the last-token path is used at generation, full sequence at
      training.
   c. **Loss**: shifted cross-entropy (G.1).
   d. **Backward**: `loss.backward()` — gradients flow through the GPU body into
      the LoRA params (cross-device autograd if `lm_head`/`embed` are CPU).
   e. `clip_grad_norm_(trainable, 1.0)` → `optimizer.step()` →
      `zero_grad(set_to_none=True)` → record `loss.item()` → `empty_cache()`.
4. `extract_adapter_state(model)` snapshots the updated 384 tensors; returns
   `LocalTrainResult(updated_state, loss_trajectory, num_samples, class_distribution)`.

---

## H. Inference pipeline — `inference/`

### H.1 The single external call — `nemotron_parse_client.parse_document_image`

- **What it does:** transcribes a scanned report **image** to text/markdown.
- **Trigger:** called once at the start of `run_inference`.
- **Endpoint (config):** `https://integrate.api.nvidia.com/v1/chat/completions`,
  model `nvidia/nemoretriever-parse`, OpenAI-compatible chat/completions.
- **Request:** the image is base64-encoded into a `data:` URL; the message content
  is `[{type:text, text: CONTROL_PROMPT}, {type:image_url, image_url:{url}}]` where
  `CONTROL_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>
  <predict_no_text_in_pic>"` (NVIDIA's documented control tokens — natural
  language degrades this model). `temperature=0.0`, `max_tokens=4096`. The API key
  is read from `NVIDIA_API_KEY` (or passed explicitly) and is **never logged or
  written to disk**.
- **Response handling:** reads `choices[0].message.content`; if absent but a
  `tool_calls` blob exists, falls back to its `function.arguments`. Returns
  `ParseResult(raw_content, model, status, error)`. All failures (no key, missing
  file, non-200, unexpected shape) return `status="error"` rather than raising.

### H.2 Deterministic field mapping — `field_mapping.map_parsed_to_schema`

Pure rule-based extraction (no model call) from the parsed text into the training
schema dict. Rules (`_RULES`) use regex to pull `age, resting_bp_mmHg,
cholesterol_mg_dl, max_heart_rate, st_depression_oldpeak, num_major_vessels`;
separate handling maps `sex` (male/female), `chest_pain_type` (typical/atypical/
non-anginal/asymptomatic), and two boolean-ish fields (`exercise_induced_angina`,
`fasting_blood_sugar_gt_120`) to `True/False` via a positive-token set.

**Verified on a synthetic sample** (real unit-test output):
```
input: "Patient sex: male, age: 63 years. Resting blood pressure: 145.
        Cholesterol: 233. Max heart rate: 150. ST depression: 2.3.
        Chest pain: typical angina. Exercise-induced angina: no."
output: {'age': 63, 'resting_bp_mmHg': 145.0, 'cholesterol_mg_dl': 233.0,
         'max_heart_rate': 150.0, 'st_depression_oldpeak': 2.3, 'sex': 'male',
         'chest_pain_type': 'typical angina', 'exercise_induced_angina': False}
```
`build_record_from_mapping` wraps the dict into a `ClinicalRecord` with empty
`label` (unknown — that's what we predict).

### H.3 Local answer generation — `infer.run_inference`

1. Parse image (H.1); on error, return early with the error.
2. Map fields (H.2), log how many were found.
3. `build_prompt(record, label_space)`; if a `question` is given, append
   `Doctor's question: <q>`; then append an instruction to provide **(1) primary
   answer, (2) other clinically relevant findings, (3) a grounded plain-language
   explanation**.
4. Load the trained model (`_load_trained_model` loads `artifacts/global_adapter.pt`
   into the LoRA params if present; otherwise warns and uses base + fresh LoRA).
5. `generate(lm, prompt, max_new_tokens=160)` — greedy, last-token-only projection
   through `lm_head` each step.
6. `parse_model_output` extracts `predicted_label`; returns `InferenceResult
   (predicted_label, answer, mapped_fields, parse_status, raw_model_output)`.

The "additional findings + grounded explanation" behavior is **elicited by the
prompt instruction**, not enforced by code — its quality depends on the fine-tuned
model (Section I).

---

## I. What is NOT implemented, is partial, or unverified

This section is deliberately complete.

1. **Dual Nemotron = one model, two roles (deviation).** Two separate 4B instances
   do not fit in the VRAM budget, so the trust agent reuses the same weights with
   `disable_adapter()`. This is a real deviation from "a second instance," made to
   satisfy the memory constraint.

2. **nemotron-parse call is UNVERIFIED end-to-end.** No live API call was ever
   executed (the key was rotated and not re-supplied). The request/response code is
   written against NVIDIA's published docs; the exact returned `content` shape and
   the field-mapping regexes are **best-effort and untested against real output**.
   The `tool_calls` fallback is a defensive guess.

3. **Field-mapping coverage is partial.** `map_parsed_to_schema` extracts ~10
   fields with hand-written regexes tuned to a synthetic example. It does not
   cover `resting_ecg`, `st_slope`, `thalassemia`, `height/weight`, or PTB-XL
   free-text, and will likely need tuning against real parsed reports.

4. **Privacy accounting is per-round only.** `dp.py` provides a per-round Laplace
   mechanism with an **L2** clip (not the tightest L1 form). There is **no
   multi-round RDP/composition accounting**, no privacy budget ceiling, and no
   formal ε-total tracking.

5. **Trust scores from the base model are uncalibrated.** The mechanism works, but
   raw model scores are noisy (observed 0.0–0.95 across runs). No validation that
   the model's numeric score correlates with actual update quality beyond the
   nan-exclusion case; the deterministic heuristic is the reliable fallback.

6. **`model.safetensors` is a required prerequisite, not produced by runtime
   code.** The one-time `.bin`→safetensors conversion must exist on disk; if
   absent, loading fails. The conversion script itself is not part of the package.

7. **No accuracy/eval numbers exist.** Only *training loss decreases* and *VRAM*
   have been measured. There is no held-out accuracy, no MIA/GIA evaluation, no
   convergence study, and no full multi-round `--rounds 3` run on the full dataset
   recorded. Runs so far were tiny smoke/integration tests (limit 90–300 records,
   1 round, 2–3 steps/client).

8. **`expandable_segments` is largely a no-op on Windows**, so peak *reserved*
   memory (~5.9–6.0 GB) runs ~1.6 GB above peak *allocated* (~4.3 GB) due to
   allocator fragmentation. True live usage is within the 4.8 GB budget, but on a
   6 GB card the default `cuda` layout leaves thin headroom; `lm_head_device="cpu"`
   is the fallback (~2.4 GB) if other apps contend for VRAM.

9. **Batch size is fixed at 1** and `LOCAL_MAX_STEPS=30` caps each client's
   per-round training — chosen for VRAM/runtime, not tuned for convergence.

10. **Inference "other findings/explanation" is prompt-elicited, not guaranteed.**
    There is no code that verifies the model actually surfaces incidental findings
    or grounds its explanation in specific values; that depends entirely on the
    (lightly) fine-tuned model's behavior.

---

## J. Component dependency summary

| Component | Consumes | Produces |
|---|---|---|
| loaders | CSV files | `ClinicalRecord[]` |
| partition | label strings | per-client index lists |
| `ClientNode.local_train` | `LoadedModel`, global adapter state | updated 384-tensor state, loss trajectory, class dist |
| `split_and_protect` | updated adapter state, DP/quant params, RNG | `ClientPayload` (quantized shared halves) + private half |
| `build_summary` | counts, class dist, loss traj, update norm | `UpdateSummary` (adds entropy) |
| `score_update` | `LoadedModel` (adapter disabled) + `UpdateSummary` text | `TrustResult` (score 0–1) |
| `entropy_importance` | `UpdateSummary[]` | `nu_i` per client |
| `aggregate` | global state + payloads + summaries + trust | new global adapter state |
| `orchestrator` | clients, model | rounds of the above, saved `global_adapter.pt` + `training_report.json` |
| inference | image, question, trained adapter, API key | parsed fields + model answer + predicted label |

---

*End of report. Every number above is either a code constant (cited to `config.py`
or the relevant module) or a value logged during a real development run on the
target machine. Sections marked "honesty note" / Section I flag everything that is
partial, deviated, or unverified.*
