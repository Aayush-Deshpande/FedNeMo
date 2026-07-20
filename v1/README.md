# FedNeMo

Local, in-process **federated fine-tuning of NVIDIA Nemotron-Mini-4B-Instruct**
for medical text classification (symptom description → disease), with a full
privacy-preserving federation stack: **FedRand** split-adapter, **Laplace
differential privacy**, **quantized** transmission, and **entropy + trust-weighted
aggregation**.

Everything runs as local Python in a single process — **no servers, no MCP, no
cloud orchestration**. The only outbound network call anywhere is a single request
to NVIDIA's hosted **nemotron-parse** model at inference time (to transcribe a
scanned report image). Training is 100% local.

## Result (held-out, Symptom2Disease — 1,200 records, 24 classes)

| Metric | Value |
|---|---|
| **Accuracy** | **87.5%** |
| **Macro-F1** | **0.86** |
| Unparseable outputs | 0.0% |
| Setup | 5 nodes · 4 rounds · DP on · 8-bit quant · LoRA r=16 |

Full breakdown: [`docs/RESULTS.md`](docs/RESULTS.md) · auto-generated detailed
report: [`docs/REPORT_final.md`](docs/REPORT_final.md).

---

## ⚠️ Model setup (required — the model is NOT in this repo)

The base checkpoint **`Nemotron-Mini-4B-Instruct` (~16 GB)** is far too large for
GitHub and is intentionally **not committed**. You must place it locally before
running:

```
v1/
└── Nemotron-Mini-4B-Instruct/     <-- put the downloaded HF checkpoint here
    ├── config.json
    ├── model.safetensors          (preferred; used automatically if present)
    ├── tokenizer.json, tokenizer.model, ...
```

The code looks for it at `v1/Nemotron-Mini-4B-Instruct/` by default. To keep it
elsewhere, set the env var:

```bash
set FEDNEMO_MODEL_PATH=E:\path\to\Nemotron-Mini-4B-Instruct   # Windows
```

`model.safetensors` is preferred (torch 2.5.x refuses to load `pytorch_model.bin`
under CVE-2025-32434). If you only have the `.bin`, convert it once to safetensors.

---

## Project structure

```
v1/
├── README.md · requirements.txt · .env.example
├── fednemo/                 # source package
│   ├── config.py            # ALL tunables + paths (edit here)
│   ├── audit.py             # per-node/round/run audit trail
│   ├── data/                # loaders, record type, IID partition, holdout
│   ├── model/               # VRAM-bounded Nemotron loader + prompt serialization
│   ├── federated/           # client, orchestrator, fedrand, dp, quantization,
│   │                        #   aggregator, trust_agent, privacy_accounting
│   ├── inference/           # nemotron-parse client, field mapping, infer, sample report
│   ├── eval/                # held-out metrics, comm accounting, GIA check
│   └── scripts/             # CLI entry points
├── data/                    # Symptom2Disease.csv
├── adapters/                # trained global adapter (the model output)
├── artifacts/               # generated run outputs (audit, eval, holdout, ...)
├── docs/                    # RESULTS.md, REPORT_final.md, evidence/
└── Nemotron-Mini-4B-Instruct/   # (you place this; gitignored)
```

---

## Functionality — every component we use

### Data (`fednemo/data/`)
- **`symptom_loader.py`** — loads `data/Symptom2Disease.csv` (or any text/label
  CSV/JSON via `load_text_classification(path, text_field, label_field)`) into a
  neutral `ClinicalRecord` list.
- **`record.py`** — dataset-agnostic `ClinicalRecord(record_id, label, free_text,
  features, source)`. Swapping datasets only means a new loader.
- **`partition.py`**
  - `stratified_holdout(...)` — reserves a class-balanced held-out set *before*
    partitioning (never seen by any node).
  - `iid_partition(...)` — random, **balanced, equal-parts** split across nodes
    (each node gets an equal share of every class).
- **`record_io.py`** — save/load records as JSON (used for the held-out set).

### Model (`fednemo/model/`)
- **`nemotron_local.py`** — the VRAM-bounded loader for a 6 GB GPU:
  - 4-bit NF4 transformer body on GPU; 256k-vocab `embed_tokens` on CPU;
    `lm_head` on GPU (4-bit) or CPU (config `LM_HEAD_DEVICE`).
  - Manual forward (`embed → body → lm_head`) so each stage runs where intended;
    cross-device autograd; `generate()` for inference.
- **`serialization.py`** — turns a record into an instruction prompt
  (`build_prompt`), the supervised target (`build_target`), registers the dataset
  label space (`set_label_space`), and parses model output back to a label
  (`parse_model_output`). Also a paraphrased prompt for robustness tests.

### Federated core (`fednemo/federated/`)
- **`client.py` — `ClientNode`** — one node: fine-tunes the LoRA adapter on its
  shard. Trains on the **full shard, shuffled each round**, with **gradient
  accumulation** (effective batch). Supports optional label-flip / exploding-update
  poisoning for trust validation.
- **`fedrand.py`** — the FedRand split: per layer per round, `Bernoulli(ρ)` picks
  whether to **share A or B** (the other stays private, persisted across rounds).
  Also extracts/loads LoRA adapter state.
- **`dp.py`** — **Laplace** differential privacy. `relative` mode calibrates noise
  to a fraction (`noise_ratio`) of the update's own magnitude (keeps SNR sane);
  `absolute` mode is the formal `C/ε` mechanism.
- **`quantization.py`** — adaptive per-tensor affine quantization to `QUANT_BITS`
  (2 or 8) with scale/zero-point; used on transmitted matrices.
- **`aggregator.py`** — combines per-slot updates weighted by `trust × entropy
  importance`; slots with no contributor keep their prior value.
- **`trust_agent.py`** — the "second Nemotron role": the base model (adapter
  disabled) scores each update's trustworthiness from a text summary; blended with
  a deterministic heuristic so a bad model read can't wrongly zero a node.
- **`privacy_accounting.py`** — RDP composition across rounds → reports
  `ε_total` (much tighter than naïve) with a configurable budget ceiling.
- **`orchestrator.py`** — runs the sequential round loop, optional distributed-DP
  noise reporting, and writes the audit trail.

### Audit (`fednemo/audit.py`)
Writes a transparent per-node/round/run trail: assigned dataset, dataset summary,
and for each layer the trained A/B → after-DP → after-quant values plus which
matrix was sent vs dropped, timing, and privacy budget.

### Evaluation (`fednemo/eval/`)
- **`evaluate.py`** — runs the trained adapter on the held-out set with
  **constrained decoding** (snaps output to a valid class → 0% unparseable).
- **`metrics.py`** — accuracy, macro-F1, per-class precision/recall/F1, confusion.
- **`comm_accounting.py`** — communication savings vs a 32-bit full-FedAvg baseline.
- **`gia.py`** — minimal gradient-inversion sanity check substantiating FedRand's
  structural privacy claim.

### Inference (`fednemo/inference/`)
- **`nemotron_parse_client.py`** — the single external API call: image →
  transcribed text (requires `NVIDIA_API_KEY`).
- **`field_mapping.py`** — deterministic rule-based mapping of parsed report text
  into the model's input schema (no model call).
- **`infer.py`** — end-to-end: image → parse → field-map → local model → prediction.
- **`make_sample_report.py`** — renders a synthetic report image to test the path.

---

## Setup

```bash
pip install -r requirements.txt
# place the checkpoint at v1/Nemotron-Mini-4B-Instruct/  (see "Model setup" above)
cp .env.example .env            # set NVIDIA_API_KEY only if you use inference
```

## Commands

```bash
# 1) Federated training  -> writes adapters/global_adapter_<tag>.pt + audit trail
python -m fednemo.scripts.run_training --tag run --rounds 4 --quant-bits 8

#    key flags: --clients 5  --rounds 4  --quant-bits {2,8}  --epsilon 4.0
#               --dp-mode {relative,absolute}  --lm-head-device {cuda,cpu}
#               --no-audit  --class-weighted  --limit N

# 2) Held-out evaluation  -> accuracy + per-class F1 + eval_<tag>.json
python -m fednemo.scripts.run_eval --tag run

# 3) Detailed report  -> docs/REPORT_<tag>.md (nodes/rounds/steps, gradients,
#    DP, quant before/after, which A/B dropped, accuracy, timing, resources)
python -m fednemo.scripts.generate_report --tag run

# 4) Communication-savings accounting
python -m fednemo.scripts.run_comm_report --tag run

# 5) Inference on a scanned report image (needs NVIDIA_API_KEY)
python -m fednemo.scripts.run_inference --image report.png --question "..."
```

## Key config knobs (`fednemo/config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `NUM_CLIENTS` | 5 | federated nodes |
| `NUM_ROUNDS` | 4 | federation rounds |
| `LOCAL_MAX_STEPS` | 0 | 0 = full shard/round |
| `GRAD_ACCUM_STEPS` | 4 | effective batch size |
| `LORA_RANK` / `LORA_ALPHA` | 16 / 32 | adapter capacity |
| `FEDRAND_SHARE_PROB` | 0.5 | P(share A) per layer |
| `DP_MODE` / `DP_NOISE_RATIO` | relative / 0.25 | DP calibration |
| `QUANT_BITS` | 2 | transmission quantization |
| `LM_HEAD_DEVICE` | cuda | lm_head placement (speed vs VRAM) |

## Honest notes

- The **inference path** is implemented but **not yet verified end-to-end against
  the live nemotron-parse API** (needs a valid `NVIDIA_API_KEY`); field mapping is
  tuned on synthetic report text.
- Reported ε_total (RDP) ≈ 16 over 4 rounds is a *moderate* privacy budget.
- `relative`-mode DP is a signal-calibrated heuristic, not pure (ε,0)-DP
  (see `fednemo/federated/dp.py`).
