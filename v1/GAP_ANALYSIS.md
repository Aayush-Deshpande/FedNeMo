# FedNeMo — Gap Analysis: Original Design (MDs/, Original-docs/, Papers/) vs. Current Implementation (`v1/fednemo/`)

> **Purpose.** An honest, code-grounded comparison of what the original design
> documents describe versus what the current lean implementation actually does.
> Not a status update — a gap analysis. Every classification cites code (or its
> absence).
>
> **Legend:**
> **[1] FULLY IMPLEMENTED** · **[2] IMPLEMENTED BUT DEVIATED** ·
> **[3] NOT IMPLEMENTED** · **[4] PARTIAL / UNVERIFIED**
>
> **Important framing.** The original docs (MDs 01–08, Original-docs/) describe a
> **completely different target system**: NVFlare + NeMo 2.0 orchestrating a 30B
> Mamba-2/MoE Nemotron across multi-GPU clusters on MIMIC/MedQA/PubMedQA, with a
> live GIA-attack Streamlit demo and a 5-paper research roadmap. The current code
> is a deliberate, much leaner rebuild (per the later rebuild brief): pure
> in-process Python, dense Nemotron-Mini-4B, PTB-XL + UCI, single 6 GB GPU. So a
> large fraction of the original design is intentionally **NOT IMPLEMENTED** —
> that is by design, not oversight, but it is catalogued here in full.

---

## 1. Federated framework / orchestration

| Feature / Claim | Source | Status | Evidence / Explanation | To complete |
|---|---|---|---|---|
| NVIDIA FLARE as core orchestrator (Controller-Worker, server/client) | MDs 01 §2, 02 §6.2, 05 §5, 06 | **[3] NOT IMPLEMENTED** | No `nvflare` import anywhere. Orchestration is a plain Python loop in `federated/orchestrator.py` (`for client in clients`). | Would require rebuilding on the FLARE runtime — a different architecture; not needed for the current local design. |
| DXO Filter pipeline (`FedRandFilter`, `LaplacianDPFilter`, `AdaptiveQuantFilter`, `ModelController`, `PrivacyAccountant` as `nvflare.apis.filter.Filter` subclasses) | MDs 01 §4 | **[3] NOT IMPLEMENTED** | The *logic* exists as plain functions (`fedrand.split_and_protect`, `dp.clip_and_add_laplace`, `quantization.quantize`, `aggregator.aggregate`) but not as DXO Filters. No `process_dxo`, `DataKind`, `MetaKey`. | Port functions into FLARE Filter classes if FLARE integration is ever desired. |
| `flare.patch(trainer)` zero-refactor federated conversion | MDs 02 §6.2 | **[3] NOT IMPLEMENTED** | No FLARE, no Lightning trainer. Training is a manual AdamW loop in `client.local_train`. | N/A under current design. |
| In-process sequential client simulation | MDs 02 §7.2 (option "c: sequential round-robin"), rebuild brief | **[1] FULLY IMPLEMENTED** | `orchestrator.run_training` trains clients strictly one-at-a-time reusing one `LoadedModel`; docs explicitly listed this as a feasibility option. | Done. |

## 2. Training framework & model

| Feature / Claim | Source | Status | Evidence / Explanation | To complete |
|---|---|---|---|---|
| NeMo 2.0 (`ModelTransform`, `llm.peft.LoRA`, Megatron, BF16/FP8) | MDs 02 §6.1, 07 §4.4 | **[3] NOT IMPLEMENTED** | Uses raw `transformers.AutoModelForCausalLM` + `peft.LoraConfig` + `bitsandbytes` in `model/nemotron_local.py`. No `nemo` import. | Would require re-hosting the loader on NeMo; large rewrite. |
| Base model = Nemotron-3 Nano 30B-A3B (hybrid Mamba-2 / Transformer / Latent MoE) | MDs 01 §3, 02 §3, 07 §4 | **[2] DEVIATED** | Actual model is `nvidia/Nemotron-Mini-4B-Instruct` = `NemotronForCausalLM`, a **dense** transformer (hidden 3072, 32 layers, GQA, squared-ReLU MLP, 256k vocab). Confirmed from checkpoint `config.json`. Chosen for single-GPU feasibility (docs 02 §2 recommended Mini-4B for the demo anyway). | To match the 30B claim would need cluster hardware; out of scope. |
| LoRA on Mamba-2 projections (`in_proj`, `out_proj`, `x_proj`, `dt_proj`) + extension-to-SSM novelty claim | MDs 01 §Stage4a, 02 §4, 07 §4.1 | **[3] NOT IMPLEMENTED (N/A)** | Dense model has no Mamba layers. `LORA_TARGET_MODULES = [q,k,v,o,up,down]_proj` only. The entire "FedRand extended to SSM" novelty is absent because the model isn't hybrid. | Only relevant if the 30B hybrid model is adopted. |
| Mamba-2 fused-kernel three-tier compatibility strategy + startup LoRA-injection validation gate | MDs 02 §5 | **[3] NOT IMPLEMENTED (N/A)** | No fused-kernel problem exists for a dense model; no validation gate in code. | N/A unless hybrid model adopted. |
| LoRA rank r=32 | MDs 01 §Stage3 | **[2] DEVIATED** | `LORA_RANK = 8` (config.py), not 32. Chosen smaller for VRAM/speed. Trainable = 11,534,336 params (0.2745%). | Trivial to raise `LORA_RANK`. |
| Catastrophic-forgetting mitigation via MoE sparsity | MDs 01 §5 | **[3] NOT IMPLEMENTED (N/A)** | Dense model has no MoE. LoRA rank constraint (a stated mechanism) is present; MoE-based one is not applicable. | N/A. |
| 4-bit / QLoRA quantized loading | MDs 02 §7.3 | **[1] FULLY IMPLEMENTED** | `bitsandbytes` NF4 4-bit body on GPU; measured load 1721 MiB. | Done. |

## 3. Datasets

| Feature / Claim | Source | Status | Evidence / Explanation | To complete |
|---|---|---|---|---|
| MIMIC-III/IV clinical notes | MDs 05 §6, 07 §5 | **[3] NOT IMPLEMENTED** | No MIMIC anywhere. | Would need MIMIC access + a notes pipeline. |
| MedQA / PubMedQA / Hindi notes / dermatology (5-hospital mix) | MDs 07 §5 (Table 2) | **[3] NOT IMPLEMENTED** | None present. | N/A under current dataset choice. |
| PTB-XL (primary) | rebuild brief | **[1] FULLY IMPLEMENTED** | `data/ptbxl_loader.py`: reads `ptbxl_database.csv` + `scp_statements.csv`; superclass label via `scp_codes → diagnostic_class` argmax. Verified 21,430 labelled records (NORM 9257 / MI 4059 / CD 3435 / STTC 3370 / HYP 1309). | Done. |
| UCI Heart Disease (secondary schema) | rebuild brief | **[1] FULLY IMPLEMENTED** | `data/uci_loader.py`: 13 features, `num`→5 text labels. | Done. |
| CSV-only, no ECG waveform features | rebuild brief | **[1] FULLY IMPLEMENTED (with stated limitation)** | No `.dat`/`.hea` read. Consequence (no signal features) documented in loader + TECHNICAL_REPORT §A. | Inherent limitation, accepted. |
| FedPS federated preprocessing (aggregated stats: scaling, KLL sketches, Box-Cox, Bayesian imputation) | MDs 01 §Stage2, 07 §4.3 | **[3] NOT IMPLEMENTED** | No FedPS module. Preprocessing is per-record local serialization only. | Build a `fedps` module if cross-site harmonization is needed. |
| Federated Tokenizer Consistency Protocol | MDs 01 §3.2.3 | **[3] NOT IMPLEMENTED** | All clients share one tokenizer trivially (same local model), but the frequency-analysis / vocab-extension protocol is absent. | N/A for single-model demo. |

## 4. Non-IID partitioning

| Feature / Claim | Source | Status | Evidence / Explanation | To complete |
|---|---|---|---|---|
| Dirichlet non-IID partition (label + volume skew), alpha 0.3–0.5, min-shard floor | rebuild brief; MDs 05 (Dirichlet mentioned) | **[1] FULLY IMPLEMENTED** | `data/partition.py`: per-class Dir(α=0.4), resample until `min_records_per_client=40`, logs realized table. Real run showed client0=143/150 NORM, volume 26→150. | Done. |

## 5. Privacy mechanisms

| Feature / Claim | Source | Status | Evidence / Explanation | To complete |
|---|---|---|---|---|
| FedRand randomized A/B subparameter split, ρ=0.5, private half retained | MDs 01 §Stage4a, 03 §4, 07 §4.1 | **[1] FULLY IMPLEMENTED** | `fedrand.split_and_protect`: per-layer `Bernoulli(0.5)`, shares one matrix, keeps the other in `private_state` persisted across rounds. Invariant "never both A&B" unit-tested. | Done (matches spec design). |
| Laplace DP noise on shared matrix, `Lap(C/ε)`, clip norm C, ε per round | MDs 01 §Stage4b, 03 §5 | **[2] DEVIATED** | `dp.clip_and_add_laplace` uses **L2** clip (C=1.0), not the L1 clip the (ε,0)-DP form requires; ε=4.0. Deviation documented in `dp.py` docstring. | Switch to L1 clip for a formally tight (ε,0)-DP claim. |
| Adaptive quantization: cosine-annealing downlink bit schedule (16→4 bit) | MDs 01 §3.4.2, 07 §4.2 | **[3] NOT IMPLEMENTED** | No cosine schedule; no downlink quantization at all. | Add scheduler if variable-bit downlink desired. |
| Adaptive quantization: Shannon-entropy per-client uplink bit-width (INT8/6/4/2) | MDs 01 §3.4.2 | **[2] DEVIATED** | `quantization.quantize` is **fixed 2-bit** for all clients with adaptive per-tensor scale/zero-point (per rebuild brief). Entropy is computed but used only for *aggregation weight*, not bit-width. | Make bit-width a function of entropy if the original scheme is wanted. |
| Unbiasedness `E[Q(θ+η)] = θ` | MDs 01 §3.4.1 | **[4] PARTIAL / UNVERIFIED** | Stochastic rounding is NOT used (deterministic `round`), so exact unbiasedness is not guaranteed; not empirically checked. | Use stochastic quantization + test expectation. |
| RDP / moments accountant, ε_total across rounds, `dp-accounting`, budget ceiling, adaptive noise scaling, hard termination | MDs 03 §6, 04 §3.1 | **[3] NOT IMPLEMENTED** | Only per-round Laplace. No composition, no ε_total, no ceiling, no `dp-accounting` import. This was flagged in the docs as "the single most critical technical gap." | Integrate `dp-accounting`, track ε_total, add ceiling + termination. |
| User-level vs example-level DP (clip whole local update) | MDs 03 §8 | **[4] PARTIAL** | DP is applied to the whole shared matrix update (client-level-ish), but not framed/verified as user-level DP; per-sample vs per-update distinction not formalized. | Define + document the exact DP unit. |
| Secure Aggregation (optional mode) | MDs 03 §9 | **[3] NOT IMPLEMENTED** | Docs marked it optional/future; absent. | Add SA workflow if targeting ANA-GIA. |
| Theorems 1–3 (noise-amplification elimination, sensitivity reduction) | MDs 03 §3–4 | **[3] NOT IMPLEMENTED (theory)** | Formal proofs are paper content, not code. The mechanism they motivate (single-matrix noising via FedRand) IS implemented, but the O(1/ε²) claim is not measured. | Empirically validate the amplification claim. |
| Gradient clipping before noise | MDs 03 §5 | **[1] FULLY IMPLEMENTED** | Two clips: `clip_grad_norm_(1.0)` during training (`client.py`) and L2 update clip in `dp.py`. | Done. |

## 6. Data quality (Stage 1)

| Feature / Claim | Source | Status | Evidence / Explanation | To complete |
|---|---|---|---|---|
| MTAE (multi-task autoencoder) sample-level outlier filtering | MDs 01 §Stage1, 07 §4 | **[3] NOT IMPLEMENTED** | No autoencoder anywhere. | Build MTAE module if data-quality filtering is in scope. |
| Federated OCSVM on loss statistics | MDs 01 §3.1.3 | **[3] NOT IMPLEMENTED** | Absent. | Build with MTAE. |

## 7. Aggregation & trust

| Feature / Claim | Source | Status | Evidence / Explanation | To complete |
|---|---|---|---|---|
| Entropy-aware client importance `ν_i = λ·H/H_max + (1−λ)·|D_i|/N_max` | MDs 01 §Stage4b/§5, 07 §4.2 | **[1] FULLY IMPLEMENTED** | `aggregator.entropy_importance` implements this exactly, λ=0.5. Real values e.g. {0:0.634,1:0.293,2:0.619}. | Done. |
| Entropy-weighted global aggregation (weighted FedAvg over slots) | MDs 01 §Stage5 | **[1] FULLY IMPLEMENTED** | `aggregator.aggregate` weights each (layer,matrix) slot by `trust·ν`, normalizes, keeps prior value for empty slots. | Done. |
| Trust-scoring reasoning agent (2nd Nemotron scores update trustworthiness) | rebuild brief (NOT in original MDs) | **[2] DEVIATED / NEW** | `trust_agent.score_update` reuses the SAME model with `disable_adapter()` (not a 2nd instance — VRAM). Real: Client-0 nan update → Trust 0.000 → excluded. **This feature is new in the rebuild; the original MDs had no trust agent.** | See priority list re: calibration. |
| Trust-agent calibration / accuracy validation | (implied best-practice; MDs 08 stresses testing claims) | **[4] PARTIAL / UNVERIFIED** | Scores are uncalibrated; observed 0.0–0.95 noisy across runs. Only the nan-exclusion behaves as intended. Heuristic fallback exists. | Validate score↔quality correlation; consider structured scoring. |

## 8. Attack / privacy validation

| Feature / Claim | Source | Status | Evidence / Explanation | To complete |
|---|---|---|---|---|
| Live GIA attack demo (IG/DLG attack, before/after, SSIM, token-reconstruction) | MDs 03 §10, 05 §6, 06 §5 | **[3] NOT IMPLEMENTED** | No attack code of any kind anywhere in the repo. Entirely absent. | Implement an IG/DLG attack harness + reconstruction metric. |
| MIA measurement (LiRA / LOSS), modality-transfer validation | MDs 03 §7 | **[3] NOT IMPLEMENTED** | Absent. No MIA experiments. | Build MIA harness; report text-specific numbers. |
| GIAShield monitor / attack simulation module | MDs 01 §2 | **[3] NOT IMPLEMENTED** | Absent. | As above. |

## 9. Evaluation & benchmarking

| Feature / Claim | Source | Status | Evidence / Explanation | To complete |
|---|---|---|---|---|
| Clinical accuracy benchmarks (PubMedQA, MedQA, ICD-10 prediction, summarization) | MDs 01 §5, 05 §6, 07 §5 | **[3] NOT IMPLEMENTED** | Zero accuracy measurement in the codebase. Only training-loss decrease is logged. | Build a held-out eval harness (see priorities). |
| Communication savings measurement (75%+ vs FedAvg) | MDs 01 §3.4.2, 07 §4.2 | **[3] NOT IMPLEMENTED** | Quantization + FedRand exist, but no byte-accounting comparing to a 32-bit FedAvg baseline. | Add transmitted-bytes accounting. |
| Privacy-utility curve (accuracy vs ε) | MDs 05 §7, 06 §2 | **[3] NOT IMPLEMENTED** | Requires an accuracy metric (absent) + ε sweep. | Depends on accuracy + RDP work. |
| Catastrophic-forgetting eval (PIQA / ARC vs medical) | MDs 01 §5, 05 §7, 06 §3 | **[3] NOT IMPLEMENTED** | No general-reasoning benchmark run. | Add a small held-out general-task probe. |
| Multiple full rounds at scale (50–100 rounds, up to 1000 clients) | MDs 01 §3.4.2, 03 §6.1, 07 | **[4] PARTIAL** | Config supports N rounds (`NUM_ROUNDS=3`, `NUM_CLIENTS=5`) but only tiny smoke/integration runs were executed (limit 90–300 records, 1 round, 2–3 steps/client). No large multi-round run recorded. | Run a real multi-round training (see priorities). |

## 10. Demo / UX / framing

| Feature / Claim | Source | Status | Evidence / Explanation | To complete |
|---|---|---|---|---|
| Streamlit dashboard (topology, comm efficiency, privacy budget, GIA panel, forgetting curves) | MDs 05 §7, 06 §1 | **[3] NOT IMPLEMENTED** | No UI code. | Build Streamlit app if a demo is needed. |
| India / ABDM / DPDP framing | MDs 05 §7, 06 §4 | **[3] NOT IMPLEMENTED (N/A code)** | Presentation content, not code. | Presentation-only. |
| Publication roadmap (5 papers) | MDs 07 §7, 08 §2 | **[3] NOT IMPLEMENTED (N/A code)** | Strategy content, not code. | N/A. |

## 11. Inference pipeline (mostly from the rebuild brief, not original MDs)

| Feature / Claim | Source | Status | Evidence / Explanation | To complete |
|---|---|---|---|---|
| Single nemotron-parse API call (image → text) | rebuild brief | **[4] PARTIAL / UNVERIFIED** | `inference/nemotron_parse_client.py` is fully written against NVIDIA's documented OpenAI-compatible API, but **has never been called successfully end-to-end** (no API key supplied; key was rotated). Response-shape handling is defensive/best-effort. | Run one real call with a valid key + real scanned report. |
| Deterministic field-mapping (parsed text → training schema) | rebuild brief | **[4] PARTIAL** | `inference/field_mapping.py` extracts ~10 fields via regex, tuned to a synthetic example (unit-tested on that). Does NOT cover `resting_ecg`, `st_slope`, `thalassemia`, `height/weight`, free-text. | Expand regex coverage; test on real parse output. |
| Local inference: question + report → answer + incidental findings + grounded explanation | rebuild brief | **[4] PARTIAL** | `infer.run_inference` builds the prompt and generates, but the "other findings + grounded explanation" behavior is **prompt-elicited, not enforced/verified**; depends on a (barely) fine-tuned model. Full path never run end-to-end (blocked on parse call). | Verify after a real training run + a real parse call. |
| TensorRT-LLM inference optimization / NIM serving | MDs 02, 05 §5, 07 §9 | **[3] NOT IMPLEMENTED** | Neither used. | Out of scope for local design. |
| NeMo Curator / NeMo Guardrails | MDs 05 §5, 07 §9 | **[3] NOT IMPLEMENTED** | Not used. | Out of scope. |

---

## Summary counts

- **[1] Fully implemented (matches design intent):** FedRand split, Dirichlet non-IID partition, entropy-aware importance + weighted aggregation, 4-bit loading, gradient clipping, PTB-XL/UCI CSV loaders, sequential simulation. (~7 core items)
- **[2] Implemented but deviated:** model (30B hybrid → dense 4B), LoRA targets/rank, Laplace clip (L1→L2), quantization scheme (adaptive-bit → fixed 2-bit), trust agent (2 instances → 1 with `disable_adapter`).
- **[3] Not implemented:** NVFlare, NeMo 2.0, Mamba/MoE + SSM-LoRA + fused-kernel strategy, MIMIC/MedQA/PubMedQA, FedPS, tokenizer protocol, MTAE+OCSVM, cosine-annealing downlink quant, RDP/composition/budget, secure aggregation, GIA/MIA attack demos, all accuracy/comm/forgetting/privacy-utility evaluations, Streamlit dashboard, TensorRT-LLM/NIM/Curator/Guardrails. (largest bucket)
- **[4] Partial / unverified:** nemotron-parse call, field-mapping coverage, end-to-end inference, multi-round-at-scale, quantization unbiasedness, user-level DP framing, trust calibration.

**Net honest read:** the current system is a working *core federated-training loop* (FedRand + Laplace DP + 2-bit quant + entropy/trust-weighted aggregation on a dense 4B model, single GPU) with a *written-but-unproven inference path*. The privacy *theory*, the *attack validation*, and *all quantitative evaluation* from the original design are absent — which means the project's headline claims (privacy protection, communication savings, model quality) are currently **not yet measured or verified**.

---

## TO BE IMPLEMENTED NEXT — prioritized

Ranked by what makes the *current* system credible, demonstrable, and defensible — not by old-doc box-checking.

**P0 — Makes existing claims verifiable (do these first):**

1. **Run a real full-scale training run** (e.g. `--dataset ptbxl --limit 2000 --rounds 5`, larger `LOCAL_MAX_STEPS`, all 5 clients) and save the adapter + `training_report.json`.
   *Why #1:* everything else (accuracy, inference, privacy-utility) depends on a genuinely trained adapter. Right now only 1-round, 2–3-step smoke tests exist. Cheap, unblocks the rest. Low risk.

2. **Produce a held-out accuracy number.** Add an eval harness: hold out a labelled split, run `generate` + `parse_model_output`, report per-class accuracy/F1 for PTB-XL (5-class) and UCI.
   *Why #2:* the project currently has **zero** quality evidence. A single honest accuracy number (even if modest) converts "it trains" into "it works." Directly addresses the biggest defensibility gap.

3. **Run one real nemotron-parse call end-to-end** with a valid key and a real scanned report image, then fix `field_mapping` against the actual returned markdown.
   *Why #3:* the entire inference story is unverified. One successful call de-risks the single external dependency and validates the response-shape assumptions (currently guesses). Then expand field-mapping coverage (P1).

**P1 — Strengthens defensibility of the core contribution:**

4. **Communication-savings measurement.** Add byte-accounting: transmitted (2-bit, FedRand-halved) payload vs a 32-bit full-adapter FedAvg baseline. This is a concrete, easy, honest number that backs a headline claim (docs claimed 75%+).
   *Why:* cheap, quantitative, directly supports a stated selling point; no model quality needed.

5. **Expand field-mapping regex coverage** to the remaining schema fields (`resting_ecg`, `st_slope`, `thalassemia`, height/weight) and test against real parse output from P0#3.
   *Why:* makes the inference path actually usable on real reports, not just the synthetic test string.

6. **Trust-agent validation.** Inject a known-bad client (e.g. shuffled labels or exploding update) and confirm it receives a low trust score and reduced aggregation weight, across several runs. Report the score distribution.
   *Why:* the trust agent is a novel current-system feature but is uncalibrated; a small controlled experiment turns "noisy scores" into "demonstrably gates bad updates."

**P2 — Closes the biggest theory gap from the original docs:**

7. **Multi-round DP composition (RDP).** Integrate Google `dp-accounting`, track ε_total across rounds, expose a budget ceiling. Also switch the DP clip from L2 to L1 for a formally correct (ε,0)-DP per-round claim.
   *Why:* the original docs called per-round-only DP "the single most credibility-destroying omission." It's the difference between "we add noise" and "we have a stated total privacy budget."

**P3 — High-effort, demo/impact, only if targeting the original pitch:**

8. **A minimal GIA reconstruction check** (attempt to reconstruct an input from an unprotected vs FedRand+DP update; report a simple similarity metric). Even a small honest version substantiates the central privacy claim; the full live Streamlit attack demo is optional polish.
9. **Privacy-utility curve** (accuracy vs ε sweep) — depends on P0#2 + P2#7.
10. **Streamlit dashboard** — presentation-only; lowest priority for correctness/defensibility.

**Explicitly deprioritized (do NOT do unless the target changes):** porting to NVFlare/NeMo, adopting the 30B Mamba-2 model, MIMIC/MedQA datasets, FedPS, MTAE+OCSVM, secure aggregation. These belong to the *old* architecture and conflict with the current local, single-GPU, CSV-only design.

---

*Every status above is traceable to code presence/absence in `v1/fednemo/` and to the cited MDs. Items marked N/A are inapplicable because of the deliberate architecture change (dense model, no FLARE/NeMo), not because they were missed.*
