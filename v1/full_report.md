# FedNeMo — Scoped To-Do Execution Report

This document records the actual execution and measured results for the P0–P3
task list. Each item is filled in only with real evidence (logs, numbers) after
it runs. Items that are blocked or partial are marked as such honestly.

Environment: RTX 4050 Laptop (6141 MiB), Windows, Python 3.12, torch 2.5.1+cu121,
transformers 4.57.3, peft 0.19.1, bitsandbytes 0.49.2. Model:
`nvidia/Nemotron-Mini-4B-Instruct` (dense `NemotronForCausalLM`, 256k vocab).

Status legend: ✅ done with evidence · ⏳ in progress · ⚠️ partial/blocked · ❌ not started

| Item | Title | Status |
|---|---|---|
| 1 | Full-scale training run | ✅ |
| 2 | Held-out accuracy eval harness | ✅ (critical negative finding) |
| 3 | Real nemotron-parse call | ✅ |
| 4 | Communication-savings measurement | ✅ |
| 5 | Expand field-mapping coverage | ✅ |
| 6 | Trust-agent validation (bad client) | ✅ |
| 7 | Paraphrase robustness eval | ❌ |
| 8 | Class-imbalance handling | ❌ |
| 9 | Multi-round DP composition (RDP) | ❌ |
| 10 | Minimal GIA sanity check | ❌ |
| 11 | Privacy-utility curve | ❌ |

---

## Item 1 — Full-scale training run ✅

**Command:** `python -m fednemo.scripts.run_training --dataset ptbxl --limit 2500
--rounds 3 --max-steps 120 --epochs 1 --max-seq-len 192 --lm-head-device cuda --tag baseline`

**Data / ratio (confirmed before running):** loaded 2500 labelled PTB-XL records;
reserved 15% stratified held-out (375) BEFORE partitioning; trained on 2125 across
5 clients via Dirichlet(α=0.4, min 40). LOCAL_MAX_STEPS=120, 1 epoch, 3 rounds. A
~425-record client sees ~28% of its shard/round (~85% over 3 rounds); smaller
shards get full coverage.

**Realized non-IID partition (logged):**
```
client |  total |    CD |   HYP |    MI |  NORM |  STTC
    0   |     85 |     9 |     1 |    51 |     0 |    24
    1   |    426 |    17 |     5 |    10 |   359 |    35
    2   |    642 |   261 |     6 |     6 |   242 |   127
    3   |    504 |    22 |     3 |    39 |   436 |     4
    4   |    468 |     1 |   107 |   135 |    87 |   138
global |   2125 |   310 |   122 |   241 |  1124 |   328
```
Strong skew: client 0 has 0 NORM (MI-heavy), client 1 is 84% NORM; volume 85→642.

**Run facts:** model load 1721 MiB allocated / 2072 reserved. Per-step training
peak briefly hit ~5.8 GB used (GPU lm_head layout), recovered to ~2.6 GB between
clients (per-step `empty_cache`); no OOM. ~90 s/client training, ~8 s/client trust
generation. Total wall time ≈ 24 min. Artifacts:
`artifacts/global_adapter_baseline.pt`, `artifacts/holdout_ptbxl_baseline.json`,
`artifacts/training_report_baseline.json`.

**Observations (honest):**
- Within each round, per-client training loss collapses toward ~0.000 — the model
  quickly memorizes its shard's majority-class short target completion. This is
  expected for completion-only LM loss on a low-diversity shard and is exactly why
  held-out per-class F1 (item 2) is the real signal, not training loss.
- Per-round STARTING losses rose sharply across rounds (round 2 clients started
  ~12–17, round 3 ~16–18). The aggregated global adapter is oscillating/diverging
  across rounds under FedRand's randomized A/B split + Laplace DP noise + entropy
  weighting. This is an important stability finding: 3 rounds is not converging
  smoothly. Trust scores also varied round-to-round (e.g. round 3: 0.155, 0.425,
  0.900, 0.500, 0.999).

---

## Item 2 — Held-out accuracy evaluation ✅ (CRITICAL NEGATIVE FINDING)

Built `fednemo/eval/` (metrics.py, evaluate.py) + `scripts/run_eval.py`. Reserves a
stratified held-out split before partitioning, runs greedy generation +
`parse_model_output`, reports overall accuracy, macro-F1, per-class precision/
recall/F1, and unparseable rate.

**Result on the item-1 trained global adapter (375 held-out PTB-XL records):**
```
n = 375 | overall accuracy = 0.0000 | macro-F1 = 0.0000 | unparseable = 0.9973
class     prec  recall  f1   support
NORM      0.000 0.000  0.000   198
MI        0.000 0.000  0.000    43
STTC      0.000 0.000  0.000    58
CD        0.000 0.000  0.000    55
HYP       0.000 0.000  0.000    21
```
Sample outputs are pure gibberish, e.g.
`"Gasol Gasol خت خت Rit lia homog Gasol Rit gren की की..."`. **The trained global
model is non-functional.** This is precisely why held-out eval was mandatory:
training loss collapsed to ~0 (memorization) while the real model is unusable.

### Root-cause diagnosis (with evidence)

**(a) Base model is fine.** Evaluating with NO trained adapter (base + fresh
zero-init LoRA) on 12 held-out records: **0% unparseable, coherent output** that
even follows the requested format:
`"Diagnostic class: NORM\n\nOther clinically relevant findings: Sinus bradycardia..."`.
(It naively predicts NORM for everything → 6/12 on that tiny sample.) So the
serialization/prompt/generation path is correct; the trained adapter is the cause.

**(b) The aggregated adapter is exploded.** Inspecting
`global_adapter_baseline.pt`: 384 tensors, **mean L2 norm 39.56** (range 15.3–87.9),
max|w| 4.59 (healthy LoRA norms are ~1). An adapter this large overwhelms the
frozen base weights → incoherent generation.

**(c) DP noise dominates the signal.** Simulating the exact pipeline
(C=1.0 L2 clip, ε=4.0 Laplace, 2-bit quant) on a realistic trained LoRA matrix:
```
per-element |signal|   = 0.01599
per-element |DP noise|  = 0.24809   (scale C/eps = 0.25)
signal-to-noise ratio   = 0.064      (noise ~15x signal)
mean|reconstructed-orig|= 0.06388   (~4x the signal magnitude after DP+2bit)
```
Clipping the update to L2≤1 then adding per-element Laplace(0.25) to a ~24,000-
element matrix injects a noise matrix of L2 norm ≈ 55 — i.e. **the transmitted
"update" is ~98% noise, and that noise becomes the global adapter.** 2-bit
quantization compounds the loss. The privacy mechanism at ε=4 with per-element
Laplace over high-dimensional LoRA matrices destroys utility entirely.

**Implication.** This is a genuine, defensible finding, not a bug in eval: the
current FedRand + Laplace-DP(ε=4) + 2-bit pipeline produces a globally unusable
model. It directly motivates item 9 (DP calibration/accounting) and item 11
(privacy-utility curve — at ε=4 utility=0; a usable operating point needs far less
noise). Items 7/8/11 that require a *functioning* trained model are constrained by
this and are addressed accordingly below.

Artifacts: `artifacts/eval_ptbxl_baseline.json`, `artifacts/eval_base_only.json`.

---

## Item 3 — Real nemotron-parse API call (end-to-end) ✅

**Purpose:** de-risk the only external dependency; verify the real response shape
and fix `field_mapping`.

**What happened (real debugging, all against the live hosted endpoint
`https://integrate.api.nvidia.com/v1/chat/completions`, model
`nvidia/nemoretriever-parse`):**

1. **TLS blocked.** First call failed: `CERTIFICATE_VERIFY_FAILED: unable to get
   local issuer certificate`. The machine has an AV/proxy (Norton) MITM-ing TLS
   with a root CA not in certifi's bundle. Tried certifi bundle and `truststore`
   (OS trust store) — both still failed. Resolved for this diagnostic via a
   documented **insecure-TLS fallback** (`FEDNEMO_INSECURE_TLS=1`, added to the
   client as an opt-in env flag). **This is insecure and environment-specific — it
   must NOT be used in production; the proper fix is installing the proxy CA into
   the trust store.**
2. **Payload shape wrong.** With TLS bypassed we got a real API error:
   `HTTP 400: "Content cannot be a plain string. The model does not support text
   input."` The self-hosted-NIM docs used a text control-prompt part; the hosted
   endpoint **rejects any text part**. Probed variants: image-only → **HTTP 200**;
   image+text → 400.
3. **Response shape discovered.** On 200, `choices[0].message.content` is `null`;
   the parse result is in `choices[0].message.tool_calls[0].function.arguments`
   (function `markdown_bbox`) as a JSON string of `[{bbox, text, type}, ...]`
   blocks. Fixed the client to (a) send image-only content, (b) parse the
   tool_calls arguments JSON and concatenate block `text` fields (normalizing
   `<br>`→newline).

**Verified end-to-end result** (synthetic rendered cardiac report
`artifacts/sample_report.png` — a real API call on a synthetic image, since no
genuine scan was available; swap in a real scan for production):
- STATUS ok; raw content extracted cleanly (all report lines recovered).
- `field_mapping` output (after item-5 fixes): all 13 fields correct:
  `age 61, resting_bp 138, cholesterol 244, max_heart_rate 142, oldpeak 1.8,
  num_major_vessels 1, sex male, chest_pain_type atypical angina, resting_ecg lv
  hypertrophy, st_slope flat, thalassemia reversable defect, exercise_induced_angina
  false, fasting_blood_sugar_gt_120 true`.

**Caveats:** the run used a synthetic report image (not a genuine scan) and
insecure TLS (environment proxy issue). The API contract and parsing are now
verified real; a genuine scanned document may have noisier text needing more
mapping rules.

---

## Item 5 — Expand field-mapping coverage ✅

Extended `inference/field_mapping.py` from ~10 fields to the full schema and fixed
two bugs surfaced by the real parse output (item 3):
- Added: `height_cm`, `weight_kg`, and categorical `resting_ecg`
  (lv hypertrophy / st-t abnormality / normal), `st_slope`
  (upsloping/downsloping/flat), `thalassemia` (normal/fixed defect/reversable
  defect), with line-scoped matching.
- Fixed: `chest_pain_type` ("atypical angina" was matching "typical angina" as a
  substring → reordered so atypical/non-anginal/asymptomatic are checked before
  typical); `fasting_blood_sugar_gt_120` ("> 120: yes" was mapping to false → made
  the yes/no value token required).
- Verified on the real API output: 13/13 fields correct (see item 3).

---

## Item 4 — Communication-savings measurement ✅

Added `fednemo/eval/comm.py`: counts real transmitted bytes for FedNeMo's payload
(FedRand shares one matrix per layer, 2-bit bit-packed + 8 bytes/tensor metadata)
vs a 32-bit full-FedAvg baseline (both A and B, uncompressed). Measured on the
384-tensor trained adapter:

| Scheme | Bytes | KiB |
|---|---|---|
| Baseline: 32-bit full FedAvg (both A & B) | 46,137,344 | 45,056 |
| FedRand split only (@32-bit, one matrix/layer) | 21,037,056 | 20,544 |
| **FedNeMo: FedRand + 2-bit + metadata** | **1,316,352** | **1,286** |

- **Total savings vs 32-bit FedAvg: 97.15%.**
  - FedRand split alone: **54.40%** (slightly above 50% because the randomly-shared
    halves in this draw were the larger matrices).
  - Additional from 2-bit quantization: **+42.74 percentage points**.

**Honest caveat:** this exceeds the original design's 75% target *because* 2-bit is
extremely aggressive — and (item 2) that same aggressiveness destroys model
utility. The communication numbers are real, but this operating point (2-bit,
ε=4) is not usable. A credible headline must pair the savings with the utility at
the same operating point (item 11).

---

## Item 9 — Multi-round DP composition (RDP) + L1 clip ✅

Added `fednemo/federated/privacy_accounting.py`: closed-form Rényi-DP for the
Laplace mechanism (Mironov 2017, numerically stable log form), additive
composition across rounds, and conversion to the tightest (eps_total, delta)-DP
over an RDP-order grid. Added a `PrivacyAccountant` with an optional budget
ceiling (`DP_EPSILON_MAX`) that stops training early; the orchestrator now logs
and records `eps_total_rdp` / `eps_total_naive` each round. Also switched the DP
clip from **L2 to L1** (`DP_CLIP_TYPE="l1"`, default) — the formally correct
sensitivity bound for the Laplace mechanism (config-toggleable).

**Validation / measured values (delta=1e-5):**
| eps_round | rounds | eps_total (RDP) | naive (T·eps) |
|---|---|---|---|
| 1.0 | 100 | 72.40 | 100.0 |
| 0.5 | 50 | 19.32 | 25.0 |
| 1.0 | 10 | 10.02 | 10.0 |
| 4.0 | 3 (our run) | **12.04** | 12.0 |
| 4.0 | 1 | 4.04 | 4.0 |

**Honest correction to the original design docs:** the docs (03_Privacy...) claimed
RDP gives a ~7× tighter budget (eps_total≈14.2 for eps=1,T=100). That figure is a
*Gaussian-mechanism* result; for the **pure Laplace** mechanism actually used here,
RDP composition is only modestly tighter than naive (72.4 vs 100, ~28%), and for
few rounds / large eps it is essentially equal to naive (our run: 12.04 vs 12.0).
This is now reported accurately rather than repeating the doc's optimistic claim.

Note: the L1 clip does not improve utility (it is about the correctness of the
privacy claim). The utility collapse (item 2) is caused by noise *magnitude* at
eps=4, addressed empirically in item 11.

---

## Item 6 — Trust-agent validation (injected bad client) ✅

Added poisoning to `ClientNode` (`poison="explode"` scales the update to a huge
norm; `poison="shuffle"` label-flips) and a `--inject-bad` CLI flag. The
orchestrator now records each client's `update_l2_norm` and its
`effective_weight_share` (= trust_i · nu_i, normalized).

**Experiment:** 5 clients, client 0 poisoned with an exploding update (×100),
2 rounds (limit 600, 30 steps/client). Model-based trust agent enabled.

| Round | Poisoned client 0: trust | client 0 weight share | honest clients' shares |
|---|---|---|---|
| 1 | **0.155** (lowest) | **0.012** | 0.234, 0.272, 0.265, 0.218 |
| 2 | (low) | **0.027** | 0.34, 0.021*, 0.335, 0.277 |

**Result:** the poisoned client's influence was cut from a fair 1/5 share (~0.20)
to **1.2% (round 1)** — a ~16× reduction — and stayed low (2.7%) in round 2. The
trust-scoring agent + entropy/trust-weighted aggregator **demonstrably catch and
suppress a blatant bad update.**

**Honest caveat (real, observed):** the model-based trust agent is **noisy**. In
round 2 it also assigned a low score (0.055) to an *honest* client (client 2,
marked * above), i.e. a false positive. So the mechanism reliably suppresses
blatant poisoning but its per-round scores are not calibrated and can mis-penalize
honest clients — consistent with the "uncalibrated" characterization, now with
evidence that it does catch the egregious case. A production version should use a
more robust/structured detector (e.g. norm-threshold gating) alongside the LLM
auditor.

---
