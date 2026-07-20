# FedNeMo — Results (Symptom2Disease)

Federated fine-tuning of **Nemotron-Mini-4B-Instruct** across **5 nodes**, with
**FedRand** split-adapter privacy, **Laplace differential privacy**, **8-bit
quantized** transmission, and **entropy + trust-weighted aggregation** — all
active. Evaluated on a stratified **held-out set never seen by any node**.

## Headline

| Metric | Value |
|---|---|
| **Accuracy** | **87.5%** (168/192) |
| **Macro-F1** | **0.862** |
| **Unparseable outputs** | **0.0%** |
| Held-out size | 192 (stratified, 24 classes × 8) |
| Classes | 24 diseases |
| Random baseline | ~4% |
| Reference (centralized fine-tuned BERT) | ~89% |

This lands ~1.5 points below a *centralized, non-private* BERT — while running
**federated + differentially private + quantized**. That gap is the honest cost
of privacy, and it is small.

## Configuration

| Setting | Value |
|---|---|
| Base model | nvidia/Nemotron-Mini-4B-Instruct (4-bit NF4 body on GPU) |
| Nodes | 5 (trained strictly sequentially, single 6 GB GPU) |
| Data split | IID, stratified, balanced (~200 records/node) |
| Rounds | 4 |
| Per-round coverage | full shard, shuffled each round |
| LoRA | rank 16, alpha 32 (23.1M trainable params, 0.55%) |
| FedRand | Bernoulli(0.5) A/B split per layer per round |
| DP | Laplace, relative mode, noise ≈ 25% of signal RMS |
| Distributed DP | on (aggregate noise reduced ~2.24× by 5-node averaging) |
| Quantization | 8-bit adaptive per-tensor |
| Privacy budget | ε_total(RDP) ≈ 16.0 over 4 rounds (δ=1e-5) |
| Decoding (eval) | constrained to the 24 valid labels |

## The journey (why 87.5%, not 63%)

An earlier run scored 63% / macro-F1 0.61. The gains came from fixing real issues:

1. **Full-shard coverage + shuffle** — previously each node trained on only the
   *same first 60* of its ~200 records every round (140 never seen). Now every
   record is used, reshuffled each round. **Biggest single lever.**
2. **Recovered a wrongly-excluded node** — the trust agent was zeroing one node's
   entire contribution due to a parsing misfire; a heuristic blend fixed it, so
   all 5 nodes now contribute (~0.20 weight each).
3. **LoRA rank 8 → 16** — doubled adaptation capacity.
4. **4 rounds + distributed-DP noise averaging.**
5. **Constrained decoding** — 0% unparseable, always a valid class.

## Per-class highlights (8 test cases each)

- **Perfect (F1 = 1.00):** Arthritis, Bronchial Asthma, Cervical spondylosis,
  Dimorphic Hemorrhoids, Jaundice, Varicose Veins, urinary tract infection
- **Excellent (F1 ≥ 0.9):** Acne, Malaria, Pneumonia, Typhoid,
  gastroesophageal reflux disease, peptic ulcer disease
- **Strong:** Migraine 0.89, Chicken pox 0.86, Psoriasis 0.82, diabetes 0.80,
  Dengue 0.80
- **Weak:** allergy 0.55, drug reaction 0.63, **Common Cold 0.00** (confused with
  Pneumonia / Bronchial Asthma — genuinely overlapping symptom text)

Full per-class metrics: [`results/eval_final.json`](results/eval_final.json).

## Honest notes

- This is the **private, federated** number, not a centralized upper bound.
- ε_total ≈ 16 (RDP) is a *moderate* privacy budget; tightening ε would trade
  some accuracy (privacy–utility curve).
- The trust agent falls back to a deterministic heuristic on the 24-class
  summaries (the base model's free-text score doesn't always parse); the blend
  keeps aggregation balanced regardless.
- Common Cold at F1 0.0 is a real weakness worth noting, not hidden.

## Reproduce

```
pip install -r requirements.txt
python -m fednemo.scripts.run_training --tag final --rounds 4 --quant-bits 8 --max-seq-len 160
python -m fednemo.scripts.run_eval --tag final --max-new-tokens 16
```
(Requires the local Nemotron-Mini-4B checkpoint; see README.)
