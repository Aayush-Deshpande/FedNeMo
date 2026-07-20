# FedNeMo — Detailed Run Report  (`final`)

Auto-generated from the audit trail, training report, and held-out evaluation.

## 1. Federation configuration

| Field | Value |
|---|---|
| Nodes (clients) | 5 |
| Rounds | 4 |
| Steps per node/round | full shard each round (max_steps=0) |
| Local epochs | 1 |
| Gradient accumulation | n/a |
| LoRA rank / alpha | 16 / 32 |
| FedRand share prob (ρ) | 0.5 |
| DP mode / ε per round | relative / 4.0 |
| Quantization bits | 8 |
| lm_head device | cuda |

## 2. Dataset

- **File:** `Symptom2Disease.csv`
- **Total records:** 1200
- **Classes:** 24
- **Columns:** ['Unnamed: 0', 'label', 'text']

**dataset.head():**

| # | label | text |
|---|---|---|
| 0 | Psoriasis | I have been experiencing a skin rash on my arms, legs, and torso for the past few weeks. I... |
| 1 | Psoriasis | My skin has been peeling, especially on my knees, elbows, and scalp. This peeling is often... |
| 2 | Psoriasis | I have been experiencing joint pain in my fingers, wrists, and knees. The pain is often ac... |
| 3 | Psoriasis | There is a silver like dusting on my skin, especially on my lower back and scalp. This dus... |
| 4 | Psoriasis | My nails have small dents or pits in them, and they often feel inflammatory and tender to ... |

## 3. Data split across nodes (IID, balanced)

| Node | Records | Class distribution |
|---|---|---|
| 0 | 216 | Acne:9, drug reaction:9, urinary tract infection:9, Dengue:9, Jaundice:9, Bronchial Asthma:9 ... |
| 1 | 216 | Dimorphic Hemorrhoids:9, Cervical spondylosis:9, Hypertension:9, Chicken pox:9, Common Cold:9, Jaundice:9 ... |
| 2 | 192 | urinary tract infection:8, Bronchial Asthma:8, Fungal infection:8, Varicose Veins:8, Acne:8, Impetigo:8 ... |
| 3 | 192 | urinary tract infection:8, diabetes:8, Malaria:8, Typhoid:8, peptic ulcer disease:8, Impetigo:8 ... |
| 4 | 192 | Dengue:8, drug reaction:8, urinary tract infection:8, Dimorphic Hemorrhoids:8, peptic ulcer disease:8, Chicken pox:8 ... |

## 4. Held-out accuracy

- **Accuracy:** 87.50%
- **Macro-F1:** 0.8624
- **Unparseable:** 0.00%
- **Held-out N:** 192

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Acne | 0.889 | 1.000 | 0.941 | 8 |
| Arthritis | 1.000 | 1.000 | 1.000 | 8 |
| Bronchial Asthma | 1.000 | 1.000 | 1.000 | 8 |
| Cervical spondylosis | 1.000 | 1.000 | 1.000 | 8 |
| Chicken pox | 1.000 | 0.750 | 0.857 | 8 |
| Common Cold | 0.000 | 0.000 | 0.000 | 8 |
| Dengue | 0.667 | 1.000 | 0.800 | 8 |
| Dimorphic Hemorrhoids | 1.000 | 1.000 | 1.000 | 8 |
| Fungal infection | 1.000 | 0.875 | 0.933 | 8 |
| Hypertension | 0.857 | 0.750 | 0.800 | 8 |
| Impetigo | 1.000 | 1.000 | 1.000 | 8 |
| Jaundice | 1.000 | 1.000 | 1.000 | 8 |
| Malaria | 1.000 | 0.875 | 0.933 | 8 |
| Migraine | 0.800 | 1.000 | 0.889 | 8 |
| Pneumonia | 0.889 | 1.000 | 0.941 | 8 |
| Psoriasis | 0.778 | 0.875 | 0.824 | 8 |
| Typhoid | 1.000 | 0.875 | 0.933 | 8 |
| Varicose Veins | 1.000 | 1.000 | 1.000 | 8 |
| allergy | 0.429 | 0.750 | 0.545 | 8 |
| diabetes | 0.857 | 0.750 | 0.800 | 8 |
| drug reaction | 0.625 | 0.625 | 0.625 | 8 |
| gastroesophageal reflux disease | 1.000 | 0.875 | 0.933 | 8 |
| peptic ulcer disease | 0.889 | 1.000 | 0.941 | 8 |
| urinary tract infection | 1.000 | 1.000 | 1.000 | 8 |

## 5. Per-round summary (trust, weights, privacy budget)

| Round | Trust (per node) | Effective weight share | ε_total (RDP) |
|---|---|---|---|
| 1 | 0:0.90, 1:0.90, 2:0.90, 3:0.90, 4:0.90 | 0:0.21, 1:0.21, 2:0.20, 3:0.20, 4:0.20 | 4.04 |
| 2 | 0:0.66, 1:0.48, 2:0.90, 3:0.64, 4:0.67 | 0:0.20, 1:0.15, 2:0.26, 3:0.19, 4:0.20 | 8.04 |
| 3 | 0:0.65, 1:0.68, 2:0.65, 3:0.65, 4:0.63 | 0:0.21, 1:0.22, 2:0.20, 3:0.19, 4:0.19 | 12.04 |
| 4 | 0:0.65, 1:0.65, 2:0.65, 3:0.64, 4:0.65 | 0:0.21, 1:0.21, 2:0.20, 3:0.19, 4:0.20 | 16.03 |

## 6. Gradient pipeline per node, per round

For each node/round: local training result, FedRand A/B split counts, DP noise added, and a **representative layer** shown through the full pipeline (trained → after-DP → after-quantization), including which matrix was **sent** vs **dropped**. (Full per-layer detail for all 192 layers lives in `artifacts/audit_final/node_*/round_*/report.json`.)

### Node 0

**Round 1** — trained 216 records, loss 3.4244484901428223 → 0.0874861627817154, train time 158.1s | FedRand: sent A=100 / B=92 | DP mode=relative noise_ratio=0.25 update‖·‖₂=23.333

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent B**, dropped A
  - trained A: shape=[16, 9216] l2=2.3329 absμ=0.00525 min=-0.0130 max=0.0128  sample [+0.00180, +0.00200, -0.00322, +0.00712, +0.00970, -0.00214 ...]
  - trained B: shape=[3072, 16] l2=0.1988 absμ=0.00073 min=-0.0038 max=0.0031  sample [+0.00061, -0.00153, -0.00130, -0.00098, +0.00003, -0.00001 ...]
  - after DP (noise_scale=0.0002242, signal_rms=0.00089679): shape=[3072, 16] l2=0.2106 absμ=0.00077 min=-0.0039 max=0.0035 sample [+0.00068, -0.00158, -0.00112, -0.00095, +0.00013, +0.00024 ...]
  - after quant (8-bit, 256 levels, scale=2.897e-05): shape=[3072, 16] l2=0.2106 absμ=0.00077 min=-0.0039 max=0.0035 sample [+0.00067, -0.00156, -0.00113, -0.00096, +0.00012, +0.00023 ...]

**Round 2** — trained 216 records, loss 0.018001558259129524 → 0.0006406735628843307, train time 157.07s | FedRand: sent A=97 / B=95 | DP mode=relative noise_ratio=0.25 update‖·‖₂=23.444

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent A**, dropped B
  - trained A: shape=[16, 9216] l2=2.3476 absμ=0.00527 min=-0.0141 max=0.0148  sample [+0.00099, +0.00173, -0.00378, +0.00646, +0.00797, -0.00381 ...]
  - trained B: shape=[3072, 16] l2=0.2364 absμ=0.00086 min=-0.0043 max=0.0046  sample [+0.00040, -0.00158, -0.00038, +0.00006, -0.00009, -0.00059 ...]
  - after DP (noise_scale=0.0015284, signal_rms=0.00611361): shape=[16, 9216] l2=2.4899 absμ=0.00549 min=-0.0250 max=0.0237 sample [+0.00205, +0.00308, -0.00778, +0.00578, +0.00976, -0.00401 ...]
  - after quant (8-bit, 256 levels, scale=0.00019133): shape=[16, 9216] l2=2.4901 absμ=0.00549 min=-0.0251 max=0.0237 sample [+0.00211, +0.00306, -0.00784, +0.00574, +0.00976, -0.00402 ...]

**Round 3** — trained 216 records, loss 0.004685115069150925 → 0.00039352549356408417, train time 121.5s | FedRand: sent A=104 / B=88 | DP mode=relative noise_ratio=0.25 update‖·‖₂=24.630

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent A**, dropped B
  - trained A: shape=[16, 9216] l2=2.5082 absμ=0.00552 min=-0.0253 max=0.0247  sample [+0.00385, +0.00468, -0.00604, +0.00698, +0.01178, -0.00224 ...]
  - trained B: shape=[3072, 16] l2=0.2968 absμ=0.00107 min=-0.0053 max=0.0058  sample [+0.00214, -0.00249, -0.00100, +0.00057, +0.00026, -0.00087 ...]
  - after DP (noise_scale=0.00163293, signal_rms=0.00653171): shape=[16, 9216] l2=2.6568 absμ=0.00577 min=-0.0276 max=0.0300 sample [+0.00540, +0.00589, -0.00743, +0.00715, +0.01162, -0.00075 ...]
  - after quant (8-bit, 256 levels, scale=0.00022591): shape=[16, 9216] l2=2.6569 absμ=0.00577 min=-0.0276 max=0.0300 sample [+0.00542, +0.00587, -0.00745, +0.00723, +0.01152, -0.00068 ...]

**Round 4** — trained 216 records, loss 0.0061029065400362015 → 0.0005249664536677301, train time 87.98s | FedRand: sent A=95 / B=97 | DP mode=relative noise_ratio=0.25 update‖·‖₂=23.883

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent A**, dropped B
  - trained A: shape=[16, 9216] l2=2.4313 absμ=0.00540 min=-0.0175 max=0.0186  sample [-0.00054, +0.00081, -0.00604, +0.00493, +0.00888, -0.00210 ...]
  - trained B: shape=[3072, 16] l2=0.3384 absμ=0.00122 min=-0.0057 max=0.0062  sample [+0.00151, -0.00288, -0.00003, -0.00025, +0.00027, -0.00137 ...]
  - after DP (noise_scale=0.00158289, signal_rms=0.00633157): shape=[16, 9216] l2=2.5797 absμ=0.00564 min=-0.0265 max=0.0258 sample [+0.00174, +0.00245, -0.00671, +0.00396, +0.00959, -0.00204 ...]
  - after quant (8-bit, 256 levels, scale=0.00020535): shape=[16, 9216] l2=2.5799 absμ=0.00564 min=-0.0265 max=0.0259 sample [+0.00164, +0.00246, -0.00678, +0.00390, +0.00965, -0.00205 ...]

### Node 1

**Round 1** — trained 216 records, loss 2.361859083175659 → 0.07449667155742645, train time 140.5s | FedRand: sent A=111 / B=81 | DP mode=relative noise_ratio=0.25 update‖·‖₂=24.596

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent B**, dropped A
  - trained A: shape=[16, 9216] l2=2.3334 absμ=0.00525 min=-0.0127 max=0.0128  sample [+0.00228, +0.00251, -0.00399, +0.00740, +0.00909, -0.00244 ...]
  - trained B: shape=[3072, 16] l2=0.1983 absμ=0.00073 min=-0.0034 max=0.0032  sample [+0.00056, -0.00114, -0.00024, -0.00127, +0.00031, -0.00064 ...]
  - after DP (noise_scale=0.00022359, signal_rms=0.00089436): shape=[3072, 16] l2=0.2104 absμ=0.00077 min=-0.0035 max=0.0044 sample [+0.00093, -0.00114, -0.00024, -0.00115, +0.00022, -0.00062 ...]
  - after quant (8-bit, 256 levels, scale=3.098e-05): shape=[3072, 16] l2=0.2104 absμ=0.00077 min=-0.0035 max=0.0044 sample [+0.00093, -0.00115, -0.00025, -0.00115, +0.00022, -0.00062 ...]

**Round 2** — trained 216 records, loss 0.011993063613772392 → 0.355606347322464, train time 151.89s | FedRand: sent A=85 / B=107 | DP mode=relative noise_ratio=0.25 update‖·‖₂=22.020

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent B**, dropped A
  - trained A: shape=[16, 9216] l2=2.3538 absμ=0.00528 min=-0.0145 max=0.0143  sample [+0.00166, +0.00143, -0.00413, +0.00641, +0.00905, -0.00271 ...]
  - trained B: shape=[3072, 16] l2=0.2492 absμ=0.00090 min=-0.0042 max=0.0042  sample [+0.00168, -0.00110, -0.00094, -0.00054, -0.00012, -0.00027 ...]
  - after DP (noise_scale=0.00028105, signal_rms=0.00112418): shape=[3072, 16] l2=0.2642 absμ=0.00096 min=-0.0050 max=0.0046 sample [+0.00234, -0.00081, -0.00113, -0.00050, -0.00042, -0.00028 ...]
  - after quant (8-bit, 256 levels, scale=3.77e-05): shape=[3072, 16] l2=0.2642 absμ=0.00096 min=-0.0051 max=0.0046 sample [+0.00234, -0.00079, -0.00113, -0.00049, -0.00042, -0.00030 ...]

**Round 3** — trained 216 records, loss 0.0575442798435688 → 0.0005513275973498821, train time 102.56s | FedRand: sent A=99 / B=93 | DP mode=relative noise_ratio=0.25 update‖·‖₂=24.026

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent A**, dropped B
  - trained A: shape=[16, 9216] l2=2.3734 absμ=0.00531 min=-0.0158 max=0.0148  sample [+0.00180, +0.00112, -0.00498, +0.00679, +0.00819, -0.00221 ...]
  - trained B: shape=[3072, 16] l2=0.2713 absμ=0.00098 min=-0.0052 max=0.0049  sample [+0.00109, -0.00069, -0.00046, -0.00161, +0.00058, -0.00115 ...]
  - after DP (noise_scale=0.00154519, signal_rms=0.00618075): shape=[16, 9216] l2=2.5199 absμ=0.00554 min=-0.0240 max=0.0242 sample [-0.00308, +0.00225, -0.00311, +0.00589, +0.00646, -0.00018 ...]
  - after quant (8-bit, 256 levels, scale=0.00018892): shape=[16, 9216] l2=2.5201 absμ=0.00554 min=-0.0240 max=0.0242 sample [-0.00302, +0.00227, -0.00302, +0.00586, +0.00642, -0.00019 ...]

**Round 4** — trained 216 records, loss 0.00035704905167222023 → 0.0017705801874399185, train time 90.43s | FedRand: sent A=100 / B=92 | DP mode=relative noise_ratio=0.25 update‖·‖₂=24.450

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent A**, dropped B
  - trained A: shape=[16, 9216] l2=2.4306 absμ=0.00540 min=-0.0176 max=0.0188  sample [-0.00060, +0.00148, -0.00658, +0.00619, +0.00878, -0.00172 ...]
  - trained B: shape=[3072, 16] l2=0.3165 absμ=0.00114 min=-0.0063 max=0.0062  sample [+0.00171, +0.00012, -0.00048, -0.00146, +0.00030, -0.00267 ...]
  - after DP (noise_scale=0.00158241, signal_rms=0.00632963): shape=[16, 9216] l2=2.5795 absμ=0.00564 min=-0.0277 max=0.0241 sample [-0.00046, +0.00103, -0.00702, +0.00029, +0.00960, +0.00210 ...]
  - after quant (8-bit, 256 levels, scale=0.00020322): shape=[16, 9216] l2=2.5796 absμ=0.00564 min=-0.0276 max=0.0242 sample [-0.00041, +0.00102, -0.00711, +0.00020, +0.00955, +0.00203 ...]

### Node 2

**Round 1** — trained 192 records, loss 2.896134853363037 → 0.22085562348365784, train time 137.88s | FedRand: sent A=98 / B=94 | DP mode=relative noise_ratio=0.25 update‖·‖₂=23.104

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent B**, dropped A
  - trained A: shape=[16, 9216] l2=2.3316 absμ=0.00524 min=-0.0131 max=0.0128  sample [+0.00203, +0.00110, -0.00360, +0.00787, +0.00754, -0.00478 ...]
  - trained B: shape=[3072, 16] l2=0.2012 absμ=0.00074 min=-0.0033 max=0.0032  sample [+0.00167, -0.00001, -0.00194, -0.00046, +0.00040, +0.00064 ...]
  - after DP (noise_scale=0.0002269, signal_rms=0.00090759): shape=[3072, 16] l2=0.2131 absμ=0.00078 min=-0.0039 max=0.0035 sample [+0.00183, -0.00041, -0.00175, +0.00034, +0.00057, +0.00021 ...]
  - after quant (8-bit, 256 levels, scale=2.938e-05): shape=[3072, 16] l2=0.2131 absμ=0.00078 min=-0.0039 max=0.0036 sample [+0.00182, -0.00041, -0.00176, +0.00035, +0.00056, +0.00021 ...]

**Round 2** — trained 192 records, loss 0.5984511375427246 → 0.03954263776540756, train time 138.39s | FedRand: sent A=96 / B=96 | DP mode=relative noise_ratio=0.25 update‖·‖₂=23.267

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent B**, dropped A
  - trained A: shape=[16, 9216] l2=2.3456 absμ=0.00527 min=-0.0144 max=0.0137  sample [+0.00234, +0.00064, -0.00320, +0.00757, +0.00797, -0.00345 ...]
  - trained B: shape=[3072, 16] l2=0.2324 absμ=0.00084 min=-0.0042 max=0.0039  sample [+0.00194, -0.00052, -0.00196, -0.00070, -0.00002, -0.00019 ...]
  - after DP (noise_scale=0.00026209, signal_rms=0.00104836): shape=[3072, 16] l2=0.2464 absμ=0.00089 min=-0.0056 max=0.0043 sample [+0.00226, -0.00041, -0.00185, -0.00069, -0.00056, -0.00091 ...]
  - after quant (8-bit, 256 levels, scale=3.889e-05): shape=[3072, 16] l2=0.2464 absμ=0.00089 min=-0.0056 max=0.0043 sample [+0.00226, -0.00043, -0.00187, -0.00070, -0.00055, -0.00089 ...]

**Round 3** — trained 192 records, loss 0.003099133027717471 → 8.64566391101107e-05, train time 107.91s | FedRand: sent A=102 / B=90 | DP mode=relative noise_ratio=0.25 update‖·‖₂=24.325

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent B**, dropped A
  - trained A: shape=[16, 9216] l2=2.3601 absμ=0.00529 min=-0.0144 max=0.0145  sample [+0.00136, -0.00069, -0.00400, +0.00711, +0.00694, -0.00468 ...]
  - trained B: shape=[3072, 16] l2=0.2629 absμ=0.00095 min=-0.0054 max=0.0047  sample [+0.00122, -0.00159, -0.00112, -0.00120, -0.00075, -0.00107 ...]
  - after DP (noise_scale=0.00029645, signal_rms=0.00118581): shape=[3072, 16] l2=0.2783 absμ=0.00101 min=-0.0059 max=0.0054 sample [+0.00088, -0.00157, -0.00112, -0.00084, -0.00010, -0.00151 ...]
  - after quant (8-bit, 256 levels, scale=4.442e-05): shape=[3072, 16] l2=0.2783 absμ=0.00101 min=-0.0059 max=0.0054 sample [+0.00089, -0.00155, -0.00111, -0.00084, -0.00009, -0.00151 ...]

**Round 4** — trained 192 records, loss 0.00036133191315457225 → 1.1920926823449918e-07, train time 110.57s | FedRand: sent A=103 / B=89 | DP mode=relative noise_ratio=0.25 update‖·‖₂=24.686

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent A**, dropped B
  - trained A: shape=[16, 9216] l2=2.3717 absμ=0.00531 min=-0.0155 max=0.0159  sample [+0.00205, -0.00026, -0.00275, +0.00823, +0.00761, -0.00374 ...]
  - trained B: shape=[3072, 16] l2=0.2925 absμ=0.00106 min=-0.0054 max=0.0052  sample [+0.00120, -0.00152, -0.00096, -0.00093, +0.00025, -0.00145 ...]
  - after DP (noise_scale=0.00154407, signal_rms=0.00617629): shape=[16, 9216] l2=2.5130 absμ=0.00553 min=-0.0252 max=0.0291 sample [-0.00124, -0.00053, -0.00134, +0.00765, +0.00747, -0.00264 ...]
  - after quant (8-bit, 256 levels, scale=0.00021322): shape=[16, 9216] l2=2.5130 absμ=0.00553 min=-0.0252 max=0.0292 sample [-0.00128, -0.00043, -0.00128, +0.00768, +0.00746, -0.00256 ...]

### Node 3

**Round 1** — trained 192 records, loss 1.9267313480377197 → 0.3559474050998688, train time 115.24s | FedRand: sent A=97 / B=95 | DP mode=relative noise_ratio=0.25 update‖·‖₂=23.003

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent A**, dropped B
  - trained A: shape=[16, 9216] l2=2.3348 absμ=0.00525 min=-0.0129 max=0.0130  sample [+0.00140, +0.00152, -0.00422, +0.00708, +0.00876, -0.00421 ...]
  - trained B: shape=[3072, 16] l2=0.2033 absμ=0.00075 min=-0.0030 max=0.0030  sample [+0.00072, -0.00121, -0.00070, -0.00094, +0.00136, +0.00086 ...]
  - after DP (noise_scale=0.00152008, signal_rms=0.0060803): shape=[16, 9216] l2=2.4750 absμ=0.00547 min=-0.0224 max=0.0233 sample [-0.00030, -0.00165, -0.00382, +0.00733, +0.00756, -0.00308 ...]
  - after quant (8-bit, 256 levels, scale=0.00017912): shape=[16, 9216] l2=2.4750 absμ=0.00547 min=-0.0224 max=0.0233 sample [-0.00036, -0.00161, -0.00376, +0.00734, +0.00752, -0.00304 ...]

**Round 2** — trained 192 records, loss 0.002169873798266053 → 0.014321397058665752, train time 111.87s | FedRand: sent A=90 / B=102 | DP mode=relative noise_ratio=0.25 update‖·‖₂=22.481

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent B**, dropped A
  - trained A: shape=[16, 9216] l2=2.4123 absμ=0.00537 min=-0.0175 max=0.0189  sample [-0.00139, -0.00177, -0.00536, +0.00599, +0.00877, -0.00378 ...]
  - trained B: shape=[3072, 16] l2=0.2496 absμ=0.00091 min=-0.0042 max=0.0043  sample [+0.00043, -0.00103, -0.00026, -0.00022, +0.00055, +0.00087 ...]
  - after DP (noise_scale=0.0002815, signal_rms=0.00112599): shape=[3072, 16] l2=0.2657 absμ=0.00096 min=-0.0056 max=0.0048 sample [+0.00045, -0.00105, -0.00029, -0.00025, +0.00135, +0.00007 ...]
  - after quant (8-bit, 256 levels, scale=4.09e-05): shape=[3072, 16] l2=0.2657 absμ=0.00096 min=-0.0056 max=0.0048 sample [+0.00045, -0.00106, -0.00029, -0.00024, +0.00135, +0.00008 ...]

**Round 3** — trained 192 records, loss 0.0034884351771324873 → 0.0010133873438462615, train time 84.97s | FedRand: sent A=96 / B=96 | DP mode=relative noise_ratio=0.25 update‖·‖₂=23.621

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent B**, dropped A
  - trained A: shape=[16, 9216] l2=2.4276 absμ=0.00539 min=-0.0173 max=0.0204  sample [-0.00138, -0.00088, -0.00517, +0.00601, +0.00902, -0.00374 ...]
  - trained B: shape=[3072, 16] l2=0.2568 absμ=0.00093 min=-0.0045 max=0.0046  sample [+0.00231, -0.00188, -0.00165, -0.00120, -0.00038, -0.00075 ...]
  - after DP (noise_scale=0.00028952, signal_rms=0.0011581): shape=[3072, 16] l2=0.2715 absμ=0.00098 min=-0.0048 max=0.0050 sample [+0.00289, -0.00180, -0.00172, -0.00003, +0.00039, -0.00111 ...]
  - after quant (8-bit, 256 levels, scale=3.839e-05): shape=[3072, 16] l2=0.2715 absμ=0.00098 min=-0.0048 max=0.0050 sample [+0.00288, -0.00180, -0.00173, -0.00004, +0.00038, -0.00111 ...]

**Round 4** — trained 192 records, loss 0.00017523366841487586 → 0.014071057550609112, train time 117.08s | FedRand: sent A=99 / B=93 | DP mode=relative noise_ratio=0.25 update‖·‖₂=24.327

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent A**, dropped B
  - trained A: shape=[16, 9216] l2=2.4439 absμ=0.00542 min=-0.0178 max=0.0196  sample [-0.00129, -0.00076, -0.00540, +0.00619, +0.00923, -0.00360 ...]
  - trained B: shape=[3072, 16] l2=0.2960 absμ=0.00107 min=-0.0056 max=0.0054  sample [+0.00208, -0.00238, -0.00074, -0.00134, -0.00017, -0.00232 ...]
  - after DP (noise_scale=0.00159107, signal_rms=0.00636427): shape=[16, 9216] l2=2.5902 absμ=0.00566 min=-0.0238 max=0.0253 sample [-0.00236, -0.00055, -0.00489, +0.00644, +0.00779, -0.00418 ...]
  - after quant (8-bit, 256 levels, scale=0.00019255): shape=[16, 9216] l2=2.5903 absμ=0.00566 min=-0.0237 max=0.0254 sample [-0.00231, -0.00058, -0.00481, +0.00635, +0.00770, -0.00424 ...]

### Node 4

**Round 1** — trained 192 records, loss 3.913295269012451 → 0.3700585663318634, train time 136.85s | FedRand: sent A=97 / B=95 | DP mode=relative noise_ratio=0.25 update‖·‖₂=22.992

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent A**, dropped B
  - trained A: shape=[16, 9216] l2=2.3317 absμ=0.00525 min=-0.0128 max=0.0129  sample [+0.00261, +0.00181, -0.00347, +0.00755, +0.00811, -0.00371 ...]
  - trained B: shape=[3072, 16] l2=0.1947 absμ=0.00072 min=-0.0032 max=0.0032  sample [+0.00011, -0.00100, -0.00185, -0.00192, +0.00023, -0.00046 ...]
  - after DP (noise_scale=0.00151806, signal_rms=0.00607224): shape=[16, 9216] l2=2.4732 absμ=0.00546 min=-0.0211 max=0.0305 sample [-0.00135, -0.00104, -0.00586, +0.00658, +0.01059, -0.00476 ...]
  - after quant (8-bit, 256 levels, scale=0.00020257): shape=[16, 9216] l2=2.4734 absμ=0.00546 min=-0.0211 max=0.0306 sample [-0.00142, -0.00101, -0.00587, +0.00648, +0.01053, -0.00466 ...]

**Round 2** — trained 192 records, loss 0.037866510450839996 → 0.005390209145843983, train time 119.07s | FedRand: sent A=90 / B=102 | DP mode=relative noise_ratio=0.25 update‖·‖₂=22.556

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent B**, dropped A
  - trained A: shape=[16, 9216] l2=2.4136 absμ=0.00537 min=-0.0179 max=0.0211  sample [+0.00005, -0.00119, -0.00406, +0.00781, +0.00971, -0.00330 ...]
  - trained B: shape=[3072, 16] l2=0.2492 absμ=0.00090 min=-0.0047 max=0.0045  sample [+0.00120, -0.00113, -0.00254, -0.00256, +0.00023, -0.00048 ...]
  - after DP (noise_scale=0.00028105, signal_rms=0.00112421): shape=[3072, 16] l2=0.2641 absμ=0.00096 min=-0.0051 max=0.0049 sample [+0.00137, -0.00117, -0.00263, -0.00256, +0.00013, -0.00020 ...]
  - after quant (8-bit, 256 levels, scale=3.912e-05): shape=[3072, 16] l2=0.2641 absμ=0.00096 min=-0.0051 max=0.0049 sample [+0.00137, -0.00117, -0.00262, -0.00254, +0.00012, -0.00020 ...]

**Round 3** — trained 192 records, loss 0.0003527304797898978 → 0.04390665888786316, train time 82.55s | FedRand: sent A=81 / B=111 | DP mode=relative noise_ratio=0.25 update‖·‖₂=21.796

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent A**, dropped B
  - trained A: shape=[16, 9216] l2=2.4329 absμ=0.00540 min=-0.0180 max=0.0203  sample [-0.00082, -0.00218, -0.00428, +0.00642, +0.00893, -0.00329 ...]
  - trained B: shape=[3072, 16] l2=0.2636 absμ=0.00095 min=-0.0051 max=0.0046  sample [+0.00153, -0.00090, -0.00247, -0.00108, -0.00022, +0.00032 ...]
  - after DP (noise_scale=0.00158391, signal_rms=0.00633566): shape=[16, 9216] l2=2.5797 absμ=0.00564 min=-0.0251 max=0.0257 sample [-0.00442, -0.00480, -0.00742, +0.00590, +0.00882, -0.00364 ...]
  - after quant (8-bit, 256 levels, scale=0.00019898): shape=[16, 9216] l2=2.5798 absμ=0.00564 min=-0.0251 max=0.0257 sample [-0.00438, -0.00478, -0.00736, +0.00597, +0.00876, -0.00358 ...]

**Round 4** — trained 192 records, loss 0.00012674718163907528 → 0.0017838341882452369, train time 134.23s | FedRand: sent A=93 / B=99 | DP mode=relative noise_ratio=0.25 update‖·‖₂=23.522

- **Representative layer:** `base_model.model.model.layers.0.mlp.down_proj` — **sent B**, dropped A
  - trained A: shape=[16, 9216] l2=2.4309 absμ=0.00540 min=-0.0177 max=0.0191  sample [-0.00119, +0.00147, -0.00625, +0.00667, +0.00828, -0.00291 ...]
  - trained B: shape=[3072, 16] l2=0.3110 absμ=0.00112 min=-0.0062 max=0.0056  sample [+0.00187, -0.00054, -0.00310, -0.00101, -0.00031, +0.00103 ...]
  - after DP (noise_scale=0.00035065, signal_rms=0.0014026): shape=[3072, 16] l2=0.3304 absμ=0.00119 min=-0.0061 max=0.0060 sample [+0.00182, -0.00109, -0.00289, -0.00071, -0.00031, +0.00167 ...]
  - after quant (8-bit, 256 levels, scale=4.746e-05): shape=[3072, 16] l2=0.3304 absμ=0.00119 min=-0.0061 max=0.0060 sample [+0.00180, -0.00109, -0.00290, -0.00071, -0.00033, +0.00166 ...]

## 7. Resources & timing

- **Total wall time:** 2450.04 s
- **GPU:** NVIDIA RTX 4050 Laptop (6 GB); model body 4-bit NF4 on GPU, 256k-vocab embed on CPU
- **Peak VRAM (load):** ~1.7 GB allocated (see training logs for peak)
- **Python:** 3.12.0 | Platform: Windows
- **Privacy budget (final):** ε_total(RDP) ≈ 16.03 over 4 rounds (δ=1e-5)

---
*Generated by `fednemo.scripts.generate_report`.*