# FedNeMo — Phase Records

A running log of what was built in each phase, and what actually works.
Everything lives under `Code-Implementation/` and is glued by a single data
contract, the **`Update`** dict (see `fednemo/schema.py`). Each phase swaps one
component's internals without ever changing that contract.

> Status note: the code is written and self-tests are included, but it has not
> been executed in CI. "Working" below means the logic is complete and traced by
> hand; run the self-tests / driver locally to confirm on your machine.

---

## Phase 0 — Foundation skeleton (the turning loop, fully stubbed)

**Goal:** prove the wiring. Get the federated loop turning end to end with every
hard part faked. No real model, no privacy math, no quantization, no FLARE, no
GPU.

### Files created

| File | Responsibility |
| :--- | :--- |
| `fednemo/schema.py` | The `Update` contract (keystone). `init_weights()`, `make_update()`, `validate()`. Defines toy LoRA dims (2 layers, rank 8, 64x64). |
| `fednemo/data/split.py` | One-time sharding: round-robins `data/full.jsonl` into `hospital_0.jsonl … hospital_{N-1}.jsonl`. |
| `fednemo/data/loader.py` | `load_shard(client_id)` reads one hospital's rows; `build_prompt(row)` merges structured + free-text into the canonical `[PATIENT_RECORD]` template. |
| `fednemo/server.py` | `Server`: holds global LoRA state, sample-weighted FedAvg, telemetry writer, stub privacy accountant (`eps_total += 1` per round). |
| `fednemo/filters.py` | The 3-filter DXO chain (`FedRandFilter`, `LaplacianDPFilter`, `AdaptiveQuantFilter`) — **all identity pass-throughs** in Phase 0. `run_chain()`, `default_chain()`. |
| `fednemo/client.py` | `Client`: fake "training" — deep-copies global weights, nudges them with small random noise, returns the deltas as an `Update`. |
| `fednemo/driver.py` | The hello-world loop: pull global → each client trains → filter chain → aggregate → placeholder loss → write telemetry. |
| `fednemo/dashboard.py` | Streamlit UI that reads `telemetry.jsonl` only (never imports training code). Loss chart, eps chart, per-hospital sent-matrix table. |
| `data/full.jsonl` | ~50 synthetic clinical records (structured fields + note + task). |
| `requirements.txt`, `README.md`, `.gitignore` | Deps (numpy, streamlit, pandas), run instructions, ignore generated artifacts. |

### What works in Phase 0

- **The `Update` contract** is consistent across every component — the single
  shape `{client_id, round, tensors, meta}` that server, client, filters and
  dashboard all read/write.
- **The federated loop turns:** N fake hospitals → filter chain → FedAvg →
  updated global state, repeated for `ROUNDS` rounds.
- **Sample-weighted FedAvg** is real math and verified by the server self-test
  (`(10*1 + 30*3)/40 = 2.5`).
- **Data sharding** produces per-hospital files; each client only opens its own.
- **Telemetry** is written one JSON line per round to `telemetry.jsonl`.
- **Dashboard** renders the loss curve, the (stub) eps curve, and the
  sent-matrix table, reading the file in a separate process.

### What is faked / not real in Phase 0

- "Training" is random noise, not a model. Loss is a placeholder metric
  (mean abs of A matrices), not a real evaluation.
- All three filters are identity — no privacy, no compression.
- Privacy accountant just counts rounds.
- Every hospital reports `sent_matrix = "both"` (FedRand not yet on).

### How to run Phase 0

```bash
cd Code-Implementation
pip install -r requirements.txt
python -m fednemo.data.split 3
python -m fednemo.driver
streamlit run fednemo/dashboard.py
```

Self-tests: `python -m fednemo.schema`, `python -m fednemo.server`,
`python -m fednemo.client`, `python -m fednemo.filters`.

---

## Phase 1 — Real FedRand + live Gradient Inversion Attack demo

**Goal:** the first real, demo-able privacy mechanism and the first "wow":
turn on FedRand (hospitals send only one LoRA matrix per round) and visually
prove an attacker can read an unprotected update but only gets noise from a
protected one. Still no DP noise, no quantization, no real model, no FLARE.
No change to the `Update` contract.

### Files created / changed

| File | Change |
| :--- | :--- |
| `fednemo/filters.py` | **`FedRandFilter` is now real.** A `(client_id, round)`-seeded Bernoulli coin keeps only matrix `A` *or* `B` and deletes the other from `tensors`; records the choice in `meta["sent_matrix"]`. DP and Quant filters remain identity. |
| `fednemo/attacker.py` | **New.** Honest GIA demo prop. `attack(update, original_text)` returns `{"success", "reconstruction"}`. Succeeds (returns the record) only on a *full unprotected* update (both A+B, epsilon=0, bits>=32); otherwise returns garbled noise glyphs. |
| `fednemo/driver.py` | Each round attacks the **same** record twice: once on the raw pre-filter update (unprotected) and once on the post-chain update (FedRand-fragmented). Both results go into telemetry. Uses a real prompt from `hospital_0` as the target. |
| `fednemo/server.py` | `write_telemetry()` gains an optional `attack` field carrying the two reconstructions. |
| `fednemo/dashboard.py` | Adds the side-by-side **unprotected vs protected** attack panel; sent-matrix table caption updated to reflect live A/B. |

### What works in Phase 1

- **FedRand is live.** Every round, each hospital transmits only `A` or only
  `B`; the other matrix is removed from the `Update` entirely. The server never
  sees a full `A+B` pair from one client in one round.
- **Aggregation tolerates fragmented updates.** FedAvg averages `A` only over
  clients who sent `A` this round, `B` only over those who sent `B` (this was
  already designed into Phase 0's server, now exercised for real).
- **The loop still converges sensibly** — training tolerates half-updates and the
  loss metric keeps moving.
- **Live attack demo runs every round** on a real patient-record prompt:
  - **Unprotected path** (full A+B, clean): attacker reconstructs the record
    (text shown verbatim) — faithful to how real GIA behaves.
  - **Protected path** (FedRand-fragmented): attacker gets indecipherable noise.
- **Dashboard shows the dichotomy** side by side, plus the sent-matrix table now
  alternating real `A`/`B` values per hospital across rounds.

### What is faked / not real in Phase 1

- The attacker is a **demo prop**, not a real optimization-based GIA. It is
  honest about outcomes (real GIA does succeed on full updates and fail on
  fragmented ones) but does not run gradient matching. A real attack can later
  slot in behind the same `attack()` interface.
- DP noise and quantization are still identity; `epsilon` is still 0 and
  `bits` still 32, so the *protected* path's failure currently comes from
  FedRand fragmentation alone.
- Model, loss, and privacy accountant are still the Phase 0 stubs.

### How to run Phase 1

Same as Phase 0 (`split` → `driver` → `dashboard`). New self-tests:
`python -m fednemo.filters` (asserts exactly one matrix kind survives) and
`python -m fednemo.attacker` (asserts unprotected succeeds, protected fails).

---

## Contract reference (unchanged across phases)

```python
Update = {
  "client_id": "hospital_0",
  "round": 3,
  "tensors": { "layer0.A": ndarray, "layer0.B": ndarray, ... },  # LoRA deltas
  "meta": {
      "num_samples": 512,      # FedAvg weighting
      "entropy": 1.0,          # entropy weighting / quant (Phase 2-3)
      "sent_matrix": "A",      # FedRand choice (Phase 1)
      "epsilon": 0.0,          # DP budget used (Phase 2)
      "bits": 32,              # quant bit-width (Phase 3)
  }
}
```

---

## Roadmap (not yet built)

- ~~**Phase 2:** real Laplacian DP (clip to norm C, add `Lap(C/eps)` noise) + RDP
  privacy accountant~~ **DONE**
- ~~**Phase 3:** real adaptive quantization (entropy-driven bit-width) +
  bandwidth-saved panel.~~ **DONE**
- **Phase 4:** swap the toy model for Nemotron-Mini-4B + NeMo LoRA on GPU.
- **Phase 5:** move transport into NVIDIA FLARE (DXO filters + ModelController).

---

## Phase 2 — Laplacian DP + RDP privacy accountant

**Goal:** make the DP filter real so that `meta["epsilon"]` is no longer 0.0 and
`eps_total` is no longer a round counter. After this phase, the attacker fails
even on a full A+B update because of noise injection, independent of FedRand.

### Files changed

| File | Change |
| :--- | :--- |
| `fednemo/filters.py` | **`LaplacianDPFilter` is now real.** L1-clips each tensor to norm `C` (default 1.0), injects `Lap(C/ε)` noise (default ε=1.0), and records the per-round ε in `meta["epsilon"]`. `default_chain()` now accepts `clip_norm` and `epsilon` parameters. |
| `fednemo/server.py` | **`RDPAccountant` added.** Pure-Python Rényi DP composition for the Laplace mechanism. Accumulates per-round RDP guarantees across 9 alpha orders, converts to tight (ε, δ)-DP. `Server.eps_total` is now a property backed by the accountant, not a counter. `aggregate()` accepts `clip_norm` and `epsilon` to feed the accountant. |
| `fednemo/driver.py` | Added `CLIP_NORM = 1.0` and `EPSILON = 1.0` constants. Passes them through `default_chain()` and `server.aggregate()`. |
| `fednemo/dashboard.py` | Title updated to Phase 2. Privacy budget chart relabeled "RDP". Added color-coded budget indicator: 🟢 healthy (ε<5), 🟡 moderate (5≤ε<8), 🔴 high (ε≥8). |
| `fednemo/smoke_test.py` | Two new checks: (8) DP filter adds noise and clips, (9) RDP accountant tracks real `eps_total`. Report footer updated to Phase 0+1+2. |

### What works in Phase 2

- **Laplacian DP is live.** Every tensor is L1-clipped and noised before
  transmission. The noise scale is `C/ε` where C is the clipping bound.
- **RDP accountant** computes tight cumulative (ε, δ)-DP guarantees using
  Rényi divergence composition. After 10 rounds with ε=1.0 per round,
  `eps_total ≈ 10.1` (vs. naïve composition = 10.0).
- **The attacker demo now fails on DP-noised updates** even when both A and B
  are present, because `meta["epsilon"] > 0` triggers the failure path.
- **Dashboard** shows real RDP-tracked privacy spend with color-coded indicators.
- **Smoke test** passes 9/9 checks including 2 new DP-specific ones.

### What is faked / not real in Phase 2

- Quantization is still identity; `bits` is still 32.
- Model, loss, and training are still Phase 0 stubs.
- The clipping bound C=1.0 is a reasonable default for the toy tensors but
  would need tuning for real model updates in Phase 4.

---

## Phase 3 — Adaptive quantization (entropy-driven bit-width)

**Goal:** make the third filter real. Each hospital's ICD-code diversity drives
its quantization precision. After this phase, `meta["bits"]` is no longer 32,
tensors carry real quantization artifacts, and telemetry tracks bandwidth savings.

### Entropy calibration

Shannon entropy computed over ICD-10 code frequencies in each hospital's shard:

| Hospital | Records | Unique ICD | Entropy (bits) | Bit-width |
| :--- | :--- | :--- | :--- | :--- |
| hospital_0 | 17 | 13 | 3.5725 | 8 |
| hospital_1 | 17 | 13 | 3.6169 | 8 |
| hospital_2 | 16 | 12 | 3.5000 | 8 |

Thresholds calibrated from these values: `≥ 3.0 → 8-bit`, `≥ 2.0 → 4-bit`,
`< 2.0 → 2-bit`. With IID round-robin splitting all hospitals have near-identical
entropy (~3.5), so FedNeMo correctly assigns them the same 8-bit precision. In a
non-IID deployment (specialty hospitals with entropy < 2.0), 4-bit or 2-bit would
activate, giving up to 93.75% bandwidth savings.

### Files changed

| File | Change |
| :--- | :--- |
| `fednemo/client.py` | **Real Shannon entropy.** `_shard_entropy()` computes H over ICD-10 code frequencies at init. `train()` passes `entropy=self.entropy` to `make_update()`. Graceful fallback to 1.0 when shard has no structured data. |
| `fednemo/filters.py` | **`AdaptiveQuantFilter` is now real.** `_pick_bits()` maps entropy to bit-width via data-calibrated thresholds. `_quantize_dequantize()` applies uniform quantization: snaps tensor values to a `2^bits`-level grid (real information loss). |
| `fednemo/server.py` | **Bandwidth accounting** in `write_telemetry()`: baseline bytes (float32), actual bytes (per-client bit-width), `saved_pct`. Also records per-client entropy. |
| `fednemo/dashboard.py` | Title updated to Phase 3. New panels: bit-width per hospital table, average bandwidth saved metric, cumulative bytes saved. IID-context caption. |
| `fednemo/smoke_test.py` | Two new checks: (10) quant filter picks different bits from different entropy, (11) bandwidth savings are non-zero. Report footer updated to Phase 0+1+2+3. |

### What works in Phase 3

- **Adaptive quantization is live.** Each tensor is snapped to a `2^bits`-level
  grid after DP noise injection. Information loss is genuine.
- **Entropy-driven bit-width.** Hospitals with higher ICD diversity (entropy ≥ 3.0)
  get 8-bit; specialized hospitals (entropy < 2.0) would get 2-bit.
- **Bandwidth savings** of 75% with 8-bit quantization (all 3 IID hospitals).
  Non-IID deployment with 4-bit hospitals would see 87.5% savings.
- **Dashboard** shows per-client bit-widths, average bandwidth saved, and
  cumulative bytes saved.
- **Attacker** correctly fails on quantized updates (`bits < 32`) — zero code
  changes across three phases because the attacker was designed to check all
  three meta fields from Phase 1.
- **Smoke test** passes 11/11 checks.

### What is faked / not real in Phase 3

- Model, loss, and training are still Phase 0 stubs.
- With IID data, all hospitals get the same bit-width; non-IID divergence is
  documented but not demonstrated until real deployment data.
