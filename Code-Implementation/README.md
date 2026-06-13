# FedNeMo — Code Implementation

This is the **source directory** for the FedNeMo implementation. It is built in
phases. Everything here right now is **Phase 0**: a working but fully stubbed
federated-learning skeleton.

## What Phase 0 is

Phase 0 builds the *turning loop* and nothing else:

- a **server** holding LoRA-shaped global weights, doing sample-weighted FedAvg;
- **N identical fake hospitals** that "train" by nudging numbers with noise;
- a **3-stage filter chain** (FedRand / DP / Quant) where every filter is a
  pass-through identity for now;
- a **telemetry file** (`telemetry.jsonl`) the server appends to each round;
- a **Streamlit dashboard** that reads that file.

There is **no real model, no privacy math, no quantization, no FLARE, no GPU**.
When loss numbers move across rounds and the dashboard draws them, Phase 0 is done.

The single contract that glues every component together is the `Update` dict
(see `fednemo/schema.py`). Every later phase swaps one component's internals
without ever changing that contract.

## Layout

```
fednemo/
  schema.py        # the Update contract + init_weights
  data/split.py    # one-time: full.jsonl -> hospital_N.jsonl
  data/loader.py   # read a shard, build prompts
  server.py        # global state + FedAvg + telemetry write
  filters.py       # FedRand/DP/Quant -- all identity for now
  client.py        # fake "training", returns an Update
  driver.py        # the loop that ties it all together
  dashboard.py     # streamlit, reads telemetry.jsonl
data/full.jsonl    # ~50 synthetic rows
requirements.txt
```

## How to run

From inside `Code-Implementation/`:

```bash
pip install -r requirements.txt

# 1. split the synthetic dataset into N hospital shards
python -m fednemo.data.split 3

# 2. run the federated loop (writes telemetry.jsonl)
python -m fednemo.driver

# 3. (separate terminal) watch the dashboard
streamlit run fednemo/dashboard.py
```

## Definition of done for Phase 0

`python -m fednemo.driver` runs N fake hospitals through 10 rounds without error,
`telemetry.jsonl` has 10 lines, and the dashboard renders a moving loss curve and
a per-hospital sent-matrix table. No real ML, and that's correct.

## What comes next (not now)

- **Phase 1**: make `FedRandFilter` real (drop one matrix) + fake attack panel.
- **Phase 2**: real Laplacian DP + RDP accountant.
- **Phase 3**: real adaptive quantization.
- **Phase 4**: swap toy model for Nemotron-Mini-4B + NeMo LoRA.
- **Phase 5**: move transport into NVIDIA FLARE.
