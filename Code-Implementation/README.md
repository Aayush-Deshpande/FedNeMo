# FedNeMo — Code Implementation

This is the **source directory** for the FedNeMo implementation. It is built in
phases. See `phases-records.md` for exactly what each phase added and what works.

## Quickest way to confirm it works (after cloning)

```bash
cd Code-Implementation
pip install -r requirements.txt
python -m fednemo.smoke_test
```

Then open **`RUN_REPORT.md`**. It is written in plain English: if `RESULT` says
`✅ WORKING`, every check passed and the Phase 0 + Phase 1 pipeline is sound on
your machine. If something is broken, the report names the failing check (or
shows the crash traceback). You do NOT need Streamlit or a GPU for this.

## What the phases are (short version)

- **Phase 0**: the federated loop turns, fully stubbed (fake training, identity
  filters). Proves the wiring.
- **Phase 1**: real FedRand (each hospital sends only one LoRA matrix per round)
  + a live Gradient Inversion Attack demo (unprotected = readable record,
  protected = noise).

Full detail in `phases-records.md`.

## Layout

```
fednemo/
  schema.py        # the Update contract + init_weights
  data/split.py    # one-time: full.jsonl -> hospital_N.jsonl
  data/loader.py   # read a shard, build prompts
  server.py        # global state + FedAvg + telemetry write
  filters.py       # FedRand (real) -> DP (stub) -> Quant (stub)
  client.py        # fake "training", returns an Update
  attacker.py      # live GIA demo prop
  driver.py        # the loop that ties it all together
  dashboard.py     # streamlit, reads telemetry.jsonl
  smoke_test.py    # one command -> RUN_REPORT.md verdict
data/full.jsonl    # ~50 synthetic rows
requirements.txt
```

## Run the full loop + dashboard

```bash
cd Code-Implementation
pip install -r requirements.txt
python -m fednemo.data.split 3      # make 3 hospital shards
python -m fednemo.driver            # run the federated loop -> telemetry.jsonl
streamlit run fednemo/dashboard.py  # watch it (optional)
```

## Two readable outputs

- **`RUN_REPORT.md`** — human-readable pass/fail verdict from `smoke_test.py`.
  This is the file to open to answer "is it working?".
- **`telemetry.jsonl`** — one JSON line per round (machine-shaped), consumed by
  the dashboard.

## What comes next (not now)

- **Phase 2**: real Laplacian DP + RDP accountant.
- **Phase 3**: real adaptive quantization.
- **Phase 4**: swap toy model for Nemotron-Mini-4B + NeMo LoRA.
- **Phase 5**: move transport into NVIDIA FLARE.
