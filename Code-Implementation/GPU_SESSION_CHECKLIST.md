# FedNeMo — GPU Session Checklist (AMD MI300X / ROCm)

Phase 4 real run: fine-tune **nvidia/Nemotron-Mini-4B-Instruct** with real LoRA.
Everything below is copy-paste. Goal: minimize idle billing — the code is
already written and CPU-validated; this session only swaps in the real model.

> The code is backend-agnostic. On ROCm, `torch.cuda.is_available()` returns
> True and the device string `"cuda"` maps to the AMD GPU — no code changes.

---

## 0. One-time setup

```bash
cd Code-Implementation

# ROCm PyTorch (NOT in requirements.txt — it's accelerator-specific):
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# Everything else:
pip install -r requirements.txt
```

## 1. Verify the GPU is visible

```bash
python -c "import torch; print('gpu:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
Expect `gpu: True  AMD Instinct MI300X`. If False, stop and fix the ROCm install.

## 2. HuggingFace auth (Nemotron-Mini may require accepting its license)

```bash
huggingface-cli login        # paste token; accept the model card terms on the web first
```

## 3. Confirm / adjust LoRA target_modules against the REAL model

Nemotron-Mini-4B is Llama-architecture-based, so the defaults in `schema.py`
(`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) should match.
Verify before a long run:

```bash
python -c "from transformers import AutoModelForCausalLM; m=AutoModelForCausalLM.from_pretrained('nvidia/Nemotron-Mini-4B-Instruct'); print(sorted({n.split('.')[-1] for n,_ in m.named_modules() if any(k in n for k in ['proj'])}))"
```
If the printed names differ, edit the single constant `TARGET_MODULES` in
`fednemo/schema.py` to match, then continue.

## 4. Prepare hospital shards from MedQA-USMLE

```bash
# 3 hospitals, 200 rows each (tune to taste):
python -m fednemo.data.medqa 3 --per 200
```

## 5. (Optional) quick 2-round sanity run on the real model

```bash
python -m fednemo.driver hf
```
Watch for non-NaN loss and `eps_total` rising. Ctrl-C after a couple rounds if
it looks healthy.

## 6. Full simulation (50–60 rounds)

Edit the run parameters at the top of `fednemo/driver.py` if needed
(`ROUNDS`, `N_CLIENTS`, `SEQ_LEN`, `LOCAL_STEPS`, `MODEL_NAME`), then:

```bash
python -c "from fednemo.driver import run; run(backend_kind='hf', rounds=60, n_clients=3)"
```

Outputs `telemetry.jsonl` (one line/round).

## 7. Inspect results

```bash
streamlit run fednemo/dashboard.py     # loss, RDP budget, FedRand, bandwidth, live GIA panel
```

---

## Notes / knobs

- **VRAM:** MI300X has 192 GB — the 4B model in fp16 + LoRA is comfortable; no
  quantization needed. `HFBackend` defaults to fp16 on GPU, fp32 on CPU.
- **Throughput:** `LOCAL_STEPS` and `SEQ_LEN` (driver.py) drive per-round time.
  Start small, scale up once a round completes cleanly.
- **Only `TARGET_MODULES` in `schema.py` is model-specific.** Nothing else
  should need editing for the real model.
- If a round errors on a specific shard, lower `SEQ_LEN` or `--per` (shorter
  sequences / fewer rows) and re-run.
