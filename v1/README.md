# FedNeMo (v1)

Local, in-process federated fine-tuning of **NVIDIA Nemotron-Mini-4B-Instruct**
for a medical anomaly-detection use case (ECG / cardiac risk).

Everything runs as local Python functions in one process. **No servers, no MCP,
no cloud orchestration.** The only outbound network call in the entire system is a
single request to NVIDIA's hosted **nemotron-parse** model at inference time, to
transcribe a scanned report image. Training is 100% local and CSV-only.

## What it does

- **5 simulated client nodes**, each holding a non-IID CSV data shard.
- Each client fine-tunes a **LoRA adapter** on its own shard only.
- A local aggregator runs **FedRand** (randomized A/B split-adapter), **Laplace
  differential privacy**, **2-bit adaptive quantization**, **entropy-aware
  weighting**, and **trust-scoring** (a second logical Nemotron role) to produce
  a new global adapter.
- The global adapter is redistributed each round.

## Hard constraints honored

- Single-GPU (RTX 4050, 6 GB). **Peak project VRAM verified ~2.4 GB (< 3 GB).**
- **Clients train strictly sequentially** (one at a time), never in parallel.
- Laplace DP (not Gaussian). Fixed **2-bit** quantization with adaptive per-tensor
  scale/zero-point. No OCR. No training on images/parsed output.

## VRAM strategy (configurable device placement)

Nemotron-Mini-4B has a 256,000-token vocabulary. `embed_tokens` and `lm_head` are
~1.5 GB each in bf16 and cannot be cleanly 4-bit quantized. `embed_tokens` always
stays on CPU (a cheap gather; a GPU copy would cost ~1.5 GB). `lm_head` placement
is configurable via `CONFIG.lm_head_device`:

| Component | `lm_head_device="cuda"` (default, fast) | `lm_head_device="cpu"` (low-VRAM) |
|---|---|---|
| `embed_tokens` (256k×3072) | CPU float32 (gather) | CPU float32 (gather) |
| 32 decoder layers | GPU 4-bit NF4 | GPU 4-bit NF4 |
| final norm | GPU | GPU |
| `lm_head` (256k×3072) | **GPU 4-bit** | **CPU float32** |
| step time | ~0.5-0.8 s | ~2.4 s |
| peak allocated | ~4.3 GB | ~2.2 GB |
| peak reserved* | ~5.9 GB | ~2.4 GB |

The forward pass is executed manually (embed → body → lm_head) so each stage runs
on the intended device. Cross-device autograd links the loss to the GPU LoRA
params (verified: gradients flow, loss decreases).

*Peak "reserved" is PyTorch's caching-allocator high-water mark (reusable memory),
not live usage. On Windows it runs ~1.6 GB above "allocated" due to fragmentation
(`expandable_segments` is largely a no-op on Windows). True live usage is the
"allocated" figure. On a 6 GB card the `cuda` layout leaves thin headroom - switch
`lm_head_device` to `"cpu"` if other apps contend for VRAM.

## Setup

```bash
pip install -r requirements.txt
```

The local checkpoint is expected at `D:\Code\models\Nemotron-Mini-4B-Instruct`
(override with the `FEDNEMO_MODEL_PATH` env var). A one-time conversion of
`pytorch_model.bin` → `model.safetensors` is required (torch 2.5.1 refuses to load
`.bin` under CVE-2025-32434; safetensors is unaffected). If `model.safetensors`
is missing, convert it once locally.

## Run training (local, CSV-only)

```bash
python -m fednemo.scripts.run_training --dataset ptbxl --limit 400 --rounds 3
python -m fednemo.scripts.run_training --dataset uci   --limit 400 --rounds 3
```

Output: per-client non-IID class distributions are logged, and the trained global
adapter is saved to `v1/artifacts/global_adapter.pt`.

## Run inference (single nemotron-parse call)

```bash
set NVIDIA_API_KEY=nvapi-...            # Windows cmd
python -m fednemo.scripts.run_inference --image report.png --question "Any MI risk?" --schema uci
```

Flow: image → nemotron-parse (1 network call) → deterministic field mapping →
prompt → local trained Nemotron → {primary answer, incidental findings, grounded
explanation}.

## Datasets

- **PTB-XL** (primary): `ptbxl_database.csv` + `scp_statements.csv`. Diagnostic
  superclass label (NORM/MI/STTC/CD/HYP) derived from `scp_codes` via the
  `diagnostic_class` mapping (standard PTB-XL benchmarking approach).
  **Limitation (accepted):** CSV-only means no waveform features — the model sees
  demographics + free-text report + device metadata only.
- **UCI Heart Disease** (secondary): different schema, validates pipeline
  generalization. Label = `num` (0–4 severity).

## Non-IID partitioning

`data/partition.py` uses a per-class **Dirichlet(alpha=0.4)** draw producing both
label skew and volume skew, with a resampled minimum-shard guarantee so no client
is trivially small. Realized distributions are logged for verification.

## Privacy notes

- Laplace mechanism: L2-clip each shared matrix to `C` then add `Lap(0, C/epsilon)`.
  This gives per-round (epsilon, 0)-DP for the released matrix. The tightest
  L1-sensitivity form uses an L1 clip; we use an L2 clip as a standard practical
  bound. **Multi-round privacy composition accounting (RDP) is not implemented** —
  reported epsilon is per-round only.
- FedRand ensures the aggregator never sees both A and B of the same layer from
  the same client in the same round (verified invariant).

## Known limitations / honesty notes

See the top of each module and the delivery summary. In particular: the
nemotron-parse response shape and the field-mapping regexes were written against
NVIDIA's published API docs but not verified against a live API call.
