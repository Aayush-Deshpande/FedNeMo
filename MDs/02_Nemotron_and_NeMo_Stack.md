# FedNeMo: Nemotron Model and NeMo/FLARE Technical Stack

> **What this document covers:** The distinction between NeMo and Nemotron, Nemotron-3 Nano architecture, model selection for demo vs. production, LoRA target module mapping, the Mamba-2 fused kernel engineering problem and its solution, NeMo 2.0 and FLARE integration code, library version pinning, and compute/GPU considerations. For algorithmic pipeline stages see `01_FedNeMo_Core_Architecture.md`.

---

## 1. NeMo vs. Nemotron — The Distinction

These are frequently confused. They serve entirely different roles.

**NeMo** is a **framework**. Think PyTorch Lightning but purpose-built for large language models. It is NVIDIA's open-source toolkit for building, training, fine-tuning, and deploying LLMs.

What NeMo provides:
- Infrastructure for distributed training across multiple GPUs (tensor parallelism, pipeline parallelism)
- Built-in support for fine-tuning methods: LoRA, P-Tuning, Prefix Tuning via `ModelTransform`
- Model checkpointing, mixed precision (BF16/FP8), gradient accumulation
- Clean Python API to define training jobs
- Integration with TensorRT-LLM for optimized inference after training
- NeMo Guardrails — separate module for safe, controlled LLM outputs
- Megatron-LM core for multi-GPU parallelism

NeMo is **the kitchen**. It provides all the tools. You bring the model.

**Nemotron** is NVIDIA's **family of actual LLMs** — pre-trained models with weights, tokenizers, and architectures that you fine-tune.

The relationship:
```
NeMo (framework)
    └── loads and fine-tunes → Nemotron (model)
                                   └── LoRA adapters sit on top
                                           └── FedNeMo's federated loop wraps everything
```

NeMo handles all GPU-level mechanics. Nemotron is what you are teaching new medical knowledge to. FedNeMo's code lives around both of them — intercepting the LoRA parameter flow between hospitals and the server.

---

## 2. Model Selection: Demo vs. Production

**For the hackathon demo: Nemotron-Mini-4B**
**For production target: Nemotron-3 Nano (30B-A3B)**

This distinction is critical. Using Nemotron-3 Nano (30B) in the demo is a hardware feasibility problem.

| Aspect | Nemotron-Mini-4B | Nemotron-3 Nano (30B-A3B) |
| :--- | :--- | :--- |
| Parameters | ~4 billion | 31.6 billion total, ~3.5B active |
| VRAM per client | Fits on a single A100 (80GB) with multiple processes | Requires ~3+ high-memory GPUs for 5 simultaneous clients |
| LoRA fine-tuning | Practical for multi-client simulation | Requires QLoRA + model parallelism for demo hardware |
| Architecture | Standard Transformer | Hybrid Mamba-2 / Transformer / Latent MoE |
| Track B compliance | ✅ | ✅ (the technical target) |
| Hackathon demo feasibility | ✅ Recommended | ❌ Infeasible on typical A100 hardware without GPUs × 3+ |

**Recommended framing in the demo:** Use Nemotron-Mini-4B as the primary demo model. Explicitly state that FedNeMo's framework scales to Nemotron-3 Nano as the production deployment target. This is what NVIDIA's own FLARE examples do — demo on smaller models, document scalability to large ones.

---

## 3. Nemotron-3 Nano Architecture (Production Target)

| Specification | Value |
| :--- | :--- |
| **Total Parameters** | 31.6 billion |
| **Active Parameters per Token** | ~3.5 billion (3.2B base + embeddings) |
| **Architecture** | Hybrid Mamba-2 / Transformer / Latent MoE |
| **Layer Configuration** | 23 Mamba-2 + MoE layers, 6 Attention layers (interleaved) |
| **Expert Design** | 128 routed experts + 1 shared expert per MoE layer |
| **Active Experts per Token** | 5–6 (sparse activation) |
| **Context Window** | Up to 262,144 (256K) tokens |
| **Supported Precision** | FP8, BF16 (FP8 retains ~99% of BF16 accuracy) |
| **Inference Throughput** | 3.3× higher than comparable MoE models (e.g., Qwen3-30B-A3B) |
| **Release Date** | December 2025 |
| **Languages** | English, German, Spanish, French, Italian, Japanese, Arabic, Chinese, Korean, Hindi |

### 3.1 Hybrid Backbone: Three Interlocking Innovations

**Mamba-2 State Space Models (23 of 29 layers):** SSMs operate with linear-time complexity $O(L)$ relative to sequence length $L$, compared to the quadratic $O(L^2)$ cost of standard self-attention. The Mamba-2 recurrence: $h_t = \bar{A}h_{t-1} + \bar{B}x_t$, $y_t = Ch_t$, where $\bar{A}$ and $\bar{B}$ are discretized state matrices. This allows processing of 256K-token context windows — necessary for longitudinal patient history spanning decades.

**Targeted Transformer Attention (6 strategic layers):** Pure SSMs struggle with complex long-range factual recall requiring all-to-all token comparison. Six Transformer attention layers are interleaved at calculated depths for high-fidelity information routing — e.g., connecting a medication on page 3 of a discharge summary to an adverse reaction documented on page 47.

**Latent Mixture-of-Experts (Latent MoE):** Traditional MoE routes tokens from dense hidden dimensions into isolated experts, requiring massive memory bandwidth proportional to expert count. Latent MoE compresses token representations into lower-dimensional latent spaces before routing to experts. This activates 5–6 of 128 specialized experts per token without increasing inference compute costs, delivering throughput closer to a 3B dense model than a 30B model.

**Clinical relevance of MoE for multi-department use:** The MoE router activates different expert subsets for different clinical domains. A Cardiology report activates different experts than an Oncology report. A single model instance handles all hospital departments simultaneously without cross-contamination. No separate model instances per department are needed.

---

## 4. LoRA Target Module Selection for Hybrid Architecture

Standard LoRA tutorials assume pure Transformer architectures and target only `q_proj`, `k_proj`, `v_proj`. This covers only 6 of 29 layers in Nemotron-3 Nano, ignoring 79% of the network's processing power.

FedNeMo targets both Transformer and Mamba-2 projection layers:

| Target Module | Layer Type | Role | Rationale |
| :--- | :--- | :--- | :--- |
| `linear_qkv` | Transformer Attention | Fused Query-Key-Value projection | Standard attention adaptation; captures clinical entity relationships |
| `linear_proj` | Transformer Attention | Output projection post-attention | Adapts representation space for domain-specific outputs |
| `in_proj` / `x_proj` | Mamba-2 SSM | Input projection into state-space | Adapts how clinical tokens enter SSM recurrence; critical for domain vocabulary |
| `out_proj` | Mamba-2 SSM | Output projection from state-space | Adapts SSM output for downstream clinical tasks |
| `dt_proj` | Mamba-2 SSM | Time-step discretization | Controls temporal dynamics; fine-tunes sequential clinical event processing |

This comprehensive targeting ensures the model integrates clinical semantics across all 3.5 billion active parameters.

**Basis:** MambaPEFT (arXiv, 2024) and SDLoRA (ICML, 2025) demonstrate that applying LoRA strictly to sparse attention layers of a hybrid model ignores the majority of the network's processing capacity. Targeting Mamba-2 layers alongside attention layers is the correct approach for hybrid architectures.

---

## 5. Mamba-2 Fused Kernel Compatibility — The Engineering Problem

This is the most significant engineering risk in FedNeMo's implementation. If unaddressed, it causes silent failure of the entire "extension of FedRand to SSM architectures" claim.

**The problem:** Nemotron-3 Nano's Mamba-2 implementation uses highly optimized fused CUDA kernels — specifically `causal_conv1d_cuda` and `selective_scan_cuda` — that execute the state-space recurrence entirely in GPU-level C++ code, bypassing standard PyTorch `nn.Linear` forward passes.

When a LoRA adapter is injected via standard Python-level module replacement (default mechanism in `peft`), the fused kernel may call the underlying weight tensor directly, silently ignoring the LoRA perturbation $\Delta W = BA$. If this occurs:
- LoRA adapters on Mamba-2 layers produce **zero weight deltas**.
- The "extension of FedRand to SSM architectures" claim **collapses entirely**.
- The failure is **silent** — training proceeds normally but the Mamba-2 layers learn nothing.

This is a documented problem in the `peft` library community as of 2026.

### 5.1 Three-Tier Compatibility Strategy

**Tier 1 — Framework-Native Injection (Primary path):**

NeMo 2.0's `ModelTransform` operates at the Megatron Core level, controlling weight initialization and module construction *before* fused kernels are compiled. When LoRA is applied via `llm.peft.LoRA` with explicit `target_modules` specifying Mamba-2 projections, the NeMo framework injects the low-rank perturbation directly into the weight tensor that the fused kernel reads. The fused kernel operates on the LoRA-modified weights $W_0 + BA$ natively.

**Requires:** NeMo Framework version ≥ 2.3, which includes validated Mamba-2 LoRA support.

**Tier 2 — Pre-Kernel Weight Materialization (Fallback for HuggingFace peft):**

If using `peft` for prototyping, implement a custom `pre_forward_hook` that materializes the LoRA perturbation into the base weight tensor before the fused kernel executes:

```python
def mamba_lora_pre_hook(module, input):
    """Materializes LoRA delta into base weight before fused CUDA kernel call."""
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
        delta_w = module.lora_B.weight @ module.lora_A.weight
        module.weight.data.add_(module.scaling * delta_w)
        module._lora_applied = True

def mamba_lora_post_hook(module, input, output):
    """Removes materialized LoRA delta after forward pass to preserve optimizer state."""
    if getattr(module, '_lora_applied', False):
        delta_w = module.lora_B.weight @ module.lora_A.weight
        module.weight.data.sub_(module.scaling * delta_w)
        module._lora_applied = False
```

The hook pair ensures the fused kernel sees LoRA-modified weights during forward computation; the base weight is restored afterward to maintain correct gradient computation for only the LoRA parameters.

**Tier 3 — Automated Validation Gate (Always runs at startup):**

Regardless of which injection tier is active, FedNeMo executes an automated LoRA injection validation at the start of every federated training job:

```python
def validate_lora_injection(model, target_modules):
    """Verifies all targeted layers have non-trivial LoRA gradient flow."""
    dummy_input = torch.randint(0, model.config.vocab_size, (1, 128)).cuda()
    loss = model(dummy_input, labels=dummy_input).loss
    loss.backward()
    
    for name, param in model.named_parameters():
        if any(t in name for t in target_modules) and 'lora' in name:
            grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
            if grad_norm < 1e-10:
                raise RuntimeError(
                    f"FATAL: LoRA adapter '{name}' has zero gradient. "
                    f"Fused CUDA kernel is bypassing LoRA injection. "
                    f"Falling back to Tier 2 pre-kernel materialization."
                )
            logger.info(f"LoRA validation PASSED: {name} | grad_norm={grad_norm:.6f}")
```

If Tier 1 fails, system automatically falls back to Tier 2 with a logged warning. If both fail, training halts with an explicit error — preventing empty weight deltas from propagating through the federation.

### 5.2 Library Version Pinning

| Library | Minimum Version | Reason |
| :--- | :--- | :--- |
| `mamba_ssm` | ≥ 2.2.5 | Includes `selective_scan_ref` fallback for non-fused execution |
| `causal_conv1d` | ≥ 1.5.2 | Compatible with PyTorch hook-based weight interception |
| `nemo_toolkit` | ≥ 2.3.0 | Native Mamba-2 LoRA support via `ModelTransform` |
| `peft` | ≥ 0.14.0 | Explicit `MambaConfig` recognition for hybrid architectures |
| `transformers` | ≥ 4.50.0 | Hybrid `Nemotron3ForCausalLM` class registration |

---

## 6. NeMo 2.0 and FLARE Integration Code

### 6.1 NeMo 2.0 ModelTransform + LoRA Configuration

```python
from nemo.collections import llm
import nemo.lightning as nl

# Define LoRA targeting both Transformer and Mamba-2 layers
lora = llm.peft.LoRA(
    target_modules=[
        'linear_qkv',   # Transformer: fused QKV attention projection
        'linear_proj',  # Transformer: attention output projection
        'in_proj',      # Mamba-2: input projection into SSM
        'out_proj',     # Mamba-2: output projection from SSM
        'x_proj',       # Mamba-2: state-space input projection
        'dt_proj',      # Mamba-2: time-step discretization
    ],
    dim=32,             # LoRA rank r
    dropout=0.05
)

# Initialize trainer with LoRA as a PyTorch Lightning callback
trainer = nl.Trainer(
    strategy=nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    ),
    callbacks=[lora],
    max_epochs=1,
)

# Initialize frozen Nemotron model with LoRA transform
nemotron_model = llm.Nemotron3Model(model_transform=lora)
```

### 6.2 FLARE Federated Client Conversion

```python
import nvflare.client as flare

# Convert the local trainer into a federated FLARE client
# Zero changes to the training loop — patching is transparent
flare.patch(trainer)

# The patched trainer now:
# 1. Receives global weights from the NVFlare controller
# 2. Executes the local fitting process
# 3. Extracts and transmits resulting parameter updates through DXO filters
```

The `flare.patch(trainer)` call is the key FLARE integration point. NeMo's training loop continues unchanged; FLARE intercepts weight exchange and routes updates through the FedRandFilter → LaplacianDPFilter → AdaptiveQuantFilter chain before transmission.

### 6.3 FedNeMo's Position in the Codebase

```
NeMo (framework) — handles all GPU mechanics, mixed precision, gradient computation
    └── loads and fine-tunes → Nemotron (model)
                                   └── LoRA adapters (trained, fragmented, compressed)
                                           └── FedNeMo wraps the parameter exchange:
                                               ├── FedRandFilter     ← Stage 4a
                                               ├── LaplacianDPFilter ← Stage 4b (noise)
                                               ├── AdaptiveQuantFilter ← Stage 4b (compress)
                                               └── ModelController   ← Stage 5 (server)
```

FedNeMo's novel contribution is not inside `llm.finetune()` — it is the federated coordination and privacy layer around it.

---

## 7. Compute and GPU Considerations

### 7.1 The CUDA Constraint

NeMo requires CUDA. AMD GPUs cannot run NeMo directly (AMD uses ROCm).

If you have AMD AI developer credits: use them for preprocessing, benchmarking CPU-heavy components (FedPS statistics, MTAE training on embeddings), and algorithm correctness verification. AMD credits cannot run the NeMo + Nemotron fine-tuning loop.

### 7.2 NVIDIA Compute Access Timeline

| Option | Access Path | Cost | Notes |
| :--- | :--- | :--- | :--- |
| Lambda Labs | Cloud A100s | ~$1.10/hour | Most cost-effective; run hospital simulation here |
| Google Colab Pro+ | A100 access | Subscription | Fast iteration on smaller experiments |
| NVIDIA Pune Workshop (June 26) | On-site hardware | Free (apply) | Pre-hackathon relationship building; mention FedNeMo specifically |
| Hackathon hardware (July 24–25) | NVIDIA-provided | Free for selected teams | Guaranteed access during final event |

**Development strategy:** Build and test the federated training logic and all algorithmic components on CPU first (it is just Python + math). The algorithm correctness does not need a GPU. Port to NVIDIA cloud for actual NeMo + Nemotron runs and scale testing.

### 7.3 Nemotron-Mini-4B Practical Footprint

On a single A100 (80GB): comfortably fits 2–3 simultaneous federated client processes with QLoRA. Suitable for the 3–5 hospital simulation in the demo. This is the correct target for development and demonstration.

Nemotron-3 Nano (30B-A3B): requires 3+ high-memory A100s for 5 simultaneous clients even with QLoRA. This is the production scalability target, not the demo model.
