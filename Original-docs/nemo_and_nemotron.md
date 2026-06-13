Good question to ask before building anything. Let me be completely direct.

---

## NeMo

NeMo is a **framework**. Think of it like PyTorch Lightning but built specifically for large language models. It's NVIDIA's open-source toolkit for building, training, fine-tuning, and deploying LLMs.

What it actually gives you:

- Infrastructure for distributed training across multiple GPUs (tensor parallelism, pipeline parallelism)
- Built-in support for fine-tuning methods like LoRA, P-Tuning, Prefix Tuning
- Model checkpointing, mixed precision (BF16), gradient accumulation
- A clean Python API to define training jobs
- Integration with TensorRT-LLM for optimized inference after training
- NeMo Guardrails — a separate module for making LLM outputs safe and controlled

NeMo is the **kitchen**. It has all the tools. You bring the model.

---

## Nemotron

Nemotron is NVIDIA's **family of actual LLMs**. These are pre-trained models with weights, tokenizers, and architectures — the things you actually fine-tune.

Current lineup that matters for us:

**Nemotron-Mini-4B** — 4 billion parameters. Fits on a single A100. Fast to fine-tune. This is your practical choice for the hackathon.

**Nemotron-4 15B / 340B** — Larger frontier models. Too heavy for a hackathon demo with multiple simulated hospitals running in parallel.

**Nemotron-H** — Brand new. Hybrid architecture combining Mamba (SSM) layers with standard attention layers. This is NVIDIA's answer to efficiency at scale. Mentioned this earlier as the cutting-edge angle — still relevant but riskier to work with since documentation is thin.

Nemotron is the **dish you cook in the kitchen**.

---

## The Relationship

```
NeMo (framework)
    └── loads and fine-tunes → Nemotron (model)
                                   └── LoRA adapters sit on top
                                           └── YOUR federated training loop wraps everything
```

NeMo handles all the GPU-level mechanics. Nemotron is what you're actually teaching new medical knowledge to. Your FedNeMo code lives in the middle — intercepting the LoRA parameter flow between hospitals and the server.

---

## Which One Do You Need

Both, but for different reasons.

You need **Nemotron-Mini-4B** because Track B explicitly requires using a Nemotron model. It's also genuinely the right size — small enough to run 5 instances simultaneously for your hospital simulation, large enough to produce meaningful clinical NLP results.

You need **NeMo** because it's the fine-tuning framework the judges expect you to use, and because it gives you LoRA support out of the box so you're not writing transformer fine-tuning from scratch. Your FedNeMo code hooks into NeMo's training loop rather than reinventing it.

---

## How They Actually Go Into FedNeMo

Here's what the code structure looks like in plain terms.

NeMo 2.0 fine-tuning in its standard (non-federated) form looks like this:

```python
from nemo.collections.llm import api as llm
from nemo.collections.llm.peft import LoRA

# Load Nemotron-Mini-4B
model = llm.import_ckpt("nvidia/nemotron-mini-4b")

# Attach LoRA adapters (rank 8)
peft = LoRA(dim=8, alpha=16)

# Standard fine-tuning
llm.finetune(
    model=model,
    data=hospital_data_module,
    trainer=trainer,
    peft=peft
)
```

Your FedNeMo wrapper intercepts this at the point where LoRA parameters would normally just be saved locally. Instead, after each local training round on each hospital's data, you:

```python
# After local training on Hospital i
lora_A, lora_B = extract_lora_params(model)  # pull out LoRA matrices

# FedRand: randomly decide which half is public this round
if random() < 0.5:
    public_params = lora_A   # send A, keep B private
else:
    public_params = lora_B   # send B, keep A private

# Laplacian DP: add noise before transmission
noised_params = add_laplacian_noise(public_params, sensitivity, epsilon)

# Adaptive quantization: compress based on this hospital's entropy score
quantized_params = adaptive_quantize(noised_params, bit_length[hospital_i])

# Send to server
server.receive(hospital_id=i, params=quantized_params)
```

On the server side:

```python
# Aggregate from all hospitals
aggregated = fedavg_aggregate(all_hospital_params)

# Update global LoRA weights
update_global_model(aggregated)

# Broadcast back
for hospital in hospitals:
    hospital.receive_global_model(aggregated)
```

NeMo handles everything inside the `llm.finetune()` call — GPU management, mixed precision, gradient computation. You're building the federated coordination layer around it, not inside it.

---

## The One Problem You Need to Know About

Your AMD AI developer credits run ROCm, not CUDA. **NeMo requires CUDA.** AMD GPUs cannot run NeMo directly.

This means your AMD credits are useful for preprocessing work, benchmarking CPU-heavy components, and non-NeMo experimentation — but for the actual NeMo + Nemotron fine-tuning you need NVIDIA GPUs.

Practical solutions for development before the hackathon:

**Lambda Labs** — cloud NVIDIA A100s, around $1.10/hour. Most cost-effective for this. Run your hospital simulation there.

**Google Colab Pro+** — has A100 access, useful for quick iteration.

**The June 26 workshop at Pune NVIDIA office** — you applied for this. This is your best shot at getting hands-on NVIDIA hardware before the final. Go to this, talk to the engineers there, mention FedNeMo specifically. That's a pre-hackathon relationship building opportunity.

**The hackathon itself (July 24-25)** — NVIDIA provides compute access for all selected teams during the final days.

So the development plan is: build and test the federated training logic on CPU/AMD first (it's just Python + math), then port to NVIDIA cloud for actual NeMo + Nemotron runs. The algorithm correctness doesn't need a GPU. The scale tests do.

---

## Bottom Line

NeMo = the framework you build inside. Nemotron-Mini-4B = the model you're fine-tuning. You need both. Your actual novel contribution (the federated privacy layer) lives around them, not inside them. That's the right mental model to start building from.
