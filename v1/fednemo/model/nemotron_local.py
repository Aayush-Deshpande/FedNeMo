"""Local Nemotron-Mini-4B loader with a VRAM-bounded layout.

Verified layout (peak training footprint ~2.3 GB on an RTX 4050, < 3 GB budget):
  - embed_tokens (256k x 3072 nn.Embedding)  -> CPU, float32  (cheap gather)
  - transformer body (32 decoder layers)      -> GPU, 4-bit NF4 (LoRA trains here)
  - final norm                                 -> GPU
  - lm_head (256k x 3072)                      -> CPU, float32  (fast MKL matmul)

Keeping the 256k-vocab embed + lm_head off-GPU is what makes training fit: their
bf16 tensors are ~1.5 GB each and, on GPU, the backward through lm_head must
materialize the full weight, spiking well past 3 GB. On CPU in float32 the matmul
is fast (MKL) and the huge logits tensor never touches the GPU.

The forward pass is executed manually (embed -> body -> lm_head) so we control
exactly where each stage runs. Cross-device autograd links the CPU loss to the
GPU LoRA parameters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_module
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..config import CONFIG

logger = logging.getLogger("fednemo.model")

GPU = 0


@dataclass
class LoadedModel:
    """Bundle of the loaded PEFT model plus direct handles to split modules."""
    model: torch.nn.Module           # PEFT-wrapped NemotronForCausalLM
    tokenizer: object
    embed: torch.nn.Module           # embed_tokens on CPU (float32)
    body: torch.nn.Module            # NemotronModel (layers on GPU, embed detached)
    lm_head: torch.nn.Module         # lm_head on GPU (4-bit) or CPU (float32)
    lm_head_device: str = "cuda"     # where lm_head runs ("cuda" or "cpu")

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]


def _bnb_config(quantize_lm_head: bool) -> BitsAndBytesConfig:
    kwargs = dict(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    if quantize_lm_head:
        # Override the default skip so lm_head is quantized to 4-bit on GPU
        # (~0.4 GB) instead of staying bf16. embed_tokens (nn.Embedding on CPU)
        # is never quantizable by bnb anyway, so skipping it is a no-op there.
        kwargs["llm_int8_skip_modules"] = ["embed_tokens"]
    return BitsAndBytesConfig(**kwargs)


def load_nemotron(attach_lora: bool = True) -> LoadedModel:
    """Load the local Nemotron checkpoint in a VRAM-bounded split layout.

    lm_head placement is driven by CONFIG.lm_head_device:
      - "cuda": lm_head is 4-bit on GPU (fast; ~4.4 GB peak; needs ~4.8 GB budget)
      - "cpu" : lm_head is float32 on CPU (slower matmul; ~2.4 GB peak)
    embed_tokens always stays on CPU (cheap gather; GPU copy would cost ~1.5 GB).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the 4-bit transformer body.")

    lm_head_on_gpu = CONFIG.lm_head_device.lower() == "cuda"

    model_path = CONFIG.model_path
    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # embed_tokens -> CPU always; lm_head -> GPU or CPU per config; body+norm -> GPU.
    device_map = {
        "model.embed_tokens": "cpu",
        "lm_head": GPU if lm_head_on_gpu else "cpu",
        "model.norm": GPU,
    }
    # NemotronForCausalLM has 32 decoder layers (see config.json num_hidden_layers).
    for i in range(32):
        device_map[f"model.layers.{i}"] = GPU

    logger.info("Loading Nemotron body in 4-bit NF4 (lm_head on %s, embed on CPU)...",
                "GPU" if lm_head_on_gpu else "CPU")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=_bnb_config(quantize_lm_head=lm_head_on_gpu),
        device_map=device_map,
        dtype=torch.bfloat16,
    )

    if attach_lora:
        lora_cfg = LoraConfig(
            r=CONFIG.lora_rank,
            lora_alpha=CONFIG.lora_alpha,
            lora_dropout=CONFIG.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=CONFIG.lora_target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        base = model.base_model.model            # NemotronForCausalLM
    else:
        base = model

    embed = base.model.embed_tokens
    lm_head = base.lm_head

    # Detach accelerate offload hooks from CPU-resident modules so they run
    # purely on CPU (otherwise the hook would shuttle the weight to GPU each call).
    remove_hook_from_module(embed)
    embed.weight.data = embed.weight.data.float()   # fast CPU gather
    if not lm_head_on_gpu:
        remove_hook_from_module(lm_head)
        lm_head.weight.data = lm_head.weight.data.float()  # fast MKL matmul

    body = base.model  # NemotronModel; embed is a submodule but we call it manually

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model loaded. Trainable params: %s", f"{trainable:,}")
    _log_vram("after load")

    return LoadedModel(
        model=model, tokenizer=tokenizer, embed=embed, body=body, lm_head=lm_head,
        lm_head_device="cuda" if lm_head_on_gpu else "cpu",
    )


def forward_logits(
    lm: LoadedModel,
    input_ids: torch.Tensor,
    last_token_only: bool = False,
) -> torch.Tensor:
    """Manual forward: CPU embed -> GPU body -> (GPU|CPU) lm_head.

    input_ids: LongTensor [batch, seq] on CPU. Returns logits on the lm_head's
    device (kept in the head's native dtype to save memory; callers upcast the
    small 2D view they actually need).

    last_token_only: if True, only the final position is projected through the
    256k-vocab head. Used during generation - avoids materializing a full
    [b, s, vocab] tensor for every decoding step.
    """
    inputs_embeds = lm.embed(input_ids)                 # CPU float32 [b, s, 3072]
    inputs_embeds = inputs_embeds.to(GPU, dtype=torch.bfloat16)
    hidden = lm.body(inputs_embeds=inputs_embeds).last_hidden_state  # GPU bf16
    if last_token_only:
        hidden = hidden[:, -1:, :]                      # [b, 1, 3072]
    if lm.lm_head_device == "cuda":
        logits = lm.lm_head(hidden)                     # GPU, head dtype
    else:
        hidden_cpu = hidden.float().to("cpu")           # keep autograd link CPU-side
        logits = lm.lm_head(hidden_cpu)                 # CPU float32
    return logits


def compute_causal_loss(
    lm: LoadedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Next-token cross-entropy with -100 masking.

    Optimization: the loss only depends on the completion tokens (labels != -100),
    which are ~10 of ~150 positions. So we run the body over the full sequence
    (needed for context) but project ONLY the labeled positions through the
    256k-vocab lm_head. This cuts the expensive vocab projection ~15x and keeps
    the logits tensor tiny (big speed + memory win vs projecting every position).
    """
    inputs_embeds = lm.embed(input_ids)                       # CPU float32 [b, s, 3072]
    inputs_embeds = inputs_embeds.to(GPU, dtype=torch.bfloat16)
    hidden = lm.body(inputs_embeds=inputs_embeds).last_hidden_state  # GPU [b, s, 3072]

    shift_hidden = hidden[:, :-1, :]                          # position t predicts t+1
    shift_labels = labels[:, 1:].to(hidden.device)
    mask = shift_labels != -100
    if mask.sum() == 0:                                       # safety: nothing to learn
        return hidden.sum() * 0.0

    sel_hidden = shift_hidden[mask]                           # [num_labeled, 3072]
    sel_labels = shift_labels[mask]                           # [num_labeled]

    if lm.lm_head_device == "cuda":
        logits = lm.lm_head(sel_hidden).float()              # [num_labeled, vocab]
    else:
        logits = lm.lm_head(sel_hidden.float().to("cpu"))
        sel_labels = sel_labels.to("cpu")
    return F.cross_entropy(logits, sel_labels)


@torch.no_grad()
def generate(lm: LoadedModel, prompt: str, max_new_tokens: int = 96) -> str:
    """Greedy generation using the manual split forward (inference path)."""
    tok = lm.tokenizer
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=CONFIG.max_seq_len)
    input_ids = enc["input_ids"]  # CPU
    eos_id = tok.eos_token_id
    for _ in range(max_new_tokens):
        logits = forward_logits(lm, input_ids, last_token_only=True)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
        if next_id == eos_id:
            break
        if input_ids.shape[1] > CONFIG.max_seq_len + max_new_tokens:
            break
    full = tok.decode(input_ids[0], skip_special_tokens=True)
    return full[len(tok.decode(enc["input_ids"][0], skip_special_tokens=True)):].strip()


def _log_vram(tag: str) -> None:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(GPU) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(GPU) / (1024 ** 2)
        logger.info("[VRAM %s] allocated=%.0f MiB reserved=%.0f MiB", tag, alloc, reserved)
