"""Model backends behind the Update contract (Phase 4).

A `ModelBackend` is the *only* thing that knows how a hospital "trains". Every
backend produces and consumes the exact same shapes the rest of the pipeline
already speaks:

  - global weights:  nested dict  {layer_id: {"A": ndarray, "B": ndarray}}
  - an update's tensors: flat dict {f"{layer_id}.A": ndarray, f"{layer_id}.B": ndarray}
                          (LoRA *deltas*, i.e. trained_weights - received_global)

Two backends:

  StubBackend  -- the Phase 0-3 fake trainer, verbatim: deep-copy the global
                  weights, nudge with small Gaussian noise, return the deltas.
                  Pure NumPy, no model download, toy LoRA dims from schema.

  HFBackend    -- real LoRA fine-tune of a HuggingFace causal LM via peft.
                  Backend-agnostic across CUDA / ROCm / CPU (device is chosen
                  from torch.cuda.is_available(); ROCm exposes itself as cuda).

server.py / filters.py / attacker.py / dashboard.py are untouched: they operate
generically on the tensor dicts above. The HFBackend registers its real LoRA
layer ids into schema.LAYERS (via schema.set_layers) so server.py's
`for layer in LAYERS` aggregation just works for a real model.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np

from . import schema


class ModelBackend(ABC):
    """Interface every backend implements."""

    @abstractmethod
    def init_global_weights(self) -> dict:
        """Return the initial global LoRA state: {layer: {"A": nd, "B": nd}}."""

    @abstractmethod
    def train_step(self, global_weights: dict, shard: list, rnd: int,
                   client_id: str, seed: int | None = None) -> tuple[dict, int]:
        """Locally fine-tune from `global_weights`; return (delta_tensors, num_samples).

        delta_tensors is a flat dict {f"{layer}.A": nd, f"{layer}.B": nd}.
        """

    @abstractmethod
    def eval_loss(self, global_weights: dict) -> float:
        """A scalar that should move as the global model changes."""


# ---------------------------------------------------------------------------
# StubBackend -- Phase 0-3 fake training, unchanged behavior.
# ---------------------------------------------------------------------------
class StubBackend(ModelBackend):
    def __init__(self, seed: int | None = None, lr: float = 0.01):
        self.lr = lr
        self.rng = np.random.default_rng(seed)

    def init_global_weights(self) -> dict:
        return schema.init_weights()

    def train_step(self, global_weights, shard, rnd, client_id, seed=None):
        tensors = {}
        for layer in schema.LAYERS:
            for mat in ("A", "B"):
                base = global_weights[layer][mat]
                delta = self.lr * self.rng.standard_normal(base.shape)
                tensors[f"{layer}.{mat}"] = delta  # local - base == noise
        return tensors, len(shard)

    def eval_loss(self, global_weights) -> float:
        vals = [abs(global_weights[l]["A"]).mean() for l in schema.LAYERS]
        return float(sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# HFBackend -- real LoRA fine-tune via transformers + peft.
# ---------------------------------------------------------------------------
def _pick_device():
    import torch
    # ROCm builds report availability through the cuda API as well.
    return "cuda" if torch.cuda.is_available() else "cpu"


def _clean_layer_id(param_name: str) -> str:
    """'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'
    -> 'model.layers.0.self_attn.q_proj'."""
    name = param_name
    for marker in (".lora_A", ".lora_B"):
        if marker in name:
            name = name.split(marker)[0]
            break
    if name.startswith("base_model.model."):
        name = name[len("base_model.model."):]
    return name


class HFBackend(ModelBackend):
    def __init__(self, model_name: str | None = None,
                 target_modules: list[str] | None = None,
                 rank: int = schema.LORA_RANK,
                 device: str | None = None,
                 dtype=None,
                 seq_len: int = 128,
                 lr: float = 5e-4,
                 local_steps: int = 2,
                 train_batch: int = 2,
                 prompt_fn=None,
                 max_eval: int = 6,
                 model=None,
                 tokenizer=None):
        import torch
        from peft import LoraConfig, get_peft_model

        self.torch = torch
        self.device = device or _pick_device()
        if dtype is None:
            dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.seq_len = seq_len
        self.lr = lr
        self.local_steps = local_steps
        self.train_batch = train_batch
        self.max_eval = max_eval
        self.prompt_fn = prompt_fn or (lambda row: row.get("text", str(row)))
        target_modules = target_modules or list(schema.TARGET_MODULES)

        # Tokenizer / base model: either injected (offline tests) or downloaded.
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None)

        if model is not None:
            base = model
        else:
            from transformers import AutoModelForCausalLM
            base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        cfg = LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias="none",
            target_modules=target_modules, task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, cfg).to(self.device)

        # Map clean layer id -> {"A": Parameter, "B": Parameter}.
        self._params: dict[str, dict] = {}
        for name, p in self.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                lid = _clean_layer_id(name)
                mat = "A" if "lora_A" in name else "B"
                self._params.setdefault(lid, {})[mat] = p
        # Keep only layers that have BOTH A and B (paired LoRA layers).
        self.layer_ids = sorted(
            lid for lid, d in self._params.items() if "A" in d and "B" in d
        )
        if not self.layer_ids:
            raise RuntimeError(
                "No paired LoRA layers found. Check target_modules "
                f"({target_modules}) against the model's module names."
            )
        # Register real ids so server.py's `for layer in LAYERS` aggregates them.
        schema.set_layers(self.layer_ids)
        self._eval_texts: list[str] = []

    # --- weight I/O ---------------------------------------------------------
    def init_global_weights(self) -> dict:
        return self.get_weights()

    def get_weights(self) -> dict:
        out = {}
        for lid in self.layer_ids:
            out[lid] = {
                "A": self._params[lid]["A"].detach().float().cpu().numpy().copy(),
                "B": self._params[lid]["B"].detach().float().cpu().numpy().copy(),
            }
        return out

    def set_weights(self, global_weights: dict) -> None:
        torch = self.torch
        with torch.no_grad():
            for lid in self.layer_ids:
                for mat in ("A", "B"):
                    p = self._params[lid][mat]
                    arr = global_weights[lid][mat]
                    p.data.copy_(torch.as_tensor(arr, dtype=p.dtype,
                                                 device=p.device))

    # --- batching -----------------------------------------------------------
    def _texts(self, shard: list) -> list[str]:
        return [self.prompt_fn(row) for row in shard if row]

    def _loss_on(self, texts: list[str]) -> "object":
        """Mean LM loss over `texts` (one sequence at a time, summed)."""
        torch = self.torch
        if not texts:
            return torch.tensor(0.0, device=self.device)
        total = torch.tensor(0.0, device=self.device)
        for t in texts:
            enc = self.tokenizer(t, truncation=True, max_length=self.seq_len,
                                 return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=input_ids)
            total = total + out.loss
        return total / len(texts)

    # --- training -----------------------------------------------------------
    def train_step(self, global_weights, shard, rnd, client_id, seed=None):
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed * 1000 + rnd)
        self.set_weights(global_weights)

        # Remember a few examples for eval_loss (first time we see data).
        texts = self._texts(shard)
        if not self._eval_texts and texts:
            self._eval_texts = texts[:self.max_eval]

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=self.lr)
        self.model.train()
        for step in range(self.local_steps):
            start = (step * self.train_batch) % max(1, len(texts))
            batch = texts[start:start + self.train_batch] or texts[:self.train_batch]
            if not batch:
                break
            loss = self._loss_on(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

        new = self.get_weights()
        tensors = {}
        for lid in self.layer_ids:
            tensors[f"{lid}.A"] = new[lid]["A"] - global_weights[lid]["A"]
            tensors[f"{lid}.B"] = new[lid]["B"] - global_weights[lid]["B"]
        return tensors, len(shard)

    def eval_loss(self, global_weights) -> float:
        torch = self.torch
        self.set_weights(global_weights)
        self.model.eval()
        with torch.no_grad():
            val = self._loss_on(self._eval_texts)
        return float(val.detach().cpu())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_backend(kind: str = "stub", *, seed: int | None = None, **kwargs) -> ModelBackend:
    """kind: 'stub' (CPU, no download) or 'hf' (real model via transformers/peft)."""
    if kind == "stub":
        return StubBackend(seed=seed)
    if kind == "hf":
        return HFBackend(**kwargs)
    raise ValueError(f"unknown backend kind: {kind!r}")
