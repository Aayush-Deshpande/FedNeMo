"""Minimal gradient-inversion (GIA) sanity check.

We test the CORE structural privacy claim of FedRand directly, at the LoRA update
level, rather than running a heavy full-model DLG optimization.

The quantity an honest-but-curious server wants is each layer's effective weight
update  dW = B @ A  (which drives what the client's data taught the model).

- UNPROTECTED (vanilla FedAvg): the server receives BOTH A and B, so it
  reconstructs dW exactly -> cosine(recon, true) = 1.0, relative error = 0.
- FEDNEMO (FedRand + Laplace DP + 2-bit quant): the server receives only ONE of
  {A, B} per layer (the other is private and never transmitted), and the shared
  one is DP-noised + quantized. To form dW the adversary MUST guess the missing
  matrix. We give it the strongest simple guess (a random matrix at the correct
  scale) and measure how close the reconstruction is.

Metric: cosine similarity and relative Frobenius error between reconstructed dW
and the true dW, averaged over layers. Low cosine / high error => the update is
not recoverable => the inversion is underdetermined, substantiating the claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from ..config import CONFIG
from ..federated.dp import clip_and_add_laplace
from ..federated.fedrand import LORA_A_SUFFIX, LORA_B_SUFFIX, AdapterState, layer_keys
from ..federated.quantization import dequantize, quantize


@dataclass
class GIAResult:
    n_layers: int
    unprotected_cos: float
    unprotected_relerr: float
    fednemo_cos: float
    fednemo_relerr: float


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _relerr(recon: torch.Tensor, true: torch.Tensor) -> float:
    d = (recon - true).norm().item()
    t = true.norm().item()
    return d / t if t > 0 else 0.0


def run_gia(state: AdapterState, seed: int = 0, max_layers: int = 48) -> GIAResult:
    """Compare update reconstruction: unprotected (has A&B) vs FedNeMo (one matrix
    + DP + quant, guess the other)."""
    g = torch.Generator(); g.manual_seed(seed)
    up_cos: List[float] = []
    up_err: List[float] = []
    fn_cos: List[float] = []
    fn_err: List[float] = []

    for layer in layer_keys(state)[:max_layers]:
        A = state.get(layer + LORA_A_SUFFIX)
        B = state.get(layer + LORA_B_SUFFIX)
        if A is None or B is None:
            continue
        A = A.float(); B = B.float()
        dW_true = B @ A  # [out, in]

        # --- unprotected: adversary has both A and B ---
        dW_up = B @ A
        up_cos.append(_cos(dW_up, dW_true))
        up_err.append(_relerr(dW_up, dW_true))

        # --- FedNeMo: share one matrix (+DP+quant); private half is unknown ---
        share_a = bool(torch.bernoulli(torch.tensor(CONFIG.fedrand_share_prob), generator=g).item())
        if share_a:
            shared = A
            dp = clip_and_add_laplace(shared, CONFIG.dp_clip_norm, CONFIG.dp_epsilon, CONFIG.dp_clip_type)
            A_obs = dequantize(quantize(dp.tensor, bits=CONFIG.quant_bits)).view_as(A)
            # B is private -> adversary guesses it (random at matching scale)
            B_guess = torch.randn(B.shape, generator=g) * (B.std().item() + 1e-6)
            dW_fn = B_guess @ A_obs
        else:
            shared = B
            dp = clip_and_add_laplace(shared, CONFIG.dp_clip_norm, CONFIG.dp_epsilon, CONFIG.dp_clip_type)
            B_obs = dequantize(quantize(dp.tensor, bits=CONFIG.quant_bits)).view_as(B)
            A_guess = torch.randn(A.shape, generator=g) * (A.std().item() + 1e-6)
            dW_fn = B_obs @ A_guess
        fn_cos.append(_cos(dW_fn, dW_true))
        fn_err.append(_relerr(dW_fn, dW_true))

    n = len(up_cos)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return GIAResult(
        n_layers=n,
        unprotected_cos=mean(up_cos), unprotected_relerr=mean(up_err),
        fednemo_cos=mean(fn_cos), fednemo_relerr=mean(fn_err),
    )
