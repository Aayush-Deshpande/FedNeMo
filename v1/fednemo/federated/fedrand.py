"""FedRand: randomized LoRA subparameter split with per-round privacy.

For every LoRA layer, each client flips a Bernoulli(rho) coin per round:
  - z = 1  -> the client SHARES its updated A matrix; keeps B private (local).
  - z = 0  -> the client SHARES its updated B matrix; keeps A private (local).

The aggregator therefore never receives both A and B of the same layer from the
same client in the same round, keeping any gradient-inversion optimization
underdetermined. The non-shared ("private") matrix persists on the client across
rounds and is re-used as that client's local half.

This module provides the full mechanism end-to-end:
  - adapter state extraction / injection (peft LoRA tensors)
  - per-layer Bernoulli split into public / private
  - Laplace DP noise on the public matrix (dp.py)
  - 2-bit adaptive quantization of the noised public matrix (quantization.py)
  - packaging into a transmittable payload

No stubs: every step operates on real peft LoRA parameters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .dp import clip_and_add_laplace
from .quantization import QuantizedTensor, dequantize, quantize

logger = logging.getLogger("fednemo.fedrand")

# A LoRA layer is identified by the module path; peft stores A/B as:
#   <path>.lora_A.default.weight   and   <path>.lora_B.default.weight
LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"


AdapterState = Dict[str, torch.Tensor]  # param name -> tensor (CPU float32)


def extract_adapter_state(model: torch.nn.Module) -> AdapterState:
    """Snapshot all LoRA A/B weights as CPU float32 tensors."""
    state: AdapterState = {}
    for name, param in model.named_parameters():
        if name.endswith(LORA_A_SUFFIX) or name.endswith(LORA_B_SUFFIX):
            state[name] = param.detach().to("cpu", dtype=torch.float32).clone()
    return state


@torch.no_grad()
def load_adapter_state(model: torch.nn.Module, state: AdapterState) -> None:
    """Write A/B weights from `state` back into the model's LoRA parameters."""
    name_to_param = dict(model.named_parameters())
    for name, tensor in state.items():
        if name not in name_to_param:
            raise KeyError(f"LoRA param {name} not found in model.")
        p = name_to_param[name]
        p.data.copy_(tensor.to(p.device, dtype=p.dtype))


def layer_keys(state: AdapterState) -> List[str]:
    """Return the set of LoRA layer base-paths present in an adapter state."""
    keys = set()
    for name in state:
        if name.endswith(LORA_A_SUFFIX):
            keys.add(name[: -len(LORA_A_SUFFIX)])
        elif name.endswith(LORA_B_SUFFIX):
            keys.add(name[: -len(LORA_B_SUFFIX)])
    return sorted(keys)


@dataclass
class SharedMatrix:
    """A single client's shared (public) matrix for one LoRA layer, post DP+quant."""
    layer: str
    which: str                       # "A" or "B"
    quantized: QuantizedTensor
    clipped_norm: float
    noise_scale: float


@dataclass
class ClientPayload:
    """Everything a client transmits to the aggregator for one round."""
    client_id: int
    shared: List[SharedMatrix] = field(default_factory=list)
    # summary stats for entropy weighting + trust scoring (text-based)
    num_samples: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    loss_trajectory: List[float] = field(default_factory=list)
    update_l2_norm: float = 0.0


def split_and_protect(
    client_id: int,
    updated_state: AdapterState,
    share_prob: float,
    clip_norm: float,
    epsilon: float,
    quant_bits: int,
    rng: torch.Generator,
    clip_type: str = "l1",
    capture: Optional[List[dict]] = None,
) -> Tuple[ClientPayload, AdapterState]:
    """Perform the full FedRand split + DP + quantization for one client update.

    Args:
        updated_state: the client's LoRA A/B weights AFTER local training.
        rng: torch Generator for reproducible Bernoulli draws.
        capture: if a list is passed, a per-layer audit record (trained A/B ->
            sent/dropped -> post-DP -> post-quant) is appended for each layer.

    Returns:
        (payload, private_state) where:
          - payload contains the noised+quantized public matrices to transmit;
          - private_state holds the matrices the client keeps (not transmitted),
            so they persist locally across rounds.
    """
    payload = ClientPayload(client_id=client_id)
    private_state: AdapterState = {}

    total_sq = 0.0
    for layer in layer_keys(updated_state):
        a_name = layer + LORA_A_SUFFIX
        b_name = layer + LORA_B_SUFFIX
        a = updated_state.get(a_name)
        b = updated_state.get(b_name)
        if a is None or b is None:
            continue

        share_a = bool(torch.bernoulli(torch.tensor(share_prob), generator=rng).item())
        if share_a:
            public, which, public_name, private_name, private_tensor = a, "A", a_name, b_name, b
        else:
            public, which, public_name, private_name, private_tensor = b, "B", b_name, a_name, a

        # keep the private half locally (persist across rounds)
        private_state[private_name] = private_tensor.clone()

        # DP (Laplace) on the public matrix, then 2-bit adaptive quantization
        dp = clip_and_add_laplace(public, clip_norm=clip_norm, epsilon=epsilon, clip_type=clip_type)
        qt = quantize(dp.tensor, bits=quant_bits)
        payload.shared.append(
            SharedMatrix(
                layer=layer, which=which, quantized=qt,
                clipped_norm=dp.clipped_norm, noise_scale=dp.noise_scale,
            )
        )
        total_sq += float(torch.sum(public ** 2).item())

        if capture is not None:
            from ..audit import build_layer_capture
            capture.append(build_layer_capture(
                layer=layer, which=which, trained_a=a, trained_b=b,
                shared_pre_dp=public, post_dp=dp, dequantized=dequantize(qt), qt=qt,
            ))

    payload.update_l2_norm = total_sq ** 0.5
    return payload, private_state


def reconstruct_shared(payload: ClientPayload) -> Dict[Tuple[str, str], torch.Tensor]:
    """Dequantize a payload's shared matrices into {(layer, which): tensor}."""
    out: Dict[Tuple[str, str], torch.Tensor] = {}
    for sm in payload.shared:
        out[(sm.layer, sm.which)] = dequantize(sm.quantized)
    return out
