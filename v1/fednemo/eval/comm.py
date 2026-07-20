"""Communication-cost accounting: FedNeMo (FedRand + 2-bit quant) vs a 32-bit
full-FedAvg baseline transmitting both LoRA matrices uncompressed.

Baseline (per client, per round): every LoRA tensor (all A and B) at 32 bits.
FedNeMo (per client, per round): FedRand shares exactly ONE matrix per layer
(≈ half the tensors), each quantized to `quant_bits` bits, plus per-tensor
metadata (scale float32 + zero_point int32 = 8 bytes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from ..config import CONFIG
from ..federated.fedrand import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    AdapterState,
    layer_keys,
    split_and_protect,
)


@dataclass
class CommReport:
    baseline_bytes: int          # 32-bit full FedAvg, both matrices
    fednemo_bytes: int           # FedRand half + quant + metadata
    fedrand_only_bytes: int      # FedRand half at 32-bit (isolates the split effect)
    savings_pct: float
    quant_bits: int
    per_tensor_meta_bytes: int


def baseline_fedavg_bytes(state: AdapterState) -> int:
    """All A and B tensors at 32 bits/element."""
    total = 0
    for name, t in state.items():
        if name.endswith(LORA_A_SUFFIX) or name.endswith(LORA_B_SUFFIX):
            total += t.numel() * 4
    return total


def measure_payload_bytes(state: AdapterState, quant_bits: int, meta_bytes: int = 8) -> Dict[str, int]:
    """Simulate one FedRand split+quant and count transmitted bytes."""
    rng = torch.Generator(); rng.manual_seed(CONFIG.seed)
    payload, _ = split_and_protect(
        client_id=0, updated_state=state, share_prob=CONFIG.fedrand_share_prob,
        clip_norm=CONFIG.dp_clip_norm, epsilon=CONFIG.dp_epsilon,
        quant_bits=quant_bits, rng=rng,
    )
    quant_bytes = 0
    fp32_half_bytes = 0
    for sm in payload.shared:
        n = 1
        for d in sm.quantized.shape:
            n *= d
        # packed codes at quant_bits/element (bit-packed), rounded up to bytes
        quant_bytes += (n * quant_bits + 7) // 8 + meta_bytes
        fp32_half_bytes += n * 4
    return {"quant_bytes": quant_bytes, "fp32_half_bytes": fp32_half_bytes}


def comm_report(state: AdapterState, quant_bits: int | None = None) -> CommReport:
    quant_bits = quant_bits or CONFIG.quant_bits
    baseline = baseline_fedavg_bytes(state)
    m = measure_payload_bytes(state, quant_bits)
    fednemo = m["quant_bytes"]
    savings = 100.0 * (1.0 - fednemo / baseline) if baseline else 0.0
    return CommReport(
        baseline_bytes=baseline, fednemo_bytes=fednemo,
        fedrand_only_bytes=m["fp32_half_bytes"], savings_pct=savings,
        quant_bits=quant_bits, per_tensor_meta_bytes=8,
    )
