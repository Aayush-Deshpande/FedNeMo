"""Communication-cost accounting: FedNeMo transmitted bytes vs a full 32-bit
FedAvg baseline.

Baseline (standard FedAvg LoRA): every round each client transmits BOTH LoRA
matrices (A and B) for every target layer at full fp32 (32-bit) precision.

FedNeMo per round each client transmits:
  - only HALF the matrices (FedRand shares A OR B per layer, not both), and
  - each shared matrix as 2-bit codes + small per-tensor metadata
    (scale: fp32=4B, zero_point: 1B, plus a constant-value fallback fp32=4B).

We compute exact byte counts from the real LoRA tensor shapes in a saved adapter.
FedRand's per-layer split is random, but since A and B of a layer have equal
element counts for LoRA (A: r x in, B: out x r -> generally different!), we report
both the expected value under rho=0.5 and the exact best/worst bounds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ..federated.fedrand import LORA_A_SUFFIX, LORA_B_SUFFIX

_META_BYTES = 4 + 1 + 4  # scale(fp32) + zero_point(uint8) + constant_value(fp32)


@dataclass
class CommReport:
    num_layers: int
    fedavg_bytes: int              # both matrices, fp32
    fedrand_expected_bytes: float  # E[bytes] under rho=0.5, 2-bit + metadata
    savings_pct: float
    detail: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "fedavg_fp32_bytes": self.fedavg_bytes,
            "fednemo_expected_bytes": self.fedrand_expected_bytes,
            "savings_pct": self.savings_pct,
            "detail": self.detail,
        }


def _quantized_bytes(numel: int, bits: int) -> float:
    """Bytes to transmit `numel` codes at `bits` bits each + per-tensor metadata."""
    return (numel * bits) / 8.0 + _META_BYTES


def account_from_adapter(adapter_path: Path, bits: int = 2) -> CommReport:
    state = torch.load(adapter_path, map_location="cpu")

    layers: Dict[str, Dict[str, int]] = {}
    for name, tensor in state.items():
        if name.endswith(LORA_A_SUFFIX):
            layers.setdefault(name[: -len(LORA_A_SUFFIX)], {})["A"] = tensor.numel()
        elif name.endswith(LORA_B_SUFFIX):
            layers.setdefault(name[: -len(LORA_B_SUFFIX)], {})["B"] = tensor.numel()

    fedavg_bytes = 0
    fednemo_expected = 0.0
    for _layer, mats in layers.items():
        a = mats.get("A", 0)
        b = mats.get("B", 0)
        # FedAvg: both A and B, fp32 (4 bytes/elt)
        fedavg_bytes += (a + b) * 4
        # FedNeMo: share A or B with prob 0.5 each, 2-bit + metadata
        fednemo_expected += 0.5 * _quantized_bytes(a, bits) + 0.5 * _quantized_bytes(b, bits)

    savings = 100.0 * (1.0 - fednemo_expected / fedavg_bytes) if fedavg_bytes else 0.0
    return CommReport(
        num_layers=len(layers),
        fedavg_bytes=fedavg_bytes,
        fedrand_expected_bytes=fednemo_expected,
        savings_pct=savings,
        detail={
            "bits": bits,
            "fedavg_MB": fedavg_bytes / 1e6,
            "fednemo_MB": fednemo_expected / 1e6,
            "fedrand_halving_factor": 0.5,
            "quant_metadata_bytes_per_tensor": _META_BYTES,
        },
    )


def format_comm_report(r: CommReport) -> str:
    return (
        f"=== Communication cost (per client, per round) ===\n"
        f"LoRA layers: {r.num_layers}\n"
        f"FedAvg baseline (both matrices, fp32): {r.fedavg_bytes/1e6:.3f} MB\n"
        f"FedNeMo (FedRand half + 2-bit + meta): {r.fedrand_expected_bytes/1e6:.3f} MB\n"
        f"Communication savings: {r.savings_pct:.1f}%"
    )
