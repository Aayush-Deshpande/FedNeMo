"""Adaptive fixed-width quantization of transmitted matrices.

Fixed 2-bit width with adaptive per-tensor scale and zero-point (asymmetric
uniform affine quantization). This is the classic adaptive-scale scheme (not the
cosine-annealed variable-bit scheme from the superseded design docs).

Quantize:   q = round(x / scale + zero_point),  clamped to [0, 2^bits - 1]
Dequantize: x_hat = (q - zero_point) * scale
where  scale = (max - min) / (2^bits - 1),  zero_point = round(-min / scale).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuantizedTensor:
    q: torch.Tensor          # uint8 codes (values in [0, 2^bits - 1])
    scale: float
    zero_point: int
    bits: int
    shape: torch.Size
    constant_value: float = 0.0   # used only when the source tensor was constant

    def num_levels(self) -> int:
        return 2 ** self.bits


def quantize(x: torch.Tensor, bits: int = 2) -> QuantizedTensor:
    """Adaptive per-tensor affine quantization to `bits` bits."""
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8].")
    x = x.detach().to("cpu", dtype=torch.float32)
    x_min = float(x.min().item())
    x_max = float(x.max().item())
    levels = 2 ** bits - 1

    if x_max == x_min:
        # Degenerate (constant tensor): store the constant explicitly; codes are 0.
        q = torch.zeros_like(x, dtype=torch.uint8)
        return QuantizedTensor(
            q=q, scale=0.0, zero_point=0, bits=bits, shape=x.shape, constant_value=x_min
        )

    scale = (x_max - x_min) / levels
    zero_point = int(round(-x_min / scale))
    zero_point = max(0, min(levels, zero_point))

    q = torch.clamp(torch.round(x / scale + zero_point), 0, levels).to(torch.uint8)
    return QuantizedTensor(q=q, scale=scale, zero_point=zero_point, bits=bits, shape=x.shape)


def dequantize(qt: QuantizedTensor) -> torch.Tensor:
    """Reconstruct the float tensor from its quantized representation."""
    if qt.scale == 0.0:
        # constant tensor: reconstruct the stored constant value everywhere
        return torch.full(qt.shape, qt.constant_value, dtype=torch.float32)
    return (qt.q.to(torch.float32) - qt.zero_point) * qt.scale
