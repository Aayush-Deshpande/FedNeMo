"""DXO filter chain (client -> server).

The one rule that makes this whole design work: every filter takes an Update and
returns an Update (same shape in, same shape out). That uniform contract lets us
chain them and replace any one with real math later without touching the others.

Phase status:
  - FedRandFilter      -> REAL (Phase 1): keeps only A or B per a Bernoulli coin
  - LaplacianDPFilter  -> identity (Phase 2: clip + Laplacian noise)
  - AdaptiveQuantFilter-> identity (Phase 3: entropy-driven bit-width)

Exact order, always: FedRand -> DP -> Quant.
"""
from __future__ import annotations

import random

import numpy as np


class FedRandFilter:
    """Keep only one LoRA matrix (A or B) per round; drop the other entirely.

    The server never sees a hospital's A and B together in the same round, so an
    attacker cannot assemble the full delta-W = B @ A needed to invert gradients.

    The coin is seeded by (client_id, round) so a run is reproducible and the
    dashboard/attacker see a consistent choice for the same Update.
    """

    def __init__(self, rho: float = 0.5):
        self.rho = rho  # P(keep A)

    def __call__(self, update: dict) -> dict:
        rng = random.Random(f"{update['client_id']}-{update['round']}")
        keep = "A" if rng.random() < self.rho else "B"
        update["tensors"] = {
            k: v for k, v in update["tensors"].items() if k.endswith(f".{keep}")
        }
        update["meta"]["sent_matrix"] = keep
        return update


class LaplacianDPFilter:
    """Clip each tensor to L1-norm C, then add Lap(C/eps) noise.

    Phase 2: real local differential privacy.  Sensitivity is bounded by
    clipping, then Laplacian noise is added at scale C/eps.  After this
    filter, meta["epsilon"] records the per-round budget consumed.
    """

    def __init__(self, clip_norm: float = 1.0, epsilon: float = 1.0):
        self.clip_norm = clip_norm
        self.epsilon = epsilon

    def __call__(self, update: dict) -> dict:
        for key in list(update["tensors"]):
            t = update["tensors"][key]
            # L1 clip
            norm = float(np.abs(t).sum())
            if norm > self.clip_norm:
                t = t * (self.clip_norm / norm)
            # Laplace noise: scale = C / eps
            scale = self.clip_norm / self.epsilon
            t = t + np.random.laplace(0, scale, size=t.shape)
            update["tensors"][key] = t
        update["meta"]["epsilon"] = self.epsilon
        return update


class AdaptiveQuantFilter:
    """Pick bit-width from the client's data-entropy, then quantize tensors.

    Phase 3: real adaptive quantization.  Higher entropy (more diverse patient
    population) → more precision needed → higher bit-width.  Lower entropy
    (specialized clinic) → aggressive compression is safe.

    Thresholds calibrated from real ICD-code entropy in the demo data:
    all 3 IID hospitals have entropy ≈ 3.5, landing at 8-bit.  With non-IID
    specialty hospitals (entropy < 2.0), 4-bit or 2-bit would kick in.
    """

    @staticmethod
    def _pick_bits(entropy: float) -> int:
        """Entropy → bit-width mapping (data-calibrated thresholds)."""
        if entropy >= 3.0:
            return 8
        if entropy >= 2.0:
            return 4
        return 2

    @staticmethod
    def _quantize_dequantize(tensor, bits: int):
        """Uniform quantization: snap values to a 2^bits-level grid."""
        levels = 2 ** bits
        t_min, t_max = float(tensor.min()), float(tensor.max())
        if t_min == t_max:
            return tensor  # constant tensor, nothing to quantize
        scale = (t_max - t_min) / (levels - 1)
        quantized = np.round((tensor - t_min) / scale)
        return quantized * scale + t_min

    def __call__(self, update: dict) -> dict:
        bits = self._pick_bits(update["meta"]["entropy"])
        for key in list(update["tensors"]):
            update["tensors"][key] = self._quantize_dequantize(
                update["tensors"][key], bits
            )
        update["meta"]["bits"] = bits
        return update


def run_chain(update: dict, filters: list) -> dict:
    for f in filters:
        update = f(update)
    return update


def default_chain(clip_norm: float = 1.0, epsilon: float = 1.0) -> list:
    """Canonical order: FedRand -> DP -> Quant."""
    return [FedRandFilter(), LaplacianDPFilter(clip_norm, epsilon), AdaptiveQuantFilter()]


if __name__ == "__main__":
    import numpy as np
    from .schema import make_update, validate

    u = make_update(
        "h0", 0,
        {"layer0.A": np.zeros((8, 64)), "layer0.B": np.zeros((64, 8))},
        num_samples=5,
    )
    out = run_chain(u, default_chain())
    assert validate(out)
    sent = out["meta"]["sent_matrix"]
    # exactly one matrix kind should remain after FedRand
    kinds = {k.split(".")[-1] for k in out["tensors"]}
    assert kinds == {sent}, (kinds, sent)
    print("FedRand OK: kept", sent, "| tensors", list(out["tensors"]))
