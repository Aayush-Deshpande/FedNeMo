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
    """Phase 0/1: identity. Phase 2: clip to norm C, add Lap(C/eps) noise."""

    def __call__(self, update: dict) -> dict:
        update["meta"]["epsilon"] = 0.0
        return update


class AdaptiveQuantFilter:
    """Phase 0/1: identity. Phase 3: pick bit-width from entropy, quantize."""

    def __call__(self, update: dict) -> dict:
        update["meta"]["bits"] = 32
        return update


def run_chain(update: dict, filters: list) -> dict:
    for f in filters:
        update = f(update)
    return update


def default_chain() -> list:
    """Canonical order: FedRand -> DP -> Quant."""
    return [FedRandFilter(), LaplacianDPFilter(), AdaptiveQuantFilter()]


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
