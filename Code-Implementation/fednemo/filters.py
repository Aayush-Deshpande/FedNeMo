"""DXO filter chain (client -> server). ALL filters are identity in Phase 0.

The one rule that makes this whole design work: every filter takes an Update and
returns an Update (same shape in, same shape out). That uniform contract lets us
chain them and replace any one with real math later without touching the others.

Real implementations land in:
  - FedRandFilter      -> Phase 1 (drop one matrix)
  - LaplacianDPFilter  -> Phase 2 (clip + Laplacian noise)
  - AdaptiveQuantFilter-> Phase 3 (entropy-driven bit-width)

Exact order, always: FedRand -> DP -> Quant.
"""
from __future__ import annotations


class FedRandFilter:
    """Phase 0: identity. Phase 1: keep only A or B per a Bernoulli coin flip."""

    def __call__(self, update: dict) -> dict:
        update["meta"]["sent_matrix"] = "both"
        return update


class LaplacianDPFilter:
    """Phase 0: identity. Phase 2: clip to norm C, add Lap(C/eps) noise."""

    def __call__(self, update: dict) -> dict:
        update["meta"]["epsilon"] = 0.0
        return update


class AdaptiveQuantFilter:
    """Phase 0: identity. Phase 3: pick bit-width from entropy, quantize."""

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

    u = make_update("h0", 0, {"layer0.A": np.zeros((8, 64))}, num_samples=5)
    out = run_chain(u, default_chain())
    assert validate(out)
    print("filter chain OK (all identity):", out["meta"])
