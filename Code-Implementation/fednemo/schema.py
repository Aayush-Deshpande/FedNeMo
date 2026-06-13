"""The Update contract -- the single data object every component agrees on.

This is the keystone of the whole system. Server, client, filters, attacker and
dashboard all read or write this exact dict shape. Lock it down; never break it.

In later phases this becomes a FLARE DXO, but the shape stays identical.
"""
from __future__ import annotations

import numpy as np

# --- Toy LoRA dimensions (small so it runs in seconds on CPU) ---------------
# Real values get swapped in Phase 4 when Nemotron-Mini-4B arrives.
LAYERS = ["layer0", "layer1"]   # two fake "targeted layers"
R, D, K = 8, 64, 64             # rank, out_dim, in_dim


def init_weights() -> dict:
    """Zero-initialized global LoRA state shared by server and clients.

    Shape: {layer: {"A": (R, K), "B": (D, R)}}
    """
    return {
        layer: {
            "A": np.zeros((R, K), dtype=np.float64),
            "B": np.zeros((D, R), dtype=np.float64),
        }
        for layer in LAYERS
    }


def make_update(client_id: str, rnd: int, tensors: dict, num_samples: int,
                entropy: float = 1.0) -> dict:
    """Build a well-formed Update.

    `tensors` maps "<layer>.<A|B>" -> np.ndarray (the LoRA *deltas*).
    `meta` carries everything the server / filters / dashboard need.
    """
    return {
        "client_id": client_id,
        "round": rnd,
        "tensors": tensors,
        "meta": {
            "num_samples": num_samples,
            "entropy": entropy,
            "sent_matrix": "both",   # FedRand sets this to "A" or "B" in Phase 1
            "epsilon": 0.0,          # DP fills this in Phase 2
            "bits": 32,              # Quant fills this in Phase 3
        },
    }


def validate(update: dict) -> bool:
    """Assert an Update has the right shape. Cheap insurance everywhere."""
    assert set(update) == {"client_id", "round", "tensors", "meta"}, \
        f"bad top-level keys: {set(update)}"
    assert isinstance(update["tensors"], dict), "tensors must be a dict"
    for key, val in update["tensors"].items():
        assert isinstance(val, np.ndarray), f"tensor {key} is not an ndarray"
    meta = update["meta"]
    for field in ("num_samples", "entropy", "sent_matrix", "epsilon", "bits"):
        assert field in meta, f"meta missing {field}"
    return True


if __name__ == "__main__":
    # tiny self-test
    g = init_weights()
    u = make_update("hospital_0", 0, {"layer0.A": g["layer0"]["A"]}, num_samples=10)
    assert validate(u)
    print("schema OK:", list(g), "| tensor shapes",
          {k: v.shape for k, v in g["layer0"].items()})
