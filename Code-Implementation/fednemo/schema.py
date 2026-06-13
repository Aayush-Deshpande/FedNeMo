"""The Update contract -- the single data object every component agrees on.

This is the keystone of the whole system. Server, client, filters, attacker and
dashboard all read or write this exact dict shape. Lock it down; never break it.

In later phases this becomes a FLARE DXO, but the shape stays identical.
"""
from __future__ import annotations

import numpy as np

# --- Toy LoRA dimensions (small so it runs in seconds on CPU) ---------------
# Used by StubBackend (Phase 0-3). The real model path (HFBackend, Phase 4)
# does NOT use these: it registers real layer ids at runtime via set_layers().
LAYERS = ["layer0", "layer1"]   # two fake "targeted layers"
R, D, K = 8, 64, 64             # rank, out_dim, in_dim

# --- Real LoRA config (Phase 4, HFBackend) ----------------------------------
# Nemotron-Mini-4B-Instruct is Llama-architecture-based, so the standard
# Llama projection names apply. Verified/overridden at GPU-session time by
# loading the real model and printing named_modules() (see GPU_SESSION_CHECKLIST.md).
LORA_RANK = 32
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]


def set_layers(layer_ids: list[str]) -> None:
    """Register the real per-module LoRA layer ids at runtime.

    Mutates the module-level LAYERS list *in place* so that components which
    did `from .schema import LAYERS` (e.g. server.py) observe the update
    without re-import. This is how the HFBackend tells the server which
    layer ids to aggregate over, keeping the Update contract intact for a
    real model with heterogeneous LoRA shapes.
    """
    LAYERS[:] = list(layer_ids)


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
