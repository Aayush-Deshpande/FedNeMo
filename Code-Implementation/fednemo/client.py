"""Hospital / client node. IID data => every client is identical code.

Phase 0 "training" is fake: deep-copy the global weights, nudge them with small
random noise, and return the deltas. This proves the loop wiring without any ML.
In Phase 4 the body of `train` is replaced by a real NeMo LoRA fine-tuning step;
the input (global weights) and output (an Update of deltas) stay identical.
"""
from __future__ import annotations

import copy

import numpy as np

from .data.loader import load_shard
from .schema import LAYERS, make_update


class Client:
    def __init__(self, client_id: str, lr: float = 0.01, seed: int | None = None):
        self.client_id = client_id
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        try:
            self.shard = load_shard(client_id)
        except FileNotFoundError:
            # allow running before split.py for quick smoke tests
            self.shard = [{} for _ in range(10)]

    def train(self, global_weights: dict, rnd: int = 0) -> dict:
        """Fake local training: nudge weights, return deltas as an Update."""
        local = copy.deepcopy(global_weights)
        tensors = {}
        for layer in LAYERS:
            for mat in ("A", "B"):
                base = global_weights[layer][mat]
                noise = self.lr * self.rng.standard_normal(base.shape)
                local[layer][mat] = base + noise
                tensors[f"{layer}.{mat}"] = local[layer][mat] - base  # delta
        return make_update(
            client_id=self.client_id,
            rnd=rnd,
            tensors=tensors,
            num_samples=len(self.shard),
        )


if __name__ == "__main__":
    from .schema import init_weights, validate

    c = Client("hospital_0", seed=0)
    u = c.train(init_weights(), rnd=0)
    assert validate(u)
    print("client OK:", u["client_id"], "| num_samples", u["meta"]["num_samples"],
          "| tensors", list(u["tensors"]))
