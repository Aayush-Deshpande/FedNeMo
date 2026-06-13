"""Orchestrator / server: holds global LoRA state, aggregates, writes telemetry.

Phase 0 aggregation is plain sample-weighted FedAvg, in-memory, single process.
The privacy accountant is a stub that just counts rounds. In Phase 2 the stub is
replaced by a real RDP accountant; in Phase 5 the transport moves into FLARE.
The aggregation math is kept pure so those swaps don't touch it.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .schema import LAYERS, init_weights

TELEMETRY_PATH = os.path.join(os.path.dirname(__file__), "..", "telemetry.jsonl")
TELEMETRY_PATH = os.path.normpath(TELEMETRY_PATH)


class Server:
    def __init__(self, init_state: dict | None = None):
        self.g = init_state if init_state is not None else init_weights()
        self.eps_total = 0.0

    def get_global(self) -> dict:
        """What clients pull at the start of a round."""
        return self.g

    def aggregate(self, updates: list[dict]) -> dict:
        """Sample-weighted FedAvg.

        Only averages matrices that are actually present in the Update. This
        matters from Phase 1 on, when FedRand means a client sends only A or
        only B in a given round.
        """
        for layer in LAYERS:
            for mat in ("A", "B"):
                key = f"{layer}.{mat}"
                contribs = [
                    (u["meta"]["num_samples"], u["tensors"][key])
                    for u in updates
                    if key in u["tensors"]
                ]
                if not contribs:
                    continue
                total = sum(n for n, _ in contribs)
                avg_delta = sum(n * t for n, t in contribs) / total
                # updates carry deltas; apply them to current global state
                self.g[layer][mat] = self.g[layer][mat] + avg_delta

        self.eps_total += 1.0  # STUB accountant -- replaced in Phase 2
        return self.g

    def write_telemetry(self, rnd: int, updates: list[dict], loss: float) -> None:
        """Append one JSON line per round. The dashboard tails this file.

        The dashboard NEVER imports this module -- file is the only coupling, so
        the dashboard can never slow training down.
        """
        record = {
            "round": rnd,
            "loss": float(loss),
            "eps_total": float(self.eps_total),
            "sent": {u["client_id"]: u["meta"]["sent_matrix"] for u in updates},
            "bits": {u["client_id"]: u["meta"]["bits"] for u in updates},
        }
        with open(TELEMETRY_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def reset_telemetry() -> None:
        if os.path.exists(TELEMETRY_PATH):
            os.remove(TELEMETRY_PATH)


if __name__ == "__main__":
    # self-test: two clients, sample-weighted mean of deltas applied to zeros
    from .schema import make_update

    s = Server()
    d1 = {"layer0.A": np.ones((8, 64)), "layer0.B": np.ones((64, 8))}
    d2 = {"layer0.A": np.full((8, 64), 3.0), "layer0.B": np.full((64, 8), 3.0)}
    u1 = make_update("h0", 0, d1, num_samples=10)
    u2 = make_update("h1", 0, d2, num_samples=30)
    s.aggregate([u1, u2])
    # weighted mean = (10*1 + 30*3) / 40 = 2.5
    assert np.allclose(s.g["layer0"]["A"], 2.5), s.g["layer0"]["A"][0, 0]
    print("server FedAvg OK: weighted mean =", s.g["layer0"]["A"][0, 0])
