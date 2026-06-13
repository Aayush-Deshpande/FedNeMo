"""Orchestrator / server: holds global LoRA state, aggregates, writes telemetry.

Aggregation is plain sample-weighted FedAvg, in-memory, single process. From
Phase 1 on, FedRand means a client sends only A or only B in a given round, so
aggregation only averages matrices that are actually present.

The privacy accountant is a stub that just counts rounds. In Phase 2 the stub is
replaced by a real RDP accountant; in Phase 5 the transport moves into FLARE.
The aggregation math is kept pure so those swaps don't touch it.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .schema import LAYERS, init_weights

TELEMETRY_PATH = os.path.join(os.path.dirname(__file__), "..", "telemetry.jsonl")
TELEMETRY_PATH = os.path.normpath(TELEMETRY_PATH)


class RDPAccountant:
    """Rényi DP composition across rounds for the Laplace mechanism.

    Accumulates per-round RDP guarantees at several alpha orders, then
    converts to (eps, delta)-DP via the standard RDP-to-DP conversion.
    Pure Python, no external library needed.
    """

    def __init__(self, delta: float = 1e-5):
        self.delta = delta
        self._alphas = [1.5, 2, 3, 4, 5, 8, 16, 32, 64]
        self._rdp_eps = [0.0] * len(self._alphas)
        self._rounds = 0

    def step(self, epsilon_per_round: float, clip_norm: float) -> None:
        """Record one round of Laplace mechanism usage."""
        b = clip_norm / epsilon_per_round  # Laplace scale
        for i, alpha in enumerate(self._alphas):
            rdp_eps = (1 / (alpha - 1)) * math.log(
                (alpha / (2 * alpha - 1)) * math.exp((alpha - 1) / b)
                + ((alpha - 1) / (2 * alpha - 1)) * math.exp(-alpha / b)
            )
            self._rdp_eps[i] += rdp_eps
        self._rounds += 1

    @property
    def eps_total(self) -> float:
        """Convert accumulated RDP to (eps, delta)-DP; return tightest eps."""
        if self._rounds == 0:
            return 0.0
        return min(
            rdp + math.log(1 / self.delta) / (alpha - 1)
            for alpha, rdp in zip(self._alphas, self._rdp_eps)
        )


class Server:
    def __init__(self, init_state: dict | None = None):
        self.g = init_state if init_state is not None else init_weights()
        self.accountant = RDPAccountant()

    @property
    def eps_total(self) -> float:
        return self.accountant.eps_total

    def get_global(self) -> dict:
        """What clients pull at the start of a round."""
        return self.g

    def aggregate(self, updates: list[dict],
                  clip_norm: float = 1.0, epsilon: float = 1.0) -> dict:
        """Sample-weighted FedAvg.

        Only averages matrices that are actually present in the Update. With
        FedRand, in any given round some hospitals send only A and some only B;
        if no hospital sent a given matrix this round, it simply isn't updated.
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

        self.accountant.step(epsilon, clip_norm)
        return self.g

    def write_telemetry(self, rnd: int, updates: list[dict], loss: float,
                        attack: dict | None = None) -> None:
        """Append one JSON line per round. The dashboard tails this file.

        The dashboard NEVER imports this module -- file is the only coupling, so
        the dashboard can never slow training down.

        `attack` (Phase 1+) carries the live GIA demo result:
            {"original": str,
             "unprotected": {"success": bool, "reconstruction": str},
             "protected":   {"success": bool, "reconstruction": str}}
        """
        record = {
            "round": rnd,
            "loss": float(loss),
            "eps_total": float(self.eps_total),
            "sent": {u["client_id"]: u["meta"]["sent_matrix"] for u in updates},
            "bits": {u["client_id"]: u["meta"]["bits"] for u in updates},
        }
        if attack is not None:
            record["attack"] = attack
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
