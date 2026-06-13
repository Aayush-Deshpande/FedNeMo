"""The driver loop.

Ties everything together. Each round:
  pull global -> every client trains -> each Update runs the filter chain ->
  server aggregates -> compute placeholder loss -> run the live GIA demo ->
  write telemetry.

Phase 1 additions:
  - FedRand is now live (clients send only A or only B per round).
  - A live attack demo runs each round on one sample record: the SAME update is
    attacked once BEFORE the filter chain (unprotected) and once AFTER
    (protected), proving FedRand neutralizes reconstruction.

Usage (from Code-Implementation/):
    python -m fednemo.driver
"""
from __future__ import annotations

import copy

from .attacker import attack
from .client import Client
from .data.loader import build_prompt, load_shard
from .filters import default_chain, run_chain
from .schema import LAYERS, init_weights
from .server import Server

N_CLIENTS = 3
ROUNDS = 10


def placeholder_loss(global_state: dict) -> float:
    """Not a real loss -- just a number that moves so the dashboard has a line.

    Mean absolute value of all A matrices. Replaced by real eval in Phase 4.
    """
    vals = [abs(global_state[l]["A"]).mean() for l in LAYERS]
    return sum(vals) / len(vals)


def _sample_text() -> str:
    """One representative record's prompt, used as the GIA demo target."""
    try:
        rows = load_shard("hospital_0")
        if rows:
            return build_prompt(rows[0])
    except FileNotFoundError:
        pass
    return "Patient reports increased thirst and fatigue over two weeks."


def run(n_clients: int = N_CLIENTS, rounds: int = ROUNDS) -> None:
    Server.reset_telemetry()
    clients = [Client(f"hospital_{i}", seed=i) for i in range(n_clients)]
    server = Server(init_weights())
    chain = default_chain()
    target_text = _sample_text()

    for rnd in range(rounds):
        g = server.get_global()

        # Each client trains; keep a pre-filter copy of the first client's raw
        # update so the attacker can see what an UNPROTECTED capture looks like.
        raw_updates = [c.train(g, rnd) for c in clients]
        raw_capture = copy.deepcopy(raw_updates[0])  # full A+B, clean

        # Run the privacy/compression chain (FedRand drops one matrix, etc.).
        updates = [run_chain(u, chain) for u in raw_updates]
        protected_capture = updates[0]                # fragmented by FedRand

        server.aggregate(updates)
        loss = placeholder_loss(server.get_global())

        # Live GIA demo: attack the same record before vs after protection.
        attack_result = {
            "original": target_text,
            "unprotected": attack(raw_capture, target_text, seed=rnd),
            "protected": attack(protected_capture, target_text, seed=rnd),
        }

        server.write_telemetry(rnd, updates, loss, attack=attack_result)
        sent = {u["client_id"]: u["meta"]["sent_matrix"] for u in updates}
        print(f"round {rnd:2d} | loss={loss:.5f} | "
              f"eps_total={server.eps_total:.1f} | sent={sent}")

    print(f"\ndone: {rounds} rounds, {n_clients} hospitals. "
          f"telemetry written. final eps_total={server.eps_total:.1f}")


if __name__ == "__main__":
    run()
