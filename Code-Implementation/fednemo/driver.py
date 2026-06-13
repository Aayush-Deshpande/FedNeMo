"""The driver loop -- this is the Phase 0 'hello world'.

Ties everything together: build N clients, init the server, and each round:
  pull global -> every client trains -> each Update runs the filter chain ->
  server aggregates -> compute a placeholder loss -> write telemetry.

When this runs end to end and telemetry.jsonl fills up, the skeleton is proven.

Usage (from Code-Implementation/):
    python -m fednemo.driver
"""
from __future__ import annotations

from .client import Client
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


def run(n_clients: int = N_CLIENTS, rounds: int = ROUNDS) -> None:
    Server.reset_telemetry()
    clients = [Client(f"hospital_{i}", seed=i) for i in range(n_clients)]
    server = Server(init_weights())
    chain = default_chain()

    for rnd in range(rounds):
        g = server.get_global()
        updates = [run_chain(c.train(g, rnd), chain) for c in clients]
        server.aggregate(updates)
        loss = placeholder_loss(server.get_global())
        server.write_telemetry(rnd, updates, loss)
        print(f"round {rnd:2d} | loss={loss:.5f} | eps_total={server.eps_total:.1f}")

    print(f"\ndone: {rounds} rounds, {n_clients} hospitals. "
          f"telemetry written. final eps_total={server.eps_total:.1f}")


if __name__ == "__main__":
    run()
