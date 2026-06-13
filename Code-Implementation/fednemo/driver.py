"""The driver loop.

Ties everything together. Each round:
  pull global -> every client trains (via its ModelBackend) -> each Update runs
  the filter chain -> server aggregates -> compute eval loss -> run the live GIA
  demo -> write telemetry.

Backends:
  - "stub" (default): Phase 0-3 fake training, pure CPU/NumPy, no model download.
  - "hf": real LoRA fine-tune of a HuggingFace causal LM (Phase 4). One shared
    model instance serves all simulated hospitals sequentially.

Usage (from Code-Implementation/):
    python -m fednemo.driver            # stub backend
    python -m fednemo.driver hf         # real model (downloads weights)
"""
from __future__ import annotations

import copy
import sys

from .attacker import attack
from .client import Client
from .data.loader import build_prompt, load_shard
from .filters import default_chain, run_chain
from .model_backend import make_backend
from .schema import init_weights
from .server import Server

N_CLIENTS = 3
ROUNDS = 10
CLIP_NORM = 1.0   # L1 clipping bound for DP
EPSILON = 1.0     # per-round privacy budget

# --- HFBackend (Phase 4) settings -------------------------------------------
MODEL_NAME = "nvidia/Nemotron-Mini-4B-Instruct"
SEQ_LEN = 128
LOCAL_STEPS = 2


def _sample_text(prompt_fn=None) -> str:
    """One representative record's prompt, used as the GIA demo target."""
    builder = prompt_fn or build_prompt
    try:
        rows = load_shard("hospital_0")
        if rows:
            return builder(rows[0])
    except FileNotFoundError:
        pass
    return "Patient reports increased thirst and fatigue over two weeks."


def run(backend_kind: str = "stub", n_clients: int = N_CLIENTS,
        rounds: int = ROUNDS, clip_norm: float = CLIP_NORM,
        epsilon: float = EPSILON, model_name: str = MODEL_NAME) -> None:
    Server.reset_telemetry()

    if backend_kind == "hf":
        from .data.medqa import build_medqa_prompt
        backend = make_backend(
            "hf", model_name=model_name, prompt_fn=build_medqa_prompt,
            seq_len=SEQ_LEN, local_steps=LOCAL_STEPS,
        )
        clients = [Client(f"hospital_{i}", backend=backend, seed=i)
                   for i in range(n_clients)]
        eval_backend = backend
        server = Server(backend.init_global_weights())
        target_text = _sample_text(prompt_fn=build_medqa_prompt)
    else:
        clients = [Client(f"hospital_{i}", seed=i) for i in range(n_clients)]
        eval_backend = clients[0].backend
        server = Server(init_weights())
        target_text = _sample_text()

    chain = default_chain(clip_norm=clip_norm, epsilon=epsilon)

    for rnd in range(rounds):
        g = server.get_global()

        # Each client trains; keep a pre-filter copy of the first client's raw
        # update so the attacker can see what an UNPROTECTED capture looks like.
        raw_updates = [c.train(g, rnd) for c in clients]
        raw_capture = copy.deepcopy(raw_updates[0])  # full A+B, clean

        # Run the privacy/compression chain (FedRand drops one matrix, etc.).
        updates = [run_chain(u, chain) for u in raw_updates]
        protected_capture = updates[0]                # fragmented by FedRand

        server.aggregate(updates, clip_norm=clip_norm, epsilon=epsilon)
        loss = eval_backend.eval_loss(server.get_global())

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
    kind = sys.argv[1] if len(sys.argv) > 1 else "stub"
    run(backend_kind=kind)
