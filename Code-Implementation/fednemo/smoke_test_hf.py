"""Phase 4 CPU smoke test: prove the real-model plumbing works end-to-end.

Runs the FULL pipeline -- HFBackend (real transformers+peft LoRA) feeding the
unchanged server / filters / attacker / telemetry -- on a tiny CPU model and a
tiny MedQA shard. It does NOT produce a useful model; it proves every interface
between the real backend and the Phase 0-3 machinery lines up.

Uses a tiny *Llama-architecture* stand-in so the exact same target_modules
(q_proj, k_proj, ... down_proj) that Nemotron-Mini-4B uses are exercised.

Run (from Code-Implementation/, no GPU needed):
    python -m fednemo.smoke_test_hf
"""
from __future__ import annotations

import copy
import os
import traceback

from .attacker import attack
from .client import Client
from .data.medqa import build_medqa_prompt, prepare_medqa_shards
from .filters import default_chain, run_chain
from .model_backend import make_backend
from .schema import TARGET_MODULES, validate
from .server import Server

N_CLIENTS = 2
ROUNDS = 2


def _build_offline_tiny_llama():
    """Construct a tiny Llama-architecture model + trivial tokenizer locally.

    No network: exercises the exact target_modules (q_proj..down_proj) that
    Nemotron-Mini-4B uses, but with random ~thousands-of-params weights on CPU.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    vocab = 256
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg)

    class CharTokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"

        def __call__(self, text, truncation=True, max_length=48,
                     return_tensors="pt"):
            ids = [min(ord(c), vocab - 1) for c in text][:max_length] or [0]
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }

    return model, CharTokenizer()


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} -- {detail}")

    try:
        counts = prepare_medqa_shards(N_CLIENTS, per_shard=6)
        check("MedQA shards prepared", sum(counts.values()) > 0, str(counts))

        print("\nBuilding tiny offline Llama-arch model (CPU, no download)...")
        model, tokenizer = _build_offline_tiny_llama()
        backend = make_backend(
            "hf", model=model, tokenizer=tokenizer, target_modules=list(TARGET_MODULES),
            prompt_fn=build_medqa_prompt, seq_len=48, local_steps=2, train_batch=2,
        )
        check("HFBackend built + LoRA layers found",
              len(backend.layer_ids) > 0,
              f"{len(backend.layer_ids)} LoRA layers, e.g. {backend.layer_ids[0]}")

        clients = [Client(f"hospital_{i}", backend=backend, seed=i)
                   for i in range(N_CLIENTS)]
        server = Server(backend.init_global_weights())
        chain = default_chain(clip_norm=1.0, epsilon=1.0)

        Server.reset_telemetry()
        target_text = build_medqa_prompt
        first_row_text = clients[0].shard[0]
        target_text = build_medqa_prompt(first_row_text)

        losses = []
        for rnd in range(ROUNDS):
            g = server.get_global()
            raw_updates = [c.train(g, rnd) for c in clients]
            for u in raw_updates:
                validate(u)
            raw_capture = copy.deepcopy(raw_updates[0])

            updates = [run_chain(u, chain) for u in raw_updates]
            protected_capture = updates[0]

            # FedRand: exactly one matrix kind survives.
            kinds = {k.split(".")[-1] for k in protected_capture["tensors"]}
            kept = protected_capture["meta"]["sent_matrix"]
            if rnd == 0:
                check("FedRand keeps exactly one matrix kind", kinds == {kept},
                      f"kept {kept}, kinds {kinds}")

            server.aggregate(updates, clip_norm=1.0, epsilon=1.0)
            loss = backend.eval_loss(server.get_global())
            losses.append(loss)

            unprot = attack(raw_capture, target_text, seed=rnd)
            prot = attack(protected_capture, target_text, seed=rnd)
            if rnd == 0:
                check("Attack RECONSTRUCTS unprotected update",
                      unprot["success"] is True, "full A+B, clean")
                check("Attack FAILS on protected update",
                      prot["success"] is False, "FedRand+DP+Quant")

            server.write_telemetry(rnd, updates, loss, attack={
                "original": target_text, "unprotected": unprot, "protected": prot,
            })
            print(f"  round {rnd} | loss={loss:.5f} | eps_total={server.eps_total:.3f}")

        import math
        check("eval_loss returns finite floats",
              all(isinstance(x, float) and math.isfinite(x) for x in losses),
              f"losses={[round(x, 4) for x in losses]}")
        check("RDP accountant advanced", server.eps_total > 0,
              f"eps_total={server.eps_total:.4f}")

        tel = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                            "telemetry.jsonl"))
        with open(tel) as f:
            n_lines = sum(1 for line in f if line.strip())
        check("Telemetry written one line per round", n_lines == ROUNDS,
              f"{n_lines} lines")

    except Exception:
        print("\nSMOKE TEST CRASHED:\n")
        print(traceback.format_exc())
        return 1

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print(f"\nPhase 4 HF smoke test: {passed}/{total} checks passed "
          f"-- {'WORKING' if passed == total else 'SOME FAILED'}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
