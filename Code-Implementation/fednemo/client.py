"""Hospital / client node. IID data => every client is identical code.

A Client owns a data shard and delegates "training" to a ModelBackend:
  - StubBackend (default): Phase 0-3 fake training (NumPy noise deltas).
  - HFBackend: real LoRA fine-tune of a HuggingFace causal LM (Phase 4).

The Client's job is unchanged from the caller's view: given the global LoRA
weights, return a well-formed Update of deltas. Only the backend differs.
"""
from __future__ import annotations

import math
from collections import Counter

from .data.loader import load_shard
from .model_backend import ModelBackend, StubBackend
from .schema import make_update


class Client:
    def __init__(self, client_id: str, backend: ModelBackend | None = None,
                 lr: float = 0.01, seed: int | None = None):
        self.client_id = client_id
        self.seed = seed
        # Backward compatible: no backend => the Phase 0-3 stub trainer.
        self.backend = backend if backend is not None else StubBackend(seed=seed, lr=lr)
        try:
            self.shard = load_shard(client_id)
        except FileNotFoundError:
            # allow running before any split for quick smoke tests
            self.shard = [{} for _ in range(10)]
        self.entropy = self._shard_entropy(self.shard)

    @staticmethod
    def _shard_entropy(shard: list[dict]) -> float:
        """Shannon entropy (bits) over class frequencies in this shard.

        Uses ICD-10 codes when present (clinical demo data); falls back to
        MedQA answer-letter distribution; finally 1.0 when no signal exists.
        """
        def entropy(values):
            counts = Counter(values)
            total = sum(counts.values())
            return -sum((c / total) * math.log2(c / total) for c in counts.values())

        icds = [
            row.get("structured", {}).get("icd")
            for row in shard
            if row.get("structured", {}).get("icd") is not None
        ]
        if icds:
            return entropy(icds)
        answers = [row.get("answer_idx") for row in shard if row.get("answer_idx")]
        if answers:
            return entropy(answers)
        return 1.0

    def train(self, global_weights: dict, rnd: int = 0) -> dict:
        """Run one local round via the backend; return an Update of deltas."""
        tensors, num_samples = self.backend.train_step(
            global_weights, self.shard, rnd, self.client_id, seed=self.seed
        )
        return make_update(
            client_id=self.client_id,
            rnd=rnd,
            tensors=tensors,
            num_samples=num_samples,
            entropy=self.entropy,
        )


if __name__ == "__main__":
    from .schema import init_weights, validate

    c = Client("hospital_0", seed=0)
    u = c.train(init_weights(), rnd=0)
    assert validate(u)
    print("client OK:", u["client_id"], "| num_samples", u["meta"]["num_samples"],
          "| entropy", f"{c.entropy:.4f}",
          "| tensors", list(u["tensors"]))
