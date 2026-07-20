"""A simulated client node: holds a local non-IID data shard and fine-tunes the
shared LoRA adapter on it. Clients are trained strictly sequentially by the
orchestrator (never in parallel) to respect the VRAM budget.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ..config import CONFIG
from ..data.record import ClinicalRecord
from ..model.nemotron_local import LoadedModel, compute_causal_loss
from ..model.serialization import build_prompt, build_target, label_space_for
from .fedrand import AdapterState, extract_adapter_state, load_adapter_state

logger = logging.getLogger("fednemo.client")


@dataclass
class LocalTrainResult:
    updated_state: AdapterState
    loss_trajectory: List[float]
    num_samples: int
    class_distribution: Dict[str, int]


class ClientNode:
    def __init__(self, client_id: int, records: List[ClinicalRecord]):
        self.client_id = client_id
        self.records = records
        # persistent private LoRA half from FedRand (set by orchestrator each round)
        self.private_state: AdapterState = {}
        # optional per-class loss weights (inverse frequency); {} => uniform
        self.class_weights: Dict[str, float] = {}
        self.round_counter = 0  # for per-round shuffle reproducibility
        # optional poisoning for the trust-agent validation experiment (item 6):
        #   None | "explode" (scale update to a huge norm) | "shuffle" (random labels)
        self.poison: Optional[str] = None
        self.poison_scale: float = 100.0

    def class_distribution(self) -> Dict[str, int]:
        return dict(Counter(r.label for r in self.records))

    def _maybe_poison_labels(self) -> None:
        """For poison='shuffle': randomly permute this client's record labels once
        (label-flipping data poisoning)."""
        if self.poison == "shuffle" and self.records:
            import random
            rng = random.Random(1234 + self.client_id)
            labels = [r.label for r in self.records]
            rng.shuffle(labels)
            for r, lab in zip(self.records, labels):
                r.label = lab

    def _encode(self, lm: LoadedModel, record: ClinicalRecord):
        """Build (input_ids, labels) with loss masked to the target completion only."""
        tok = lm.tokenizer
        label_space = label_space_for(record.source)
        prompt = build_prompt(record, label_space)
        target = build_target(record) + tok.eos_token

        prompt_ids = tok(prompt, add_special_tokens=True).input_ids
        target_ids = tok(target, add_special_tokens=False).input_ids

        # Always preserve the full target completion; truncate the PROMPT from the
        # LEFT if needed (keeps the trailing "Answer:" cue). This prevents an
        # all -100 label sequence, which would otherwise yield a nan loss.
        max_len = CONFIG.max_seq_len
        target_ids = target_ids[: max(1, max_len - 1)]  # never let target alone overflow
        room_for_prompt = max_len - len(target_ids)
        if room_for_prompt < 1:
            room_for_prompt = 1
        if len(prompt_ids) > room_for_prompt:
            prompt_ids = prompt_ids[-room_for_prompt:]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)

        # safety: guarantee at least one supervised (non -100) token
        if all(l == -100 for l in labels):
            labels[-1] = input_ids[-1]

        input_ids_t = torch.tensor([input_ids], dtype=torch.long)   # CPU
        labels_t = torch.tensor([labels], dtype=torch.long)         # CPU
        return input_ids_t, labels_t

    def local_train(
        self,
        lm: LoadedModel,
        global_state: AdapterState,
    ) -> LocalTrainResult:
        """Fine-tune the adapter on this client's shard for the configured budget.

        The client starts from the global adapter, but overwrites its private
        FedRand half (if any) with its locally retained matrices before training.
        """
        # label-flipping poison (item 6) applied once before training
        self._maybe_poison_labels()

        # start from global, then restore this client's persistent private half
        start_state = {k: v.clone() for k, v in global_state.items()}
        start_state.update(self.private_state)
        load_adapter_state(lm.model, start_state)

        lm.model.train()
        trainable = lm.trainable_parameters()
        optimizer = torch.optim.AdamW(trainable, lr=CONFIG.learning_rate)

        # coverage: train on the FULL shard each round (max_steps=0 => no cap),
        # shuffled per round so every record is used (fixes the old "first-60-only"
        # bug). grad_accum_steps records are accumulated per optimizer step
        # (effective batch size) for gradient stability.
        import random
        n = len(self.records)
        cap = CONFIG.local_max_steps if CONFIG.local_max_steps and CONFIG.local_max_steps > 0 else n * CONFIG.local_epochs
        accum = max(1, CONFIG.grad_accum_steps)

        loss_traj: List[float] = []
        seen = 0
        micro = 0
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(CONFIG.local_epochs):
            order = list(range(n))
            if CONFIG.shuffle_each_round:
                random.Random(1000 * self.client_id + 7 * self.round_counter + epoch).shuffle(order)
            for i in order:
                if seen >= cap:
                    break
                record = self.records[i]
                input_ids, labels = self._encode(lm, record)
                loss = compute_causal_loss(lm, input_ids, labels)
                if self.class_weights:
                    loss = loss * self.class_weights.get(record.label, 1.0)
                (loss / accum).backward()      # scale for accumulation
                loss_traj.append(float(loss.item()))
                seen += 1
                micro += 1
                if micro % accum == 0:         # optimizer step every `accum` records
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if seen >= cap:
                break
        # flush any remaining accumulated gradient
        if micro % accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        self.round_counter += 1
        updated = extract_adapter_state(lm.model)

        # exploding-update poison (item 6): scale the update to a huge norm
        if self.poison == "explode":
            for k in updated:
                updated[k] = updated[k] * self.poison_scale
            logger.info("Client %d POISONED (explode x%.0f).", self.client_id, self.poison_scale)

        logger.info(
            "Client %d trained %d records | loss %.3f -> %.3f | n=%d",
            self.client_id, seen,
            loss_traj[0] if loss_traj else float("nan"),
            loss_traj[-1] if loss_traj else float("nan"),
            len(self.records),
        )
        return LocalTrainResult(
            updated_state=updated,
            loss_trajectory=loss_traj,
            num_samples=len(self.records),
            class_distribution=self.class_distribution(),
        )
