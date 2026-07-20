"""Data partitioning across federated nodes.

Provides a stratified held-out split and an IID (random, balanced, equal-parts)
partition across nodes. Non-IID (Dirichlet) partitioning was removed - the
federation uses balanced equal splits so every node has sufficient, comparable
data to train on.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple, TypeVar

import numpy as np

logger = logging.getLogger("fednemo.partition")

T = TypeVar("T")


def stratified_holdout(
    records: Sequence[T],
    labels: Sequence[str],
    holdout_frac: float,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Split indices into (train_idx, holdout_idx), stratified by label.

    The held-out set preserves per-class proportions and is disjoint from the
    training pool, so it is never seen by any node's shard.
    """
    rng = np.random.default_rng(seed)
    by_class: Dict[str, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_class[lab].append(i)

    train_idx: List[int] = []
    holdout_idx: List[int] = []
    for lab, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_hold = max(1, int(round(len(idxs) * holdout_frac))) if len(idxs) > 1 else 0
        holdout_idx.extend(idxs[:n_hold])
        train_idx.extend(idxs[n_hold:])
    rng.shuffle(train_idx)
    rng.shuffle(holdout_idx)
    logger.info(
        "Stratified holdout: %d train, %d holdout (frac=%.2f)",
        len(train_idx), len(holdout_idx), holdout_frac,
    )
    return train_idx, holdout_idx


def iid_partition(
    labels: Sequence[str],
    num_clients: int,
    seed: int = 42,
    stratified: bool = True,
) -> List[List[int]]:
    """Random split into `num_clients` (near-)equal parts.

    - stratified=True (default): each node gets an equal share of EACH class, so
      all nodes have the same balanced class mix (true IID, guaranteed balance).
    - stratified=False: a plain global shuffle then equal chunking.

    Every node ends up with ~= n/num_clients records.
    """
    labels = list(labels)
    n = len(labels)
    if n == 0:
        raise ValueError("No samples to partition.")
    rng = np.random.default_rng(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    if stratified:
        by_class: Dict[str, List[int]] = defaultdict(list)
        for i, lab in enumerate(labels):
            by_class[lab].append(i)
        for _lab, idxs in by_class.items():
            idxs = list(idxs)
            rng.shuffle(idxs)
            for j, idx in enumerate(idxs):
                client_indices[j % num_clients].append(idx)
    else:
        idxs = list(range(n))
        rng.shuffle(idxs)
        for cid, chunk in enumerate(np.array_split(np.array(idxs), num_clients)):
            client_indices[cid] = chunk.tolist()

    for c in client_indices:
        rng.shuffle(c)
    logger.info("IID partition (stratified=%s): sizes=%s",
                stratified, [len(c) for c in client_indices])
    return client_indices


def log_partition_stats(labels: Sequence[str], client_indices: List[List[int]]) -> None:
    """Emit per-node record counts and class distribution (verifiable balance)."""
    labels = list(labels)
    classes = sorted(set(labels))
    logger.info("=" * 68)
    logger.info("IID partition - realized per-node distribution")
    logger.info("=" * 68)
    # abbreviate long class names for the header
    def abbr(c: str) -> str:
        return (c[:8]) if len(c) > 8 else c
    header = "node |  total | " + " | ".join(f"{abbr(c):>8}" for c in classes)
    logger.info(header)
    for cid, idxs in enumerate(client_indices):
        counts = Counter(labels[i] for i in idxs)
        row = f" {cid:>3} | {len(idxs):>6} | " + " | ".join(f"{counts.get(c, 0):>8}" for c in classes)
        logger.info(row)
    logger.info("=" * 68)
