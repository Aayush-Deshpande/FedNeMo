"""Global aggregation of FedRand-split client payloads.

Because FedRand means each client shares only A OR B per layer, aggregation is
done per (layer, matrix) slot: for each slot we collect the contributing clients'
dequantized matrices and combine them with a weight that blends:

    w_i  =  trust_i  *  nu_i

where nu_i is the entropy-aware client importance:

    nu_i = lambda_h * (H(D_i)/H_max)  +  (1 - lambda_h) * (|D_i| / N_max)

Slots with no contributor this round keep their previous global value. This
never leaves a matrix undefined and faithfully reflects the randomized split.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch

from .fedrand import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    AdapterState,
    ClientPayload,
    reconstruct_shared,
)
from .trust_agent import TrustResult, UpdateSummary

logger = logging.getLogger("fednemo.aggregator")

LAMBDA_H = 0.5  # balance between entropy diversity and dataset volume


def entropy_importance(
    summaries: List[UpdateSummary],
    lambda_h: float = LAMBDA_H,
) -> Dict[int, float]:
    """Compute nu_i for each client from label entropy and dataset volume."""
    if not summaries:
        return {}
    n_max = max(s.num_samples for s in summaries) or 1
    nu: Dict[int, float] = {}
    for s in summaries:
        vol = s.num_samples / n_max
        nu[s.client_id] = lambda_h * s.entropy_ratio + (1.0 - lambda_h) * vol
    return nu


def aggregate(
    global_state: AdapterState,
    payloads: List[ClientPayload],
    summaries: List[UpdateSummary],
    trust: Dict[int, TrustResult],
    lambda_h: float = LAMBDA_H,
) -> Tuple[AdapterState, Dict[str, float]]:
    """Aggregate client payloads into a new global adapter state.

    Returns (new_global_state, per_slot_contributor_counts_for_logging).
    """
    nu = entropy_importance(summaries, lambda_h=lambda_h)

    # slot -> list of (weight, tensor)
    slot_updates: Dict[Tuple[str, str], List[Tuple[float, torch.Tensor]]] = {}
    for payload in payloads:
        cid = payload.client_id
        w = max(0.0, trust[cid].score) * max(1e-6, nu.get(cid, 0.0))
        shared = reconstruct_shared(payload)
        for (layer, which), tensor in shared.items():
            slot_updates.setdefault((layer, which), []).append((w, tensor))

    new_state: AdapterState = {k: v.clone() for k, v in global_state.items()}
    contributor_log: Dict[str, float] = {}

    for (layer, which), contributions in slot_updates.items():
        suffix = LORA_A_SUFFIX if which == "A" else LORA_B_SUFFIX
        param_name = layer + suffix
        if param_name not in new_state:
            logger.warning("Aggregation slot %s not in global state; skipping.", param_name)
            continue
        total_w = sum(w for w, _ in contributions)
        if total_w <= 0:
            continue
        acc = torch.zeros_like(new_state[param_name])
        for w, tensor in contributions:
            acc += (w / total_w) * tensor.to(acc.dtype)
        new_state[param_name] = acc
        contributor_log[param_name] = float(len(contributions))

    return new_state, contributor_log
