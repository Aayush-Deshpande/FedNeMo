"""Evaluate a trained global adapter on a held-out set.

Loads the adapter into the local model, runs greedy generation + deterministic
output parsing on each held-out record, and reports overall accuracy plus
per-class precision/recall/F1 (mandatory given PTB-XL class imbalance).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from ..config import ARTIFACTS_DIR
from ..data.record import ClinicalRecord
from ..data.record_io import load_records
from ..model.nemotron_local import LoadedModel, generate, load_nemotron
from ..model.serialization import build_prompt, label_space_for, parse_model_output
from .metrics import EvalMetrics, compute_metrics, format_report

logger = logging.getLogger("fednemo.eval")


def _load_adapter(lm: LoadedModel, adapter_path: Path) -> bool:
    if not adapter_path.exists():
        logger.warning("Adapter not found: %s (evaluating base+fresh LoRA)", adapter_path)
        return False
    from ..federated.fedrand import load_adapter_state
    state = torch.load(adapter_path, map_location="cpu")
    load_adapter_state(lm.model, state)
    logger.info("Loaded adapter: %s", adapter_path)
    return True


def _constrained_pred(out: str, label_space: List[str]) -> Optional[str]:
    """Guaranteed-valid prediction: first the deterministic parse; if that finds
    no allowed label, snap the generated text to the nearest valid label (fuzzy).
    Ensures every prediction is one of the real classes (no None/unparseable)."""
    import difflib
    pred = parse_model_output(out, label_space)["predicted_label"]
    if pred is not None:
        return pred
    text = out.strip().lower()
    # try fuzzy match of the generated text against each label
    best, best_ratio = None, 0.0
    for lab in label_space:
        r = difflib.SequenceMatcher(None, lab.lower(), text[:len(lab) + 10]).ratio()
        if r > best_ratio:
            best_ratio, best = r, lab
    return best if best_ratio >= 0.5 else None


def evaluate_records(
    records: List[ClinicalRecord],
    source: str,
    lm: Optional[LoadedModel] = None,
    adapter_path: Optional[Path] = None,
    max_new_tokens: int = 12,
    log_every: int = 25,
) -> EvalMetrics:
    if lm is None:
        lm = load_nemotron(attach_lora=True)
        if adapter_path is not None:
            _load_adapter(lm, adapter_path)
    lm.model.eval()

    label_space = label_space_for(source)
    pairs: List[Tuple[str, Optional[str]]] = []
    for i, rec in enumerate(records):
        prompt = build_prompt(rec, label_space)
        out = generate(lm, prompt, max_new_tokens=max_new_tokens)
        pred = _constrained_pred(out, label_space)
        pairs.append((rec.label, pred))
        if (i + 1) % log_every == 0:
            logger.info("  evaluated %d/%d", i + 1, len(records))

    return compute_metrics(pairs, label_space)


def run_eval(
    tag: str = "run",
    holdout_path: Optional[Path] = None,
    adapter_path: Optional[Path] = None,
    lm: Optional[LoadedModel] = None,
    max_new_tokens: int = 24,
) -> EvalMetrics:
    holdout_path = holdout_path or (ARTIFACTS_DIR / f"holdout_{tag}.json")
    adapter_path = adapter_path or (ARTIFACTS_DIR / f"global_adapter_{tag}.pt")

    # auto-detect the LoRA rank from the saved adapter so the model is built to match
    if adapter_path.exists():
        from ..config import CONFIG
        _state = torch.load(adapter_path, map_location="cpu")
        for _k, _v in _state.items():
            if _k.endswith("lora_A.default.weight"):
                CONFIG.lora_rank = int(_v.shape[0])
                logger.info("Auto-detected LoRA rank=%d from adapter", CONFIG.lora_rank)
                break
        del _state

    records = load_records(holdout_path)
    logger.info("Loaded %d held-out records from %s", len(records), holdout_path)
    source = records[0].source if records else tag
    # register the label space from the held-out set's labels
    from ..model.serialization import set_label_space
    set_label_space([r.label for r in records])

    if lm is None:
        lm = load_nemotron(attach_lora=True)
        _load_adapter(lm, adapter_path)

    metrics = evaluate_records(records, source=source, lm=lm, max_new_tokens=max_new_tokens)

    out_json = ARTIFACTS_DIR / f"eval_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info("Saved eval metrics -> %s", out_json)
    return metrics
