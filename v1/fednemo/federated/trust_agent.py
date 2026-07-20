"""Trust-scoring reasoning agent.

The second "logical" Nemotron instance. Because two separate 4B instances cannot
coexist within the 3 GB VRAM budget, this reuses the SAME loaded model with the
LoRA adapter DISABLED (base weights only) and a distinct reasoning prompt. It
receives a *text summary* of each client update (loss trajectory, class
distribution, update norm, entropy) - never raw tensors - and returns a trust
score in [0, 1] plus a short rationale.

The model output is parsed for a numeric score; if parsing fails we fall back to
a deterministic heuristic over the same summary so aggregation never breaks.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..model.nemotron_local import LoadedModel, generate

logger = logging.getLogger("fednemo.trust")


@dataclass
class UpdateSummary:
    client_id: int
    num_samples: int
    class_distribution: Dict[str, int]
    loss_trajectory: List[float]
    update_l2_norm: float
    label_entropy: float
    entropy_ratio: float


@dataclass
class TrustResult:
    client_id: int
    score: float
    rationale: str
    source: str  # "model" or "heuristic"


def label_entropy(class_distribution: Dict[str, int]) -> float:
    total = sum(class_distribution.values())
    if total == 0:
        return 0.0
    h = 0.0
    for count in class_distribution.values():
        if count <= 0:
            continue
        p = count / total
        h -= p * math.log2(p)
    return h


def build_summary(
    client_id: int,
    num_samples: int,
    class_distribution: Dict[str, int],
    loss_trajectory: List[float],
    update_l2_norm: float,
    num_classes: int,
) -> UpdateSummary:
    h = label_entropy(class_distribution)
    h_max = math.log2(num_classes) if num_classes > 1 else 1.0
    return UpdateSummary(
        client_id=client_id,
        num_samples=num_samples,
        class_distribution=class_distribution,
        loss_trajectory=loss_trajectory,
        update_l2_norm=update_l2_norm,
        label_entropy=h,
        entropy_ratio=(h / h_max) if h_max > 0 else 0.0,
    )


def _summary_text(s: UpdateSummary) -> str:
    traj = ", ".join(f"{v:.3f}" for v in s.loss_trajectory) if s.loss_trajectory else "n/a"
    dist = ", ".join(f"{k}:{v}" for k, v in s.class_distribution.items())
    loss_drop = (
        s.loss_trajectory[0] - s.loss_trajectory[-1]
        if len(s.loss_trajectory) >= 2 else 0.0
    )
    return (
        f"Client {s.client_id} federated update summary:\n"
        f"- training samples: {s.num_samples}\n"
        f"- class distribution: {dist}\n"
        f"- label entropy: {s.label_entropy:.3f} (ratio to max: {s.entropy_ratio:.2f})\n"
        f"- local loss trajectory: {traj} (net drop: {loss_drop:.3f})\n"
        f"- update L2 norm: {s.update_l2_norm:.4f}\n"
    )


def _heuristic_score(s: UpdateSummary) -> float:
    """Deterministic trust proxy used as fallback / when model parse fails.

    Rewards: a loss that actually decreased, a moderate (not exploding) update
    norm, and a non-degenerate class distribution.
    """
    score = 0.5
    if len(s.loss_trajectory) >= 2:
        drop = s.loss_trajectory[0] - s.loss_trajectory[-1]
        score += max(-0.25, min(0.25, drop * 0.5))
    # penalize exploding / vanishing updates
    if s.update_l2_norm > 50.0 or s.update_l2_norm < 1e-6:
        score -= 0.25
    # reward some label diversity
    score += 0.15 * s.entropy_ratio
    return float(max(0.0, min(1.0, score)))


def _parse_score(text: str) -> Optional[float]:
    """Strict: only accept an explicit 'score/trust: <number>'. No greedy
    first-float fallback (that misfired on numbers inside the rationale, e.g.
    a class-distribution '0', wrongly zeroing a good client)."""
    m = re.search(r"(?:score|trust)\s*[:=]\s*(0?\.\d+|1(?:\.0+)?|\d{1,3})", text.lower())
    if not m:
        return None
    val = float(m.group(1))
    if val > 1.0:  # model may answer on a 0-100 scale
        val = val / 100.0
    return max(0.0, min(1.0, val))


def score_update(lm: Optional[LoadedModel], summary: UpdateSummary) -> TrustResult:
    """Score one client update. Uses the model (adapter disabled) if provided."""
    heuristic = _heuristic_score(summary)

    if lm is None:
        return TrustResult(summary.client_id, heuristic, "model unavailable; heuristic used", "heuristic")

    prompt = (
        "You are auditing a federated learning client's model update for "
        "trustworthiness. A trustworthy update comes from clean, diverse data and "
        "shows a decreasing training loss with a reasonable (non-exploding) update "
        "magnitude. Read the summary and output a single trust score between 0.0 "
        "(untrustworthy) and 1.0 (fully trustworthy), then one sentence of reason.\n\n"
        f"{_summary_text(summary)}\n"
        "Respond exactly as:\nScore: <number>\nReason: <one sentence>\n"
    )
    try:
        # reasoning role = base model without the fine-tuned adapter.
        # Cap at 8 tokens: we only need the "Score: <n>" line, not a long
        # rationale - this cuts trust-scoring time ~3x.
        with lm.model.disable_adapter():
            out = generate(lm, prompt, max_new_tokens=8)
        parsed = _parse_score(out)
        if parsed is not None:
            # Blend the (noisy) 4B model score with the deterministic heuristic.
            # This keeps the model's judgement influential while preventing a
            # single bad read (e.g. the base model writing "Score: 0.0" on a
            # verbose 24-class summary) from wrongly zeroing a healthy node's
            # entire contribution. A genuinely bad update is still down-weighted
            # because the heuristic also penalizes it (no loss drop / exploding norm).
            blended = 0.5 * parsed + 0.5 * heuristic
            return TrustResult(
                summary.client_id, round(blended, 4),
                f"model={parsed:.2f} heuristic={heuristic:.2f} -> blend={blended:.2f} | "
                + out.strip()[:200],
                "model+heuristic",
            )
        logger.warning("Trust agent output unparseable for client %d; using heuristic.", summary.client_id)
    except Exception as exc:  # never let the auditor break aggregation
        logger.warning("Trust agent failed for client %d (%s); using heuristic.", summary.client_id, exc)

    return TrustResult(summary.client_id, heuristic, "fallback heuristic", "heuristic")
