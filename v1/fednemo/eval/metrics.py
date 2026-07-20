"""Manual classification metrics (no sklearn dependency).

Computes per-class precision/recall/F1/support, overall accuracy, macro-F1, and
a confusion matrix from lists of (true, predicted) labels. `predicted` may be
None (unparseable model output) - those count as errors and are tracked
separately as the "unparseable rate".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ClassMetric:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvalMetrics:
    n: int
    accuracy: float
    macro_f1: float
    unparseable: int
    unparseable_rate: float
    per_class: List[ClassMetric] = field(default_factory=list)
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "unparseable": self.unparseable,
            "unparseable_rate": self.unparseable_rate,
            "per_class": [vars(c) for c in self.per_class],
            "confusion": self.confusion,
        }


def compute_metrics(
    pairs: List[Tuple[str, Optional[str]]],
    label_space: List[str],
) -> EvalMetrics:
    n = len(pairs)
    unparseable = sum(1 for _, p in pairs if p is None)

    # confusion[true][pred] ; include an "UNPARSED" pred bucket
    preds_space = list(label_space) + ["UNPARSED"]
    confusion = {t: {p: 0 for p in preds_space} for t in label_space}
    correct = 0
    for true, pred in pairs:
        p_key = pred if pred in label_space else "UNPARSED"
        if true in confusion:
            confusion[true][p_key] += 1
        if pred == true:
            correct += 1

    accuracy = correct / n if n else 0.0

    per_class: List[ClassMetric] = []
    f1s = []
    for lab in label_space:
        tp = confusion.get(lab, {}).get(lab, 0)
        # predicted positives for lab across all true classes
        fp = sum(confusion[t].get(lab, 0) for t in label_space if t != lab)
        support = sum(confusion.get(lab, {}).values())
        fn = support - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class.append(ClassMetric(lab, precision, recall, f1, support))
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return EvalMetrics(
        n=n, accuracy=accuracy, macro_f1=macro_f1,
        unparseable=unparseable, unparseable_rate=(unparseable / n if n else 0.0),
        per_class=per_class, confusion=confusion,
    )


def format_report(m: EvalMetrics, title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"=== {title} ===")
    lines.append(f"n={m.n}  accuracy={m.accuracy:.3f}  macro-F1={m.macro_f1:.3f}  "
                 f"unparseable={m.unparseable} ({m.unparseable_rate:.1%})")
    lines.append(f"{'class':<24} {'prec':>6} {'recall':>7} {'f1':>6} {'support':>8}")
    for c in m.per_class:
        lines.append(f"{c.label:<24} {c.precision:>6.3f} {c.recall:>7.3f} {c.f1:>6.3f} {c.support:>8}")
    return "\n".join(lines)
