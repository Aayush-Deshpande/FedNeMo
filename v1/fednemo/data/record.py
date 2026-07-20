"""Neutral, dataset-agnostic record type shared across FedNeMo.

Any dataset loader produces a list of ClinicalRecord. The federated pipeline,
serialization, audit, and eval all operate on this type - so swapping datasets
only means writing a new loader, nothing else changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ClinicalRecord:
    """One record ready for prompt serialization.

    - record_id: unique id
    - features: optional structured key->value fields (may be empty {})
    - label: the target class (string)
    - free_text: the free-text input the model reasons over (e.g. symptom text)
    - source: dataset tag (informational)
    """
    record_id: str
    label: str
    free_text: str = ""
    features: Dict[str, object] = field(default_factory=dict)
    source: str = "generic"
