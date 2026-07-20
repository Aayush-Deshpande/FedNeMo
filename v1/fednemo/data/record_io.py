"""Persist/load ClinicalRecord lists as JSON (for held-out eval sets)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from .record import ClinicalRecord


def save_records(records: List[ClinicalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2, ensure_ascii=False)


def load_records(path: Path) -> List[ClinicalRecord]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return [ClinicalRecord(**row) for row in rows]
