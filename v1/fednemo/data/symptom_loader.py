"""Loader for the Symptom2Disease dataset (text -> disease classification).

CSV columns: [unnamed index], 'label' (disease name), 'text' (symptom description).
1200 rows, 24 balanced disease classes (50 each). The diagnostic signal lives
entirely in the text, which is exactly what a generative LLM classifies well.

Also provides a generic CSV/JSON loader so any text-classification dataset can be
dropped in by naming its text and label columns.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config import DATASETS_DIR
from .record import ClinicalRecord

SYMPTOM_CSV = DATASETS_DIR / "Symptom2Disease.csv"


def load_symptom_records(limit: Optional[int] = None) -> List[ClinicalRecord]:
    return load_text_classification(
        SYMPTOM_CSV, text_field="text", label_field="label",
        source="symptom2disease", limit=limit,
    )


def load_text_classification(
    path: Path,
    text_field: str,
    label_field: str,
    source: str = "generic",
    limit: Optional[int] = None,
) -> List[ClinicalRecord]:
    """Generic loader: read a CSV or JSON with a text column and a label column.

    Supports .csv and .json (list-of-objects or {"data": [...]}) files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        rows = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)

    if text_field not in df.columns or label_field not in df.columns:
        raise KeyError(
            f"Expected columns '{text_field}' and '{label_field}' in {path.name}; "
            f"found {list(df.columns)}"
        )

    records: List[ClinicalRecord] = []
    for i, row in df.iterrows():
        text = row.get(text_field)
        label = row.get(label_field)
        if pd.isna(text) or pd.isna(label):
            continue
        records.append(
            ClinicalRecord(
                record_id=f"{source}_{i}",
                label=str(label).strip(),
                free_text=str(text).strip(),
                features={},
                source=source,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records
