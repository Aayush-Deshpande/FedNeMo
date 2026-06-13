"""Read a hospital's shard and build the canonical training prompt.

In Phase 0 the prompt is built but the fake model ignores it -- we build it now
so the contract exists and Phase 4 (real model) can just consume it.
"""
from __future__ import annotations

import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DATA_DIR = os.path.normpath(DATA_DIR)


def load_shard(client_id: str) -> list[dict]:
    """Load one hospital's rows. client_id is e.g. 'hospital_0'."""
    path = os.path.join(DATA_DIR, f"{client_id}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python -m fednemo.data.split N` first."
        )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(row: dict) -> str:
    """Merge structured metadata + free-text note into the canonical template."""
    s = row.get("structured", {})
    return (
        "[PATIENT_RECORD]\n"
        f"Demographics: {s.get('age', '?')}y, {s.get('sex', '?')}, "
        f"BMI: {s.get('bmi', '?')}\n"
        f"Lab Values: {s.get('labs', '?')}\n"
        f"ICD-10 Codes: {s.get('icd', '?')}\n"
        f"Medications: {s.get('medications', '?')}\n\n"
        "Clinical Note:\n"
        f"{row.get('note', '')}\n\n"
        f"Task: {row.get('task', '')}\n"
        "[/PATIENT_RECORD]"
    )


if __name__ == "__main__":
    rows = load_shard("hospital_0")
    print(f"hospital_0 has {len(rows)} rows. First prompt:\n")
    print(build_prompt(rows[0]))
