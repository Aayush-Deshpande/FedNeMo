"""MedQA-USMLE data layer for Phase 4 (real model path).

Reads the local MedQA-USMLE US question set and shards it into per-hospital
jsonl files (same `hospital_{i}.jsonl` convention that `loader.load_shard`
reads), then builds a causal-LM training prompt from each row.

The dataset is local (no network); rows look like:
    {"question": str, "options": {"A": str, ... "E": str},
     "answer": str, "answer_idx": "A".."E", "meta_info": str}

Usage (from Code-Implementation/):
    python -m fednemo.data.medqa 3            # 3 shards, default 40 rows/shard
    python -m fednemo.data.medqa 3 --per 100  # 3 shards, 100 rows/shard
"""
from __future__ import annotations

import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DATA_DIR = os.path.normpath(DATA_DIR)

# Local MedQA-USMLE US train split (workspace root / MedQA-USMLE/...).
MEDQA_US_TRAIN = os.path.normpath(
    os.path.join(DATA_DIR, "..", "..", "MedQA-USMLE", "questions", "US", "train.jsonl")
)

_OPTION_KEYS = ["A", "B", "C", "D", "E"]


def build_medqa_prompt(row: dict) -> str:
    """Render one MedQA row into the canonical training/eval prompt."""
    opts = row.get("options", {}) or {}
    lines = [f"{k}. {opts[k]}" for k in _OPTION_KEYS if k in opts]
    answer_idx = row.get("answer_idx", "")
    answer = row.get("answer", "")
    return (
        "[MEDQA]\n"
        f"Question: {row.get('question', '')}\n"
        "Options:\n"
        + "\n".join(lines)
        + f"\nAnswer: {answer_idx}. {answer}\n"
        "[/MEDQA]"
    )


def prepare_medqa_shards(n: int = 3, per_shard: int = 40,
                         source_path: str | None = None) -> dict:
    """Round-robin the first `n * per_shard` MedQA rows into n hospital shards.

    Writes data/hospital_{i}.jsonl (overwriting any existing shard files).
    Returns a {client_id: count} map.
    """
    src = source_path or MEDQA_US_TRAIN
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"MedQA source not found at {src}. Expected the MedQA-USMLE folder "
            "at the workspace root."
        )

    take = n * per_shard
    rows: list[dict] = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= take:
                break

    shards: list[list[dict]] = [[] for _ in range(n)]
    for i, row in enumerate(rows):
        shards[i % n].append(row)

    counts = {}
    for i, shard in enumerate(shards):
        out_path = os.path.join(DATA_DIR, f"hospital_{i}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in shard:
                f.write(json.dumps(row) + "\n")
        counts[f"hospital_{i}"] = len(shard)
    return counts


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    per = 40
    if "--per" in sys.argv:
        per = int(sys.argv[sys.argv.index("--per") + 1])
    result = prepare_medqa_shards(n, per)
    print(f"prepared {n} MedQA shards ({per} rows each):", result)
    # show one prompt
    from .loader import load_shard
    print("\nsample prompt:\n")
    print(build_medqa_prompt(load_shard("hospital_0")[0]))
