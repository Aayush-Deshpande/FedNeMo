"""One-time sharding: split data/full.jsonl into N per-hospital shards.

Data is IID for Phase 0, so we just round-robin rows. Each hospital only ever
opens its own file, which is what makes it *look* like separate institutions.

Usage (from Code-Implementation/):
    python -m fednemo.data.split 3
"""
from __future__ import annotations

import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DATA_DIR = os.path.normpath(DATA_DIR)


def split(n: int) -> dict:
    full_path = os.path.join(DATA_DIR, "full.jsonl")
    with open(full_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    shards: list[list] = [[] for _ in range(n)]
    for i, row in enumerate(rows):
        shards[i % n].append(row)

    counts = {}
    for i, shard in enumerate(shards):
        out_path = os.path.join(DATA_DIR, f"hospital_{i}.jsonl")
        with open(out_path, "w") as f:
            for row in shard:
                f.write(json.dumps(row) + "\n")
        counts[f"hospital_{i}"] = len(shard)
    return counts


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    result = split(n)
    print(f"split full.jsonl into {n} shards:", result)
