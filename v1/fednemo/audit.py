"""Per-node audit trail for the FedNeMo pipeline (portfolio / explainability).

Writes a transparent, human-readable record of the ENTIRE federated pipeline for
every node, every round:

  artifacts/audit_<tag>/
    run_config.json                 # full config snapshot
    node_<i>/
      assigned_dataset.json         # the exact records this node was given
      dataset_summary.json          # counts + per-class distribution
      round_<k>/
        report.json                 # timing, per-layer A/B values, DP noise,
                                    # quantization, which matrix sent vs dropped
    global/
      round_<k>.json                # aggregation: trust, weights, eps budget
    run_summary.json                # end-to-end summary + total time

NOTE (honesty): saving raw gradients/weights like this is intentionally NOT what
a real privacy-preserving deployment would do (it would defeat the point). It is
enabled here purely to make the mechanism inspectable for a portfolio.

To keep files readable, each tensor is stored as summary STATS for all layers
plus a small SAMPLE of actual values; set `full_values=True` to also dump the
complete tensors as binary .pt alongside the JSON.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger("fednemo.audit")

SAMPLE_N = 16  # how many actual values to include per tensor in the JSON report


def tensor_report(t: torch.Tensor, sample_n: int = SAMPLE_N) -> dict:
    """Compact, JSON-serializable summary of a tensor: shape, norms, stats, sample."""
    t = t.detach().to("cpu", dtype=torch.float32)
    flat = t.flatten()
    n = flat.numel()
    return {
        "shape": list(t.shape),
        "numel": int(n),
        "l2_norm": float(torch.linalg.vector_norm(flat, ord=2).item()),
        "l1_norm": float(torch.linalg.vector_norm(flat, ord=1).item()),
        "mean": float(flat.mean().item()) if n else 0.0,
        "std": float(flat.std().item()) if n > 1 else 0.0,
        "min": float(flat.min().item()) if n else 0.0,
        "max": float(flat.max().item()) if n else 0.0,
        "abs_mean": float(flat.abs().mean().item()) if n else 0.0,
        "sample_values": [round(float(x), 6) for x in flat[:sample_n].tolist()],
    }


class RunAudit:
    """Manages the audit folder tree and writes reports."""

    def __init__(self, tag: str, artifacts_dir: Path, config_snapshot: dict,
                 full_values: bool = False):
        safe = tag or "run"
        self.root = artifacts_dir / f"audit_{safe}"
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "global").mkdir(exist_ok=True)
        self.full_values = full_values
        self.t0 = time.time()
        with open(self.root / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(config_snapshot, f, indent=2)
        logger.info("Audit trail -> %s", self.root)

    def node_dir(self, node_id: int) -> Path:
        d = self.root / f"node_{node_id}"
        d.mkdir(exist_ok=True)
        return d

    def save_assigned_dataset(self, node_id: int, records: List) -> None:
        """records: list of ClinicalRecord (dataclass)."""
        d = self.node_dir(node_id)
        rows = [asdict(r) for r in records]
        with open(d / "assigned_dataset.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        from collections import Counter
        dist = Counter(r.label for r in records)
        summary = {
            "node_id": node_id,
            "num_records": len(records),
            "num_classes_present": len(dist),
            "class_distribution": dict(dist),
            "sources": dict(Counter(r.source for r in records)),
        }
        with open(d / "dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def save_client_round(self, node_id: int, rnd: int, report: dict,
                          transmitted_tensors: Optional[Dict[str, torch.Tensor]] = None,
                          trained_state: Optional[Dict[str, torch.Tensor]] = None) -> None:
        rdir = self.node_dir(node_id) / f"round_{rnd}"
        rdir.mkdir(exist_ok=True)
        with open(rdir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        if self.full_values and transmitted_tensors is not None:
            torch.save(transmitted_tensors, rdir / "transmitted_full.pt")
        if self.full_values and trained_state is not None:
            torch.save(trained_state, rdir / "trained_adapter_full.pt")

    def save_global_round(self, rnd: int, report: dict) -> None:
        with open(self.root / "global" / f"round_{rnd}.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def finalize(self, summary: dict) -> None:
        summary = dict(summary)
        summary["total_wall_time_s"] = round(time.time() - self.t0, 2)
        with open(self.root / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Audit finalized (%.1fs) -> %s", summary["total_wall_time_s"], self.root)


def build_layer_capture(layer: str, which: str,
                        trained_a: torch.Tensor, trained_b: torch.Tensor,
                        shared_pre_dp: torch.Tensor, post_dp,
                        dequantized: torch.Tensor, qt) -> dict:
    """Assemble the per-layer audit record shown to the user:
    trained A & B -> which sent/dropped -> post-DP (noise) -> post-quant."""
    dropped = "B" if which == "A" else "A"
    return {
        "layer": layer,
        "sent_matrix": which,
        "dropped_matrix": dropped,
        "trained_A": tensor_report(trained_a),
        "trained_B": tensor_report(trained_b),
        "sent_pre_dp": tensor_report(shared_pre_dp),
        "after_dp_noise": {
            **tensor_report(post_dp.tensor),
            "noise_scale": round(float(post_dp.noise_scale), 8),
            "signal_rms": round(float(post_dp.signal_rms), 8),
            "clipped_norm": round(float(post_dp.clipped_norm), 6),
        },
        "after_quantization": {
            **tensor_report(dequantized),
            "bits": qt.bits,
            "num_levels": qt.num_levels(),
            "scale": round(float(qt.scale), 8),
            "zero_point": int(qt.zero_point),
        },
    }
