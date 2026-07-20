"""CLI: report communication savings from a saved adapter's LoRA shapes.

    python -m fednemo.scripts.run_comm_report --tag ptbxl_baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import ARTIFACTS_DIR, CONFIG
from ..eval.comm_accounting import account_from_adapter, format_comm_report


def main() -> None:
    ap = argparse.ArgumentParser(description="FedNeMo communication accounting")
    ap.add_argument("--tag", default="", help="artifact tag used at training time")
    ap.add_argument("--bits", type=int, default=CONFIG.quant_bits)
    args = ap.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    adapter_path = ARTIFACTS_DIR / f"global_adapter{suffix}.pt"
    if not adapter_path.exists():
        raise SystemExit(f"Adapter not found: {adapter_path}")

    report = account_from_adapter(adapter_path, bits=args.bits)
    print(format_comm_report(report))

    out = ARTIFACTS_DIR / f"comm_report{suffix}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
