"""CLI: evaluate a trained adapter on its held-out set.

    python -m fednemo.scripts.run_eval --tag run
"""
from __future__ import annotations

import argparse
import logging
import sys

from ..eval.evaluate import run_eval
from ..eval.metrics import format_report


def main() -> None:
    ap = argparse.ArgumentParser(description="FedNeMo held-out evaluation")
    ap.add_argument("--tag", default="run", help="artifact tag used at training time")
    ap.add_argument("--lora-rank", type=int, default=None,
                    help="LoRA rank the adapter was trained at (must match training)")
    ap.add_argument("--lm-head-device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    from ..config import CONFIG
    if args.lora_rank is not None:
        CONFIG.lora_rank = args.lora_rank
    if args.lm_head_device is not None:
        CONFIG.lm_head_device = args.lm_head_device

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S", stream=sys.stdout,
    )

    metrics = run_eval(tag=args.tag, max_new_tokens=args.max_new_tokens)
    print("\n" + format_report(metrics, title=f"[{args.tag}] held-out"))


if __name__ == "__main__":
    main()
