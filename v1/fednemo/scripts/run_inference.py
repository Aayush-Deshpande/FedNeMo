"""CLI: run inference on a scanned medical report image.

    set NVIDIA_API_KEY=nvapi-...
    python -m fednemo.scripts.run_inference --image report.png --question "Any MI risk?"

Makes exactly one external call (nemotron-parse); everything else is local.
"""
from __future__ import annotations

import argparse
import logging
import sys

from ..inference.infer import run_inference


def main() -> None:
    ap = argparse.ArgumentParser(description="FedNeMo inference on a report image")
    ap.add_argument("--image", required=True, help="path to a scanned report image")
    ap.add_argument("--question", default="", help="doctor's question (optional)")
    ap.add_argument("--schema", choices=["uci", "ptbxl"], default="uci",
                    help="which training feature schema to normalize toward")
    ap.add_argument("--api-key", default=None, help="override NVIDIA_API_KEY env var")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout,
    )

    result = run_inference(
        image_path=args.image, question=args.question,
        source_schema=args.schema, api_key=args.api_key,
    )

    print("\n=== INFERENCE RESULT ===")
    if result.parse_status != "ok":
        print(f"Document parsing failed: {result.parse_error}")
        return
    print(f"Mapped fields: {result.mapped_fields}")
    print(f"Predicted label: {result.predicted_label}")
    print("\nModel answer:\n" + result.answer)


if __name__ == "__main__":
    main()
