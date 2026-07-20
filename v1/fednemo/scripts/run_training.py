"""CLI: run federated fine-tuning on a text-classification dataset.

    python -m fednemo.scripts.run_training --tag run1 --rounds 3 --max-steps 60

Default dataset is Symptom2Disease (symptom text -> disease). A stratified
held-out split is reserved BEFORE the IID (equal, balanced) partition across
nodes. With --audit (default ON) a full per-node/per-round/per-run report is
written to artifacts/audit_<tag>/.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..config import ARTIFACTS_DIR, CONFIG
from ..data.partition import iid_partition, log_partition_stats, stratified_holdout
from ..data.record_io import save_records
from ..data.symptom_loader import SYMPTOM_CSV, load_text_classification
from ..federated.client import ClientNode
from ..federated.orchestrator import run_training
from ..model.serialization import set_label_space


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="FedNeMo federated fine-tuning")
    ap.add_argument("--data", default=str(SYMPTOM_CSV), help="path to a CSV/JSON dataset")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--source", default="symptom2disease")
    ap.add_argument("--limit", type=int, default=None, help="cap records (default: all)")
    ap.add_argument("--rounds", type=int, default=CONFIG.num_rounds)
    ap.add_argument("--clients", type=int, default=CONFIG.num_clients)
    ap.add_argument("--max-steps", type=int, default=CONFIG.local_max_steps)
    ap.add_argument("--epochs", type=int, default=CONFIG.local_epochs)
    ap.add_argument("--max-seq-len", type=int, default=CONFIG.max_seq_len)
    ap.add_argument("--lora-rank", type=int, default=CONFIG.lora_rank)
    ap.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    ap.add_argument("--epsilon", type=float, default=CONFIG.dp_epsilon)
    ap.add_argument("--clip-norm", type=float, default=CONFIG.dp_clip_norm)
    ap.add_argument("--quant-bits", type=int, default=CONFIG.quant_bits)
    ap.add_argument("--dp-mode", choices=["relative", "absolute"], default=CONFIG.dp_mode)
    ap.add_argument("--noise-ratio", type=float, default=CONFIG.dp_noise_ratio)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--class-weighted", action="store_true")
    ap.add_argument("--lm-head-device", choices=["cuda", "cpu"], default=CONFIG.lm_head_device)
    ap.add_argument("--no-trust-model", action="store_true")
    ap.add_argument("--no-audit", action="store_true", help="disable the per-node audit report")
    ap.add_argument("--audit-full", action="store_true", help="also dump full tensors (.pt)")
    ap.add_argument("--tag", default="run", help="artifact/report tag")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    _setup_logging(not args.quiet)

    CONFIG.num_rounds = args.rounds
    CONFIG.num_clients = args.clients
    CONFIG.local_max_steps = args.max_steps
    CONFIG.local_epochs = args.epochs
    CONFIG.max_seq_len = args.max_seq_len
    CONFIG.lora_rank = args.lora_rank
    CONFIG.learning_rate = args.lr
    CONFIG.dp_epsilon = args.epsilon
    CONFIG.dp_clip_norm = args.clip_norm
    CONFIG.quant_bits = args.quant_bits
    CONFIG.dp_mode = args.dp_mode
    CONFIG.dp_noise_ratio = args.noise_ratio
    CONFIG.class_weighted_loss = args.class_weighted
    CONFIG.lm_head_device = args.lm_head_device

    records = load_text_classification(
        Path(args.data), text_field=args.text_field, label_field=args.label_field,
        source=args.source, limit=args.limit,
    )
    if not records:
        raise SystemExit(f"No records loaded from {args.data}")

    labels_all = [r.label for r in records]
    set_label_space(labels_all)   # register the dataset's label space
    num_classes = len(set(labels_all))
    logger.info("Loaded %d records, %d classes.", len(records), num_classes)

    # 1) stratified held-out split BEFORE partitioning
    train_idx, holdout_idx = stratified_holdout(
        records, labels_all, holdout_frac=args.holdout_frac, seed=CONFIG.seed,
    )
    train_records = [records[i] for i in train_idx]
    holdout_records = [records[i] for i in holdout_idx]

    save_records(holdout_records, ARTIFACTS_DIR / f"holdout_{args.tag}.json")

    # 2) IID (equal, balanced) partition across nodes
    train_labels = [r.label for r in train_records]
    parts = iid_partition(train_labels, num_clients=args.clients, seed=CONFIG.seed, stratified=True)
    log_partition_stats(train_labels, parts)

    clients = [ClientNode(cid, [train_records[i] for i in idxs]) for cid, idxs in enumerate(parts)]

    report = run_training(
        clients, num_classes=num_classes, use_trust_model=not args.no_trust_model,
        artifact_tag=args.tag, audit=not args.no_audit, audit_full=args.audit_full,
    )
    print("\n=== TRAINING COMPLETE ===")
    print(f"Dataset: {args.data} | {len(records)} records | {num_classes} classes")
    print(f"Train {len(train_records)} | holdout {len(holdout_records)}")
    print(f"Global adapter: {report['adapter_path']}")
    if not args.no_audit:
        print(f"Per-node report: {ARTIFACTS_DIR / ('audit_' + args.tag)}")


logger = logging.getLogger("fednemo.run_training")

if __name__ == "__main__":
    main()
