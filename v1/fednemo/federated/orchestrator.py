"""Federated training orchestrator (fully local, in-process, sequential clients).

Round loop:
  for each round:
    for each client (STRICTLY one at a time, never parallel):
      1. client fine-tunes the shared LoRA adapter on its local shard
      2. FedRand split -> public/private; Laplace DP + 2-bit quant on public
      3. keep private half on the client for next round
      4. build a text summary of the update
    trust-score every client update (base model, adapter disabled)
    entropy-aware + trust-weighted aggregation -> new global adapter
  save final global adapter
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..config import ARTIFACTS_DIR, CONFIG
from ..audit import RunAudit
from ..model.nemotron_local import LoadedModel, load_nemotron
from .aggregator import aggregate, entropy_importance
from .client import ClientNode
from .fedrand import (
    AdapterState,
    extract_adapter_state,
    load_adapter_state,
    split_and_protect,
)
from .quantization import dequantize
from .trust_agent import UpdateSummary, build_summary, score_update
from .privacy_accounting import PrivacyAccountant

logger = logging.getLogger("fednemo.orchestrator")


def _save_adapter_state(state: AdapterState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def compute_class_weights(clients: List[ClientNode]) -> Dict[str, float]:
    """Inverse-frequency class weights over the pooled training labels.

    weight(c) = total / (num_classes * count(c)), normalized so the mean weight
    is ~1. Minority classes get > 1, majority < 1.
    """
    from collections import Counter
    counts: Counter = Counter()
    for c in clients:
        counts.update(c.class_distribution())
    total = sum(counts.values())
    k = len(counts)
    if total == 0 or k == 0:
        return {}
    return {cls: total / (k * cnt) for cls, cnt in counts.items()}


def run_training(
    clients: List[ClientNode],
    num_classes: int,
    lm: Optional[LoadedModel] = None,
    use_trust_model: bool = True,
    artifact_tag: str = "",
    audit: bool = False,
    audit_full: bool = False,
) -> Dict[str, object]:
    """Run the full federated fine-tuning loop. Returns a run report dict.

    audit: if True, write a full per-node audit trail (dataset, per-round
      trained A/B values, DP noise, quantization, which matrix sent vs dropped,
      timing) under artifacts/audit_<tag>/ - see fednemo/audit.py.
    audit_full: also dump complete tensors as binary .pt (large).
    """
    if lm is None:
        lm = load_nemotron(attach_lora=True)

    # class-weighted loss (optional): assign inverse-frequency weights to clients
    if CONFIG.class_weighted_loss:
        cw = compute_class_weights(clients)
        for c in clients:
            c.class_weights = cw
        logger.info("Class-weighted loss ENABLED. Weights: %s",
                    {k: round(v, 3) for k, v in cw.items()})

    # initial global adapter = freshly initialized LoRA (A ~ small, B = 0)
    global_state: AdapterState = extract_adapter_state(lm.model)
    logger.info("Initialized global adapter with %d LoRA tensors.", len(global_state))

    rng = torch.Generator()
    rng.manual_seed(CONFIG.seed)

    accountant = PrivacyAccountant(
        eps_round=CONFIG.dp_epsilon, delta=CONFIG.dp_delta, eps_max=CONFIG.dp_epsilon_max,
    )

    report: Dict[str, object] = {
        "config": {
            "num_clients": len(clients),
            "num_rounds": CONFIG.num_rounds,
            "local_max_steps": CONFIG.local_max_steps,
            "local_epochs": CONFIG.local_epochs,
            "max_seq_len": CONFIG.max_seq_len,
            "lora_rank": CONFIG.lora_rank,
            "dp_epsilon": CONFIG.dp_epsilon,
            "dp_clip_norm": CONFIG.dp_clip_norm,
            "quant_bits": CONFIG.quant_bits,
            "fedrand_share_prob": CONFIG.fedrand_share_prob,
            "class_weighted_loss": CONFIG.class_weighted_loss,
            "lm_head_device": CONFIG.lm_head_device,
        },
        "client_sizes": {c.client_id: len(c.records) for c in clients},
        "rounds": [],
    }

    # ---- optional per-node audit trail ----
    auditor: Optional[RunAudit] = None
    if audit:
        auditor = RunAudit(artifact_tag or "run", ARTIFACTS_DIR,
                           config_snapshot=report["config"], full_values=audit_full)
        for c in clients:
            auditor.save_assigned_dataset(c.client_id, c.records)

    for rnd in range(CONFIG.num_rounds):
        logger.info("################ ROUND %d/%d ################", rnd + 1, CONFIG.num_rounds)
        payloads = []
        summaries: List[UpdateSummary] = []

        # ---- sequential client training (one at a time) ----
        for client in clients:
            t_train0 = time.time()
            result = client.local_train(lm, global_state)
            train_s = time.time() - t_train0

            capture: Optional[List[dict]] = [] if auditor else None
            t_prot0 = time.time()
            payload, private_state = split_and_protect(
                client_id=client.client_id,
                updated_state=result.updated_state,
                share_prob=CONFIG.fedrand_share_prob,
                clip_norm=CONFIG.dp_clip_norm,
                epsilon=CONFIG.dp_epsilon,
                quant_bits=CONFIG.quant_bits,
                rng=rng,
                clip_type=CONFIG.dp_clip_type,
                capture=capture,
            )
            protect_s = time.time() - t_prot0
            payload.num_samples = result.num_samples
            payload.class_distribution = result.class_distribution
            payload.loss_trajectory = result.loss_trajectory
            # persist the private FedRand half on the client for next round
            client.private_state = private_state
            payloads.append(payload)

            # ---- write this client's per-round audit record ----
            if auditor:
                n_shared_a = sum(1 for c in capture if c["sent_matrix"] == "A")
                node_report = {
                    "client_id": client.client_id,
                    "round": rnd + 1,
                    "timing_seconds": {
                        "local_training": round(train_s, 2),
                        "split_dp_quant": round(protect_s, 3),
                    },
                    "local_training": {
                        "steps": len(result.loss_trajectory),
                        "num_samples": result.num_samples,
                        "class_distribution": result.class_distribution,
                        "loss_start": result.loss_trajectory[0] if result.loss_trajectory else None,
                        "loss_end": result.loss_trajectory[-1] if result.loss_trajectory else None,
                        "loss_trajectory": [round(x, 4) for x in result.loss_trajectory],
                    },
                    "fedrand_summary": {
                        "total_layers": len(capture),
                        "sent_A_count": n_shared_a,
                        "sent_B_count": len(capture) - n_shared_a,
                        "share_prob": CONFIG.fedrand_share_prob,
                    },
                    "dp_summary": {
                        "mode": CONFIG.dp_mode,
                        "noise_ratio": CONFIG.dp_noise_ratio,
                        "epsilon_per_round": CONFIG.dp_epsilon,
                        "update_l2_norm": payload.update_l2_norm,
                    },
                    "quantization": {"bits": CONFIG.quant_bits},
                    "per_layer": capture,
                }
                transmitted_full = None
                if audit_full:
                    transmitted_full = {
                        f"{sm.layer}.{sm.which}": dequantize(sm.quantized)
                        for sm in payload.shared
                    }
                auditor.save_client_round(
                    client.client_id, rnd + 1, node_report,
                    transmitted_tensors=transmitted_full,
                    trained_state=result.updated_state if audit_full else None,
                )

            summaries.append(
                build_summary(
                    client_id=client.client_id,
                    num_samples=result.num_samples,
                    class_distribution=result.class_distribution,
                    loss_trajectory=result.loss_trajectory,
                    update_l2_norm=payload.update_l2_norm,
                    num_classes=num_classes,
                )
            )
            # free GPU cache between clients (sequential VRAM discipline)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ---- trust scoring (second logical Nemotron role) ----
        trust = {}
        trust_model = lm if use_trust_model else None
        for s in summaries:
            tr = score_update(trust_model, s)
            trust[s.client_id] = tr
            logger.info(
                "Trust[client %d] = %.3f (%s) | %s",
                s.client_id, tr.score, tr.source, tr.rationale[:80].replace("\n", " "),
            )

        # ---- aggregation ----
        nu = entropy_importance(summaries)
        # effective per-client weight w_i = trust_i * nu_i and its normalized share
        eff_w = {s.client_id: max(0.0, trust[s.client_id].score) * max(1e-6, nu.get(s.client_id, 0.0))
                 for s in summaries}
        tot_w = sum(eff_w.values()) or 1.0
        eff_share = {cid: w / tot_w for cid, w in eff_w.items()}
        global_state, contrib = aggregate(global_state, payloads, summaries, trust)
        logger.info("Aggregated %d slots this round. Effective weight shares: %s",
                    len(contrib), {c: round(v, 3) for c, v in eff_share.items()})

        # ---- privacy accounting (RDP composition across rounds) ----
        priv = accountant.step()
        logger.info(
            "Privacy budget: eps_total(RDP)=%.3f | naive=%.1f | delta=%.0e (round %d)",
            priv.eps_total_rdp, priv.eps_total_naive, priv.delta, accountant.rounds,
        )

        # ---- distributed-DP effective noise (aggregate benefit) ----
        dist_dp = None
        if CONFIG.dp_distributed:
            import math as _math
            scales = [sm.noise_scale for p in payloads for sm in p.shared]
            n_contrib = max(1, len(payloads))
            if scales:
                mean_scale = sum(scales) / len(scales)
                eff = mean_scale / _math.sqrt(n_contrib)
                dist_dp = {"per_node_noise": round(mean_scale, 6),
                           "n_nodes": n_contrib,
                           "effective_aggregate_noise": round(eff, 6)}
                logger.info(
                    "Distributed-DP: per-node noise=%.5f -> effective aggregate noise=%.5f "
                    "(reduced ~%.2fx by averaging %d nodes)",
                    mean_scale, eff, _math.sqrt(n_contrib), n_contrib,
                )

        report["rounds"].append({
            "round": rnd + 1,
            "trust": {cid: tr.score for cid, tr in trust.items()},
            "entropy_importance": nu,
            "num_aggregated_slots": len(contrib),
            "client_final_loss": {
                s.client_id: (s.loss_trajectory[-1] if s.loss_trajectory else None)
                for s in summaries
            },
            "eps_total_rdp": priv.eps_total_rdp,
            "eps_total_naive": priv.eps_total_naive,
            "distributed_dp": dist_dp,
            "update_l2_norm": {s.client_id: s.update_l2_norm for s in summaries},
            "effective_weight_share": eff_share,
        })

        if auditor:
            auditor.save_global_round(rnd + 1, {
                "round": rnd + 1,
                "trust": {cid: tr.score for cid, tr in trust.items()},
                "trust_source": {cid: tr.source for cid, tr in trust.items()},
                "entropy_importance": nu,
                "effective_weight_share": eff_share,
                "num_aggregated_slots": len(contrib),
                "eps_total_rdp": priv.eps_total_rdp,
                "eps_total_naive": priv.eps_total_naive,
            })

        if accountant.budget_exhausted():
            logger.warning("Privacy budget ceiling (eps_max=%s) reached; stopping early.",
                           CONFIG.dp_epsilon_max)
            break

    # ---- persist final global adapter ----
    tag = f"_{artifact_tag}" if artifact_tag else ""
    out_path = ARTIFACTS_DIR / f"global_adapter{tag}.pt"
    _save_adapter_state(global_state, out_path)
    load_adapter_state(lm.model, global_state)  # leave model holding the global adapter
    report["adapter_path"] = str(out_path)
    logger.info("Saved global adapter -> %s", out_path)

    with open(ARTIFACTS_DIR / f"training_report{tag}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if auditor:
        auditor.finalize({
            "tag": artifact_tag,
            "num_clients": len(clients),
            "num_rounds_completed": len(report["rounds"]),
            "adapter_path": report["adapter_path"],
            "final_eps_total_rdp": report["rounds"][-1]["eps_total_rdp"] if report["rounds"] else None,
        })

    return report
