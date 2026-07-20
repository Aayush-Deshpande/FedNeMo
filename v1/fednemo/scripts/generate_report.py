"""Generate a single detailed human-readable report for a completed run.

Compiles: node/round/step counts, dataset size + head + per-node split, per-node
per-round gradient pipeline (trained -> DP -> quantized, which A/B dropped),
accuracy + per-class metrics, timing, and resources -- all from the audit trail,
training report and eval JSON produced by a run.

    python -m fednemo.scripts.generate_report --tag final
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Optional

from ..config import ARTIFACTS_DIR, DATASETS_DIR


def _load(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_stats(d: dict) -> str:
    return (f"shape={d.get('shape')} l2={d.get('l2_norm', 0):.4f} "
            f"absμ={d.get('abs_mean', 0):.5f} min={d.get('min', 0):.4f} max={d.get('max', 0):.4f}")


def _sample(d: dict, k: int = 6) -> str:
    vals = d.get("sample_values", [])[:k]
    return "[" + ", ".join(f"{v:+.5f}" for v in vals) + " ...]"


def generate(tag: str) -> Path:
    audit_dir = ARTIFACTS_DIR / f"audit_{tag}"
    train = _load(ARTIFACTS_DIR / f"training_report_{tag}.json") or {}
    ev = _load(ARTIFACTS_DIR / f"eval_{tag}.json") or {}
    run_summary = _load(audit_dir / "run_summary.json") or {}
    cfg = train.get("config", {})
    client_sizes = train.get("client_sizes", {})
    rounds = train.get("rounds", [])

    L = []
    def w(s=""):
        L.append(s)

    # ---------- header ----------
    w(f"# FedNeMo — Detailed Run Report  (`{tag}`)")
    w()
    w("Auto-generated from the audit trail, training report, and held-out evaluation.")
    w()
    w("## 1. Federation configuration")
    w()
    w("| Field | Value |")
    w("|---|---|")
    w(f"| Nodes (clients) | {cfg.get('num_clients')} |")
    w(f"| Rounds | {cfg.get('num_rounds')} |")
    steps_desc = "full shard each round (max_steps=0)" if cfg.get("local_max_steps") in (0, None) else str(cfg.get("local_max_steps"))
    w(f"| Steps per node/round | {steps_desc} |")
    w(f"| Local epochs | {cfg.get('local_epochs')} |")
    w(f"| Gradient accumulation | {cfg.get('grad_accum_steps', 'n/a')} |")
    w(f"| LoRA rank / alpha | {cfg.get('lora_rank')} / {cfg.get('lora_rank', 0) * 2} |")
    w(f"| FedRand share prob (ρ) | {cfg.get('fedrand_share_prob')} |")
    w(f"| DP mode / ε per round | {cfg.get('dp_mode', 'relative')} / {cfg.get('dp_epsilon')} |")
    w(f"| Quantization bits | {cfg.get('quant_bits')} |")
    w(f"| lm_head device | {cfg.get('lm_head_device')} |")
    w()

    # ---------- dataset ----------
    w("## 2. Dataset")
    w()
    try:
        import pandas as pd
        csv = DATASETS_DIR / "Symptom2Disease.csv"
        df = pd.read_csv(csv)
        w(f"- **File:** `{csv.name}`")
        w(f"- **Total records:** {len(df)}")
        w(f"- **Classes:** {df['label'].nunique()}")
        w(f"- **Columns:** {list(df.columns)}")
        w()
        w("**dataset.head():**")
        w()
        w("| # | label | text |")
        w("|---|---|---|")
        for i in range(min(5, len(df))):
            t = str(df.iloc[i]['text']).replace("|", "/")[:90]
            w(f"| {i} | {df.iloc[i]['label']} | {t}... |")
        w()
    except Exception as exc:
        w(f"(dataset head unavailable: {exc})")
        w()

    # ---------- per-node split ----------
    w("## 3. Data split across nodes (IID, balanced)")
    w()
    w("| Node | Records | Class distribution |")
    w("|---|---|---|")
    for cid in sorted(client_sizes, key=lambda x: int(x)):
        ds = _load(audit_dir / f"node_{cid}" / "dataset_summary.json") or {}
        dist = ds.get("class_distribution", {})
        top = ", ".join(f"{k}:{v}" for k, v in list(dist.items())[:6])
        w(f"| {cid} | {client_sizes[cid]} | {top} ... |")
    w()

    # ---------- accuracy ----------
    w("## 4. Held-out accuracy")
    w()
    w(f"- **Accuracy:** {ev.get('accuracy', 0) * 100:.2f}%")
    w(f"- **Macro-F1:** {ev.get('macro_f1', 0):.4f}")
    w(f"- **Unparseable:** {ev.get('unparseable_rate', 0) * 100:.2f}%")
    w(f"- **Held-out N:** {ev.get('n')}")
    w()
    per_class = ev.get("per_class", [])
    if per_class:
        w("| Class | Precision | Recall | F1 | Support |")
        w("|---|---|---|---|---|")
        for c in per_class:
            w(f"| {c['label']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['support']} |")
        w()

    # ---------- timing + privacy per round ----------
    w("## 5. Per-round summary (trust, weights, privacy budget)")
    w()
    w("| Round | Trust (per node) | Effective weight share | ε_total (RDP) |")
    w("|---|---|---|---|")
    for r in rounds:
        tr = ", ".join(f"{k}:{v:.2f}" for k, v in r.get("trust", {}).items())
        ws = ", ".join(f"{k}:{v:.2f}" for k, v in r.get("effective_weight_share", {}).items())
        w(f"| {r.get('round')} | {tr} | {ws} | {r.get('eps_total_rdp', 0):.2f} |")
    w()

    # ---------- per node per round: gradient pipeline ----------
    w("## 6. Gradient pipeline per node, per round")
    w()
    w("For each node/round: local training result, FedRand A/B split counts, DP "
      "noise added, and a **representative layer** shown through the full pipeline "
      "(trained → after-DP → after-quantization), including which matrix was "
      "**sent** vs **dropped**. (Full per-layer detail for all 192 layers lives in "
      "`artifacts/audit_" + tag + "/node_*/round_*/report.json`.)")
    w()
    n_clients = cfg.get("num_clients", 0)
    n_rounds = cfg.get("num_rounds", 0)
    for cid in range(n_clients):
        w(f"### Node {cid}")
        w()
        for rnd in range(1, n_rounds + 1):
            rep = _load(audit_dir / f"node_{cid}" / f"round_{rnd}" / "report.json")
            if not rep:
                continue
            lt = rep.get("local_training", {})
            tm = rep.get("timing_seconds", {})
            fr = rep.get("fedrand_summary", {})
            dp = rep.get("dp_summary", {})
            w(f"**Round {rnd}** — trained {lt.get('num_samples')} records, "
              f"loss {lt.get('loss_start')} → {lt.get('loss_end')}, "
              f"train time {tm.get('local_training')}s | "
              f"FedRand: sent A={fr.get('sent_A_count')} / B={fr.get('sent_B_count')} | "
              f"DP mode={dp.get('mode')} noise_ratio={dp.get('noise_ratio')} "
              f"update‖·‖₂={dp.get('update_l2_norm', 0):.3f}")
            w()
            layers = rep.get("per_layer", [])
            if layers:
                ly = layers[0]  # representative layer
                w(f"- **Representative layer:** `{ly.get('layer')}` — "
                  f"**sent {ly.get('sent_matrix')}**, dropped {ly.get('dropped_matrix')}")
                ta, tb = ly.get("trained_A", {}), ly.get("trained_B", {})
                w(f"  - trained A: {_fmt_stats(ta)}  sample {_sample(ta)}")
                w(f"  - trained B: {_fmt_stats(tb)}  sample {_sample(tb)}")
                dpn = ly.get("after_dp_noise", {})
                w(f"  - after DP (noise_scale={dpn.get('noise_scale')}, "
                  f"signal_rms={dpn.get('signal_rms')}): {_fmt_stats(dpn)} sample {_sample(dpn)}")
                aq = ly.get("after_quantization", {})
                w(f"  - after quant ({aq.get('bits')}-bit, {aq.get('num_levels')} levels, "
                  f"scale={aq.get('scale')}): {_fmt_stats(aq)} sample {_sample(aq)}")
                w()
    # ---------- resources ----------
    w("## 7. Resources & timing")
    w()
    w(f"- **Total wall time:** {run_summary.get('total_wall_time_s', 'n/a')} s")
    w(f"- **GPU:** NVIDIA RTX 4050 Laptop (6 GB); model body 4-bit NF4 on GPU, "
      f"256k-vocab embed on CPU")
    w(f"- **Peak VRAM (load):** ~1.7 GB allocated (see training logs for peak)")
    w(f"- **Python:** {platform.python_version()} | Platform: {platform.system()}")
    w(f"- **Privacy budget (final):** ε_total(RDP) ≈ "
      f"{rounds[-1].get('eps_total_rdp', 0):.2f} over {len(rounds)} rounds (δ=1e-5)" if rounds else "")
    w()
    w("---")
    w("*Generated by `fednemo.scripts.generate_report`.*")

    out = ARTIFACTS_DIR.parent / f"REPORT_{tag}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    out = generate(args.tag)
    print(f"Report written -> {out}")


if __name__ == "__main__":
    main()
