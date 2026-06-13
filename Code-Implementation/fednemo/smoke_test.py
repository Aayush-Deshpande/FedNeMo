"""One-command smoke test: proves the whole pipeline works and writes a
human-readable RUN_REPORT.md you can open and understand.

Run it after cloning:

    cd Code-Implementation
    pip install -r requirements.txt
    python -m fednemo.smoke_test

Then open RUN_REPORT.md. If every check says PASS, the system works.
If something is broken, the report says exactly which check failed and why.

This does NOT need streamlit, a GPU, or any network. Pure CPU, a few seconds.
"""
from __future__ import annotations

import copy
import datetime as _dt
import os
import traceback

import numpy as np

from .attacker import attack
from .client import Client
from .filters import FedRandFilter, default_chain, run_chain
from .schema import LAYERS, init_weights, make_update, validate
from .server import Server

REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "RUN_REPORT.md")
REPORT_PATH = os.path.normpath(REPORT_PATH)


class _Checks:
    """Collects (name, passed, detail) so the report can list every check."""

    def __init__(self):
        self.results = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.results.append((name, bool(condition), detail))

    @property
    def all_passed(self) -> bool:
        return all(ok for _, ok, _ in self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for _, ok, _ in self.results if ok)


def _run_checks() -> tuple[_Checks, dict]:
    c = _Checks()
    evidence = {}

    # 1. Schema contract holds.
    g = init_weights()
    u = make_update("hospital_0", 0, {"layer0.A": g["layer0"]["A"]}, num_samples=10)
    c.check("Update contract validates", validate(u),
            "schema.validate() accepts a well-formed Update")

    # 2. FedAvg math is correct (weighted mean 2.5).
    s = Server()
    d1 = {"layer0.A": np.ones((8, 64)), "layer0.B": np.ones((64, 8))}
    d2 = {"layer0.A": np.full((8, 64), 3.0), "layer0.B": np.full((64, 8), 3.0)}
    s.aggregate([make_update("h0", 0, d1, 10), make_update("h1", 0, d2, 30)])
    fedavg_val = float(s.g["layer0"]["A"][0, 0])
    c.check("Sample-weighted FedAvg is correct", abs(fedavg_val - 2.5) < 1e-9,
            f"(10x1 + 30x3)/40 expected 2.5, got {fedavg_val:.4f}")

    # 3. FedRand really drops one matrix.
    raw = make_update(
        "hospital_0", 1,
        {"layer0.A": np.zeros((8, 64)), "layer0.B": np.zeros((64, 8)),
         "layer1.A": np.zeros((8, 64)), "layer1.B": np.zeros((64, 8))},
        num_samples=5,
    )
    fragmented = FedRandFilter()(copy.deepcopy(raw))
    kept = fragmented["meta"]["sent_matrix"]
    kinds = {k.split(".")[-1] for k in fragmented["tensors"]}
    c.check("FedRand keeps exactly one matrix kind", kinds == {kept},
            f"kept '{kept}', tensors now {sorted(fragmented['tensors'])}")

    # 4. Attack: succeeds unprotected, fails protected.
    text = "Patient reports increased thirst and fatigue over two weeks."
    unprot = attack(copy.deepcopy(raw), text, seed=0)
    prot = attack(FedRandFilter()(copy.deepcopy(raw)), text, seed=0)
    c.check("Attack RECONSTRUCTS unprotected update", unprot["success"] is True,
            "full A+B, clean -> GIA succeeds")
    c.check("Attack FAILS on FedRand-protected update", prot["success"] is False,
            "one matrix only -> GIA gets noise")
    evidence["attack_original"] = text
    evidence["attack_unprotected"] = unprot["reconstruction"]
    evidence["attack_protected"] = prot["reconstruction"]

    # 5. Full loop turns and loss moves.
    Server.reset_telemetry()
    clients = [Client(f"hospital_{i}", seed=i) for i in range(3)]
    server = Server(init_weights())
    chain = default_chain()
    losses = []
    sent_log = []
    for rnd in range(5):
        gw = server.get_global()
        updates = [run_chain(cl.train(gw, rnd), chain) for cl in clients]
        server.aggregate(updates)
        loss = sum(abs(server.g[l]["A"]).mean() for l in LAYERS) / len(LAYERS)
        losses.append(float(loss))
        sent_log.append({u2["client_id"]: u2["meta"]["sent_matrix"] for u2 in updates})
    c.check("Federated loop runs 5 rounds without error", len(losses) == 5,
            f"losses per round: {[round(x, 5) for x in losses]}")
    c.check("Loss metric changes across rounds",
            len(set(round(x, 6) for x in losses)) > 1,
            "the global state actually updates each round")
    evidence["losses"] = losses
    evidence["sent_log"] = sent_log

    return c, evidence


def _write_report(c: _Checks, evidence: dict, error: str | None = None) -> None:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# FedNeMo — Run Report\n")
    lines.append(f"Generated: {now}\n")

    if error is not None:
        lines.append("## RESULT: ❌ BROKEN\n")
        lines.append("The smoke test crashed before finishing. "
                     "Something is wrong with the install or the code.\n")
        lines.append("### Error\n")
        lines.append("```\n" + error + "\n```\n")
        with open(REPORT_PATH, "w") as f:
            f.write("\n".join(lines))
        return

    verdict = "✅ WORKING" if c.all_passed else "❌ SOME CHECKS FAILED"
    lines.append(f"## RESULT: {verdict}\n")
    lines.append(f"Passed {c.n_passed} of {len(c.results)} checks.\n")

    lines.append("## Checks\n")
    lines.append("| # | Check | Result | Detail |")
    lines.append("| :- | :--- | :--- | :--- |")
    for i, (name, ok, detail) in enumerate(c.results, 1):
        mark = "✅ PASS" if ok else "❌ FAIL"
        lines.append(f"| {i} | {name} | {mark} | {detail} |")
    lines.append("")

    # Human-readable evidence so you can SEE it working.
    lines.append("## What the attacker saw (the headline demo)\n")
    lines.append("Original patient record (ground truth):\n")
    lines.append("```\n" + evidence.get("attack_original", "") + "\n```\n")
    lines.append("Unprotected capture — attacker reconstructs it (BAD, this is the "
                 "vulnerability FedNeMo fixes):\n")
    lines.append("```\n" + evidence.get("attack_unprotected", "") + "\n```\n")
    lines.append("FedRand-protected capture — attacker gets noise (GOOD):\n")
    lines.append("```\n" + evidence.get("attack_protected", "") + "\n```\n")

    lines.append("## The loop actually ran\n")
    losses = evidence.get("losses", [])
    lines.append("Loss metric per round (should change, not stay flat):\n")
    for i, lo in enumerate(losses):
        lines.append(f"- round {i}: {lo:.6f}")
    lines.append("")
    lines.append("Which LoRA matrix each hospital sent each round "
                 "(FedRand — should be a mix of A and B):\n")
    lines.append("| round | " + " | ".join(
        sorted(evidence.get("sent_log", [{}])[0].keys())) + " |")
    keys = sorted(evidence.get("sent_log", [{}])[0].keys())
    lines.append("| :- | " + " | ".join(":-" for _ in keys) + " |")
    for i, row in enumerate(evidence.get("sent_log", [])):
        lines.append(f"| {i} | " + " | ".join(row.get(k, "?") for k in keys) + " |")
    lines.append("")

    lines.append("---\n")
    lines.append("If RESULT says WORKING, the Phase 0 + Phase 1 pipeline is sound "
                 "on your machine. Re-run `python -m fednemo.smoke_test` anytime "
                 "to regenerate this report.\n")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    try:
        c, evidence = _run_checks()
        _write_report(c, evidence)
        verdict = "WORKING" if c.all_passed else "SOME CHECKS FAILED"
        print(f"Smoke test done: {verdict} ({c.n_passed}/{len(c.results)} checks).")
        print(f"Open the report: {REPORT_PATH}")
        for name, ok, _ in c.results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    except Exception:
        err = traceback.format_exc()
        _write_report(_Checks(), {}, error=err)
        print("Smoke test CRASHED. See RUN_REPORT.md for the traceback.")
        print(err)
        raise


if __name__ == "__main__":
    main()
