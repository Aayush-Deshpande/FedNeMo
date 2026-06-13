"""Streamlit dashboard. Reads telemetry.jsonl ONLY -- never imports training code.

That file-only coupling is deliberate: the dashboard runs in its own process and
can never slow training down. You can even build it against a hand-faked
telemetry.jsonl before the driver exists.

Usage (from Code-Implementation/):
    streamlit run fednemo/dashboard.py
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

TELEMETRY_PATH = os.path.join(os.path.dirname(__file__), "..", "telemetry.jsonl")
TELEMETRY_PATH = os.path.normpath(TELEMETRY_PATH)


def load_telemetry() -> list[dict]:
    if not os.path.exists(TELEMETRY_PATH):
        return []
    with open(TELEMETRY_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


st.set_page_config(page_title="FedNeMo — Phase 3", layout="wide")
st.title("FedNeMo — Federated Round Telemetry (Phase 3)")

rows = load_telemetry()
if not rows:
    st.warning("No telemetry yet. Run `python -m fednemo.driver` first.")
    st.stop()

df = pd.DataFrame(rows)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Training loss (placeholder metric)")
    st.line_chart(df.set_index("round")["loss"])
with col2:
    st.subheader("Cumulative privacy budget (RDP)")
    st.line_chart(df.set_index("round")["eps_total"])
    eps_val = df['eps_total'].iloc[-1]
    st.metric("eps_total", f"{eps_val:.2f}")
    # Color-coded budget indicator
    if eps_val < 5:
        st.success(f"🟢 Budget healthy: ε = {eps_val:.2f}")
    elif eps_val < 8:
        st.warning(f"🟡 Budget moderate: ε = {eps_val:.2f}")
    else:
        st.error(f"🔴 Budget high: ε = {eps_val:.2f}")

st.subheader("FedRand: which matrix each hospital sent each round")
st.caption("Values alternate between A and B (FedRand), with Laplacian DP noise "
           "and adaptive quantization applied.")
sent_df = pd.DataFrame([r["sent"] for r in rows], index=[r["round"] for r in rows])
sent_df.index.name = "round"
st.dataframe(sent_df, use_container_width=True)

# --- Adaptive Quantization --------------------------------------------------
st.subheader("Adaptive quantization: bit-width per hospital per round")
st.caption(
    "Bit-width is driven by each hospital's ICD-code Shannon entropy. "
    "With IID round-robin sharding all hospitals have near-identical entropy "
    "(≈3.5), so they correctly receive the same 8-bit precision. "
    "In a non-IID deployment (specialty hospitals), bit-widths would diverge."
)
bits_df = pd.DataFrame([r["bits"] for r in rows], index=[r["round"] for r in rows])
bits_df.index.name = "round"
st.dataframe(bits_df, use_container_width=True)

# Bandwidth savings
bw_rows = [r for r in rows if "bandwidth" in r]
if bw_rows:
    col_bw1, col_bw2 = st.columns(2)
    with col_bw1:
        saved_series = [r["bandwidth"]["saved_pct"] for r in bw_rows]
        avg_saved = sum(saved_series) / len(saved_series)
        st.metric("Avg bandwidth saved", f"{avg_saved:.1f}%")
    with col_bw2:
        total_baseline = sum(r["bandwidth"]["baseline_bytes"] for r in bw_rows)
        total_actual = sum(r["bandwidth"]["actual_bytes"] for r in bw_rows)
        st.metric("Cumulative bytes saved",
                  f"{(total_baseline - total_actual):,} bytes")

# --- Live Gradient Inversion Attack demo -----------------------------------
attack_rows = [r for r in rows if "attack" in r]
if attack_rows:
    st.subheader("Live Gradient Inversion Attack — unprotected vs FedRand-protected")
    latest = attack_rows[-1]
    a = latest["attack"]
    st.caption(f"Showing round {latest['round']}. "
               "Left = attacker captures a full unprotected update. "
               "Right = attacker captures the FedRand-fragmented update.")

    left, right = st.columns(2)
    with left:
        ok = a["unprotected"]["success"]
        st.markdown("**❌ Unprotected FedAvg (both A & B sent)**")
        st.error("Patient record RECONSTRUCTED" if ok else "reconstruction failed")
        st.code(a["unprotected"]["reconstruction"], language=None)
    with right:
        ok = a["protected"]["success"]
        st.markdown("**✅ FedNeMo protected (FedRand: one matrix)**")
        st.success("Reconstruction FAILED — noise" if not ok else "reconstructed!")
        st.code(a["protected"]["reconstruction"], language=None)

    with st.expander("Original target record (ground truth)"):
        st.code(a["original"], language=None)

st.caption(
    "Dashboard reads telemetry.jsonl only. Refresh after running more rounds. "
    "FedRand, Laplacian DP, and adaptive quantization are live (Phase 1-3). "
    "Real model (Nemotron-Mini-4B) arrives in Phase 4."
)
