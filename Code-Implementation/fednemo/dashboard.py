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


st.set_page_config(page_title="FedNeMo — Phase 0", layout="wide")
st.title("FedNeMo — Federated Round Telemetry (Phase 0)")

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
    st.subheader("Cumulative privacy budget (stub)")
    st.line_chart(df.set_index("round")["eps_total"])
    last_eps = df["eps_total"].iloc[-1]
    st.metric("eps_total", f"{last_eps:.1f}")

st.subheader("Which matrix each hospital sent (FedRand — 'both' until Phase 1)")
sent_df = pd.DataFrame([r["sent"] for r in rows], index=[r["round"] for r in rows])
sent_df.index.name = "round"
st.dataframe(sent_df, use_container_width=True)

st.caption(
    "Dashboard reads telemetry.jsonl only. Refresh the page after running more "
    "rounds. Real privacy/quantization panels arrive in Phases 2-3."
)
