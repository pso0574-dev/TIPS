# streamlit_app.py
from __future__ import annotations

import os
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Macro Risk Radar", page_icon="📉", layout="wide")
st.title("📉 Macro Risk Radar Dashboard")
st.caption("FRED API key required")

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("Settings")
lookback_days = st.sidebar.slider("Lookback period (days)", 30, 365, 180, 10)
short_window = st.sidebar.slider("Short trend window", 3, 30, 10, 1)
long_window = st.sidebar.slider("Long trend window", 10, 90, 30, 5)
show_ma = st.sidebar.checkbox("Show moving averages", True)

# Put your FRED API key in:
# 1) environment variable FRED_API_KEY
# or 2) .streamlit/secrets.toml:
#    FRED_API_KEY="your_key_here"
FRED_API_KEY = os.getenv("FRED_API_KEY", st.secrets.get("FRED_API_KEY", ""))

if not FRED_API_KEY:
    st.error(
        "FRED_API_KEY is missing. Add it to environment variables or .streamlit/secrets.toml"
    )
    st.stop()

# ------------------------------------------------------------
# Series map
# ------------------------------------------------------------
SERIES = {
    "DGS10": {
        "name": "US 10Y Treasury Yield",
        "unit": "%",
        "risk_direction": "up",
        "category": "Rates",
    },
    "BAMLC0A1CAAA": {
        "name": "ICE BofA AAA Corporate OAS",
        "unit": "%",
        "risk_direction": "up",
        "category": "Credit",
    },
    "SP500": {
        "name": "S&P 500",
        "unit": "index",
        "risk_direction": "down",
        "category": "Equity",
    },
    "VIXCLS": {
        "name": "VIX",
        "unit": "index",
        "risk_direction": "up",
        "category": "Volatility",
    },
}

# ------------------------------------------------------------
# FRED fetch
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_fred_series(series_id: str) -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()

    observations = payload.get("observations", [])
    if not observations:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(observations)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


def filter_recent(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff].copy()


def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < max(short_window, long_window) + 1:
        return {
            "latest": np.nan,
            "change_short": np.nan,
            "change_long": np.nan,
            "trend_short": "N/A",
            "trend_long": "N/A",
        }

    latest = df["value"].iloc[-1]
    prev_short = df["value"].iloc[-(short_window + 1)]
    prev_long = df["value"].iloc[-(long_window + 1)]

    change_short = latest - prev_short
    change_long = latest - prev_long

    trend_short = "Up" if change_short > 0 else "Down" if change_short < 0 else "Flat"
    trend_long = "Up" if change_long > 0 else "Down" if change_long < 0 else "Flat"

    return {
        "latest": latest,
        "change_short": change_short,
        "change_long": change_long,
        "trend_short": trend_short,
        "trend_long": trend_long,
    }


def score_risk(risk_direction: str, change_value: float) -> int:
    if pd.isna(change_value):
        return 0

    risk_move = (risk_direction == "up" and change_value > 0) or (
        risk_direction == "down" and change_value < 0
    )
    if not risk_move:
        return 0

    abs_change = abs(change_value)
    if abs_change < 0.2:
        return 1
    elif abs_change < 0.7:
        return 2
    else:
        return 3


def overall_risk_label(score: int) -> str:
    if score <= 2:
        return "Low"
    elif score <= 5:
        return "Moderate"
    elif score <= 8:
        return "High"
    return "Severe"


def make_chart(df: pd.DataFrame, title: str, unit: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["value"],
            mode="lines+markers",
            name=title,
        )
    )

    if show_ma and len(df) >= 20:
        temp = df.copy()
        temp["MA10"] = temp["value"].rolling(10).mean()
        temp["MA20"] = temp["value"].rolling(20).mean()

        fig.add_trace(
            go.Scatter(
                x=temp["date"],
                y=temp["MA10"],
                mode="lines",
                name="MA10",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=temp["date"],
                y=temp["MA20"],
                mode="lines",
                name="MA20",
            )
        )

    fig.update_layout(
        title=title,
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Date",
        yaxis_title=unit,
    )
    return fig


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
raw = {}
for sid in SERIES:
    try:
        raw[sid] = fetch_fred_series(sid)
    except Exception as e:
        st.warning(f"Failed to load {sid}: {e}")
        raw[sid] = pd.DataFrame(columns=["date", "value"])

recent = {sid: filter_recent(df, lookback_days) for sid, df in raw.items()}
metrics = {sid: compute_metrics(df) for sid, df in recent.items()}
scores = {
    sid: score_risk(SERIES[sid]["risk_direction"], metrics[sid]["change_short"])
    for sid in SERIES
}

total_score = sum(scores.values())

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Risk Score", f"{total_score} / 12")
with c2:
    st.metric("Risk Label", overall_risk_label(total_score))
with c3:
    st.metric("Signals Loaded", sum(1 for df in recent.values() if not df.empty))

st.markdown("---")

# ------------------------------------------------------------
# Cards
# ------------------------------------------------------------
cols = st.columns(4)
for i, (sid, meta) in enumerate(SERIES.items()):
    m = metrics[sid]
    with cols[i]:
        latest_txt = f"{m['latest']:.2f}" if pd.notna(m["latest"]) else "N/A"
        delta_txt = f"{m['change_short']:+.2f}" if pd.notna(m["change_short"]) else "N/A"
        st.markdown(f"### {meta['name']}")
        st.metric("Latest", latest_txt, delta_txt)
        st.write(f"Category: {meta['category']}")
        st.write(f"Score: {scores[sid]}/3")

st.markdown("---")

# ------------------------------------------------------------
# Table
# ------------------------------------------------------------
rows = []
for sid, meta in SERIES.items():
    m = metrics[sid]
    rows.append(
        {
            "Series": meta["name"],
            "Latest": round(m["latest"], 3) if pd.notna(m["latest"]) else np.nan,
            f"{short_window}D Change": round(m["change_short"], 3) if pd.notna(m["change_short"]) else np.nan,
            f"{long_window}D Change": round(m["change_long"], 3) if pd.notna(m["change_long"]) else np.nan,
            "Short Trend": m["trend_short"],
            "Long Trend": m["trend_long"],
            "Risk Score": scores[sid],
        }
    )

st.subheader("Signal Summary")
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ------------------------------------------------------------
# Charts
# ------------------------------------------------------------
st.subheader("Charts")
for sid, meta in SERIES.items():
    df = recent[sid]
    if df.empty:
        st.warning(f"No data for {meta['name']}")
        continue
    st.plotly_chart(make_chart(df, meta["name"], meta["unit"]), use_container_width=True)
