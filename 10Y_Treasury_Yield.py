# streamlit_app.py
# ============================================================
# Macro Risk Radar Dashboard
# - US 10Y Treasury Yield
# - US AAA Corporate Bond Yield
# - S&P 500
# - US Presidential Approval Rating
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit pandas numpy plotly requests
#
# Notes:
# - Uses FRED public API first (no API key required for many cases)
# - Falls back gracefully if some series fail
# ============================================================

from __future__ import annotations

import math
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Macro Risk Radar Dashboard",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Macro Risk Radar Dashboard")
st.caption("Track bond yields, credit stress, equity weakness, and political sentiment in one view.")

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Settings")

lookback_days = st.sidebar.slider("Lookback period (days)", 30, 365, 180, 10)
short_window = st.sidebar.slider("Short trend window", 3, 30, 10, 1)
long_window = st.sidebar.slider("Long trend window", 10, 90, 30, 5)
show_ma = st.sidebar.checkbox("Show moving averages", True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Risk Logic")
st.sidebar.write(
    """
- 10Y Yield rising -> tighter financial conditions
- AAA Yield rising -> higher credit stress
- S&P 500 falling -> risk-off
- Approval Rating falling -> policy uncertainty
"""
)

# ============================================================
# FRED Series map
# ============================================================
SERIES = {
    "DGS10": {
        "name": "US 10Y Treasury Yield",
        "unit": "%",
        "source": "FRED",
        "risk_direction": "up",
        "category": "Rates",
    },
    "AAA": {
        "name": "Moody's Seasoned Aaa Corporate Bond Yield",
        "unit": "%",
        "source": "FRED",
        "risk_direction": "up",
        "category": "Credit",
    },
    "SP500": {
        "name": "S&P 500 Index",
        "unit": "index",
        "source": "FRED",
        "risk_direction": "down",
        "category": "Equity",
    },
    "PRESAPPROVAL": {
        "name": "US Presidential Approval Rating",
        "unit": "%",
        "source": "FRED",
        "risk_direction": "down",
        "category": "Politics",
    },
}

# ============================================================
# Data loading
# ============================================================
@st.cache_data(ttl=60 * 60)
def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """
    Fetch FRED series as DataFrame with columns: date, value
    Works without API key in many public-access cases.
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "file_type": "json",
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    observations = data.get("observations", [])
    if not observations:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(observations)
    df = df[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


def safe_load_series(series_id: str) -> pd.DataFrame:
    try:
        return fetch_fred_series(series_id)
    except Exception as e:
        st.warning(f"Failed to load {series_id}: {e}")
        return pd.DataFrame(columns=["date", "value"])


def filter_recent(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff].copy()


def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < max(short_window, long_window) + 1:
        return {
            "latest": np.nan,
            "prev_short": np.nan,
            "prev_long": np.nan,
            "change_short": np.nan,
            "change_long": np.nan,
            "pct_change_short": np.nan,
            "pct_change_long": np.nan,
            "trend_short": "N/A",
            "trend_long": "N/A",
        }

    latest = df["value"].iloc[-1]
    prev_short = df["value"].iloc[-(short_window + 1)]
    prev_long = df["value"].iloc[-(long_window + 1)]

    change_short = latest - prev_short
    change_long = latest - prev_long

    pct_change_short = (change_short / prev_short * 100) if prev_short != 0 else np.nan
    pct_change_long = (change_long / prev_long * 100) if prev_long != 0 else np.nan

    trend_short = "Up" if change_short > 0 else "Down" if change_short < 0 else "Flat"
    trend_long = "Up" if change_long > 0 else "Down" if change_long < 0 else "Flat"

    return {
        "latest": latest,
        "prev_short": prev_short,
        "prev_long": prev_long,
        "change_short": change_short,
        "change_long": change_long,
        "pct_change_short": pct_change_short,
        "pct_change_long": pct_change_long,
        "trend_short": trend_short,
        "trend_long": trend_long,
    }


def score_risk(risk_direction: str, change_value: float) -> int:
    """
    Convert direction + change into simple risk score
    0 = benign
    1 = mild
    2 = moderate
    3 = high
    """
    if pd.isna(change_value):
        return 0

    abs_change = abs(change_value)

    # whether the move is risk-increasing
    risk_move = (risk_direction == "up" and change_value > 0) or (
        risk_direction == "down" and change_value < 0
    )

    if not risk_move:
        return 0

    if abs_change < 0.2:
        return 1
    elif abs_change < 0.7:
        return 2
    else:
        return 3


def risk_label(score: int) -> str:
    if score <= 2:
        return "Low"
    elif score <= 5:
        return "Moderate"
    elif score <= 8:
        return "High"
    else:
        return "Severe"


def direction_text(risk_direction: str) -> str:
    return "rising is risky" if risk_direction == "up" else "falling is risky"


def interpret_signal(series_name: str, risk_direction: str, change_short: float) -> str:
    if pd.isna(change_short):
        return "Not enough data."

    risk_move = (risk_direction == "up" and change_short > 0) or (
        risk_direction == "down" and change_short < 0
    )

    if not risk_move:
        return f"{series_name} is not currently moving in the main risk direction."
    else:
        return f"{series_name} is moving in the main risk direction, which raises macro stress."


def make_line_chart(df: pd.DataFrame, title: str, unit: str) -> go.Figure:
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
        df_plot = df.copy()
        df_plot["MA10"] = df_plot["value"].rolling(10).mean()
        df_plot["MA20"] = df_plot["value"].rolling(20).mean()

        fig.add_trace(
            go.Scatter(
                x=df_plot["date"],
                y=df_plot["MA10"],
                mode="lines",
                name="MA10",
                line=dict(dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_plot["date"],
                y=df_plot["MA20"],
                mode="lines",
                name="MA20",
                line=dict(dash="dot"),
            )
        )

    fig.update_layout(
        title=title,
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Date",
        yaxis_title=unit,
        legend_title="Series",
    )
    return fig


# ============================================================
# Load data
# ============================================================
with st.spinner("Loading macro data..."):
    raw_data = {sid: safe_load_series(sid) for sid in SERIES.keys()}

recent_data = {sid: filter_recent(df, lookback_days) for sid, df in raw_data.items()}
metrics = {sid: compute_metrics(df) for sid, df in recent_data.items()}

# ============================================================
# Risk computation
# ============================================================
component_scores = {}
for sid, meta in SERIES.items():
    component_scores[sid] = score_risk(meta["risk_direction"], metrics[sid]["change_short"])

total_score = sum(component_scores.values())
overall_label = risk_label(total_score)

# ============================================================
# Top summary
# ============================================================
c1, c2, c3 = st.columns([1.2, 1, 1])

with c1:
    st.subheader("Overall Macro Risk")
    st.metric("Risk Score", f"{total_score} / 12", overall_label)

with c2:
    severe_count = sum(1 for v in component_scores.values() if v == 3)
    st.metric("Severe Signals", severe_count)

with c3:
    risk_pct = total_score / 12 * 100
    st.metric("Risk Intensity", f"{risk_pct:.1f}%")

st.markdown("---")

# ============================================================
# Four key cards
# ============================================================
cols = st.columns(4)

for i, (sid, meta) in enumerate(SERIES.items()):
    m = metrics[sid]
    score = component_scores[sid]
    latest = m["latest"]
    change_short = m["change_short"]

    with cols[i]:
        st.markdown(f"### {meta['name']}")
        latest_text = f"{latest:.2f}" if pd.notna(latest) else "N/A"
        delta_text = f"{change_short:+.2f}" if pd.notna(change_short) else "N/A"
        st.metric(
            label=f"Latest ({meta['unit']})",
            value=latest_text,
            delta=delta_text,
        )
        st.caption(f"Risk logic: {direction_text(meta['risk_direction'])}")
        st.write(f"**Signal Score:** {score}/3")
        st.write(interpret_signal(meta["name"], meta["risk_direction"], change_short))

st.markdown("---")

# ============================================================
# Detailed table
# ============================================================
rows = []
for sid, meta in SERIES.items():
    m = metrics[sid]
    rows.append(
        {
            "Series": meta["name"],
            "Category": meta["category"],
            "Latest": round(m["latest"], 3) if pd.notna(m["latest"]) else np.nan,
            f"{short_window}D Change": round(m["change_short"], 3) if pd.notna(m["change_short"]) else np.nan,
            f"{long_window}D Change": round(m["change_long"], 3) if pd.notna(m["change_long"]) else np.nan,
            "Short Trend": m["trend_short"],
            "Long Trend": m["trend_long"],
            "Risk Direction": meta["risk_direction"],
            "Component Score": component_scores[sid],
        }
    )

summary_df = pd.DataFrame(rows)

st.subheader("Signal Summary Table")
st.dataframe(summary_df, use_container_width=True)

# ============================================================
# Charts
# ============================================================
st.subheader("Charts")

for sid, meta in SERIES.items():
    df = recent_data[sid]
    if df.empty:
        st.warning(f"No recent data for {meta['name']}")
        continue

    fig = make_line_chart(df, meta["name"], meta["unit"])
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Risk explanation
# ============================================================
st.subheader("Macro Interpretation")

risk_messages = []

if component_scores["DGS10"] >= 2:
    risk_messages.append("- **10Y Treasury Yield is rising**: tighter financial conditions and valuation pressure on equities.")

if component_scores["AAA"] >= 2:
    risk_messages.append("- **AAA Corporate Yield is rising**: credit conditions are worsening and financing costs are increasing.")

if component_scores["SP500"] >= 2:
    risk_messages.append("- **S&P 500 is falling**: market is shifting toward risk-off behavior.")

if component_scores["PRESAPPROVAL"] >= 2:
    risk_messages.append("- **Presidential Approval Rating is falling**: policy confidence may be weakening, adding political uncertainty.")

if not risk_messages:
    st.success("Current data does not show a strong synchronized macro stress signal.")
else:
    for msg in risk_messages:
        st.markdown(msg)

# ============================================================
# Portfolio implication
# ============================================================
st.subheader("Portfolio Implication")

if total_score <= 2:
    st.info(
        """
Macro backdrop looks relatively stable.
Possible preference:
- broad equity exposure
- balanced duration
- moderate cyclicals
"""
    )
elif total_score <= 5:
    st.warning(
        """
Some macro stress is building.
Possible preference:
- quality equities
- selective defensives
- partial inflation hedge
- avoid excessive leverage
"""
    )
elif total_score <= 8:
    st.warning(
        """
Macro risk is elevated.
Possible preference:
- reduce high-beta exposure
- hold more cash / short-duration bonds
- consider TIPS / gold / defensive sectors
"""
    )
else:
    st.error(
        """
Macro stress is severe.
Possible preference:
- defensive posture
- focus on liquidity
- reduce speculative growth exposure
- monitor credit and rates daily
"""
    )

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "Data source: FRED. Dashboard is for educational and monitoring purposes only, not investment advice."
)
