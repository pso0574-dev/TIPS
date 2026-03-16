# streamlit_app.py
# ============================================================
# Macro Risk Radar Dashboard (FRED API KEY version)
# - US 10Y Treasury Yield
# - Moody's Aaa Corporate Bond Yield
# - S&P 500
# - US Presidential Approval Rating
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit pandas numpy plotly requests
#
# FRED API Key:
#   Option 1) .streamlit/secrets.toml
#       FRED_API_KEY="YOUR_API_KEY"
#   Option 2) Environment variable
#       FRED_API_KEY=YOUR_API_KEY
# ============================================================

from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
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
st.caption("FRED API KEY based monitoring for rates, credit, equities, and politics.")

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Settings")

lookback_days = st.sidebar.slider("Lookback period (days)", 30, 365, 180, 10)
short_window = st.sidebar.slider("Short trend window", 3, 30, 10, 1)
long_window = st.sidebar.slider("Long trend window", 10, 90, 30, 5)
show_ma = st.sidebar.checkbox("Show moving averages", True)
auto_refresh = st.sidebar.checkbox("Force refresh data", False)

st.sidebar.markdown("---")
st.sidebar.markdown("### Risk Logic")
st.sidebar.write(
    """
- 10Y Yield rising → tighter financial conditions
- AAA Yield rising → higher credit stress
- S&P 500 falling → risk-off
- Approval Rating falling → political uncertainty
"""
)

# ============================================================
# FRED API KEY
# ============================================================
def get_fred_api_key() -> str:
    # 1) Streamlit secrets
    try:
        if "FRED_API_KEY" in st.secrets:
            return st.secrets["FRED_API_KEY"]
    except Exception:
        pass

    # 2) Environment variable
    return os.getenv("FRED_API_KEY", "")


FRED_API_KEY = get_fred_api_key()

if not FRED_API_KEY:
    st.error(
        "FRED_API_KEY not found. Please set it in `.streamlit/secrets.toml` "
        "or as an environment variable."
    )
    st.stop()

# ============================================================
# Series configuration
# ============================================================
SERIES: Dict[str, Dict[str, Any]] = {
    "DGS10": {
        "name": "US 10Y Treasury Yield",
        "unit": "%",
        "risk_direction": "up",
        "category": "Rates",
    },
    "AAA": {
        "name": "Moody's Seasoned Aaa Corporate Bond Yield",
        "unit": "%",
        "risk_direction": "up",
        "category": "Credit",
    },
    "SP500": {
        "name": "S&P 500 Index",
        "unit": "index",
        "risk_direction": "down",
        "category": "Equity",
    },
    # Note:
    # This series may not always be available in all setups.
    # If unavailable, the app will show a warning and continue.
    "PRESAPPROVAL": {
        "name": "US Presidential Approval Rating",
        "unit": "%",
        "risk_direction": "down",
        "category": "Politics",
    },
}

# ============================================================
# Helpers
# ============================================================
def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "streamlit-macro-risk-dashboard/1.0"})
    return session


SESSION = get_session()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_fred_series(series_id: str, api_key: str) -> pd.DataFrame:
    """
    Fetch a FRED series with API key.
    Returns DataFrame with columns: date, value
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }

    response = SESSION.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    observations = data.get("observations", [])
    if not observations:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(observations)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return df


def safe_load_series(series_id: str, api_key: str) -> pd.DataFrame:
    try:
        if auto_refresh:
            fetch_fred_series.clear()
        return fetch_fred_series(series_id, api_key)
    except Exception as e:
        st.warning(f"Failed to load {series_id}: {e}")
        return pd.DataFrame(columns=["date", "value"])


def filter_recent(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff].copy()


def compute_metrics(df: pd.DataFrame, short_w: int, long_w: int) -> dict:
    min_required = max(short_w, long_w) + 1

    if df.empty or len(df) < min_required:
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
    prev_short = df["value"].iloc[-(short_w + 1)]
    prev_long = df["value"].iloc[-(long_w + 1)]

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


def score_risk(risk_direction: str, change_value: float, series_id: str) -> int:
    """
    Simple component score:
    0 = benign
    1 = mild
    2 = moderate
    3 = high
    """
    if pd.isna(change_value):
        return 0

    risk_move = (
        (risk_direction == "up" and change_value > 0)
        or (risk_direction == "down" and change_value < 0)
    )

    if not risk_move:
        return 0

    abs_change = abs(change_value)

    # Series-specific thresholds
    if series_id == "SP500":
        if abs_change < 30:
            return 1
        elif abs_change < 100:
            return 2
        else:
            return 3
    elif series_id in ["DGS10", "AAA", "PRESAPPROVAL"]:
        if abs_change < 0.2:
            return 1
        elif abs_change < 0.7:
            return 2
        else:
            return 3

    return 1


def overall_risk_label(score: int) -> str:
    if score <= 2:
        return "Low"
    elif score <= 5:
        return "Moderate"
    elif score <= 8:
        return "High"
    return "Severe"


def direction_text(risk_direction: str) -> str:
    return "Rising is risky" if risk_direction == "up" else "Falling is risky"


def interpret_signal(name: str, risk_direction: str, change_short: float) -> str:
    if pd.isna(change_short):
        return "Not enough recent data."

    risk_move = (
        (risk_direction == "up" and change_short > 0)
        or (risk_direction == "down" and change_short < 0)
    )

    if risk_move:
        return f"{name} is moving in the macro risk direction."
    return f"{name} is not currently moving in the main risk direction."


def make_line_chart(df: pd.DataFrame, title: str, unit: str) -> go.Figure:
    plot_df = df.copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["value"],
            mode="lines+markers",
            name=title,
        )
    )

    if show_ma and len(plot_df) >= 20:
        plot_df["MA10"] = plot_df["value"].rolling(10).mean()
        plot_df["MA20"] = plot_df["value"].rolling(20).mean()

        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["MA10"],
                mode="lines",
                name="MA10",
                line=dict(dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["MA20"],
                mode="lines",
                name="MA20",
                line=dict(dash="dot"),
            )
        )

    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Date",
        yaxis_title=unit,
        legend_title="Series",
    )
    return fig


# ============================================================
# Load data
# ============================================================
with st.spinner("Loading FRED data..."):
    raw_data = {sid: safe_load_series(sid, FRED_API_KEY) for sid in SERIES}

recent_data = {sid: filter_recent(df, lookback_days) for sid, df in raw_data.items()}
metrics = {
    sid: compute_metrics(df, short_window, long_window)
    for sid, df in recent_data.items()
}

# ============================================================
# Risk score
# ============================================================
component_scores = {
    sid: score_risk(
        SERIES[sid]["risk_direction"],
        metrics[sid]["change_short"],
        sid,
    )
    for sid in SERIES
}

total_score = sum(component_scores.values())
overall_label = overall_risk_label(total_score)
risk_intensity = total_score / (len(SERIES) * 3) * 100 if SERIES else 0.0

# ============================================================
# Top summary
# ============================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Overall Macro Risk")
    st.metric("Risk Score", f"{total_score} / {len(SERIES) * 3}", overall_label)

with c2:
    severe_signals = sum(1 for v in component_scores.values() if v == 3)
    st.metric("Severe Signals", severe_signals)

with c3:
    st.metric("Risk Intensity", f"{risk_intensity:.1f}%")

st.markdown("---")

# ============================================================
# Signal cards
# ============================================================
cols = st.columns(len(SERIES))

for i, (sid, meta) in enumerate(SERIES.items()):
    m = metrics[sid]
    latest = m["latest"]
    change_short = m["change_short"]
    score = component_scores[sid]

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
# Summary table
# ============================================================
summary_rows = []
for sid, meta in SERIES.items():
    m = metrics[sid]
    summary_rows.append(
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

summary_df = pd.DataFrame(summary_rows)

st.subheader("Signal Summary Table")
st.dataframe(summary_df, use_container_width=True)

# ============================================================
# Charts
# ============================================================
st.subheader("Charts")

for sid, meta in SERIES.items():
    df = recent_data[sid]
    if df.empty:
        st.warning(f"No recent data available for {meta['name']}")
        continue

    fig = make_line_chart(df, meta["name"], meta["unit"])
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{sid}")

# ============================================================
# Interpretation
# ============================================================
st.subheader("Macro Interpretation")

messages = []

if component_scores.get("DGS10", 0) >= 2:
    messages.append("- **US 10Y Treasury Yield is rising**: tighter financial conditions and valuation pressure.")

if component_scores.get("AAA", 0) >= 2:
    messages.append("- **Aaa Corporate Yield is rising**: credit conditions are becoming less friendly.")

if component_scores.get("SP500", 0) >= 2:
    messages.append("- **S&P 500 is falling**: market tone is becoming risk-off.")

if component_scores.get("PRESAPPROVAL", 0) >= 2:
    messages.append("- **Presidential Approval is falling**: political uncertainty may increase.")

if not messages:
    st.success("Current data does not show a strong synchronized macro stress signal.")
else:
    for msg in messages:
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
- moderate cyclical exposure
"""
    )
elif total_score <= 5:
    st.warning(
        """
Some macro stress is building.

Possible preference:
- quality equities
- selective defensives
- partial inflation hedges
- avoid excessive leverage
"""
    )
elif total_score <= 8:
    st.warning(
        """
Macro risk is elevated.

Possible preference:
- reduce high-beta exposure
- raise cash or short-duration bonds
- consider TIPS, gold, or defensive sectors
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
- monitor rates and credit more closely
"""
    )

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption("Source: FRED API. For monitoring and educational purposes only.")
