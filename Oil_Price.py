# oil_macro_dashboard.py
# ------------------------------------------------------------
# Streamlit dashboard for oil direction monitoring
#
# Data design:
# - FRED: WTI / Brent / FEDFUNDS / DGS10 / dollar proxy
# - EIA : crude inventory / cushing inventory / US crude production
# - OPEC: manual event table (cut / hike / neutral)
#
# Install:
#   pip install streamlit pandas numpy requests plotly python-dateutil
#
# Run:
#   streamlit run oil_macro_dashboard.py
#
# Environment variables:
#   FRED_API_KEY=your_key
#   EIA_API_KEY=your_key
# ------------------------------------------------------------

from __future__ import annotations

import os
import math
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="Oil Macro Signal Dashboard",
    page_icon="🛢️",
    layout="wide",
)

st.title("🛢️ Oil Macro Signal Dashboard")
st.caption("WTI / Brent / Inventory / Production / Rates / Dollar / OPEC event-based signal")

# =========================
# API keys
# =========================
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
EIA_API_KEY = os.getenv("EIA_API_KEY", "")

# =========================
# User controls
# =========================
with st.sidebar:
    st.header("Settings")

    preset = st.selectbox(
        "Chart range",
        ["6M", "1Y", "3Y", "5Y", "10Y", "Max"],
        index=1,
    )

    show_brent = st.checkbox("Show Brent", value=True)
    show_wti = st.checkbox("Show WTI", value=True)

    st.markdown("---")
    st.subheader("Signal weights")
    w_inventory = st.slider("Inventory", 0.0, 3.0, 1.0, 0.1)
    w_opec = st.slider("OPEC policy", 0.0, 3.0, 1.0, 0.1)
    w_rates = st.slider("Rates", 0.0, 3.0, 1.0, 0.1)
    w_dollar = st.slider("Dollar", 0.0, 3.0, 1.0, 0.1)
    w_production = st.slider("US production", 0.0, 3.0, 1.0, 0.1)
    w_china_proxy = st.slider("Growth proxy", 0.0, 3.0, 0.5, 0.1)

# =========================
# Helpers
# =========================
def get_start_date_from_preset(preset_name: str) -> Optional[str]:
    today = datetime.today()
    mapping = {
        "6M": today - timedelta(days=183),
        "1Y": today - timedelta(days=365),
        "3Y": today - timedelta(days=365 * 3),
        "5Y": today - timedelta(days=365 * 5),
        "10Y": today - timedelta(days=365 * 10),
        "Max": None,
    }
    dt = mapping[preset_name]
    return None if dt is None else dt.strftime("%Y-%m-%d")


def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def infer_signal_label(score: float) -> str:
    if score >= 3:
        return "Strong Bull"
    if score >= 1:
        return "Bull"
    if score > -1:
        return "Neutral"
    if score > -3:
        return "Bear"
    return "Strong Bear"


def metric_delta_text(series: pd.Series, periods: int = 4) -> Tuple[Optional[float], Optional[float]]:
    s = series.dropna()
    if len(s) < periods + 1:
        return None, None
    latest = float(s.iloc[-1])
    prev = float(s.iloc[-1 - periods])
    delta = latest - prev
    return latest, delta


def normalize_score(x: float, lower: float = -5.0, upper: float = 5.0) -> float:
    x = max(lower, min(upper, x))
    return (x - lower) / (upper - lower) * 100.0


# =========================
# Data fetchers
# =========================
@st.cache_data(ttl=3600)
def fetch_fred_series(
    series_id: str,
    api_key: str,
    observation_start: Optional[str] = None,
) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame(columns=["date", "value"])

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if observation_start:
        params["observation_start"] = observation_start

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    obs = js.get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]].dropna()


@st.cache_data(ttl=3600)
def fetch_eia_v2(
    route: str,
    facets: Optional[Dict[str, List[str]]] = None,
    frequency: str = "weekly",
    data_field: str = "value",
    sort_desc: bool = False,
    length: int = 5000,
) -> pd.DataFrame:
    """
    Generic EIA v2 fetcher.

    Example:
      route = "petroleum/stoc/wstk/data/"
      route = "petroleum/crd/crpdn/data/"
    """
    if not EIA_API_KEY:
        return pd.DataFrame()

    base = f"https://api.eia.gov/v2/{route}"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": frequency,
        "data[0]": data_field,
        "length": length,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc" if sort_desc else "asc",
    }

    if facets:
        idx = 0
        for key, values in facets.items():
            for val in values:
                params[f"facets[{key}][{idx}]"] = val
                idx += 1

    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    records = js.get("response", {}).get("data", [])
    df = pd.DataFrame(records)
    if df.empty:
        return df

    if "period" in df.columns:
        df["date"] = pd.to_datetime(df["period"], errors="coerce")

    if data_field in df.columns:
        df["value"] = pd.to_numeric(df[data_field], errors="coerce")
    elif "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


# =========================
# Specific loaders
# =========================
def load_fred_block(start_date: Optional[str]) -> Dict[str, pd.DataFrame]:
    series_map = {
        "WTI": "DCOILWTICO",
        "Brent": "DCOILBRENTEU",
        "FedFunds": "FEDFUNDS",
        "US10Y": "DGS10",
        # Broad dollar proxy from FRED can vary by preference.
        # DTWEXBGS = Trade Weighted U.S. Dollar Index: Broad, Goods and Services
        "Dollar": "DTWEXBGS",
        # Growth proxy
        "SP500": "SP500",
    }
    out = {}
    for name, sid in series_map.items():
        try:
            out[name] = fetch_fred_series(sid, FRED_API_KEY, start_date)
        except Exception as e:
            st.warning(f"Failed to load FRED series {sid}: {e}")
            out[name] = pd.DataFrame(columns=["date", "value"])
    return out


def load_opec_events() -> pd.DataFrame:
    # Manual event table.
    # Expand this over time.
    events = [
        {"date": "2025-06-02", "policy": "cut", "score": 1, "note": "OPEC+ supportive / cut stance"},
        {"date": "2025-12-05", "policy": "cut", "score": 1, "note": "OPEC+ supply discipline"},
        {"date": "2026-03-01", "policy": "cut", "score": 1, "note": "OPEC+ still supporting price"},
    ]
    df = pd.DataFrame(events)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_eia_block() -> Dict[str, pd.DataFrame]:
    """
    IMPORTANT:
    EIA routes/facets can differ by route and may need adjustment
    after checking the API browser for the exact table you want.

    The defaults below are meant as a practical starting template.
    """
    data = {}

    # 1) US crude oil production (weekly)
    try:
        # Route and facets may need refinement depending on the exact series you want.
        # This is a template pattern.
        df_prod = fetch_eia_v2(
            route="petroleum/crd/crpdn/data/",
            facets=None,
            frequency="weekly",
            data_field="value",
        )
        data["US_Production"] = df_prod[["date", "value"]].dropna() if not df_prod.empty else pd.DataFrame(columns=["date", "value"])
    except Exception as e:
        st.warning(f"Failed to load EIA US production: {e}")
        data["US_Production"] = pd.DataFrame(columns=["date", "value"])

    # 2) Crude inventory
    try:
        df_inv = fetch_eia_v2(
            route="petroleum/stoc/wstk/data/",
            facets=None,
            frequency="weekly",
            data_field="value",
        )
        data["Crude_Inventory"] = df_inv[["date", "value"]].dropna() if not df_inv.empty else pd.DataFrame(columns=["date", "value"])
    except Exception as e:
        st.warning(f"Failed to load EIA crude inventory: {e}")
        data["Crude_Inventory"] = pd.DataFrame(columns=["date", "value"])

    # 3) Cushing inventory
    try:
        df_cushing = fetch_eia_v2(
            route="petroleum/stoc/wstk/data/",
            facets={"series": ["WCESTUS1"]},  # likely needs adjustment in API browser
            frequency="weekly",
            data_field="value",
        )
        data["Cushing_Inventory"] = df_cushing[["date", "value"]].dropna() if not df_cushing.empty else pd.DataFrame(columns=["date", "value"])
    except Exception as e:
        st.warning(f"Failed to load EIA cushing inventory: {e}")
        data["Cushing_Inventory"] = pd.DataFrame(columns=["date", "value"])

    return data


# =========================
# Signal model
# =========================
def latest_change_signal(series: pd.Series, lookback: int = 4, inverse: bool = False) -> float:
    """
    Returns:
      +1 if trend supports higher oil
      -1 if trend supports lower oil
       0 if insufficient data / flat
    inverse=True means higher series is bearish for oil (e.g. dollar, rates, inventory, production)
    """
    s = series.dropna()
    if len(s) < lookback + 1:
        return 0.0

    recent = s.iloc[-1]
    prev = s.iloc[-1 - lookback]
    diff = recent - prev

    if math.isclose(diff, 0, abs_tol=1e-12):
        return 0.0

    raw = 1.0 if diff > 0 else -1.0
    return -raw if inverse else raw


def current_opec_signal(opec_df: pd.DataFrame) -> float:
    if opec_df.empty:
        return 0.0
    return float(opec_df.iloc[-1]["score"])


def build_signal_snapshot(
    fred_data: Dict[str, pd.DataFrame],
    eia_data: Dict[str, pd.DataFrame],
    opec_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    # Inventory: rising inventory is bearish for oil
    inv_sig = latest_change_signal(eia_data["Crude_Inventory"]["value"], lookback=4, inverse=True)
    rows.append({"factor": "Crude Inventory", "signal_raw": inv_sig, "weight": w_inventory, "weighted": inv_sig * w_inventory})

    # OPEC: cut supportive
    opec_sig = current_opec_signal(opec_df)
    rows.append({"factor": "OPEC Policy", "signal_raw": opec_sig, "weight": w_opec, "weighted": opec_sig * w_opec})

    # Rates: higher rates bearish
    fed_sig = latest_change_signal(fred_data["FedFunds"]["value"], lookback=3, inverse=True)
    rows.append({"factor": "Fed Funds", "signal_raw": fed_sig, "weight": w_rates, "weighted": fed_sig * w_rates})

    # Dollar: stronger dollar bearish
    dxy_sig = latest_change_signal(fred_data["Dollar"]["value"], lookback=4, inverse=True)
    rows.append({"factor": "Dollar Proxy", "signal_raw": dxy_sig, "weight": w_dollar, "weighted": dxy_sig * w_dollar})

    # Production: rising production bearish
    prod_sig = latest_change_signal(eia_data["US_Production"]["value"], lookback=4, inverse=True)
    rows.append({"factor": "US Production", "signal_raw": prod_sig, "weight": w_production, "weighted": prod_sig * w_production})

    # Growth proxy: stronger SP500 is only a rough risk/growth proxy here
    # Later, replace with OECD CLI or China-specific demand proxy
    growth_sig = latest_change_signal(fred_data["SP500"]["value"], lookback=4, inverse=False)
    rows.append({"factor": "Growth Proxy", "signal_raw": growth_sig, "weight": w_china_proxy, "weighted": growth_sig * w_china_proxy})

    snap = pd.DataFrame(rows)
    return snap


# =========================
# Charts
# =========================
def make_line_chart(
    series_dict: Dict[str, pd.DataFrame],
    title: str,
    y_title: str,
) -> go.Figure:
    fig = go.Figure()

    for name, df in series_dict.items():
        if df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["value"],
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420,
        xaxis_title="Date",
        yaxis_title=y_title,
        legend_title="Series",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# =========================
# Load data
# =========================
start_date = get_start_date_from_preset(preset)

fred_data = load_fred_block(start_date)
eia_data = load_eia_block()
opec_df = load_opec_events()

# Filter date range for EIA/OPEC if preset chosen
if start_date:
    start_ts = pd.to_datetime(start_date)
    for key in eia_data:
        if not eia_data[key].empty:
            eia_data[key] = eia_data[key][eia_data[key]["date"] >= start_ts].copy()
    opec_df = opec_df[opec_df["date"] >= start_ts].copy()

# =========================
# Build signal
# =========================
snapshot = build_signal_snapshot(fred_data, eia_data, opec_df)
total_score = float(snapshot["weighted"].sum()) if not snapshot.empty else 0.0
signal_label = infer_signal_label(total_score)
signal_meter = normalize_score(total_score, -6, 6)

# =========================
# Top summary
# =========================
c1, c2, c3, c4 = st.columns(4)

wti_latest, wti_delta = metric_delta_text(fred_data["WTI"]["value"], periods=20) if not fred_data["WTI"].empty else (None, None)
brent_latest, brent_delta = metric_delta_text(fred_data["Brent"]["value"], periods=20) if not fred_data["Brent"].empty else (None, None)
inv_latest, inv_delta = metric_delta_text(eia_data["Crude_Inventory"]["value"], periods=4) if not eia_data["Crude_Inventory"].empty else (None, None)
prod_latest, prod_delta = metric_delta_text(eia_data["US_Production"]["value"], periods=4) if not eia_data["US_Production"].empty else (None, None)

c1.metric("WTI", f"{wti_latest:.2f}" if wti_latest is not None else "N/A", f"{wti_delta:.2f}" if wti_delta is not None else None)
c2.metric("Brent", f"{brent_latest:.2f}" if brent_latest is not None else "N/A", f"{brent_delta:.2f}" if brent_delta is not None else None)
c3.metric("Crude Inventory", f"{inv_latest:,.0f}" if inv_latest is not None else "N/A", f"{inv_delta:,.0f}" if inv_delta is not None else None)
c4.metric("US Production", f"{prod_latest:,.0f}" if prod_latest is not None else "N/A", f"{prod_delta:,.0f}" if prod_delta is not None else None)

# =========================
# Signal header
# =========================
st.subheader("Oil Direction Signal")
left, right = st.columns([1, 2])

with left:
    st.metric("Composite Score", f"{total_score:.2f}")
    st.metric("Signal", signal_label)
    st.progress(int(signal_meter))

with right:
    st.dataframe(snapshot, use_container_width=True, hide_index=True)

# =========================
# Price chart
# =========================
price_series = {}
if show_wti:
    price_series["WTI"] = fred_data["WTI"]
if show_brent:
    price_series["Brent"] = fred_data["Brent"]

st.plotly_chart(
    make_line_chart(price_series, "Oil Price", "USD / barrel"),
    use_container_width=True
)

# =========================
# Macro charts
# =========================
col_a, col_b = st.columns(2)

with col_a:
    st.plotly_chart(
        make_line_chart(
            {
                "Fed Funds": fred_data["FedFunds"],
                "US 10Y": fred_data["US10Y"],
            },
            "Rates",
            "%",
        ),
        use_container_width=True
    )

with col_b:
    st.plotly_chart(
        make_line_chart(
            {
                "Dollar Proxy": fred_data["Dollar"],
            },
            "Dollar",
            "Index",
        ),
        use_container_width=True
    )

# =========================
# Supply / inventory charts
# =========================
col_c, col_d = st.columns(2)

with col_c:
    st.plotly_chart(
        make_line_chart(
            {
                "Crude Inventory": eia_data["Crude_Inventory"],
                "Cushing Inventory": eia_data["Cushing_Inventory"],
            },
            "Inventory",
            "Level",
        ),
        use_container_width=True
    )

with col_d:
    st.plotly_chart(
        make_line_chart(
            {
                "US Production": eia_data["US_Production"],
            },
            "US Crude Production",
            "Level",
        ),
        use_container_width=True
    )

# =========================
# OPEC event table
# =========================
st.subheader("OPEC Event Table")
if opec_df.empty:
    st.info("No OPEC events loaded.")
else:
    st.dataframe(opec_df.sort_values("date", ascending=False), use_container_width=True, hide_index=True)

# =========================
# Notes
# =========================
with st.expander("Important implementation notes"):
    st.markdown(
        """
1. **FRED**
   - Works well for WTI, Brent, Fed Funds, 10Y, and dollar proxy.

2. **EIA**
   - API v2 routes/facets sometimes need exact route confirmation in the EIA API browser.
   - If one route fails, open the API browser and copy the exact route/facets for:
     - crude inventory
     - cushing inventory
     - US crude production

3. **OPEC**
   - For now, use a manual event table.
   - This is often better than trying to fully automate nuanced policy statements.

4. **Rig Count**
   - Add later as a separate scraper or uploaded CSV module.

5. **Best next upgrade**
   - add OECD CLI or China demand proxy
   - add YoY / 4W change calculations
   - add oil signal history backtest
        """
    )
