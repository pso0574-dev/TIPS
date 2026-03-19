# streamlit_app.py
# ============================================================
# Global Macro Monitoring Dashboard
# - Oil / Rates / Gold / BTC / Equity / FX
# - Period changes: 3M / 6M / 1Y / 5Y / 10Y
# - Summary table + charts + simple signals
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit pandas numpy plotly yfinance requests python-dateutil
#
# Optional but recommended for FRED:
#   Set FRED_API_KEY in environment variables
#   or in .streamlit/secrets.toml:
#   FRED_API_KEY="YOUR_FRED_API_KEY"
# ============================================================

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta

# ============================================================
# Streamlit page config
# ============================================================
st.set_page_config(
    page_title="Global Macro Monitoring Dashboard",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Global Macro Monitoring Dashboard")
st.caption(
    "Oil / Rates / Gold / BTC / Equity / FX • Current level + 3M / 6M / 1Y / 5Y / 10Y changes"
)

# ============================================================
# Constants
# ============================================================
TODAY = pd.Timestamp.today().normalize()
START_DATE = TODAY - relativedelta(years=11)  # enough for 10Y change calculation
DISPLAY_PERIODS = {
    "3M": relativedelta(months=3),
    "6M": relativedelta(months=6),
    "1Y": relativedelta(years=1),
    "5Y": relativedelta(years=5),
    "10Y": relativedelta(years=10),
}
GRAPH_PERIODS = {
    "3M": relativedelta(months=3),
    "6M": relativedelta(months=6),
    "1Y": relativedelta(years=1),
    "5Y": relativedelta(years=5),
    "10Y": relativedelta(years=10),
    "ALL": relativedelta(years=11),
}

# Main assets requested by user
ASSET_CONFIG = [
    # Category, Asset, Symbol, Source, Unit, Type
    {"category": "Oil", "asset": "WTI Crude Oil", "symbol": "CL=F", "source": "Yahoo", "unit": "USD/bbl", "type": "price"},
    {"category": "Oil", "asset": "Brent Crude Oil", "symbol": "BZ=F", "source": "Yahoo", "unit": "USD/bbl", "type": "price"},

    {"category": "Rate", "asset": "US 10Y Yield", "symbol": "DGS10", "source": "FRED", "unit": "%", "type": "rate"},
    {"category": "Rate", "asset": "US 2Y Yield", "symbol": "DGS2", "source": "FRED", "unit": "%", "type": "rate"},
    {"category": "Rate", "asset": "US 3M Yield", "symbol": "DGS3MO", "source": "FRED", "unit": "%", "type": "rate"},

    {"category": "Metal", "asset": "Gold", "symbol": "GC=F", "source": "Yahoo", "unit": "USD/oz", "type": "price"},
    {"category": "Commodity", "asset": "Broad Commodities ETF", "symbol": "DBC", "source": "Yahoo", "unit": "ETF", "type": "price"},

    {"category": "Crypto", "asset": "Bitcoin / EUR", "symbol": "BTC-EUR", "source": "Yahoo", "unit": "EUR", "type": "price"},

    {"category": "Equity", "asset": "S&P 500", "symbol": "^GSPC", "source": "Yahoo", "unit": "Index", "type": "price"},
    {"category": "Equity", "asset": "Dow Jones", "symbol": "^DJI", "source": "Yahoo", "unit": "Index", "type": "price"},
    {"category": "Equity", "asset": "Nasdaq 100", "symbol": "^NDX", "source": "Yahoo", "unit": "Index", "type": "price"},

    {"category": "FX", "asset": "USD/KRW", "symbol": "KRW=X", "source": "Yahoo", "unit": "KRW per USD", "type": "fx_usdkrw"},
    {"category": "FX", "asset": "EUR/KRW", "symbol": "EURKRW=X", "source": "Yahoo", "unit": "KRW per EUR", "type": "price"},
]

# FRED fallback proxies via Yahoo if no API key
FRED_FALLBACK = {
    "DGS10": "^TNX",   # 10Y yield proxy (Yahoo, *10)
    "DGS2": "^IRX",    # not a perfect 2Y proxy; kept only as last resort
    "DGS3MO": "^IRX",  # 13-week bill proxy
}

# ============================================================
# Helpers
# ============================================================
def get_fred_api_key() -> Optional[str]:
    key = None
    try:
        key = st.secrets.get("FRED_API_KEY", None)
    except Exception:
        key = None
    if not key:
        key = os.getenv("FRED_API_KEY")
    return key


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def format_current(value: Optional[float], unit: str, asset_type: str) -> str:
    if value is None:
        return "N/A"
    if asset_type == "rate":
        return f"{value:.2f}%"
    if "KRW" in unit:
        return f"{value:,.1f}"
    if unit in {"Index", "ETF"}:
        return f"{value:,.2f}"
    if unit in {"USD/bbl", "USD/oz", "EUR"}:
        return f"{value:,.2f}"
    return f"{value:,.2f}"


def format_change(value: Optional[float], asset_type: str, as_bp: bool = False) -> str:
    if value is None:
        return "N/A"
    if asset_type == "rate":
        if as_bp:
            return f"{value * 100:.0f}bp"
        return f"{value:+.2f}pp"
    return f"{value:+.2f}%"


def nearest_value(series: pd.Series, target_date: pd.Timestamp, tolerance_days: int = 7) -> Optional[float]:
    if series.empty:
        return None
    s = series.dropna().sort_index()
    if s.empty:
        return None

    # exact / nearest previous
    idx = s.index.searchsorted(target_date, side="right") - 1
    candidate = None
    if idx >= 0:
        candidate_date = s.index[idx]
        if abs((target_date - candidate_date).days) <= tolerance_days:
            candidate = safe_float(s.iloc[idx])

    # if previous is too far, nearest absolute
    if candidate is None:
        diffs = np.abs((s.index - target_date).days)
        min_i = int(np.argmin(diffs))
        if diffs[min_i] <= tolerance_days:
            candidate = safe_float(s.iloc[min_i])

    return candidate


def latest_value(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    s = series.dropna()
    if s.empty:
        return None
    return safe_float(s.iloc[-1])


def compute_pct_change(current: Optional[float], past: Optional[float]) -> Optional[float]:
    if current is None or past is None or past == 0:
        return None
    return (current / past - 1.0) * 100.0


def compute_pp_change(current: Optional[float], past: Optional[float]) -> Optional[float]:
    if current is None or past is None:
        return None
    return current - past


def normalize_to_100(series: pd.Series) -> pd.Series:
    s = series.dropna().copy()
    if s.empty:
        return s
    base = s.iloc[0]
    if base == 0 or pd.isna(base):
        return s * np.nan
    return s / base * 100.0


def build_signal(row: pd.Series, spread_10y_2y: Optional[float], spread_10y_3m: Optional[float]) -> str:
    asset = row["Asset"]
    category = row["Category"]
    ch_3m = row["3M_raw"]
    ch_1y = row["1Y_raw"]

    if category == "Rate":
        curr = row["Current_raw"]
        if curr is None:
            return "N/A"
        if asset == "US 10Y Yield":
            if curr >= 4.5:
                return "Tight"
            if curr <= 3.0:
                return "Loose"
            return "Neutral"
        if asset == "US 2Y Yield":
            return "Policy Watch"
        if asset == "US 3M Yield":
            return "Liquidity Watch"

    if asset == "WTI Crude Oil":
        if ch_3m is not None and ch_3m >= 10:
            return "Inflation Watch"
        if ch_3m is not None and ch_3m <= -10:
            return "Demand Weakness"
        return "Neutral"

    if asset == "Gold":
        if ch_1y is not None and ch_1y > 10:
            return "Risk-Off / Hedge"
        return "Neutral"

    if asset == "Bitcoin / EUR":
        if ch_3m is not None and ch_3m > 15:
            return "Risk-On"
        if ch_3m is not None and ch_3m < -15:
            return "Risk-Off"
        return "Volatile"

    if asset == "S&P 500":
        if ch_1y is not None and ch_1y > 10:
            return "Growth"
        if ch_1y is not None and ch_1y < -10:
            return "Weakness"
        return "Neutral"

    if asset == "Nasdaq 100":
        if ch_1y is not None and ch_1y > 15:
            return "Growth / Tech"
        return "Neutral"

    if asset == "USD/KRW":
        if ch_3m is not None and ch_3m > 3:
            return "KRW Weakness"
        if ch_3m is not None and ch_3m < -3:
            return "KRW Strength"
        return "Neutral"

    if asset == "EUR/KRW":
        if ch_3m is not None and ch_3m > 3:
            return "EUR Strong"
        if ch_3m is not None and ch_3m < -3:
            return "EUR Weak"
        return "Neutral"

    if asset == "US 10Y-2Y Spread" and spread_10y_2y is not None:
        if spread_10y_2y < 0:
            return "Inverted"
        if spread_10y_2y < 0.5:
            return "Flat"
        return "Steep"

    if asset == "US 10Y-3M Spread" and spread_10y_3m is not None:
        if spread_10y_3m < 0:
            return "Inverted"
        if spread_10y_3m < 0.5:
            return "Flat"
        return "Steep"

    return "Neutral"


# ============================================================
# Data loading
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_yahoo_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    Robust Yahoo loader using start/end instead of custom period strings.
    """
    try:
        df = yf.download(
            symbol,
            start=start.date(),
            end=(end + pd.Timedelta(days=1)).date(),
            auto_adjust=False,
            progress=False,
            interval="1d",
            threads=False,
        )
        if df is None or df.empty:
            return pd.Series(dtype=float)

        # Sometimes columns come as MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            if ("Adj Close", symbol) in df.columns:
                s = df[("Adj Close", symbol)]
            elif ("Close", symbol) in df.columns:
                s = df[("Close", symbol)]
            else:
                s = df.iloc[:, 0]
        else:
            if "Adj Close" in df.columns:
                s = df["Adj Close"]
            elif "Close" in df.columns:
                s = df["Close"]
            else:
                s = df.iloc[:, 0]

        s = pd.to_numeric(s, errors="coerce")
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=600, show_spinner=False)
def load_fred_series(series_id: str, start: pd.Timestamp, end: pd.Timestamp, api_key: Optional[str]) -> pd.Series:
    """
    FRED daily/available observations loader.
    """
    if not api_key:
        return pd.Series(dtype=float)

    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start.strftime("%Y-%m-%d"),
            "observation_end": end.strftime("%Y-%m-%d"),
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        obs = data.get("observations", [])
        if not obs:
            return pd.Series(dtype=float)

        df = pd.DataFrame(obs)
        if "date" not in df.columns or "value" not in df.columns:
            return pd.Series(dtype=float)

        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
        s = df.set_index("date")["value"].sort_index().dropna()
        s.index = s.index.tz_localize(None)
        return s
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=600, show_spinner=False)
def load_all_series() -> Dict[str, pd.Series]:
    api_key = get_fred_api_key()
    out: Dict[str, pd.Series] = {}

    for item in ASSET_CONFIG:
        sym = item["symbol"]
        src = item["source"]
        a_type = item["type"]

        if src == "Yahoo":
            s = load_yahoo_history(sym, START_DATE, TODAY)
        elif src == "FRED":
            s = load_fred_series(sym, START_DATE, TODAY, api_key)

            # optional fallback for missing FRED key / failure
            if s.empty:
                fb = FRED_FALLBACK.get(sym)
                if fb:
                    s_fb = load_yahoo_history(fb, START_DATE, TODAY)
                    if not s_fb.empty:
                        # Yahoo treasury proxies like ^TNX are often in yield*10
                        if fb == "^TNX":
                            s_fb = s_fb / 10.0
                        elif fb == "^IRX":
                            s_fb = s_fb / 10.0
                        s = s_fb
        else:
            s = pd.Series(dtype=float)

        out[sym] = s

    # Derived spread series
    d10 = out.get("DGS10", pd.Series(dtype=float))
    d2 = out.get("DGS2", pd.Series(dtype=float))
    d3m = out.get("DGS3MO", pd.Series(dtype=float))

    if not d10.empty and not d2.empty:
        aligned = pd.concat([d10.rename("10Y"), d2.rename("2Y")], axis=1).dropna()
        out["SPREAD_10Y_2Y"] = aligned["10Y"] - aligned["2Y"]
    else:
        out["SPREAD_10Y_2Y"] = pd.Series(dtype=float)

    if not d10.empty and not d3m.empty:
        aligned = pd.concat([d10.rename("10Y"), d3m.rename("3M")], axis=1).dropna()
        out["SPREAD_10Y_3M"] = aligned["10Y"] - aligned["3M"]
    else:
        out["SPREAD_10Y_3M"] = pd.Series(dtype=float)

    return out


# ============================================================
# Summary table builder
# ============================================================
def build_summary_table(series_map: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []

    for item in ASSET_CONFIG:
        category = item["category"]
        asset = item["asset"]
        symbol = item["symbol"]
        source = item["source"]
        unit = item["unit"]
        asset_type = item["type"]

        s = series_map.get(symbol, pd.Series(dtype=float))
        current = latest_value(s)

        row = {
            "Category": category,
            "Asset": asset,
            "Symbol": symbol,
            "Current_raw": current,
            "Unit": unit,
            "Source": source,
            "Type": asset_type,
        }

        for label, delta in DISPLAY_PERIODS.items():
            past_date = TODAY - delta
            past = nearest_value(s, past_date)
            if asset_type == "rate":
                raw_change = compute_pp_change(current, past)
                row[f"{label}_raw"] = raw_change
                row[label] = format_change(raw_change, asset_type)
            else:
                raw_change = compute_pct_change(current, past)
                row[f"{label}_raw"] = raw_change
                row[label] = format_change(raw_change, asset_type)

        row["Current"] = format_current(current, unit, asset_type)
        rows.append(row)

    # Add spread rows
    for spread_symbol, spread_name in [
        ("SPREAD_10Y_2Y", "US 10Y-2Y Spread"),
        ("SPREAD_10Y_3M", "US 10Y-3M Spread"),
    ]:
        s = series_map.get(spread_symbol, pd.Series(dtype=float))
        current = latest_value(s)
        row = {
            "Category": "Rate",
            "Asset": spread_name,
            "Symbol": spread_symbol,
            "Current_raw": current,
            "Unit": "pp",
            "Source": "Derived",
            "Type": "rate",
        }
        for label, delta in DISPLAY_PERIODS.items():
            past_date = TODAY - delta
            past = nearest_value(s, past_date)
            raw_change = compute_pp_change(current, past)
            row[f"{label}_raw"] = raw_change
            row[label] = format_change(raw_change, "rate")
        row["Current"] = "N/A" if current is None else f"{current:+.2f}pp"
        rows.append(row)

    df = pd.DataFrame(rows)

    spread_10y_2y = None
    spread_10y_3m = None
    try:
        spread_10y_2y = float(df.loc[df["Asset"] == "US 10Y-2Y Spread", "Current_raw"].iloc[0])
    except Exception:
        pass
    try:
        spread_10y_3m = float(df.loc[df["Asset"] == "US 10Y-3M Spread", "Current_raw"].iloc[0])
    except Exception:
        pass

    df["Signal"] = df.apply(
        lambda r: build_signal(r, spread_10y_2y=spread_10y_2y, spread_10y_3m=spread_10y_3m),
        axis=1,
    )

    return df


# ============================================================
# Styling
# ============================================================
def color_change_cell(val: str) -> str:
    try:
        if val == "N/A":
            return "color: #999999;"
        s = str(val).replace("%", "").replace("pp", "").replace("bp", "").replace("+", "")
        num = float(s)
        if num > 0:
            return "color: #0a7f2e; font-weight: 600;"
        if num < 0:
            return "color: #b00020; font-weight: 600;"
        return "color: #666666;"
    except Exception:
        return ""


def color_signal_cell(val: str) -> str:
    mapping = {
        "Risk-On": "#0a7f2e",
        "Growth": "#0a7f2e",
        "Growth / Tech": "#0a7f2e",
        "Risk-Off / Hedge": "#b00020",
        "Inflation Watch": "#b36b00",
        "Tight": "#b36b00",
        "Inverted": "#b00020",
        "KRW Weakness": "#b00020",
        "EUR Strong": "#0a4ea1",
    }
    color = mapping.get(val, "#444444")
    return f"color: {color}; font-weight: 600;"


# ============================================================
# Charts
# ============================================================
def filter_series_by_period(series: pd.Series, period_label: str) -> pd.Series:
    if series.empty:
        return series
    delta = GRAPH_PERIODS[period_label]
    start = TODAY - delta
    return series.loc[series.index >= start].dropna()


def make_line_chart(
    series_dict: Dict[str, pd.Series],
    title: str,
    normalize: bool = False,
    yaxis_title: str = "",
) -> go.Figure:
    fig = go.Figure()
    for name, series in series_dict.items():
        s = series.dropna()
        if s.empty:
            continue
        if normalize:
            s = normalize_to_100(s)
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=name,
            )
        )
    fig.update_layout(
        title=title,
        height=520,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Date",
        yaxis_title=yaxis_title,
    )
    return fig


def make_bar_chart(summary_df: pd.DataFrame, period_col: str, asset_filter: str) -> go.Figure:
    df = summary_df.copy()
    if asset_filter != "All":
        df = df[df["Category"] == asset_filter]

    vals = df[f"{period_col}_raw"].copy()
    chart_df = pd.DataFrame(
        {
            "Asset": df["Asset"],
            "Value": vals,
            "Type": df["Type"],
        }
    ).dropna()

    # Keep rates and prices together but readable
    fig = go.Figure()
    if not chart_df.empty:
        fig.add_trace(
            go.Bar(
                x=chart_df["Asset"],
                y=chart_df["Value"],
                text=[f"{v:+.2f}" for v in chart_df["Value"]],
                textposition="outside",
                name=period_col,
            )
        )

    fig.update_layout(
        title=f"{period_col} Change by Asset",
        height=500,
        margin=dict(l=40, r=20, t=50, b=80),
        xaxis_title="Asset",
        yaxis_title="Change (% for price assets, pp for rate assets)",
    )
    return fig


# ============================================================
# Sidebar / Controls
# ============================================================
with st.sidebar:
    st.header("Controls")
    refresh = st.button("🔄 Refresh Data", use_container_width=True)

    selected_graph_period = st.selectbox(
        "Graph Period",
        options=list(GRAPH_PERIODS.keys()),
        index=2,  # 1Y
    )

    category_options = ["All", "Oil", "Rate", "Metal", "Commodity", "Crypto", "Equity", "FX"]
    selected_category = st.selectbox("Table Category Filter", category_options, index=0)

    sort_options = [
        "Category",
        "Asset",
        "3M_raw",
        "6M_raw",
        "1Y_raw",
        "5Y_raw",
        "10Y_raw",
        "Current_raw",
    ]
    selected_sort = st.selectbox("Sort By", sort_options, index=3)

    sort_desc = st.checkbox("Descending", value=True)

    show_all_columns = st.checkbox("Show raw helper columns", value=False)
    normalize_chart = st.checkbox("Normalize multi-asset chart to 100", value=True)

    st.markdown("---")
    st.markdown("**Data notes**")
    st.caption("Rates use FRED when API key is available. Yahoo proxies may be used as fallback for missing FRED data.")

if refresh:
    st.cache_data.clear()

# ============================================================
# Load data
# ============================================================
with st.spinner("Loading market data..."):
    series_map = load_all_series()
    summary_df = build_summary_table(series_map)

# ============================================================
# Filter / sort summary table
# ============================================================
view_df = summary_df.copy()

if selected_category != "All":
    view_df = view_df[view_df["Category"] == selected_category].copy()

if selected_sort in view_df.columns:
    if selected_sort.endswith("_raw") or selected_sort == "Current_raw":
        view_df = view_df.sort_values(selected_sort, ascending=not sort_desc, na_position="last")
    else:
        view_df = view_df.sort_values(selected_sort, ascending=not sort_desc, na_position="last")

display_cols = [
    "Category",
    "Asset",
    "Symbol",
    "Current",
    "3M",
    "6M",
    "1Y",
    "5Y",
    "10Y",
    "Unit",
    "Source",
    "Signal",
]

if show_all_columns:
    raw_cols = [c for c in view_df.columns if c.endswith("_raw") or c == "Current_raw"]
    display_cols += raw_cols

st.subheader("Summary Table")

styled = (
    view_df[display_cols]
    .style
    .map(color_change_cell, subset=["3M", "6M", "1Y", "5Y", "10Y"])
    .map(color_signal_cell, subset=["Signal"])
)

st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# KPI cards
# ============================================================
st.subheader("Quick Risk Snapshot")

def get_row_value(df: pd.DataFrame, asset_name: str, col: str = "Current_raw") -> Optional[float]:
    try:
        return safe_float(df.loc[df["Asset"] == asset_name, col].iloc[0])
    except Exception:
        return None

oil_3m = get_row_value(summary_df, "WTI Crude Oil", "3M_raw")
gold_1y = get_row_value(summary_df, "Gold", "1Y_raw")
btc_3m = get_row_value(summary_df, "Bitcoin / EUR", "3M_raw")
spx_1y = get_row_value(summary_df, "S&P 500", "1Y_raw")
spread_10y_2y = get_row_value(summary_df, "US 10Y-2Y Spread", "Current_raw")

inflation_score = "High" if oil_3m is not None and oil_3m > 10 else "Moderate" if oil_3m is not None and oil_3m > 0 else "Low"
risk_sentiment = "Risk-On" if (btc_3m is not None and btc_3m > 10 and spx_1y is not None and spx_1y > 10) else "Mixed"
if gold_1y is not None and gold_1y > 10 and (spx_1y is not None and spx_1y < 5):
    risk_sentiment = "Risk-Off"

yield_curve = "Inverted" if spread_10y_2y is not None and spread_10y_2y < 0 else "Normal"
growth = "Strong" if spx_1y is not None and spx_1y > 10 else "Weak" if spx_1y is not None and spx_1y < 0 else "Moderate"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Inflation Pressure", inflation_score)
c2.metric("Risk Sentiment", risk_sentiment)
c3.metric("Yield Curve", yield_curve)
c4.metric("Growth Momentum", growth)

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Multi-Asset Trend", "Rates", "Equity", "FX", "Change Bar"]
)

with tab1:
    st.markdown("### Multi-Asset Trend")

    asset_name_to_symbol = {
        "WTI Crude Oil": "CL=F",
        "Brent Crude Oil": "BZ=F",
        "Gold": "GC=F",
        "Broad Commodities ETF": "DBC",
        "Bitcoin / EUR": "BTC-EUR",
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "Nasdaq 100": "^NDX",
        "USD/KRW": "KRW=X",
        "EUR/KRW": "EURKRW=X",
    }

    default_assets = ["WTI Crude Oil", "Gold", "Bitcoin / EUR", "S&P 500", "Nasdaq 100", "USD/KRW"]

    selected_assets = st.multiselect(
        "Select assets",
        options=list(asset_name_to_symbol.keys()),
        default=default_assets,
    )

    chart_map = {}
    for name in selected_assets:
        sym = asset_name_to_symbol[name]
        s = filter_series_by_period(series_map.get(sym, pd.Series(dtype=float)), selected_graph_period)
        if not s.empty:
            chart_map[name] = s

    fig = make_line_chart(
        chart_map,
        title=f"Multi-Asset Trend ({selected_graph_period})",
        normalize=normalize_chart,
        yaxis_title="Indexed to 100" if normalize_chart else "Price / Level",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Rates & Yield Curve")

    rate_map = {
        "US 10Y Yield": filter_series_by_period(series_map.get("DGS10", pd.Series(dtype=float)), selected_graph_period),
        "US 2Y Yield": filter_series_by_period(series_map.get("DGS2", pd.Series(dtype=float)), selected_graph_period),
        "US 3M Yield": filter_series_by_period(series_map.get("DGS3MO", pd.Series(dtype=float)), selected_graph_period),
    }
    fig_rates = make_line_chart(rate_map, title=f"US Rates ({selected_graph_period})", normalize=False, yaxis_title="%")
    st.plotly_chart(fig_rates, use_container_width=True)

    spread_map = {
        "10Y-2Y Spread": filter_series_by_period(series_map.get("SPREAD_10Y_2Y", pd.Series(dtype=float)), selected_graph_period),
        "10Y-3M Spread": filter_series_by_period(series_map.get("SPREAD_10Y_3M", pd.Series(dtype=float)), selected_graph_period),
    }
    fig_spreads = make_line_chart(spread_map, title=f"Yield Curve Spreads ({selected_graph_period})", normalize=False, yaxis_title="pp")
    st.plotly_chart(fig_spreads, use_container_width=True)

with tab3:
    st.markdown("### Equity Indices")

    equity_map = {
        "S&P 500": filter_series_by_period(series_map.get("^GSPC", pd.Series(dtype=float)), selected_graph_period),
        "Dow Jones": filter_series_by_period(series_map.get("^DJI", pd.Series(dtype=float)), selected_graph_period),
        "Nasdaq 100": filter_series_by_period(series_map.get("^NDX", pd.Series(dtype=float)), selected_graph_period),
    }
    fig_eq = make_line_chart(
        equity_map,
        title=f"US Equity Indices ({selected_graph_period})",
        normalize=True,
        yaxis_title="Indexed to 100",
    )
    st.plotly_chart(fig_eq, use_container_width=True)

with tab4:
    st.markdown("### FX")

    fx_map = {
        "USD/KRW": filter_series_by_period(series_map.get("KRW=X", pd.Series(dtype=float)), selected_graph_period),
        "EUR/KRW": filter_series_by_period(series_map.get("EURKRW=X", pd.Series(dtype=float)), selected_graph_period),
    }
    fig_fx = make_line_chart(
        fx_map,
        title=f"KRW FX Pairs ({selected_graph_period})",
        normalize=False,
        yaxis_title="KRW",
    )
    st.plotly_chart(fig_fx, use_container_width=True)

with tab5:
    st.markdown("### Period Change Comparison")
    selected_bar_period = st.selectbox("Change Period", list(DISPLAY_PERIODS.keys()), index=2, key="bar_period")
    fig_bar = make_bar_chart(summary_df, selected_bar_period, selected_category)
    st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================
# Raw data preview
# ============================================================
with st.expander("See raw latest series data"):
    preview_options = st.multiselect(
        "Choose symbols to preview",
        options=list(series_map.keys()),
        default=["CL=F", "DGS10", "GC=F", "BTC-EUR", "^GSPC", "KRW=X"],
    )
    for sym in preview_options:
        st.markdown(f"**{sym}**")
        s = series_map.get(sym, pd.Series(dtype=float))
        if s.empty:
            st.write("No data")
        else:
            st.dataframe(s.tail(10).rename(sym).to_frame(), use_container_width=True)

st.markdown("---")
st.caption(
    "Tip: For FRED rates, set FRED_API_KEY in environment variables or .streamlit/secrets.toml."
)
