# streamlit_app.py
# ============================================================
# TIPS Buy/Sell Timing Dashboard
# - Macro regime monitor for inflation hedging
# - TIPS vs TLT vs Gold vs Commodities performance
# - FRED + Yahoo Finance based
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit pandas numpy plotly yfinance requests
# ============================================================

from __future__ import annotations

from datetime import datetime, timedelta
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="TIPS Buy/Sell Timing Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 TIPS Buy/Sell Timing Dashboard")
st.caption(
    "Monitor real yield, breakeven inflation, CPI trend, and compare TIP / VTIP / SCHP / TLT / GLD / DBC / IEF"
)


# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Settings")

lookback_years = st.sidebar.slider("Price lookback (years)", 1, 15, 5, 1)
fred_years = st.sidebar.slider("Macro lookback (years)", 3, 25, 10, 1)
font_size = st.sidebar.slider("Base font size", 10, 24, 14, 1)
price_ma_short = st.sidebar.slider("Short moving average", 10, 100, 50, 5)
price_ma_long = st.sidebar.slider("Long moving average", 50, 300, 200, 10)

selected_assets = st.sidebar.multiselect(
    "ETF universe",
    ["TIP", "VTIP", "SCHP", "TLT", "IEF", "GLD", "IAU", "DBC", "PDBC", "SPY", "QQQ"],
    default=["TIP", "VTIP", "TLT", "GLD", "DBC", "IEF"],
)

use_log_scale = st.sidebar.checkbox("Use log scale on price charts", value=False)
show_drawdown = st.sidebar.checkbox("Show drawdown chart", value=True)

refresh = st.sidebar.button("🔄 Refresh data")


# ============================================================
# Constants
# ============================================================
FRED_SERIES: Dict[str, str] = {
    "DFII10": "10Y TIPS Real Yield",
    "T10YIE": "10Y Breakeven Inflation",
    "DGS10": "10Y Treasury Yield",
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "FEDFUNDS": "Fed Funds Rate",
    "UNRATE": "Unemployment Rate",
}

SERIES_COLORS = {
    "DFII10": "#1f77b4",
    "T10YIE": "#ff7f0e",
    "DGS10": "#2ca02c",
    "CPI_YOY": "#d62728",
    "FEDFUNDS": "#9467bd",
    "UNRATE": "#8c564b",
}

ETF_LABELS = {
    "TIP": "Broad TIPS",
    "VTIP": "Short-Term TIPS",
    "SCHP": "Broad TIPS",
    "TLT": "Long Treasury",
    "IEF": "Intermediate Treasury",
    "GLD": "Gold",
    "IAU": "Gold",
    "DBC": "Commodities",
    "PDBC": "Commodities",
    "SPY": "US Equities",
    "QQQ": "Nasdaq 100",
}


# ============================================================
# Helpers
# ============================================================
def fred_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


@st.cache_data(ttl=3600, show_spinner=False)
def load_fred_series(series_id: str) -> pd.Series:
    """
    Load a single FRED series from CSV and return a clean Series with DatetimeIndex.
    """
    url = fred_url(series_id)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    if df.empty or len(df.columns) < 2:
        return pd.Series(dtype="float64", name=series_id)

    df.columns = ["Date", series_id]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    df = df.dropna(subset=["Date"]).sort_values("Date")
    if df.empty:
        return pd.Series(dtype="float64", name=series_id)

    series = df.set_index("Date")[series_id]
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[~series.index.isna()]
    series = series.sort_index()
    series.name = series_id

    return series


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_fred_data(years: int) -> pd.DataFrame:
    """
    Load all required FRED series and return a clean DataFrame.
    Uses a date mask instead of df.last(...), which avoids DatetimeIndex errors.
    """
    data: Dict[str, pd.Series] = {}

    for series_id in FRED_SERIES.keys():
        try:
            data[series_id] = load_fred_series(series_id)
        except Exception:
            data[series_id] = pd.Series(dtype="float64", name=series_id)

    if not data:
        return pd.DataFrame()

    df = pd.concat(data, axis=1)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~df.index.isna()]
    df = df.sort_index()

    cutoff_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=years * 365)
    df = df.loc[df.index >= cutoff_date].copy()

    if "CPIAUCSL" in df.columns:
        df["CPI_YOY"] = df["CPIAUCSL"].pct_change(12) * 100

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(tickers: List[str], years: int) -> pd.DataFrame:
    """
    Load ETF price data from Yahoo Finance.
    Handles both single-ticker and multi-ticker download formats safely.
    """
    if not tickers:
        return pd.DataFrame()

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years + 20)

    df = yf.download(
        tickers=tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Single ticker case
    if len(tickers) == 1:
        ticker = tickers[0]

        try:
            if isinstance(df.columns, pd.MultiIndex):
                out = df[ticker][["Close"]].copy()
                out.columns = [ticker]
            else:
                out = df[["Close"]].copy()
                out.columns = [ticker]

            out.index = pd.to_datetime(out.index, errors="coerce")
            out = out[~out.index.isna()].sort_index()
            return out.dropna(how="all")
        except Exception:
            return pd.DataFrame()

    # Multi ticker case
    close_frames = []
    for ticker in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                s = df[ticker]["Close"].rename(ticker)
                close_frames.append(s)
        except Exception:
            pass

    if not close_frames:
        return pd.DataFrame()

    prices = pd.concat(close_frames, axis=1)
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices[~prices.index.isna()].sort_index()

    return prices.dropna(how="all")


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices
    return prices / prices.iloc[0] * 100


def compute_drawdown(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices
    peak = prices.cummax()
    drawdown = prices / peak - 1.0
    return drawdown * 100


def annualized_return(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 2:
        return np.nan

    total_return = series.iloc[-1] / series.iloc[0]
    years = (series.index[-1] - series.index[0]).days / 365.25

    if years <= 0 or total_return <= 0:
        return np.nan

    return (total_return ** (1 / years) - 1) * 100


def annualized_volatility(series: pd.Series) -> float:
    returns = series.pct_change().dropna()
    if len(returns) < 2:
        return np.nan
    return returns.std() * np.sqrt(252) * 100


def max_drawdown_pct(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty:
        return np.nan
    dd = series / series.cummax() - 1
    return dd.min() * 100


def sharpe_like(series: pd.Series) -> float:
    returns = series.pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(252)


def percentile_rank(value: float, series: pd.Series) -> float:
    s = series.dropna()
    if s.empty or pd.isna(value):
        return np.nan
    return (s <= value).mean() * 100


def latest_valid(series: pd.Series) -> float:
    s = series.dropna()
    return s.iloc[-1] if not s.empty else np.nan


def prev_valid_by_count(series: pd.Series, n: int = 1) -> float:
    """
    Return the nth previous valid observation by count.
    """
    s = series.dropna()
    if len(s) <= n:
        return np.nan
    return s.iloc[-1 - n]


def classify_tips_signal(df: pd.DataFrame) -> Tuple[str, str, Dict[str, float]]:
    """
    Heuristic scoring model for TIPS attractiveness.

    Positive factors:
    - High real yield
    - Reasonable breakeven inflation
    - Inflation still above target
    - Restrictive nominal/real policy backdrop

    Negative factors:
    - TIPS already too expensive via breakeven inflation
    - Inflation collapsing too much
    """
    real_yield = latest_valid(df["DFII10"]) if "DFII10" in df else np.nan
    breakeven = latest_valid(df["T10YIE"]) if "T10YIE" in df else np.nan
    cpi_yoy = latest_valid(df["CPI_YOY"]) if "CPI_YOY" in df else np.nan
    fedfunds = latest_valid(df["FEDFUNDS"]) if "FEDFUNDS" in df else np.nan
    dgs10 = latest_valid(df["DGS10"]) if "DGS10" in df else np.nan

    real_yield_3m_ago = prev_valid_by_count(df["DFII10"], n=63) if "DFII10" in df else np.nan
    breakeven_3m_ago = prev_valid_by_count(df["T10YIE"], n=63) if "T10YIE" in df else np.nan
    cpi_3m_ago = prev_valid_by_count(df["CPI_YOY"], n=3) if "CPI_YOY" in df else np.nan

    score = 0

    # 1) Real yield valuation
    if not pd.isna(real_yield):
        if real_yield >= 2.0:
            score += 3
        elif real_yield >= 1.5:
            score += 2
        elif real_yield >= 1.0:
            score += 1
        elif real_yield < 0.5:
            score -= 2

    # 2) Breakeven inflation level
    if not pd.isna(breakeven):
        if 1.8 <= breakeven <= 2.5:
            score += 2
        elif 2.5 < breakeven <= 2.8:
            score += 1
        elif breakeven > 3.0:
            score -= 1
        elif breakeven < 1.6:
            score -= 1

    # 3) Inflation still above target
    if not pd.isna(cpi_yoy):
        if cpi_yoy >= 3.0:
            score += 2
        elif cpi_yoy >= 2.2:
            score += 1
        elif cpi_yoy < 1.8:
            score -= 1

    # 4) Trend checks
    if not pd.isna(real_yield) and not pd.isna(real_yield_3m_ago):
        if real_yield > real_yield_3m_ago + 0.20:
            score += 1
        elif real_yield < real_yield_3m_ago - 0.30:
            score -= 1

    if not pd.isna(breakeven) and not pd.isna(breakeven_3m_ago):
        if breakeven > breakeven_3m_ago + 0.20:
            score += 1
        elif breakeven < breakeven_3m_ago - 0.20:
            score -= 1

    if not pd.isna(cpi_yoy) and not pd.isna(cpi_3m_ago):
        if cpi_yoy > cpi_3m_ago + 0.20:
            score += 1
        elif cpi_yoy < cpi_3m_ago - 0.50:
            score -= 1

    # 5) Nominal 10Y context
    if not pd.isna(dgs10):
        if dgs10 >= 4.5:
            score += 1
        elif dgs10 < 3.0:
            score -= 1

    # 6) Policy stance
    if not pd.isna(fedfunds) and not pd.isna(cpi_yoy):
        real_policy_proxy = fedfunds - cpi_yoy
        if real_policy_proxy >= 1.0:
            score += 1
        elif real_policy_proxy <= -1.0:
            score -= 1

    if score >= 7:
        signal = "Strong Buy"
        reason = "High real yield, acceptable breakeven inflation, and still-elevated inflation support TIPS entry."
    elif score >= 4:
        signal = "Buy / Accumulate"
        reason = "The macro backdrop is generally supportive for gradual TIPS accumulation."
    elif score >= 1:
        signal = "Neutral / Hold"
        reason = "TIPS look reasonable, but not clearly cheap. Phased buying is preferable."
    else:
        signal = "Wait / Reduce"
        reason = "The current mix of real yield, breakeven inflation, and inflation trend is not favorable."

    details = {
        "score": score,
        "real_yield": real_yield,
        "breakeven": breakeven,
        "cpi_yoy": cpi_yoy,
        "fedfunds": fedfunds,
        "dgs10": dgs10,
    }
    return signal, reason, details


def classify_macro_regime(df: pd.DataFrame) -> str:
    cpi_yoy = latest_valid(df["CPI_YOY"]) if "CPI_YOY" in df else np.nan
    breakeven = latest_valid(df["T10YIE"]) if "T10YIE" in df else np.nan
    real_yield = latest_valid(df["DFII10"]) if "DFII10" in df else np.nan
    unrate = latest_valid(df["UNRATE"]) if "UNRATE" in df else np.nan

    if pd.isna(cpi_yoy) or pd.isna(breakeven) or pd.isna(real_yield):
        return "Unknown"

    if cpi_yoy >= 3.0 and breakeven >= 2.3 and real_yield >= 1.5:
        return "Sticky Inflation / Restrictive Rates"
    if cpi_yoy >= 3.0 and breakeven < 2.2:
        return "Disinflation with Market Skepticism"
    if cpi_yoy < 2.2 and real_yield >= 1.5 and (pd.isna(unrate) or unrate <= 4.8):
        return "Late-Cycle Tight Policy"
    if cpi_yoy < 2.2 and breakeven < 2.0 and not pd.isna(unrate) and unrate >= 4.8:
        return "Growth Slowdown / Disinflation"

    return "Mixed Transition Regime"


def top_asset_in_recent_window(prices: pd.DataFrame, days: int = 126) -> Tuple[str, float]:
    if prices.empty:
        return "-", np.nan

    recent = prices.dropna(how="all").tail(days)
    if len(recent) < 2:
        return "-", np.nan

    perf = (recent.iloc[-1] / recent.iloc[0] - 1) * 100
    perf = perf.sort_values(ascending=False)
    return perf.index[0], perf.iloc[0]


def make_line_chart(
    df: pd.DataFrame,
    title: str,
    yaxis_title: str = "",
    log_scale: bool = False,
    percent: bool = False,
) -> go.Figure:
    fig = go.Figure()

    for col in df.columns:
        if df[col].dropna().empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=520,
        font=dict(size=font_size),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=70, b=40),
        yaxis_title=yaxis_title,
        xaxis_title="Date",
    )

    if log_scale:
        fig.update_yaxes(type="log")
    if percent:
        fig.update_yaxes(ticksuffix="%")

    return fig


def make_macro_combo_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if "DFII10" in df:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["DFII10"],
                mode="lines",
                name="10Y TIPS Real Yield",
                line=dict(width=2, color=SERIES_COLORS["DFII10"]),
                yaxis="y1",
            )
        )

    if "T10YIE" in df:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["T10YIE"],
                mode="lines",
                name="10Y Breakeven Inflation",
                line=dict(width=2, color=SERIES_COLORS["T10YIE"]),
                yaxis="y1",
            )
        )

    if "CPI_YOY" in df:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["CPI_YOY"],
                mode="lines",
                name="CPI YoY",
                line=dict(width=2, dash="dot", color=SERIES_COLORS["CPI_YOY"]),
                yaxis="y2",
            )
        )

    fig.update_layout(
        title="Macro Core: Real Yield vs Breakeven vs CPI YoY",
        template="plotly_white",
        height=540,
        font=dict(size=font_size),
        margin=dict(l=40, r=40, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Yield / Inflation Expectation (%)"),
        yaxis2=dict(
            title="CPI YoY (%)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )

    return fig


def make_single_asset_chart(prices: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()

    if ticker not in prices.columns or prices[ticker].dropna().empty:
        fig.update_layout(
            title=f"{ticker} data unavailable",
            template="plotly_white",
            height=500,
            font=dict(size=font_size),
        )
        return fig

    s = prices[ticker].dropna()
    ma_short = s.rolling(price_ma_short).mean()
    ma_long = s.rolling(price_ma_long).mean()

    fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=ticker, line=dict(width=2)))
    fig.add_trace(go.Scatter(x=ma_short.index, y=ma_short, mode="lines", name=f"MA {price_ma_short}", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=ma_long.index, y=ma_long, mode="lines", name=f"MA {price_ma_long}", line=dict(width=1.5)))

    fig.update_layout(
        title=f"{ticker} Price with Moving Averages",
        template="plotly_white",
        height=500,
        font=dict(size=font_size),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=40, r=20, t=70, b=40),
    )

    if use_log_scale:
        fig.update_yaxes(type="log")

    return fig


def make_signal_gauge(score: float) -> go.Figure:
    capped = max(min(score, 10), -3)
    normalized = (capped + 3) / 13 * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=normalized,
            title={"text": "TIPS Attractiveness Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.25},
                "steps": [
                    {"range": [0, 30], "color": "#f8d7da"},
                    {"range": [30, 55], "color": "#fff3cd"},
                    {"range": [55, 80], "color": "#d1ecf1"},
                    {"range": [80, 100], "color": "#d4edda"},
                ],
            },
        )
    )

    fig.update_layout(height=300, template="plotly_white", font=dict(size=font_size))
    return fig


# ============================================================
# Refresh
# ============================================================
if refresh:
    st.cache_data.clear()


# ============================================================
# Load data
# ============================================================
with st.spinner("Loading macro and market data..."):
    fred_df = load_all_fred_data(fred_years)
    price_df = load_price_data(selected_assets, lookback_years)

signal, reason, signal_details = classify_tips_signal(fred_df)
regime = classify_macro_regime(fred_df)

norm_prices = normalize_prices(price_df) if not price_df.empty else pd.DataFrame()
drawdowns = compute_drawdown(price_df) if not price_df.empty else pd.DataFrame()
leader_6m, leader_6m_perf = top_asset_in_recent_window(price_df, days=126)


# ============================================================
# KPI row
# ============================================================
col1, col2, col3, col4, col5, col6 = st.columns(6)

real_yield = signal_details.get("real_yield", np.nan)
breakeven = signal_details.get("breakeven", np.nan)
cpi_yoy = signal_details.get("cpi_yoy", np.nan)
fedfunds = signal_details.get("fedfunds", np.nan)
dgs10 = signal_details.get("dgs10", np.nan)
score = signal_details.get("score", np.nan)

col1.metric("TIPS Signal", signal)
col2.metric("10Y Real Yield", f"{real_yield:.2f}%" if pd.notna(real_yield) else "N/A")
col3.metric("10Y Breakeven", f"{breakeven:.2f}%" if pd.notna(breakeven) else "N/A")
col4.metric("CPI YoY", f"{cpi_yoy:.2f}%" if pd.notna(cpi_yoy) else "N/A")
col5.metric("Fed Funds", f"{fedfunds:.2f}%" if pd.notna(fedfunds) else "N/A")
col6.metric("10Y Treasury", f"{dgs10:.2f}%" if pd.notna(dgs10) else "N/A")

st.info(f"**Macro Regime:** {regime}")
st.write(f"**Interpretation:** {reason}")


# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Overview",
        "TIPS Timing Model",
        "Macro Dashboard",
        "ETF Relative Performance",
        "Asset Deep Dive",
    ]
)


# ============================================================
# Tab 1: Overview
# ============================================================
with tab1:
    left, right = st.columns([1.2, 1])

    with left:
        st.plotly_chart(make_macro_combo_chart(fred_df), use_container_width=True)

    with right:
        st.plotly_chart(make_signal_gauge(score), use_container_width=True)

        st.markdown("### Summary")
        st.write(
            f"""
**Current signal:** **{signal}**

**Why this matters**
- **Real Yield** measures how attractive TIPS are from a valuation perspective.
- **Breakeven Inflation** shows what inflation the market already expects.
- **CPI YoY** tells you whether realized inflation is still elevated.

**Practical read**
- High real yield + still-elevated inflation = better TIPS entry
- Falling inflation + expensive breakeven = weaker TIPS setup
- If inflation fear rises again, **TIP / VTIP / SCHP** usually become more relevant
"""
        )

    st.markdown("### Quick Asset Leadership")
    if leader_6m != "-":
        st.success(f"Best performer over the last ~6 months: **{leader_6m}** ({leader_6m_perf:.2f}%)")
    else:
        st.warning("Not enough ETF price data to compute recent leadership.")

    if not norm_prices.empty:
        st.plotly_chart(
            make_line_chart(
                norm_prices,
                "Normalized ETF Performance (Start = 100)",
                yaxis_title="Indexed Price",
                log_scale=False,
                percent=False,
            ),
            use_container_width=True,
        )


# ============================================================
# Tab 2: TIPS Timing Model
# ============================================================
with tab2:
    st.markdown("## TIPS Buy/Sell Timing Logic")

    logic_col1, logic_col2 = st.columns([1, 1])

    with logic_col1:
        st.markdown("### Rule Inputs")
        rules_df = pd.DataFrame(
            {
                "Indicator": [
                    "10Y TIPS Real Yield (DFII10)",
                    "10Y Breakeven Inflation (T10YIE)",
                    "CPI YoY",
                    "10Y Treasury Yield (DGS10)",
                    "Fed Funds minus CPI",
                ],
                "Current": [
                    f"{real_yield:.2f}%" if pd.notna(real_yield) else "N/A",
                    f"{breakeven:.2f}%" if pd.notna(breakeven) else "N/A",
                    f"{cpi_yoy:.2f}%" if pd.notna(cpi_yoy) else "N/A",
                    f"{dgs10:.2f}%" if pd.notna(dgs10) else "N/A",
                    f"{(fedfunds - cpi_yoy):.2f}%" if pd.notna(fedfunds) and pd.notna(cpi_yoy) else "N/A",
                ],
                "Comment": [
                    "Higher usually improves long-term TIPS entry valuation",
                    "Too high can mean inflation is already priced in",
                    "Above target supports demand for inflation protection",
                    "Higher nominal yields can improve future bond entry levels",
                    "Restrictive policy can support bond disinflation setups later",
                ],
            }
        )
        st.dataframe(rules_df, use_container_width=True, hide_index=True)

    with logic_col2:
        st.markdown("### TIPS Decision")
        st.metric("Signal Score", f"{score:.0f}")
        st.write(f"**Action:** {signal}")
        st.write(reason)

        st.markdown("### Suggested Interpretation")
        if signal in ["Strong Buy", "Buy / Accumulate"]:
            st.write(
                """
- Favor **gradual accumulation** over a one-time purchase
- Consider splitting between **TIP** and **VTIP**
- If you expect aggressive rate cuts, also compare with **TLT**
"""
            )
        elif signal == "Neutral / Hold":
            st.write(
                """
- TIPS look reasonable, but not clearly cheap
- Use phased entry or pair them with gold / commodities
- Watch whether breakeven inflation starts rising again
"""
            )
        else:
            st.write(
                """
- Waiting may be better than chasing
- Check whether breakeven inflation is already too high
- Consider short-duration TIPS or a diversified inflation sleeve
"""
            )

    st.markdown("### Historical Percentile Context")

    pct_col1, pct_col2, pct_col3 = st.columns(3)

    with pct_col1:
        real_pct = percentile_rank(real_yield, fred_df["DFII10"]) if "DFII10" in fred_df else np.nan
        st.metric("Real Yield Percentile", f"{real_pct:.1f}%" if pd.notna(real_pct) else "N/A")

    with pct_col2:
        breakeven_pct = percentile_rank(breakeven, fred_df["T10YIE"]) if "T10YIE" in fred_df else np.nan
        st.metric("Breakeven Percentile", f"{breakeven_pct:.1f}%" if pd.notna(breakeven_pct) else "N/A")

    with pct_col3:
        cpi_pct = percentile_rank(cpi_yoy, fred_df["CPI_YOY"]) if "CPI_YOY" in fred_df else np.nan
        st.metric("CPI YoY Percentile", f"{cpi_pct:.1f}%" if pd.notna(cpi_pct) else "N/A")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if {"DFII10", "T10YIE"}.issubset(fred_df.columns):
            st.plotly_chart(
                make_line_chart(
                    fred_df[["DFII10", "T10YIE"]],
                    "Real Yield vs Breakeven Inflation",
                    yaxis_title="%",
                    percent=True,
                ),
                use_container_width=True,
            )

    with chart_col2:
        if "CPI_YOY" in fred_df.columns:
            cpi_plot = fred_df[["CPI_YOY"]].copy()
            if "FEDFUNDS" in fred_df.columns:
                cpi_plot["FEDFUNDS"] = fred_df["FEDFUNDS"]

            st.plotly_chart(
                make_line_chart(
                    cpi_plot,
                    "CPI YoY vs Fed Funds",
                    yaxis_title="%",
                    percent=True,
                ),
                use_container_width=True,
            )


# ============================================================
# Tab 3: Macro Dashboard
# ============================================================
with tab3:
    macro_left, macro_right = st.columns(2)

    with macro_left:
        cols = [c for c in ["DGS10", "DFII10", "T10YIE"] if c in fred_df.columns]
        if cols:
            st.plotly_chart(
                make_line_chart(
                    fred_df[cols],
                    "Rates Dashboard",
                    yaxis_title="%",
                    percent=True,
                ),
                use_container_width=True,
            )

    with macro_right:
        cols = [c for c in ["CPI_YOY", "FEDFUNDS", "UNRATE"] if c in fred_df.columns]
        if cols:
            st.plotly_chart(
                make_line_chart(
                    fred_df[cols],
                    "Inflation / Policy / Labor",
                    yaxis_title="%",
                    percent=True,
                ),
                use_container_width=True,
            )

    st.markdown("### Macro Regime Playbook")
    regime_table = pd.DataFrame(
        {
            "Macro Regime": [
                "Sticky inflation / restrictive policy",
                "Inflation shock",
                "Growth slowdown / disinflation",
                "Rate-cut recession scare",
                "Soft landing",
            ],
            "Likely Better Asset": [
                "TIP / VTIP / SCHP",
                "DBC / GLD / energy-sensitive assets",
                "IEF / TIP",
                "TLT",
                "IEF / quality equities",
            ],
            "Interpretation": [
                "TIPS benefit when inflation stays above target and real yields are attractive",
                "Commodities and gold often react first to inflation shocks",
                "Intermediate duration can work better than very long duration",
                "Long Treasuries can outperform if yields fall sharply",
                "Balanced bonds may outperform inflation hedges",
            ],
        }
    )
    st.dataframe(regime_table, use_container_width=True, hide_index=True)


# ============================================================
# Tab 4: ETF Relative Performance
# ============================================================
with tab4:
    if price_df.empty:
        st.warning("ETF price data could not be loaded.")
    else:
        st.markdown("## Relative ETF Performance")

        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:
            st.plotly_chart(
                make_line_chart(
                    norm_prices,
                    "Normalized Price Performance (Start = 100)",
                    yaxis_title="Indexed Price",
                ),
                use_container_width=True,
            )

        with perf_col2:
            if show_drawdown:
                st.plotly_chart(
                    make_line_chart(
                        drawdowns,
                        "Drawdown from Previous Peak",
                        yaxis_title="Drawdown (%)",
                        percent=True,
                    ),
                    use_container_width=True,
                )

        rows = []
        for ticker in price_df.columns:
            s = price_df[ticker].dropna()
            if s.empty:
                continue

            six_month_return = np.nan
            three_month_return = np.nan

            if len(s) > 126:
                six_month_return = (s.iloc[-1] / s.iloc[-126] - 1) * 100
            if len(s) > 63:
                three_month_return = (s.iloc[-1] / s.iloc[-63] - 1) * 100

            rows.append(
                {
                    "Ticker": ticker,
                    "Role": ETF_LABELS.get(ticker, ""),
                    "Annualized Return (%)": annualized_return(s),
                    "Volatility (%)": annualized_volatility(s),
                    "Max Drawdown (%)": max_drawdown_pct(s),
                    "Sharpe-like": sharpe_like(s),
                    "6M Return (%)": six_month_return,
                    "3M Return (%)": three_month_return,
                }
            )

        stats_df = pd.DataFrame(rows)
        if not stats_df.empty:
            stats_df = stats_df.sort_values("Annualized Return (%)", ascending=False).reset_index(drop=True)

            st.markdown("### Performance Table")
            st.dataframe(
                stats_df.style.format(
                    {
                        "Annualized Return (%)": "{:.2f}",
                        "Volatility (%)": "{:.2f}",
                        "Max Drawdown (%)": "{:.2f}",
                        "Sharpe-like": "{:.2f}",
                        "6M Return (%)": "{:.2f}",
                        "3M Return (%)": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

        st.markdown("### Rolling 3-Month Return")
        rolling_63 = price_df.pct_change(63) * 100
        st.plotly_chart(
            make_line_chart(
                rolling_63.dropna(how="all"),
                "Rolling 3-Month Return",
                yaxis_title="Return (%)",
                percent=True,
            ),
            use_container_width=True,
        )


# ============================================================
# Tab 5: Asset Deep Dive
# ============================================================
with tab5:
    if price_df.empty:
        st.warning("ETF price data could not be loaded.")
    else:
        selected_ticker = st.selectbox(
            "Select ETF for detailed view",
            options=list(price_df.columns),
            index=0,
            key="deep_dive_ticker",
        )

        top_left, top_right = st.columns([1.2, 1])

        with top_left:
            st.plotly_chart(
                make_single_asset_chart(price_df, selected_ticker),
                use_container_width=True,
            )

        with top_right:
            s = price_df[selected_ticker].dropna()

            st.markdown(f"## {selected_ticker}")
            st.write(f"**Role:** {ETF_LABELS.get(selected_ticker, 'Asset')}")

            if not s.empty:
                latest_price = s.iloc[-1]
                ret_1m = (s.iloc[-1] / s.iloc[-21] - 1) * 100 if len(s) > 21 else np.nan
                ret_3m = (s.iloc[-1] / s.iloc[-63] - 1) * 100 if len(s) > 63 else np.nan
                ret_6m = (s.iloc[-1] / s.iloc[-126] - 1) * 100 if len(s) > 126 else np.nan
                ret_1y = (s.iloc[-1] / s.iloc[-252] - 1) * 100 if len(s) > 252 else np.nan

                st.metric("Latest Price", f"{latest_price:.2f}")
                st.metric("1M Return", f"{ret_1m:.2f}%" if pd.notna(ret_1m) else "N/A")
                st.metric("3M Return", f"{ret_3m:.2f}%" if pd.notna(ret_3m) else "N/A")
                st.metric("6M Return", f"{ret_6m:.2f}%" if pd.notna(ret_6m) else "N/A")
                st.metric("1Y Return", f"{ret_1y:.2f}%" if pd.notna(ret_1y) else "N/A")
                st.metric("Max Drawdown", f"{max_drawdown_pct(s):.2f}%")

            st.markdown("### Commentary")
            commentary = {
                "TIP": "Broad TIPS exposure. Useful when inflation remains sticky and real yields are attractive.",
                "VTIP": "Short-duration TIPS. Usually less rate-sensitive than TIP in uncertain rate environments.",
                "SCHP": "Low-cost broad TIPS exposure, similar to TIP.",
                "TLT": "Long-duration Treasury exposure. Usually benefits more from falling yields than from inflation protection.",
                "IEF": "Intermediate Treasury exposure. Often smoother than TLT.",
                "GLD": "Gold exposure. Often useful for monetary debasement and geopolitical stress.",
                "IAU": "Lower-cost gold exposure with a role similar to GLD.",
                "DBC": "Broad commodities. Often reacts fastest during commodity-led inflation shocks.",
                "PDBC": "Commodity strategy with a similar inflation-hedging role to DBC.",
                "SPY": "US equities. Not a pure inflation hedge, but can work in nominal growth regimes.",
                "QQQ": "Growth equities. Usually weaker when real yields rise sharply.",
            }
            st.write(commentary.get(selected_ticker, "No commentary available."))

        if "TIP" in price_df.columns and selected_ticker in price_df.columns and selected_ticker != "TIP":
            relative = price_df[selected_ticker] / price_df["TIP"] * 100
            relative_df = pd.DataFrame({f"{selected_ticker} / TIP Relative Strength": relative})

            st.plotly_chart(
                make_line_chart(
                    relative_df.dropna(),
                    f"{selected_ticker} vs TIP Relative Strength",
                    yaxis_title="Relative Index",
                ),
                use_container_width=True,
            )


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    """
### Notes
- This dashboard uses **FRED** macro data and **Yahoo Finance** ETF prices.
- The **TIPS timing signal** is a practical heuristic, not investment advice.
- In practice, many macro investors combine:
  - **TIPS** for CPI-linked inflation protection
  - **TLT / IEF** for duration and recession sensitivity
  - **GLD / IAU** for currency debasement and crisis hedging
  - **DBC / PDBC** for commodity inflation shocks

### Useful FRED Series
- **DFII10**: 10-Year Treasury Inflation-Indexed Security, Constant Maturity
- **T10YIE**: 10-Year Breakeven Inflation Rate
- **DGS10**: 10-Year Treasury Constant Maturity Rate
- **CPIAUCSL**: Consumer Price Index
- **FEDFUNDS**: Effective Federal Funds Rate
"""
)
