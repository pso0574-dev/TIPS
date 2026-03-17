# streamlit_app.py
# ============================================================
# Rates vs Equities Monitor
# - Main tab: compare representative rates with S&P 500 / QQQ
# - Adjustable time range
# - Crisis period shading
# - Moving average options
# - Additional tabs for curve, inflation, credit, and data table
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit pandas numpy plotly requests yfinance
#
# Notes:
# - FRED data is loaded with the public CSV endpoint (no API key required)
# - QQQ is loaded from yfinance
# ============================================================

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ============================================================
# Streamlit page config
# ============================================================
st.set_page_config(
    page_title="Rates vs Equities Monitor",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Rates vs Equities Monitor")
st.caption(
    "Compare representative FRED rates with S&P 500 / QQQ, including crisis markers, "
    "range controls, normalization, and moving averages."
)

# ============================================================
# Constants
# ============================================================
FRED_SERIES_META: Dict[str, Dict[str, str]] = {
    "DFF": {"name": "Effective Fed Funds Rate", "group": "Policy"},
    "DFEDTARU": {"name": "Fed Target Upper Bound", "group": "Policy"},
    "DFEDTARL": {"name": "Fed Target Lower Bound", "group": "Policy"},
    "DGS3MO": {"name": "3M Treasury Yield", "group": "Treasury"},
    "DGS2": {"name": "2Y Treasury Yield", "group": "Treasury"},
    "DGS5": {"name": "5Y Treasury Yield", "group": "Treasury"},
    "DGS10": {"name": "10Y Treasury Yield", "group": "Treasury"},
    "DGS30": {"name": "30Y Treasury Yield", "group": "Treasury"},
    "DFII10": {"name": "10Y TIPS Real Yield", "group": "Real Yield"},
    "T10YIE": {"name": "10Y Breakeven Inflation", "group": "Inflation"},
    "T10Y2Y": {"name": "10Y-2Y Spread", "group": "Curve"},
    "T10Y3M": {"name": "10Y-3M Spread", "group": "Curve"},
    "BAMLC0A0CM": {"name": "US Corporate Master OAS", "group": "Credit"},
    "BAMLC0A4CBBB": {"name": "BBB OAS", "group": "Credit"},
    "SP500": {"name": "S&P 500 Index (FRED)", "group": "Equity"},
    "UNRATE": {"name": "Unemployment Rate", "group": "Macro"},
    "CPIAUCSL": {"name": "CPI All Urban Consumers", "group": "Macro"},
}

DEFAULT_RATE_SERIES = ["DFF", "DGS2", "DGS10", "DGS30"]
AVAILABLE_MAIN_RATE_SERIES = [
    "DFF", "DFEDTARU", "DFEDTARL",
    "DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30",
    "DFII10", "T10YIE",
    "T10Y2Y", "T10Y3M",
]

CRISIS_PERIODS = [
    {"label": "Dot-com Bust", "start": "2000-03-10", "end": "2002-10-09"},
    {"label": "GFC", "start": "2007-10-09", "end": "2009-03-09"},
    {"label": "Euro Debt Stress", "start": "2011-07-01", "end": "2012-06-30"},
    {"label": "COVID Crash", "start": "2020-02-19", "end": "2020-04-30"},
    {"label": "2022 Tightening Shock", "start": "2022-01-03", "end": "2022-10-14"},
    {"label": "2023 Regional Bank Stress", "start": "2023-03-08", "end": "2023-05-15"},
]

RANGE_OPTIONS = {
    "3M": 90,
    "6M": 180,
    "1Y": 365,
    "3Y": 365 * 3,
    "5Y": 365 * 5,
    "10Y": 365 * 10,
    "Max": None,
}

RECESSIONS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

# ============================================================
# Helpers
# ============================================================
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_fred_series(series_id: str) -> pd.DataFrame:
    """
    Load a single FRED series using the public CSV download endpoint.
    No API key required.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(pd.compat.StringIO(response.text)) if hasattr(pd, "compat") and hasattr(pd.compat, "StringIO") else None
    if df is None:
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))

    if df.shape[1] < 2:
        raise ValueError(f"Unexpected FRED format for {series_id}")

    df.columns = ["Date", series_id]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_multiple_fred_series(series_ids: List[str]) -> pd.DataFrame:
    merged = None
    for sid in series_ids:
        try:
            df = load_fred_series(sid)
            merged = df if merged is None else merged.merge(df, on="Date", how="outer")
        except Exception as e:
            st.warning(f"Failed to load {sid}: {e}")
    if merged is None:
        return pd.DataFrame(columns=["Date"])
    return merged.sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_yfinance_close(ticker: str, start: str = "1990-01-01") -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame(columns=["Date", ticker])
    close_col = "Close"
    if isinstance(df.columns, pd.MultiIndex):
        close_col = ("Close", ticker) if ("Close", ticker) in df.columns else df.columns[0]
        series = df[close_col]
    else:
        series = df["Close"]
    out = pd.DataFrame({"Date": pd.to_datetime(series.index), ticker: pd.to_numeric(series.values, errors="coerce")})
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def merge_main_data() -> pd.DataFrame:
    fred_ids = list(set(AVAILABLE_MAIN_RATE_SERIES + ["SP500", "BAMLC0A4CBBB", "BAMLC0A0CM"]))
    fred_df = load_multiple_fred_series(fred_ids)
    qqq_df = load_yfinance_close("QQQ")
    merged = fred_df.merge(qqq_df, on="Date", how="outer")
    merged = merged.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    return merged


def filter_by_date(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    return df.loc[mask].copy()


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    first_valid = s.dropna()
    if first_valid.empty:
        return s
    base = first_valid.iloc[0]
    if base == 0 or pd.isna(base):
        return s
    return s / base * 100.0


def pct_change_from_start(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    first_valid = s.dropna()
    if first_valid.empty:
        return s
    base = first_valid.iloc[0]
    if base == 0 or pd.isna(base):
        return s
    return (s / base - 1.0) * 100.0


def compute_drawdown(price: pd.Series) -> pd.Series:
    price = pd.to_numeric(price, errors="coerce")
    running_max = price.cummax()
    return (price / running_max - 1.0) * 100.0


def add_crisis_shading(fig: go.Figure, show_labels: bool, yref: str = "paper") -> go.Figure:
    for crisis in CRISIS_PERIODS:
        x0 = pd.to_datetime(crisis["start"])
        x1 = pd.to_datetime(crisis["end"])
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="gray",
            opacity=0.14,
            line_width=0,
            layer="below",
        )
        if show_labels:
            fig.add_annotation(
                x=x0 + (x1 - x0) / 2,
                y=1.02,
                xref="x",
                yref=yref,
                text=crisis["label"],
                showarrow=False,
                font=dict(size=11),
            )
    return fig


def add_recession_shading(fig: go.Figure) -> go.Figure:
    for start, end in RECESSIONS:
        fig.add_vrect(
            x0=pd.to_datetime(start),
            x1=pd.to_datetime(end),
            fillcolor="lightgray",
            opacity=0.18,
            line_width=0,
            layer="below",
        )
    return fig


def ma_label(window: int) -> str:
    return f"MA{window}"


def build_line_chart(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    chart_mode: str,
    show_crisis: bool,
    show_crisis_labels: bool,
    show_recession: bool,
    use_secondary_y: bool = False,
    secondary_y_columns: List[str] | None = None,
) -> go.Figure:
    fig = go.Figure()
    secondary_y_columns = secondary_y_columns or []

    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")

        if chart_mode == "Raw":
            y = series
            y_name = col
        elif chart_mode == "Normalized (Start=100)":
            y = normalize_series(series)
            y_name = f"{col} (Norm)"
        elif chart_mode == "Change % from Start":
            y = pct_change_from_start(series)
            y_name = f"{col} (% from start)"
        else:
            y = series
            y_name = col

        trace_kwargs = dict(
            x=df["Date"],
            y=y,
            mode="lines",
            name=y_name,
            connectgaps=False,
        )

        if use_secondary_y and col in secondary_y_columns:
            trace_kwargs["yaxis"] = "y2"

        fig.add_trace(go.Scatter(**trace_kwargs))

    fig.update_layout(
        title=title,
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    if use_secondary_y:
        fig.update_layout(
            yaxis=dict(title="Rates / Spread"),
            yaxis2=dict(
                title="Equities",
                overlaying="y",
                side="right",
                showgrid=False,
            )
        )

    if show_crisis:
        fig = add_crisis_shading(fig, show_crisis_labels)
    if show_recession:
        fig = add_recession_shading(fig)

    return fig


def add_equity_mas(
    fig: go.Figure,
    df: pd.DataFrame,
    ticker: str,
    windows: List[int],
    use_secondary_axis: bool = False,
) -> go.Figure:
    if ticker not in df.columns:
        return fig

    price = pd.to_numeric(df[ticker], errors="coerce")
    for w in windows:
        ma = price.rolling(w).mean()
        trace_kwargs = dict(
            x=df["Date"],
            y=ma,
            mode="lines",
            name=f"{ticker} {ma_label(w)}",
            line=dict(dash="dot"),
        )
        if use_secondary_axis:
            trace_kwargs["yaxis"] = "y2"
        fig.add_trace(go.Scatter(**trace_kwargs))
    return fig


def latest_value_and_change(series: pd.Series) -> Tuple[float | None, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None, None
    latest = s.iloc[-1]
    if len(s) >= 2:
        prev = s.iloc[-2]
        delta = latest - prev
    else:
        delta = None
    return latest, delta


def ytd_return(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    s = df[["Date", col]].dropna().copy()
    if s.empty:
        return None
    year = pd.Timestamp.today().year
    s = s[s["Date"].dt.year == year]
    if len(s) < 2:
        return None
    start = s[col].iloc[0]
    end = s[col].iloc[-1]
    if start == 0:
        return None
    return (end / start - 1) * 100.0


def format_num(x, pct: bool = False, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}%" if pct else f"{x:.{digits}f}"


# ============================================================
# Data load
# ============================================================
with st.spinner("Loading FRED and market data..."):
    data = merge_main_data()

if data.empty:
    st.error("Failed to load data.")
    st.stop()

data = data.sort_values("Date").reset_index(drop=True)

# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Controls")

preset_range = st.sidebar.selectbox(
    "Preset Range",
    list(RANGE_OPTIONS.keys()),
    index=4,
    key="preset_range_main",
)

min_date = data["Date"].min().date()
max_date = data["Date"].max().date()

if RANGE_OPTIONS[preset_range] is None:
    default_start = min_date
else:
    default_start = max(min_date, (max_date - timedelta(days=RANGE_OPTIONS[preset_range])))

start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start,
    min_value=min_date,
    max_value=max_date,
    key="start_date_main",
)

end_date = st.sidebar.date_input(
    "End Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
    key="end_date_main",
)

if start_date > end_date:
    st.sidebar.error("Start Date must be earlier than or equal to End Date.")
    st.stop()

selected_rate_series = st.sidebar.multiselect(
    "Representative Rates",
    options=AVAILABLE_MAIN_RATE_SERIES,
    default=DEFAULT_RATE_SERIES,
    format_func=lambda x: f"{x} — {FRED_SERIES_META.get(x, {}).get('name', x)}",
    key="rate_series_main",
)

selected_equities = st.sidebar.multiselect(
    "Compare with Equities",
    options=["SP500", "QQQ"],
    default=["SP500", "QQQ"],
    format_func=lambda x: "S&P 500" if x == "SP500" else "QQQ",
    key="equity_select_main",
)

main_chart_mode = st.sidebar.radio(
    "Main Chart Mode",
    ["Raw", "Normalized (Start=100)", "Change % from Start"],
    index=1,
    key="main_chart_mode",
)

show_crisis = st.sidebar.checkbox("Show Crisis Periods", value=True, key="show_crisis_main")
show_crisis_labels = st.sidebar.checkbox("Show Crisis Labels", value=True, key="show_crisis_labels_main")
show_recession = st.sidebar.checkbox("Show Recession Shading", value=True, key="show_recession_main")

show_ma = st.sidebar.checkbox("Show Moving Averages for Equities", value=True, key="show_ma_main")
selected_ma_windows = st.sidebar.multiselect(
    "Moving Average Windows",
    options=[20, 50, 100, 200],
    default=[50, 200],
    key="ma_windows_main",
)

show_drawdown = st.sidebar.checkbox("Show Equity Drawdown Chart", value=True, key="show_drawdown_main")
show_data_table = st.sidebar.checkbox("Show Filtered Data Table", value=False, key="show_table_main")

filtered = filter_by_date(data, start_date, end_date).copy()

# Forward-fill some lower-frequency series for easier comparison in charts
for col in filtered.columns:
    if col != "Date":
        filtered[col] = filtered[col].ffill()

# ============================================================
# Snapshot metrics
# ============================================================
st.subheader("Snapshot")

col1, col2, col3, col4, col5, col6 = st.columns(6)

v_dgs10, d_dgs10 = latest_value_and_change(filtered["DGS10"]) if "DGS10" in filtered.columns else (None, None)
v_dgs2, d_dgs2 = latest_value_and_change(filtered["DGS2"]) if "DGS2" in filtered.columns else (None, None)
v_dff, d_dff = latest_value_and_change(filtered["DFF"]) if "DFF" in filtered.columns else (None, None)
v_sp500, _ = latest_value_and_change(filtered["SP500"]) if "SP500" in filtered.columns else (None, None)
v_qqq, _ = latest_value_and_change(filtered["QQQ"]) if "QQQ" in filtered.columns else (None, None)

spread_10_2 = None
if "DGS10" in filtered.columns and "DGS2" in filtered.columns:
    tmp = pd.to_numeric(filtered["DGS10"], errors="coerce") - pd.to_numeric(filtered["DGS2"], errors="coerce")
    spread_10_2 = tmp.dropna().iloc[-1] if not tmp.dropna().empty else None

col1.metric("10Y Yield", format_num(v_dgs10, pct=True), delta=format_num(d_dgs10, pct=True))
col2.metric("2Y Yield", format_num(v_dgs2, pct=True), delta=format_num(d_dgs2, pct=True))
col3.metric("Fed Funds", format_num(v_dff, pct=True), delta=format_num(d_dff, pct=True))
col4.metric("10Y-2Y Spread", format_num(spread_10_2, pct=True))
col5.metric("S&P 500 YTD", format_num(ytd_return(filtered, "SP500"), pct=True))
col6.metric("QQQ YTD", format_num(ytd_return(filtered, "QQQ"), pct=True))

# ============================================================
# Tabs
# ============================================================
tab_main, tab_curve, tab_inflation, tab_credit, tab_table = st.tabs(
    ["Main View", "Yield Curve", "Inflation / Real Rates", "Credit Stress", "Data Table"]
)

# ============================================================
# Main View
# ============================================================
with tab_main:
    st.subheader("Representative Rates vs S&P 500 / QQQ")

    if not selected_rate_series and not selected_equities:
        st.warning("Please select at least one rate or one equity series.")
    else:
        main_cols = selected_rate_series + selected_equities

        use_secondary = main_chart_mode == "Raw" and len(selected_rate_series) > 0 and len(selected_equities) > 0
        fig_main = build_line_chart(
            df=filtered,
            columns=main_cols,
            title="Main Comparison",
            chart_mode=main_chart_mode,
            show_crisis=show_crisis,
            show_crisis_labels=show_crisis_labels,
            show_recession=show_recession,
            use_secondary_y=use_secondary,
            secondary_y_columns=selected_equities if use_secondary else [],
        )

        if show_ma and selected_equities:
            for eq in selected_equities:
                fig_main = add_equity_mas(
                    fig=fig_main,
                    df=filtered,
                    ticker=eq,
                    windows=selected_ma_windows,
                    use_secondary_axis=use_secondary,
                )

        st.plotly_chart(fig_main, use_container_width=True, key="plot_main_comparison")

    st.markdown("### Separate Views")

    left, right = st.columns(2)

    with left:
        if selected_rate_series:
            fig_rates = build_line_chart(
                df=filtered,
                columns=selected_rate_series,
                title="Rates Only",
                chart_mode="Raw",
                show_crisis=show_crisis,
                show_crisis_labels=show_crisis_labels,
                show_recession=show_recession,
                use_secondary_y=False,
            )
            st.plotly_chart(fig_rates, use_container_width=True, key="plot_rates_only")
        else:
            st.info("No rate series selected.")

    with right:
        eq_to_show = selected_equities if selected_equities else []
        if eq_to_show:
            fig_eq = build_line_chart(
                df=filtered,
                columns=eq_to_show,
                title="Equities Only",
                chart_mode="Raw",
                show_crisis=show_crisis,
                show_crisis_labels=show_crisis_labels,
                show_recession=show_recession,
                use_secondary_y=False,
            )
            if show_ma:
                for eq in eq_to_show:
                    fig_eq = add_equity_mas(fig_eq, filtered, eq, selected_ma_windows, use_secondary_axis=False)
            st.plotly_chart(fig_eq, use_container_width=True, key="plot_equities_only")
        else:
            st.info("No equity series selected.")

    if show_drawdown and selected_equities:
        st.markdown("### Equity Drawdown")
        dd_fig = go.Figure()
        for eq in selected_equities:
            if eq in filtered.columns:
                dd = compute_drawdown(filtered[eq])
                dd_fig.add_trace(
                    go.Scatter(
                        x=filtered["Date"],
                        y=dd,
                        mode="lines",
                        name=f"{eq} Drawdown",
                    )
                )
        dd_fig.update_layout(
            title="Drawdown vs Previous Peak",
            height=380,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        if show_crisis:
            dd_fig = add_crisis_shading(dd_fig, show_crisis_labels)
        if show_recession:
            dd_fig = add_recession_shading(dd_fig)
        st.plotly_chart(dd_fig, use_container_width=True, key="plot_drawdown")

# ============================================================
# Yield Curve
# ============================================================
with tab_curve:
    st.subheader("Yield Curve / Inversion Monitor")

    curve_df = filtered.copy()
    curve_df["2s10s"] = pd.to_numeric(curve_df["DGS10"], errors="coerce") - pd.to_numeric(curve_df["DGS2"], errors="coerce")
    curve_df["3m10y"] = pd.to_numeric(curve_df["DGS10"], errors="coerce") - pd.to_numeric(curve_df["DGS3MO"], errors="coerce")

    c1, c2 = st.columns(2)

    with c1:
        fig_curve_levels = build_line_chart(
            df=curve_df,
            columns=["DGS3MO", "DGS2", "DGS10", "DGS30"],
            title="Treasury Yield Levels",
            chart_mode="Raw",
            show_crisis=show_crisis,
            show_crisis_labels=show_crisis_labels,
            show_recession=show_recession,
        )
        st.plotly_chart(fig_curve_levels, use_container_width=True, key="plot_curve_levels")

    with c2:
        fig_curve_spread = build_line_chart(
            df=curve_df,
            columns=["2s10s", "3m10y", "T10Y2Y", "T10Y3M"],
            title="Curve Spreads / Inversion",
            chart_mode="Raw",
            show_crisis=show_crisis,
            show_crisis_labels=show_crisis_labels,
            show_recession=show_recession,
        )
        fig_curve_spread.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_curve_spread, use_container_width=True, key="plot_curve_spreads")

# ============================================================
# Inflation / Real Rates
# ============================================================
with tab_inflation:
    st.subheader("Inflation Expectations and Real Rates")

    inf1, inf2 = st.columns(2)

    with inf1:
        fig_inf = build_line_chart(
            df=filtered,
            columns=["DGS10", "DFII10", "T10YIE"],
            title="10Y Nominal / Real / Breakeven",
            chart_mode="Raw",
            show_crisis=show_crisis,
            show_crisis_labels=show_crisis_labels,
            show_recession=show_recession,
        )
        st.plotly_chart(fig_inf, use_container_width=True, key="plot_inflation_rates")

    with inf2:
        fig_real_vs_qqq = build_line_chart(
            df=filtered,
            columns=["DFII10", "QQQ"],
            title="10Y Real Yield vs QQQ",
            chart_mode=main_chart_mode,
            show_crisis=show_crisis,
            show_crisis_labels=show_crisis_labels,
            show_recession=show_recession,
            use_secondary_y=(main_chart_mode == "Raw"),
            secondary_y_columns=["QQQ"] if main_chart_mode == "Raw" else [],
        )
        if show_ma and "QQQ" in selected_equities:
            fig_real_vs_qqq = add_equity_mas(
                fig_real_vs_qqq,
                filtered,
                "QQQ",
                selected_ma_windows,
                use_secondary_axis=(main_chart_mode == "Raw"),
            )
        st.plotly_chart(fig_real_vs_qqq, use_container_width=True, key="plot_real_vs_qqq")

# ============================================================
# Credit Stress
# ============================================================
with tab_credit:
    st.subheader("Credit Stress Monitor")

    cr1, cr2 = st.columns(2)

    with cr1:
        fig_credit = build_line_chart(
            df=filtered,
            columns=["BAMLC0A0CM", "BAMLC0A4CBBB"],
            title="Corporate Credit Spreads (OAS)",
            chart_mode="Raw",
            show_crisis=show_crisis,
            show_crisis_labels=show_crisis_labels,
            show_recession=show_recession,
        )
        st.plotly_chart(fig_credit, use_container_width=True, key="plot_credit_oas")

    with cr2:
        fig_credit_equity = build_line_chart(
            df=filtered,
            columns=["BAMLC0A4CBBB", "SP500", "QQQ"],
            title="BBB OAS vs Equities",
            chart_mode=main_chart_mode,
            show_crisis=show_crisis,
            show_crisis_labels=show_crisis_labels,
            show_recession=show_recession,
            use_secondary_y=(main_chart_mode == "Raw"),
            secondary_y_columns=["SP500", "QQQ"] if main_chart_mode == "Raw" else [],
        )
        if show_ma:
            for eq in [e for e in ["SP500", "QQQ"] if e in selected_equities]:
                fig_credit_equity = add_equity_mas(
                    fig_credit_equity,
                    filtered,
                    eq,
                    selected_ma_windows,
                    use_secondary_axis=(main_chart_mode == "Raw"),
                )
        st.plotly_chart(fig_credit_equity, use_container_width=True, key="plot_credit_vs_equities")

# ============================================================
# Data Table
# ============================================================
with tab_table:
    st.subheader("Filtered Data Table")

    display_cols = ["Date"] + [c for c in (
        selected_rate_series + ["SP500", "QQQ", "T10Y2Y", "T10Y3M", "DFII10", "T10YIE", "BAMLC0A4CBBB"]
    ) if c in filtered.columns]
    display_cols = list(dict.fromkeys(display_cols))

    st.dataframe(
        filtered[display_cols].sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
        key="dataframe_filtered",
    )

    csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="rates_vs_equities_filtered.csv",
        mime="text/csv",
        key="download_csv_filtered",
    )

# ============================================================
# Optional table below page
# ============================================================
if show_data_table:
    st.subheader("Quick Filtered Preview")
    st.dataframe(filtered.tail(50), use_container_width=True, hide_index=True, key="quick_preview_table")
