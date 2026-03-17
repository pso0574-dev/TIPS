# streamlit_app.py
from __future__ import annotations

from io import StringIO
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Rates vs Equities Monitor",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Rates vs Equities Monitor")
st.caption("Representative rates vs SPY / QQQ with preset-range synced chart updates")

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

RECESSIONS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

RANGE_OPTIONS = {
    "3M": 90,
    "6M": 180,
    "1Y": 365,
    "3Y": 365 * 3,
    "5Y": 365 * 5,
    "10Y": 365 * 10,
    "20Y": 365 * 20,
    "Max": None,
}


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return s
    base = valid.iloc[0]
    if pd.isna(base) or base == 0:
        return s
    return s / base * 100.0


def pct_change_from_start(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return s
    base = valid.iloc[0]
    if pd.isna(base) or base == 0:
        return s
    return (s / base - 1.0) * 100.0


def compute_drawdown(price: pd.Series) -> pd.Series:
    price = pd.to_numeric(price, errors="coerce")
    running_max = price.cummax()
    return (price / running_max - 1.0) * 100.0


def format_num(x, pct: bool = False, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}%" if pct else f"{x:.{digits}f}"


def latest_value_and_change(series: pd.Series) -> Tuple[float | None, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None, None
    latest = s.iloc[-1]
    delta = latest - s.iloc[-2] if len(s) >= 2 else None
    return latest, delta


def ytd_return(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    s = df[["Date", col]].dropna().copy()
    if s.empty:
        return None
    this_year = pd.Timestamp.today().year
    s = s[s["Date"].dt.year == this_year]
    if len(s) < 2:
        return None
    start = s[col].iloc[0]
    end = s[col].iloc[-1]
    if start == 0 or pd.isna(start):
        return None
    return (end / start - 1.0) * 100.0


def first_valid_ffill(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    first_idx = s.first_valid_index()
    if first_idx is None:
        return s
    out = s.copy()
    out.loc[first_idx:] = out.loc[first_idx:].ffill()
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def load_fred_series(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    df.columns = ["Date", series_id]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
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


@st.cache_data(show_spinner=False, ttl=3600)
def load_yf_close(ticker: str, start: str = "1990-01-01") -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame(columns=["Date", ticker])

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            close = df[("Close", ticker)]
        else:
            close = df.xs("Close", axis=1, level=0).iloc[:, 0]
    else:
        close = df["Close"]

    out = pd.DataFrame({
        "Date": pd.to_datetime(close.index),
        ticker: pd.to_numeric(close.values, errors="coerce"),
    })
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def load_all_data() -> pd.DataFrame:
    fred_ids = list(set(AVAILABLE_MAIN_RATE_SERIES + ["BAMLC0A0CM", "BAMLC0A4CBBB"]))
    fred_df = load_multiple_fred_series(fred_ids)

    spy_df = load_yf_close("SPY", start="1993-01-01")
    qqq_df = load_yf_close("QQQ", start="1999-03-10")

    merged = fred_df.merge(spy_df, on="Date", how="outer")
    merged = merged.merge(qqq_df, on="Date", how="outer")
    merged = merged.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    if merged.empty:
        return merged

    calendar = pd.DataFrame({
        "Date": pd.date_range(start=merged["Date"].min(), end=merged["Date"].max(), freq="B")
    })
    merged = calendar.merge(merged, on="Date", how="left")

    for col in merged.columns:
        if col != "Date":
            merged[col] = first_valid_ffill(merged[col])

    return merged


def filter_by_date(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    return df.loc[mask].copy()


def add_crisis_shading(fig: go.Figure, show_labels: bool = True) -> None:
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
                yref="paper",
                text=crisis["label"],
                showarrow=False,
                font=dict(size=10),
            )


def add_recession_shading(fig: go.Figure) -> None:
    for start, end in RECESSIONS:
        fig.add_vrect(
            x0=pd.to_datetime(start),
            x1=pd.to_datetime(end),
            fillcolor="lightgray",
            opacity=0.12,
            line_width=0,
            layer="below",
        )


def apply_mode(series: pd.Series, mode: str) -> pd.Series:
    if mode == "Raw":
        return pd.to_numeric(series, errors="coerce")
    if mode == "Normalized (Start=100)":
        return normalize_series(series)
    if mode == "Change % from Start":
        return pct_change_from_start(series)
    return pd.to_numeric(series, errors="coerce")


def build_main_chart(
    df: pd.DataFrame,
    rate_cols: List[str],
    equity_cols: List[str],
    mode: str,
    show_crisis: bool,
    show_crisis_labels: bool,
    show_recession: bool,
    ma_windows: List[int],
    show_ma: bool,
) -> go.Figure:
    fig = go.Figure()
    use_secondary_y = (mode == "Raw" and len(rate_cols) > 0 and len(equity_cols) > 0)

    for col in rate_cols:
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df["Date"], y=apply_mode(df[col], mode),
            mode="lines", name=col, connectgaps=False
        ))

    for col in equity_cols:
        if col not in df.columns:
            continue

        trace = go.Scatter(
            x=df["Date"], y=apply_mode(df[col], mode),
            mode="lines", name=col, connectgaps=False
        )
        if use_secondary_y:
            trace.yaxis = "y2"
        fig.add_trace(trace)

        if show_ma and mode == "Raw":
            price = pd.to_numeric(df[col], errors="coerce")
            for w in ma_windows:
                ma_trace = go.Scatter(
                    x=df["Date"],
                    y=price.rolling(w).mean(),
                    mode="lines",
                    name=f"{col} MA{w}",
                    line=dict(dash="dot"),
                    connectgaps=False,
                )
                if use_secondary_y:
                    ma_trace.yaxis = "y2"
                fig.add_trace(ma_trace)

    fig.update_layout(
        title="Representative Rates vs SPY / QQQ",
        hovermode="x unified",
        height=560,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )

    if use_secondary_y:
        fig.update_layout(
            yaxis=dict(title="Rates / Spread"),
            yaxis2=dict(title="Equities", overlaying="y", side="right", showgrid=False),
        )
    else:
        fig.update_yaxes(title_text=mode)

    if show_crisis:
        add_crisis_shading(fig, show_crisis_labels)
    if show_recession:
        add_recession_shading(fig)

    return fig


def build_simple_chart(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    mode: str,
    show_crisis: bool,
    show_crisis_labels: bool,
    show_recession: bool,
) -> go.Figure:
    fig = go.Figure()

    for col in columns:
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=apply_mode(df[col], mode),
            mode="lines",
            name=col,
            connectgaps=False
        ))

    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=460,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )

    if show_crisis:
        add_crisis_shading(fig, show_crisis_labels)
    if show_recession:
        add_recession_shading(fig)

    return fig


# -----------------------------
# Load data
# -----------------------------
with st.spinner("Loading data..."):
    data = load_all_data()

if data.empty:
    st.error("Data could not be loaded.")
    st.stop()

min_date = data["Date"].min().date()
max_date = data["Date"].max().date()

# -----------------------------
# Session-state synced preset range
# -----------------------------
if "preset_range" not in st.session_state:
    st.session_state["preset_range"] = "5Y"

if "start_date" not in st.session_state:
    st.session_state["start_date"] = max(min_date, max_date - timedelta(days=RANGE_OPTIONS["5Y"]))

if "end_date" not in st.session_state:
    st.session_state["end_date"] = max_date


def sync_dates_from_preset():
    preset = st.session_state["preset_range"]
    if RANGE_OPTIONS[preset] is None:
        st.session_state["start_date"] = min_date
        st.session_state["end_date"] = max_date
    else:
        st.session_state["end_date"] = max_date
        st.session_state["start_date"] = max(min_date, max_date - timedelta(days=RANGE_OPTIONS[preset]))


st.sidebar.header("Controls")

st.sidebar.selectbox(
    "Preset Range",
    list(RANGE_OPTIONS.keys()),
    key="preset_range",
    on_change=sync_dates_from_preset,
)

st.sidebar.date_input(
    "Start Date",
    min_value=min_date,
    max_value=max_date,
    key="start_date",
)

st.sidebar.date_input(
    "End Date",
    min_value=min_date,
    max_value=max_date,
    key="end_date",
)

if st.session_state["start_date"] > st.session_state["end_date"]:
    st.error("Start Date must be before End Date.")
    st.stop()

selected_rate_series = st.sidebar.multiselect(
    "Representative Rates",
    options=AVAILABLE_MAIN_RATE_SERIES,
    default=DEFAULT_RATE_SERIES,
    format_func=lambda x: f"{x} — {FRED_SERIES_META.get(x, {}).get('name', x)}"
)

selected_equities = st.sidebar.multiselect(
    "Compare with Equities",
    options=["SPY", "QQQ"],
    default=["SPY", "QQQ"]
)

chart_mode = st.sidebar.radio(
    "Chart Mode",
    ["Raw", "Normalized (Start=100)", "Change % from Start"],
    index=1
)

show_crisis = st.sidebar.checkbox("Show Crisis Periods", value=True)
show_crisis_labels = st.sidebar.checkbox("Show Crisis Labels", value=True)
show_recession = st.sidebar.checkbox("Show Recession Shading", value=True)

show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
ma_windows = st.sidebar.multiselect("MA Windows", [20, 50, 100, 200], default=[50, 200])

show_drawdown = st.sidebar.checkbox("Show Drawdown Chart", value=True)

filtered = filter_by_date(
    data,
    st.session_state["start_date"],
    st.session_state["end_date"]
)

# -----------------------------
# Snapshot
# -----------------------------
st.subheader("Snapshot")

metric_cols = st.columns(6)

v_dgs10, d_dgs10 = latest_value_and_change(filtered["DGS10"]) if "DGS10" in filtered.columns else (None, None)
v_dgs2, d_dgs2 = latest_value_and_change(filtered["DGS2"]) if "DGS2" in filtered.columns else (None, None)
v_dff, d_dff = latest_value_and_change(filtered["DFF"]) if "DFF" in filtered.columns else (None, None)

spread_10_2 = None
if "DGS10" in filtered.columns and "DGS2" in filtered.columns:
    tmp = pd.to_numeric(filtered["DGS10"], errors="coerce") - pd.to_numeric(filtered["DGS2"], errors="coerce")
    if not tmp.dropna().empty:
        spread_10_2 = tmp.dropna().iloc[-1]

metric_cols[0].metric("10Y Yield", format_num(v_dgs10, pct=True), format_num(d_dgs10, pct=True))
metric_cols[1].metric("2Y Yield", format_num(v_dgs2, pct=True), format_num(d_dgs2, pct=True))
metric_cols[2].metric("Fed Funds", format_num(v_dff, pct=True), format_num(d_dff, pct=True))
metric_cols[3].metric("10Y-2Y Spread", format_num(spread_10_2, pct=True))
metric_cols[4].metric("SPY YTD", format_num(ytd_return(filtered, "SPY"), pct=True))
metric_cols[5].metric("QQQ YTD", format_num(ytd_return(filtered, "QQQ"), pct=True))

tab_main, tab_curve, tab_inflation, tab_credit, tab_table = st.tabs(
    ["Main View", "Yield Curve", "Inflation / Real Rates", "Credit Stress", "Data Table"]
)

with tab_main:
    st.subheader("Main Comparison")

    fig_main = build_main_chart(
        df=filtered,
        rate_cols=selected_rate_series,
        equity_cols=selected_equities,
        mode=chart_mode,
        show_crisis=show_crisis,
        show_crisis_labels=show_crisis_labels,
        show_recession=show_recession,
        ma_windows=ma_windows,
        show_ma=show_ma,
    )
    st.plotly_chart(fig_main, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        if selected_rate_series:
            fig_rates = build_simple_chart(
                filtered,
                selected_rate_series,
                "Rates Only",
                "Raw",
                show_crisis,
                show_crisis_labels,
                show_recession,
            )
            st.plotly_chart(fig_rates, use_container_width=True)

    with col_b:
        if selected_equities:
            fig_eq = build_simple_chart(
                filtered,
                selected_equities,
                "Equities Only",
                "Raw",
                show_crisis,
                show_crisis_labels,
                show_recession,
            )

            if show_ma:
                for eq in selected_equities:
                    if eq in filtered.columns:
                        price = pd.to_numeric(filtered[eq], errors="coerce")
                        for w in ma_windows:
                            fig_eq.add_trace(go.Scatter(
                                x=filtered["Date"],
                                y=price.rolling(w).mean(),
                                mode="lines",
                                name=f"{eq} MA{w}",
                                line=dict(dash="dot"),
                                connectgaps=False,
                            ))
            st.plotly_chart(fig_eq, use_container_width=True)

    if show_drawdown and selected_equities:
        dd_fig = go.Figure()
        for eq in selected_equities:
            if eq in filtered.columns:
                dd_fig.add_trace(go.Scatter(
                    x=filtered["Date"],
                    y=compute_drawdown(filtered[eq]),
                    mode="lines",
                    name=f"{eq} Drawdown",
                    connectgaps=False,
                ))
        dd_fig.update_layout(
            title="Equity Drawdown vs Previous Peak",
            hovermode="x unified",
            height=380,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        if show_crisis:
            add_crisis_shading(dd_fig, show_crisis_labels)
        if show_recession:
            add_recession_shading(dd_fig)
        st.plotly_chart(dd_fig, use_container_width=True)

with tab_curve:
    st.subheader("Yield Curve")

    curve_df = filtered.copy()
    curve_df["2s10s_calc"] = pd.to_numeric(curve_df["DGS10"], errors="coerce") - pd.to_numeric(curve_df["DGS2"], errors="coerce")
    curve_df["3m10y_calc"] = pd.to_numeric(curve_df["DGS10"], errors="coerce") - pd.to_numeric(curve_df["DGS3MO"], errors="coerce")

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            build_simple_chart(curve_df, ["DGS3MO", "DGS2", "DGS10", "DGS30"], "Treasury Yield Levels", "Raw",
                               show_crisis, show_crisis_labels, show_recession),
            use_container_width=True
        )

    with c2:
        fig_curve_spreads = build_simple_chart(
            curve_df,
            ["2s10s_calc", "3m10y_calc", "T10Y2Y", "T10Y3M"],
            "Curve Spreads",
            "Raw",
            show_crisis,
            show_crisis_labels,
            show_recession,
        )
        fig_curve_spreads.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_curve_spreads, use_container_width=True)

with tab_inflation:
    st.subheader("Inflation / Real Rates")

    i1, i2 = st.columns(2)

    with i1:
        st.plotly_chart(
            build_simple_chart(filtered, ["DGS10", "DFII10", "T10YIE"], "10Y Nominal / Real / Breakeven", "Raw",
                               show_crisis, show_crisis_labels, show_recession),
            use_container_width=True
        )

    with i2:
        fig_real_vs_qqq = build_main_chart(
            df=filtered,
            rate_cols=["DFII10"],
            equity_cols=["QQQ"] if "QQQ" in filtered.columns else [],
            mode=chart_mode,
            show_crisis=show_crisis,
            show_crisis_labels=show_crisis_labels,
            show_recession=show_recession,
            ma_windows=ma_windows,
            show_ma=show_ma,
        )
        fig_real_vs_qqq.update_layout(title="10Y Real Yield vs QQQ")
        st.plotly_chart(fig_real_vs_qqq, use_container_width=True)

with tab_credit:
    st.subheader("Credit Stress")

    cr1, cr2 = st.columns(2)

    with cr1:
        st.plotly_chart(
            build_simple_chart(filtered, ["BAMLC0A0CM", "BAMLC0A4CBBB"], "Corporate Credit Spreads (OAS)", "Raw",
                               show_crisis, show_crisis_labels, show_recession),
            use_container_width=True
        )

    with cr2:
        fig_credit_vs_eq = build_main_chart(
            df=filtered,
            rate_cols=["BAMLC0A4CBBB"],
            equity_cols=[c for c in ["SPY", "QQQ"] if c in filtered.columns],
            mode=chart_mode,
            show_crisis=show_crisis,
            show_crisis_labels=show_crisis_labels,
            show_recession=show_recession,
            ma_windows=ma_windows,
            show_ma=show_ma,
        )
        fig_credit_vs_eq.update_layout(title="BBB OAS vs SPY / QQQ")
        st.plotly_chart(fig_credit_vs_eq, use_container_width=True)

with tab_table:
    st.subheader("Filtered Data")

    table_cols = ["Date"] + list(dict.fromkeys(
        selected_rate_series + selected_equities + ["T10Y2Y", "T10Y3M", "DFII10", "T10YIE", "BAMLC0A4CBBB"]
    ))
    table_cols = [c for c in table_cols if c in filtered.columns or c == "Date"]

    st.dataframe(
        filtered[table_cols].sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered[table_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="rates_vs_equities_filtered.csv",
        mime="text/csv",
    )
