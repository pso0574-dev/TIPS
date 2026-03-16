# same_as_ever_dashboard.py
# ============================================================
# Same as Ever - Investment Strategy Dashboard
# Inspired by the recurring laws of human behavior, cycles,
# uncertainty, survival, patience, and margin of safety.
#
# Run:
#   streamlit run same_as_ever_dashboard.py
#
# Install:
#   pip install streamlit yfinance pandas numpy plotly
# ============================================================

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Same as Ever Investment Dashboard",
    page_icon="📖",
    layout="wide",
)

st.title("📖 Same as Ever — Investment Strategy Dashboard")
st.caption(
    "A market interpretation framework based on recurring human behavior, "
    "tail risk, patience, survival, simplicity, and historical cycles."
)

# ============================================================
# Constants
# ============================================================
DEFAULT_TICKERS = {
    "Equity_US_Broad": "SPY",
    "Equity_US_Tech": "QQQ",
    "Long_Bond": "TLT",
    "Gold": "GLD",
    "Commodity": "DBC",
    "World_Equity": "VEU",
    "Dividend_Equity": "SCHD",
    "Short_Bond": "SHY",
    "Cash_Proxy": "BIL",
    "Bitcoin": "BTC-USD",
    "Volatility": "^VIX",
}

RISK_ASSETS = ["SPY", "QQQ", "VEU", "BTC-USD", "DBC"]
DEFENSIVE_ASSETS = ["TLT", "GLD", "SHY", "BIL", "SCHD"]

LOOKBACK_OPTIONS = {
    "1 Year": "1y",
    "2 Years": "2y",
    "3 Years": "3y",
    "5 Years": "5y",
    "10 Years": "10y",
}

# ============================================================
# Utility functions
# ============================================================
def clamp(x: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, x))


def safe_last(series: pd.Series, default=np.nan):
    return series.dropna().iloc[-1] if series.dropna().shape[0] > 0 else default


def normalize_to_100(x: float, low: float, high: float, inverse: bool = False) -> float:
    if pd.isna(x):
        return 50.0
    if high == low:
        return 50.0
    score = (x - low) / (high - low) * 100.0
    if inverse:
        score = 100.0 - score
    return clamp(score)


def percentile_rank(value: float, history: pd.Series, inverse: bool = False) -> float:
    hist = history.dropna()
    if len(hist) < 10 or pd.isna(value):
        return 50.0
    pct = (hist <= value).mean() * 100.0
    if inverse:
        pct = 100.0 - pct
    return clamp(pct)


# ============================================================
# Data layer
# ============================================================
@st.cache_data(show_spinner=False)
def download_prices(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    frames = []
    for t in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                s = df[(t, "Close")].rename(t)
            else:
                s = df["Close"].rename(t)
            frames.append(s)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    return out.dropna(how="all")


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return price_df.pct_change().replace([np.inf, -np.inf], np.nan)


def rolling_drawdown(price_series: pd.Series) -> pd.Series:
    rolling_max = price_series.cummax()
    dd = price_series / rolling_max - 1.0
    return dd


def annualized_volatility(return_series: pd.Series, window: int = 63) -> pd.Series:
    return return_series.rolling(window).std() * np.sqrt(252)


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def max_drawdown(series: pd.Series) -> float:
    dd = rolling_drawdown(series)
    return float(dd.min()) if dd.dropna().shape[0] else 0.0


def trailing_return(series: pd.Series, n_days: int) -> float:
    s = series.dropna()
    if len(s) <= n_days:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-n_days - 1] - 1.0)


def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


# ============================================================
# Law mapping
# ============================================================
LAW_TEXT = {
    1: "People become greedy. Bubbles are normal, not exceptions.",
    2: "People become fearful. Selloffs and crashes always return.",
    3: "The future is driven by surprising events. Black swans matter.",
    4: "Compounding comes from time, not constant action.",
    5: "Luck is part of success. Do not over-credit skill.",
    6: "Luck is part of failure. Do not judge only by outcomes.",
    7: "Crowds can be wrong. Consensus is not truth.",
    8: "Big events come from small probabilities. Tail risk matters.",
    9: "Forecasting is weak. Response systems matter more.",
    10: "Most outcomes come from a few events. Power laws dominate.",
    11: "People believe stories. Narrative can move prices.",
    12: "Short-term noise hides long-term trends.",
    13: "Patience is one of the best strategies.",
    14: "Markets exaggerate. Boom and panic overshoot reality.",
    15: "Hindsight bias distorts interpretation of the past.",
    16: "Margin of safety matters.",
    17: "Systems break. Fragility repeats.",
    18: "Money should buy freedom, not only status.",
    19: "Survival is the highest rule.",
    20: "Probability is not intuitive.",
    21: "Simple strategies survive longer than complex ones.",
    22: "Human behavior changes less than technology.",
    23: "History matters because cycles repeat through behavior.",
}

LAW_GROUPS = {
    "Psychology": [1, 2, 7, 11, 14, 22],
    "Tail Risk": [3, 8, 9, 17, 20],
    "Compounding": [4, 12, 13, 21],
    "Luck & Power Law": [5, 6, 10],
    "Survival": [16, 18, 19],
    "History": [15, 23],
}

# ============================================================
# Scoring engines
# ============================================================
@dataclass
class EngineScores:
    psychology: float
    tail_risk: float
    compounding: float
    power_law: float
    survival: float
    history: float

    def total(self) -> float:
        return (
            0.22 * self.psychology
            + 0.20 * self.tail_risk
            + 0.20 * self.compounding
            + 0.12 * self.power_law
            + 0.18 * self.survival
            + 0.08 * self.history
        )


def psychology_engine(prices: pd.DataFrame, rets: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    spy = prices["SPY"].dropna()
    qqq = prices["QQQ"].dropna()
    vix = prices["^VIX"].dropna() if "^VIX" in prices.columns else pd.Series(dtype=float)
    btc = prices["BTC-USD"].dropna() if "BTC-USD" in prices.columns else pd.Series(dtype=float)

    spy_rsi = safe_last(compute_rsi(spy, 14), 50.0)
    qqq_rsi = safe_last(compute_rsi(qqq, 14), 50.0)
    vix_now = safe_last(vix, 20.0)
    spy_dd = safe_last(rolling_drawdown(spy), 0.0)
    qqq_dd = safe_last(rolling_drawdown(qqq), 0.0)
    btc_6m = trailing_return(btc, 126) if len(btc) > 200 else np.nan

    # Greed indicators
    greed_from_rsi = clamp((0.6 * spy_rsi + 0.4 * qqq_rsi))
    greed_from_low_vix = normalize_to_100(vix_now, 12, 35, inverse=True)
    greed_from_btc = normalize_to_100(btc_6m if not pd.isna(btc_6m) else 0.0, -0.4, 1.0)

    # Fear indicators
    fear_from_vix = normalize_to_100(vix_now, 12, 45, inverse=False)
    fear_from_drawdown = normalize_to_100(abs(min(spy_dd, qqq_dd)), 0.0, 0.35)
    fear_from_rsi = 100 - greed_from_rsi

    greed = 0.45 * greed_from_rsi + 0.30 * greed_from_low_vix + 0.25 * greed_from_btc
    fear = 0.45 * fear_from_vix + 0.35 * fear_from_drawdown + 0.20 * fear_from_rsi

    # Book-consistent interpretation:
    # score is higher when psychology is balanced or fear gives opportunity.
    # score is lower when greed/euphoria is excessive.
    psychology_score = clamp(
        100
        - 0.75 * max(greed - 70, 0)
        - 0.20 * max(fear - 85, 0)
        + 0.25 * max(fear - 60, 0)
    )

    details = {
        "greed_score": round(greed, 1),
        "fear_score": round(fear, 1),
        "spy_rsi": round(float(spy_rsi), 1),
        "qqq_rsi": round(float(qqq_rsi), 1),
        "vix": round(float(vix_now), 2),
        "spy_drawdown_pct": round(float(spy_dd * 100), 2),
        "qqq_drawdown_pct": round(float(qqq_dd * 100), 2),
    }
    return psychology_score, details


def tail_risk_engine(prices: pd.DataFrame, rets: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    spy = prices["SPY"].dropna()
    qqq = prices["QQQ"].dropna()
    tlt = prices["TLT"].dropna() if "TLT" in prices.columns else pd.Series(dtype=float)
    vix = prices["^VIX"].dropna() if "^VIX" in prices.columns else pd.Series(dtype=float)

    spy_ret = rets["SPY"].dropna()
    qqq_ret = rets["QQQ"].dropna()

    spy_vol = safe_last(annualized_volatility(spy_ret, 63), 0.15)
    qqq_vol = safe_last(annualized_volatility(qqq_ret, 63), 0.20)
    vix_now = safe_last(vix, 20.0)

    corr_window = rets[["SPY", "QQQ", "TLT"]].dropna().tail(63)
    corr_avg = corr_window.corr().abs().mean().mean() if len(corr_window) > 20 else 0.4

    recent_tail_moves = (
        (spy_ret.tail(63) < -0.02).sum() + (qqq_ret.tail(63) < -0.025).sum()
        if len(spy_ret) >= 63 and len(qqq_ret) >= 63
        else 0
    )

    risk_pressure = (
        0.35 * normalize_to_100(spy_vol, 0.08, 0.35)
        + 0.25 * normalize_to_100(qqq_vol, 0.10, 0.45)
        + 0.25 * normalize_to_100(vix_now, 12, 45)
        + 0.15 * normalize_to_100(corr_avg, 0.2, 0.9)
    )
    tail_event_pressure = normalize_to_100(recent_tail_moves, 0, 15)

    # High engine score should mean "tail risk is well-controlled"
    tail_risk_score = clamp(100 - (0.75 * risk_pressure + 0.25 * tail_event_pressure))

    details = {
        "spy_vol_63d_pct": round(float(spy_vol * 100), 2),
        "qqq_vol_63d_pct": round(float(qqq_vol * 100), 2),
        "vix": round(float(vix_now), 2),
        "avg_abs_corr_63d": round(float(corr_avg), 3),
        "recent_tail_move_count": int(recent_tail_moves),
        "risk_pressure": round(float(risk_pressure), 1),
    }
    return tail_risk_score, details


def compounding_engine(prices: pd.DataFrame, rets: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    spy = prices["SPY"].dropna()
    qqq = prices["QQQ"].dropna()

    spy_ma50 = safe_last(moving_average(spy, 50), np.nan)
    spy_ma200 = safe_last(moving_average(spy, 200), np.nan)
    qqq_ma50 = safe_last(moving_average(qqq, 50), np.nan)
    qqq_ma200 = safe_last(moving_average(qqq, 200), np.nan)

    spy_trend = 100 if safe_last(spy, 0) > spy_ma200 else 35
    qqq_trend = 100 if safe_last(qqq, 0) > qqq_ma200 else 30
    spy_short_vs_long = 100 if spy_ma50 > spy_ma200 else 40
    qqq_short_vs_long = 100 if qqq_ma50 > qqq_ma200 else 40

    turnover_penalty_proxy = abs(safe_last(compute_rsi(spy, 14), 50) - 50) * 0.5
    noise_penalty = turnover_penalty_proxy

    score = clamp(
        0.30 * spy_trend
        + 0.25 * qqq_trend
        + 0.20 * spy_short_vs_long
        + 0.20 * qqq_short_vs_long
        + 0.05 * (100 - noise_penalty)
    )

    details = {
        "spy_above_200ma": int(safe_last(spy, 0) > spy_ma200) if not pd.isna(spy_ma200) else 0,
        "qqq_above_200ma": int(safe_last(qqq, 0) > qqq_ma200) if not pd.isna(qqq_ma200) else 0,
        "spy_ma50_gt_ma200": int(spy_ma50 > spy_ma200) if not pd.isna(spy_ma50) and not pd.isna(spy_ma200) else 0,
        "qqq_ma50_gt_ma200": int(qqq_ma50 > qqq_ma200) if not pd.isna(qqq_ma50) and not pd.isna(qqq_ma200) else 0,
        "noise_penalty": round(float(noise_penalty), 1),
    }
    return score, details


def power_law_engine(prices: pd.DataFrame, rets: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    # The idea: market returns are often driven by few outsized winners.
    # Higher score when winners still exist but the market is not fully euphoric.
    assets = [c for c in ["SPY", "QQQ", "GLD", "DBC", "TLT", "BTC-USD", "SCHD"] if c in prices.columns]
    one_year_returns = {}
    for c in assets:
        one_year_returns[c] = trailing_return(prices[c].dropna(), 252)

    vals = pd.Series(one_year_returns).dropna()
    if len(vals) == 0:
        return 50.0, {"dispersion": 0.0, "top_return_pct": 0.0}

    dispersion = vals.std()
    top_ret = vals.max()
    bottom_ret = vals.min()

    # Healthy concentration exists, but too much can imply crowding
    dispersion_score = normalize_to_100(dispersion, 0.05, 0.50)
    crowding_penalty = normalize_to_100(top_ret, 0.10, 1.20)
    score = clamp(0.65 * dispersion_score + 0.35 * (100 - 0.45 * crowding_penalty))

    details = {
        "dispersion_1y": round(float(dispersion), 3),
        "top_return_pct": round(float(top_ret * 100), 2),
        "bottom_return_pct": round(float(bottom_ret * 100), 2),
    }
    return score, details


def survival_engine(prices: pd.DataFrame, rets: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    spy = prices["SPY"].dropna()
    qqq = prices["QQQ"].dropna()
    gld = prices["GLD"].dropna() if "GLD" in prices.columns else pd.Series(dtype=float)
    tlt = prices["TLT"].dropna() if "TLT" in prices.columns else pd.Series(dtype=float)
    bil = prices["BIL"].dropna() if "BIL" in prices.columns else pd.Series(dtype=float)

    spy_mdd = abs(max_drawdown(spy))
    qqq_mdd = abs(max_drawdown(qqq))
    tlt_mdd = abs(max_drawdown(tlt)) if len(tlt) > 100 else 0.10
    gld_mdd = abs(max_drawdown(gld)) if len(gld) > 100 else 0.10

    diversification = 0.0
    comps = []
    for c in ["SPY", "QQQ", "TLT", "GLD", "BIL"]:
        if c in rets.columns:
            comps.append(c)
    if len(comps) >= 3:
        corr = rets[comps].dropna().tail(252).corr().abs()
        diversification = 1.0 - float(corr.where(~np.eye(corr.shape[0], dtype=bool)).mean().mean())
    else:
        diversification = 0.4

    safety_score = (
        0.30 * normalize_to_100(spy_mdd, 0.10, 0.60, inverse=True)
        + 0.25 * normalize_to_100(qqq_mdd, 0.15, 0.75, inverse=True)
        + 0.20 * normalize_to_100(tlt_mdd, 0.05, 0.50, inverse=True)
        + 0.10 * normalize_to_100(gld_mdd, 0.05, 0.35, inverse=True)
        + 0.15 * normalize_to_100(diversification, 0.1, 0.8)
    )
    score = clamp(safety_score)

    details = {
        "spy_mdd_pct": round(float(spy_mdd * 100), 2),
        "qqq_mdd_pct": round(float(qqq_mdd * 100), 2),
        "tlt_mdd_pct": round(float(tlt_mdd * 100), 2),
        "gld_mdd_pct": round(float(gld_mdd * 100), 2),
        "diversification_score_raw": round(float(diversification), 3),
    }
    return score, details


def history_engine(prices: pd.DataFrame, rets: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    spy = prices["SPY"].dropna()
    qqq = prices["QQQ"].dropna()
    vix = prices["^VIX"].dropna() if "^VIX" in prices.columns else pd.Series(dtype=float)

    # Compare current state vs trailing 5y range
    spy_dd = rolling_drawdown(spy)
    qqq_dd = rolling_drawdown(qqq)
    vix_hist = vix.tail(1260) if len(vix) > 1260 else vix

    curr_spy_dd = safe_last(spy_dd, 0.0)
    curr_qqq_dd = safe_last(qqq_dd, 0.0)
    curr_vix = safe_last(vix, 20.0)

    spy_dd_pct_rank = percentile_rank(curr_spy_dd, spy_dd.tail(1260), inverse=False)
    qqq_dd_pct_rank = percentile_rank(curr_qqq_dd, qqq_dd.tail(1260), inverse=False)
    vix_pct_rank = percentile_rank(curr_vix, vix_hist, inverse=False)

    # High score means "history suggests current conditions are manageable / not extreme"
    extremeness = 0.4 * vix_pct_rank + 0.3 * (100 - spy_dd_pct_rank) + 0.3 * (100 - qqq_dd_pct_rank)
    score = clamp(100 - 0.7 * extremeness + 20)

    details = {
        "spy_drawdown_percentile": round(float(spy_dd_pct_rank), 1),
        "qqq_drawdown_percentile": round(float(qqq_dd_pct_rank), 1),
        "vix_percentile": round(float(vix_pct_rank), 1),
        "historical_extremeness": round(float(extremeness), 1),
    }
    return score, details


# ============================================================
# Regime classification
# ============================================================
def classify_market_regime(scores: EngineScores, psych_details: Dict[str, float], tail_details: Dict[str, float]) -> str:
    greed = psych_details["greed_score"]
    fear = psych_details["fear_score"]
    vix = psych_details["vix"]
    risk_pressure = tail_details["risk_pressure"]

    total = scores.total()

    if greed > 75 and scores.psychology < 55 and scores.tail_risk < 55:
        return "Euphoric Bubble"
    if fear > 75 and vix > 28 and total >= 45:
        return "Panic / Capitulation"
    if scores.tail_risk < 40 and risk_pressure > 70:
        return "Structural Risk Regime"
    if scores.compounding >= 65 and scores.survival >= 60 and greed < 75:
        return "Healthy Bull"
    if fear > 55 and scores.compounding >= 50 and scores.tail_risk >= 45:
        return "Fragile Recovery"
    return "Neutral / Transitional"


# ============================================================
# Portfolio recommendation
# ============================================================
def recommend_allocation(regime: str, scores: EngineScores) -> Dict[str, int]:
    if regime == "Euphoric Bubble":
        alloc = {"Risk Assets": 40, "Defensive Assets": 30, "Cash": 25, "Optional Opportunistic": 5}
    elif regime == "Panic / Capitulation":
        alloc = {"Risk Assets": 50, "Defensive Assets": 20, "Cash": 20, "Optional Opportunistic": 10}
    elif regime == "Structural Risk Regime":
        alloc = {"Risk Assets": 25, "Defensive Assets": 40, "Cash": 30, "Optional Opportunistic": 5}
    elif regime == "Healthy Bull":
        alloc = {"Risk Assets": 65, "Defensive Assets": 15, "Cash": 10, "Optional Opportunistic": 10}
    elif regime == "Fragile Recovery":
        alloc = {"Risk Assets": 50, "Defensive Assets": 25, "Cash": 15, "Optional Opportunistic": 10}
    else:
        alloc = {"Risk Assets": 45, "Defensive Assets": 25, "Cash": 20, "Optional Opportunistic": 10}

    # Survival override
    if scores.survival < 45:
        alloc["Cash"] = min(35, alloc["Cash"] + 10)
        alloc["Risk Assets"] = max(20, alloc["Risk Assets"] - 10)

    # Tail-risk override
    if scores.tail_risk < 40:
        alloc["Cash"] = min(40, alloc["Cash"] + 5)
        alloc["Defensive Assets"] = min(45, alloc["Defensive Assets"] + 5)
        alloc["Risk Assets"] = max(15, alloc["Risk Assets"] - 10)

    return alloc


def recommend_actions(regime: str, scores: EngineScores, psych_details: Dict[str, float]) -> List[str]:
    actions = []

    if regime == "Euphoric Bubble":
        actions += [
            "Trim stretched winners gradually.",
            "Avoid aggressive chasing of high-momentum narratives.",
            "Raise cash buffer and favor quality / resilient assets.",
        ]
    elif regime == "Panic / Capitulation":
        actions += [
            "Start staged buying rather than all-in timing.",
            "Focus on durable assets and strong balance-sheet proxies.",
            "Do not use leverage into volatility spikes.",
        ]
    elif regime == "Structural Risk Regime":
        actions += [
            "Prioritize survival over return maximization.",
            "Reduce fragility and concentration risk.",
            "Prefer cash, short duration, gold, and selective defensive exposure.",
        ]
    elif regime == "Healthy Bull":
        actions += [
            "Stay invested and let compounding work.",
            "Avoid unnecessary trading driven by short-term noise.",
            "Rebalance lightly rather than frequently.",
        ]
    elif regime == "Fragile Recovery":
        actions += [
            "Increase risk exposure gradually, not aggressively.",
            "Keep some dry powder in case volatility returns.",
            "Favor broad diversification over narrow thematic bets.",
        ]
    else:
        actions += [
            "Maintain neutral positioning.",
            "Wait for either better margin of safety or trend confirmation.",
            "Keep the process simple and consistent.",
        ]

    if psych_details["greed_score"] > 80:
        actions.append("Narrative heat is elevated; do not confuse excitement with durability.")
    if psych_details["fear_score"] > 80:
        actions.append("Fear is extreme; use a rules-based plan to avoid emotional selling.")

    if scores.compounding > 70:
        actions.append("Long-term trend is supportive; patience is a feature, not inactivity.")
    if scores.survival < 45:
        actions.append("Survival score is weak; cap position sizing and avoid concentrated bets.")

    return actions


# ============================================================
# Explanation layer
# ============================================================
def build_law_based_commentary(
    scores: EngineScores,
    psych_details: Dict[str, float],
    tail_details: Dict[str, float],
    comp_details: Dict[str, float],
    power_details: Dict[str, float],
    survival_details: Dict[str, float],
    hist_details: Dict[str, float],
) -> pd.DataFrame:
    rows = []

    rows.append({
        "Law #": 1,
        "Law": "Greed returns",
        "Reading": f"Greed score is {psych_details['greed_score']}. High greed usually means expectations are running ahead of resilience.",
        "Implication": "Reduce chasing behavior when greed is extreme.",
    })
    rows.append({
        "Law #": 2,
        "Law": "Fear returns",
        "Reading": f"Fear score is {psych_details['fear_score']}. Fear often creates opportunity but also raises volatility.",
        "Implication": "Use staged entries, not heroic single-entry timing.",
    })
    rows.append({
        "Law #": 3,
        "Law": "Black swans matter",
        "Reading": f"Tail-risk pressure is {tail_details['risk_pressure']}. Unexpected shocks should always be assumed possible.",
        "Implication": "Hold optionality via cash and diversification.",
    })
    rows.append({
        "Law #": 4,
        "Law": "Compounding needs time",
        "Reading": f"Compounding score is {scores.compounding:.1f}. Long-term trend and patience matter more than activity.",
        "Implication": "Favor durable exposure over frequent trading.",
    })
    rows.append({
        "Law #": 7,
        "Law": "Crowds can be wrong",
        "Reading": f"Greed {psych_details['greed_score']} and VIX {psych_details['vix']} help identify whether consensus is complacent or fearful.",
        "Implication": "When narratives become too one-sided, expect mean reversion risk.",
    })
    rows.append({
        "Law #": 8,
        "Law": "Tail risks are real",
        "Reading": f"Recent tail-move count is {tail_details['recent_tail_move_count']}, and average correlation is {tail_details['avg_abs_corr_63d']}.",
        "Implication": "When correlations rise, diversification weakens right when it is most needed.",
    })
    rows.append({
        "Law #": 10,
        "Law": "Few events drive most outcomes",
        "Reading": f"1-year cross-asset dispersion is {power_details['dispersion_1y']}. Return concentration remains important.",
        "Implication": "Allow winners to matter, but do not let concentration become fragility.",
    })
    rows.append({
        "Law #": 12,
        "Law": "Noise hides trend",
        "Reading": f"SPY above 200MA: {comp_details['spy_above_200ma']}, QQQ above 200MA: {comp_details['qqq_above_200ma']}.",
        "Implication": "Use structure, not headlines, to judge direction.",
    })
    rows.append({
        "Law #": 16,
        "Law": "Margin of safety matters",
        "Reading": f"Survival score is {scores.survival:.1f}. Historical MDD remains a reminder of what can go wrong.",
        "Implication": "Position size should reflect downside tolerance, not upside excitement.",
    })
    rows.append({
        "Law #": 19,
        "Law": "Survival first",
        "Reading": f"SPY MDD {survival_details['spy_mdd_pct']}%, QQQ MDD {survival_details['qqq_mdd_pct']}%.",
        "Implication": "Avoid strategies that fail before compounding can work.",
    })
    rows.append({
        "Law #": 21,
        "Law": "Simple strategies last longer",
        "Reading": "This system uses a few durable signals instead of many unstable predictors.",
        "Implication": "A robust process often beats an elegant but fragile model.",
    })
    rows.append({
        "Law #": 23,
        "Law": "History rhymes through behavior",
        "Reading": f"Historical extremeness score is {hist_details['historical_extremeness']}.",
        "Implication": "Today is never identical to the past, but human reactions often repeat.",
    })

    return pd.DataFrame(rows)


# ============================================================
# Charts
# ============================================================
def make_price_chart(prices: pd.DataFrame, tickers: List[str], title: str) -> go.Figure:
    fig = go.Figure()
    norm = prices[tickers].dropna()
    if norm.empty:
        return fig
    norm = norm / norm.iloc[0] * 100

    for c in norm.columns:
        fig.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm[c],
                mode="lines",
                name=c,
            )
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500,
        xaxis_title="Date",
        yaxis_title="Indexed (Start = 100)",
        legend_title="Asset",
    )
    return fig


def make_drawdown_chart(prices: pd.DataFrame, tickers: List[str], title: str) -> go.Figure:
    fig = go.Figure()
    for c in tickers:
        if c in prices.columns:
            dd = rolling_drawdown(prices[c].dropna()) * 100
            fig.add_trace(
                go.Scatter(
                    x=dd.index,
                    y=dd,
                    mode="lines",
                    name=c,
                )
            )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=450,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        legend_title="Asset",
    )
    return fig


def make_engine_score_bar(scores: EngineScores) -> go.Figure:
    items = {
        "Psychology": scores.psychology,
        "Tail Risk": scores.tail_risk,
        "Compounding": scores.compounding,
        "Power Law": scores.power_law,
        "Survival": scores.survival,
        "History": scores.history,
        "Total": scores.total(),
    }
    fig = go.Figure(
        go.Bar(
            x=list(items.keys()),
            y=list(items.values()),
            text=[f"{v:.1f}" for v in items.values()],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Engine Scores",
        template="plotly_white",
        height=420,
        yaxis_title="Score (0-100)",
    )
    return fig


# ============================================================
# Main render
# ============================================================
def main():
    with st.sidebar:
        st.header("Settings")
        selected_period_label = st.selectbox("Lookback Period", list(LOOKBACK_OPTIONS.keys()), index=3)
        selected_period = LOOKBACK_OPTIONS[selected_period_label]

        extra_tickers = st.text_input(
            "Add Extra Tickers (comma-separated)",
            value="",
            help="Example: IWM,EFA,IEF,XLK",
        )

        show_law_table = st.checkbox("Show law-based commentary table", value=True)
        show_raw_data = st.checkbox("Show raw return table", value=False)

    tickers = list(DEFAULT_TICKERS.values())
    if extra_tickers.strip():
        more = [x.strip().upper() for x in extra_tickers.split(",") if x.strip()]
        tickers = list(dict.fromkeys(tickers + more))

    prices = download_prices(tickers, period=selected_period)

    if prices.empty or "SPY" not in prices.columns or "QQQ" not in prices.columns:
        st.error("Price download failed or required core tickers are missing.")
        return

    rets = compute_returns(prices)

    psychology_score, psych_details = psychology_engine(prices, rets)
    tail_risk_score, tail_details = tail_risk_engine(prices, rets)
    compounding_score, comp_details = compounding_engine(prices, rets)
    power_law_score, power_details = power_law_engine(prices, rets)
    survival_score, survival_details = survival_engine(prices, rets)
    history_score, hist_details = history_engine(prices, rets)

    scores = EngineScores(
        psychology=psychology_score,
        tail_risk=tail_risk_score,
        compounding=compounding_score,
        power_law=power_law_score,
        survival=survival_score,
        history=history_score,
    )

    regime = classify_market_regime(scores, psych_details, tail_details)
    alloc = recommend_allocation(regime, scores)
    actions = recommend_actions(regime, scores, psych_details)
    commentary_df = build_law_based_commentary(
        scores, psych_details, tail_details, comp_details,
        power_details, survival_details, hist_details
    )

    # ========================================================
    # Summary metrics
    # ========================================================
    st.subheader("1) Market Interpretation Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Regime", regime)
    c2.metric("Total Score", f"{scores.total():.1f} / 100")
    c3.metric("Greed / Fear", f"{psych_details['greed_score']:.1f} / {psych_details['fear_score']:.1f}")
    c4.metric("VIX", f"{psych_details['vix']:.2f}")

    # ========================================================
    # Engine score cards
    # ========================================================
    st.subheader("2) Core Engine Scores")
    s1, s2, s3 = st.columns(3)
    s1.metric("Psychology", f"{scores.psychology:.1f}")
    s1.metric("Compounding", f"{scores.compounding:.1f}")
    s2.metric("Tail Risk", f"{scores.tail_risk:.1f}")
    s2.metric("Power Law", f"{scores.power_law:.1f}")
    s3.metric("Survival", f"{scores.survival:.1f}")
    s3.metric("History", f"{scores.history:.1f}")

    st.plotly_chart(make_engine_score_bar(scores), use_container_width=True)

    # ========================================================
    # Price and drawdown charts
    # ========================================================
    st.subheader("3) Multi-Asset Context")
    chart_tickers = [c for c in ["SPY", "QQQ", "TLT", "GLD", "DBC", "BTC-USD"] if c in prices.columns]
    st.plotly_chart(
        make_price_chart(prices, chart_tickers, "Cross-Asset Relative Performance"),
        use_container_width=True
    )
    st.plotly_chart(
        make_drawdown_chart(prices, chart_tickers, "Cross-Asset Drawdown"),
        use_container_width=True
    )

    # ========================================================
    # Book reflection panel
    # ========================================================
    st.subheader("4) Same as Ever Reading of the Market")

    left, right = st.columns(2)

    with left:
        st.markdown("### Human Behavior")
        st.write(f"- **Greed score:** {psych_details['greed_score']}")
        st.write(f"- **Fear score:** {psych_details['fear_score']}")
        st.write(f"- **SPY RSI:** {psych_details['spy_rsi']}")
        st.write(f"- **QQQ RSI:** {psych_details['qqq_rsi']}")
        st.write(f"- **SPY drawdown:** {psych_details['spy_drawdown_pct']}%")
        st.write(f"- **QQQ drawdown:** {psych_details['qqq_drawdown_pct']}%")

        st.markdown("### Uncertainty / Tail Risk")
        st.write(f"- **Risk pressure:** {tail_details['risk_pressure']}")
        st.write(f"- **SPY vol (63d):** {tail_details['spy_vol_63d_pct']}%")
        st.write(f"- **QQQ vol (63d):** {tail_details['qqq_vol_63d_pct']}%")
        st.write(f"- **Average abs correlation (63d):** {tail_details['avg_abs_corr_63d']}")
        st.write(f"- **Tail move count:** {tail_details['recent_tail_move_count']}")

    with right:
        st.markdown("### Compounding / Patience")
        st.write(f"- **SPY above 200MA:** {bool(comp_details['spy_above_200ma'])}")
        st.write(f"- **QQQ above 200MA:** {bool(comp_details['qqq_above_200ma'])}")
        st.write(f"- **SPY 50MA > 200MA:** {bool(comp_details['spy_ma50_gt_ma200'])}")
        st.write(f"- **QQQ 50MA > 200MA:** {bool(comp_details['qqq_ma50_gt_ma200'])}")

        st.markdown("### Survival / History")
        st.write(f"- **SPY max drawdown:** {survival_details['spy_mdd_pct']}%")
        st.write(f"- **QQQ max drawdown:** {survival_details['qqq_mdd_pct']}%")
        st.write(f"- **Historical extremeness:** {hist_details['historical_extremeness']}")
        st.write(f"- **VIX historical percentile:** {hist_details['vix_percentile']}")

    # ========================================================
    # Allocation panel
    # ========================================================
    st.subheader("5) Strategy Recommendation")
    a1, a2 = st.columns([1, 1])

    with a1:
        st.markdown("### Suggested Allocation")
        alloc_df = pd.DataFrame({
            "Bucket": list(alloc.keys()),
            "Weight (%)": list(alloc.values()),
        })
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)

        st.markdown("### Suggested Mapping")
        st.write("- **Risk Assets:** SPY / QQQ / VEU / DBC / selective BTC")
        st.write("- **Defensive Assets:** TLT / GLD / SHY / SCHD")
        st.write("- **Cash:** BIL or unallocated cash")
        st.write("- **Optional Opportunistic:** deep selloff entries, special situations")

    with a2:
        st.markdown("### Action Rules")
        for i, action in enumerate(actions, 1):
            st.write(f"{i}. {action}")

    # ========================================================
    # Law table
    # ========================================================
    if show_law_table:
        st.subheader("6) Law-Based Commentary")
        st.dataframe(commentary_df, use_container_width=True, hide_index=True)

    # ========================================================
    # Raw data
    # ========================================================
    if show_raw_data:
        st.subheader("7) Raw Return Snapshot")
        snapshot = []
        for c in chart_tickers:
            s = prices[c].dropna()
            snapshot.append({
                "Ticker": c,
                "1M Return (%)": round(trailing_return(s, 21) * 100, 2) if len(s) > 30 else np.nan,
                "3M Return (%)": round(trailing_return(s, 63) * 100, 2) if len(s) > 80 else np.nan,
                "6M Return (%)": round(trailing_return(s, 126) * 100, 2) if len(s) > 140 else np.nan,
                "1Y Return (%)": round(trailing_return(s, 252) * 100, 2) if len(s) > 260 else np.nan,
                "Max Drawdown (%)": round(max_drawdown(s) * 100, 2),
            })
        st.dataframe(pd.DataFrame(snapshot), use_container_width=True, hide_index=True)

    # ========================================================
    # Philosophy panel
    # ========================================================
    st.subheader("7) Philosophy Behind This System")
    st.write(
        "This dashboard is deliberately designed to reflect recurring human behavior more than precise forecasting. "
        "It assumes greed, fear, overreaction, fragility, and cycles are permanent features of markets. "
        "The goal is not to perfectly predict returns, but to improve survival, patience, and decision quality."
    )

    st.info(
        "Important: this is a framework for interpretation and discipline, not personalized investment advice. "
        "It simplifies complex reality and should be combined with valuation, tax, liquidity, and personal risk constraints."
    )


if __name__ == "__main__":
    main()
