from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd
import requests
import streamlit as st

# ============================================================
# Page
# ============================================================
st.set_page_config(
    page_title="Buffett Tracker - WhaleWisdom",
    page_icon="🐋",
    layout="wide",
)

st.title("🐋 Buffett Top 10 Tracker — WhaleWisdom Based")
st.caption("Direct HTML parsing from WhaleWisdom filer page")

# ============================================================
# Constants
# ============================================================
BASE_URL = "https://whalewisdom.com/filer/berkshire-hathaway-inc"

HEADERS = {
    "User-Agent": "Mozilla/5.0 BuffettTracker/1.0 contact: your_email@example.com",
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT = 20


# ============================================================
# Helpers
# ============================================================
def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text


def extract_available_quarters(html: str) -> List[str]:
    """
    Try to extract quarter labels like 'Q4 2025' from the HTML.
    """
    found = re.findall(r"Q[1-4]\s20\d{2}", html)
    quarters = []
    for q in found:
        if q not in quarters:
            quarters.append(q)
    return quarters


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_holdings_table(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Find the table that looks like the stock holdings table.
    Expected columns often include:
    Stock / Company Name / % of Portfolio / Shares Owned / Value Owned
    """
    for df in tables:
        x = clean_columns(df)

        col_text = " | ".join(x.columns).lower()

        if (
            ("stock" in col_text or "company name" in col_text)
            and ("% of portfolio" in col_text or "value owned" in col_text)
        ):
            return x

    return None


def normalize_holdings_table(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    x = df.copy()

    rename_map = {}
    for c in x.columns:
        lc = c.lower()

        if lc == "stock":
            rename_map[c] = "Ticker"
        elif "company name" in lc:
            rename_map[c] = "Company"
        elif "% of portfolio" in lc:
            rename_map[c] = "Weight (%)"
        elif "shares owned" in lc:
            rename_map[c] = "Shares Owned"
        elif "value owned" in lc:
            rename_map[c] = "Value Owned"
        elif "date" == lc:
            rename_map[c] = "Date"

    x = x.rename(columns=rename_map)

    desired = [c for c in ["Ticker", "Company", "Weight (%)", "Shares Owned", "Value Owned", "Date"] if c in x.columns]
    x = x[desired].copy()

    # Remove repeated header rows if any
    if "Ticker" in x.columns:
        x = x[x["Ticker"].astype(str).str.lower() != "stock"].copy()

    if "Weight (%)" in x.columns:
        x["Weight_num"] = (
            x["Weight (%)"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        x["Weight_num"] = pd.to_numeric(x["Weight_num"], errors="coerce")
        x = x.sort_values("Weight_num", ascending=False)

    x = x.head(top_n).copy()
    x.insert(0, "Rank", range(1, len(x) + 1))

    if "Weight_num" in x.columns:
        x = x.drop(columns=["Weight_num"])

    return x.reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_quarter_table(quarter: Optional[str], top_n: int) -> tuple[pd.DataFrame, str]:
    """
    Load one quarter from WhaleWisdom.
    We try:
      - base page
      - query variations
    because WhaleWisdom page behavior may vary.
    """
    candidate_urls = [BASE_URL]

    if quarter:
        q = quarter.replace(" ", "%20")
        candidate_urls.extend(
            [
                f"{BASE_URL}?quarter={q}",
                f"{BASE_URL}?q={q}",
                f"{BASE_URL}?period={q}",
            ]
        )

    last_error = ""

    for url in candidate_urls:
        try:
            html = fetch_html(url)
            tables = pd.read_html(html)
            holdings = find_holdings_table(tables)

            if holdings is not None and not holdings.empty:
                normalized = normalize_holdings_table(holdings, top_n=top_n)
                if not normalized.empty:
                    return normalized, url

        except Exception as e:
            last_error = str(e)

    return pd.DataFrame(), last_error


# ============================================================
# Main
# ============================================================
try:
    html = fetch_html(BASE_URL)
except Exception as e:
    st.error(f"Failed to load WhaleWisdom page: {e}")
    st.stop()

quarters = extract_available_quarters(html)

if not quarters:
    st.warning("Could not detect quarter labels from WhaleWisdom page. Using generic mode.")
    quarters = ["Current"]

st.sidebar.header("Settings")
top_n = st.sidebar.slider("Top N", 5, 15, 10)

default_q = quarters[:3] if len(quarters) >= 3 else quarters
selected_quarters = st.sidebar.multiselect(
    "Select up to 3 quarters",
    options=quarters,
    default=default_q,
    max_selections=3,
)

if not selected_quarters:
    st.warning("Please select at least one quarter.")
    st.stop()

results = []

for q in selected_quarters:
    with st.spinner(f"Loading {q}..."):
        table, meta = load_quarter_table(None if q == "Current" else q, top_n)
        results.append((q, table, meta))

st.subheader("Top Holdings Comparison")
cols = st.columns(len(results))

for col, (quarter, table, meta) in zip(cols, results):
    with col:
        st.markdown(f"### {quarter}")
        if table.empty:
            st.warning("Could not parse holdings table.")
            st.caption(f"Debug: {meta}")
        else:
            st.dataframe(table, use_container_width=True, hide_index=True, height=420)

st.subheader("Merged View")
merged = None

for quarter, table, meta in results:
    if table.empty:
        continue
    t = table.copy()
    t.columns = [f"{c} ({quarter})" if c != "Rank" else "Rank" for c in t.columns]
    if merged is None:
        merged = t
    else:
        merged = merged.merge(t, on="Rank", how="outer")

if merged is not None:
    st.dataframe(merged, use_container_width=True, hide_index=True)
else:
    st.info("No merged table available.")

st.subheader("Detected Quarters")
st.write(quarters)

st.caption("This version parses WhaleWisdom HTML directly, so it may break if the page layout changes.")
