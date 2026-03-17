from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from lxml import etree

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Buffett Tracker (Free)",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Buffett Tracker (Free Version)")
st.caption("Official SEC 13F + free price charts via Yahoo Finance")

# ============================================================
# Constants
# ============================================================
BERKSHIRE_CIK = "0001067983"
SEC_SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{BERKSHIRE_CIK}.json"

# IMPORTANT:
# SEC asks for a descriptive User-Agent containing contact info.
# Replace with your real email if you use this heavily.
SEC_HEADERS = {
    "User-Agent": "SangohPark BuffettTracker/1.0 contact@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

REQUEST_SLEEP = 0.2


# ============================================================
# Helpers
# ============================================================
@dataclass
class FilingInfo:
    accession_number: str
    filing_date: str
    report_date: str
    primary_doc: str
    accession_nodash: str


def safe_get(url: str, headers: dict, timeout: int = 30) -> requests.Response:
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    time.sleep(REQUEST_SLEEP)
    return r


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_submissions_json() -> dict:
    r = safe_get(SEC_SUBMISSIONS_URL, SEC_HEADERS)
    return r.json()


def get_recent_13f_filings(submissions: dict, max_count: int = 8) -> List[FilingInfo]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_docs = recent.get("primaryDocument", [])

    filings: List[FilingInfo] = []

    for form, acc, fdate, rdate, pdoc in zip(
        forms, accession_numbers, filing_dates, report_dates, primary_docs
    ):
        if form == "13F-HR":
            filings.append(
                FilingInfo(
                    accession_number=acc,
                    filing_date=fdate,
                    report_date=rdate,
                    primary_doc=pdoc,
                    accession_nodash=acc.replace("-", ""),
                )
            )

    return filings[:max_count]


def filing_index_url(filing: FilingInfo) -> str:
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(BERKSHIRE_CIK)}/{filing.accession_nodash}/"
    )


def filing_primary_xml_url(filing: FilingInfo) -> str:
    # Often primary doc is xml/html; kept for reference if needed
    return filing_index_url(filing) + filing.primary_doc


@st.cache_data(show_spinner=False, ttl=60 * 60)
def discover_information_table_xml(filing: FilingInfo) -> Optional[str]:
    """
    Find the XML file for the Information Table in a filing folder.
    """
    index_json_url = filing_index_url(filing) + "index.json"
    r = safe_get(index_json_url, SEC_HEADERS)
    data = r.json()

    items = data.get("directory", {}).get("item", [])
    names = [item.get("name", "") for item in items]

    # Prefer xml files that look like information table files
    preferred_patterns = [
        r"infotable.*\.xml$",
        r"informationtable.*\.xml$",
        r".*form13f.*xml$",
        r".*\.xml$",
    ]

    for pattern in preferred_patterns:
        for name in names:
            if re.search(pattern, name, flags=re.IGNORECASE):
                # Avoid index/meta files if possible
                if "primary_doc" in name.lower() and pattern != preferred_patterns[-1]:
                    continue
                return filing_index_url(filing) + name

    return None


def _clean_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _parse_xml_text(root, xpath_list: List[str]) -> str:
    for xp in xpath_list:
        val = root.xpath(xp)
        if val:
            if isinstance(val[0], etree._Element):
                return _clean_text(val[0].text)
            return _clean_text(val[0])
    return ""


@st.cache_data(show_spinner=False, ttl=60 * 60)
def parse_13f_information_table(xml_url: str) -> pd.DataFrame:
    r = safe_get(xml_url, SEC_HEADERS)
    content = r.content

    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(content, parser=parser)

    # Handle possible namespaces
    nsmap = root.nsmap.copy() if root.nsmap else {}
    if None in nsmap:
        nsmap["ns"] = nsmap.pop(None)

    # Try with and without namespace
    info_tables = root.xpath(".//ns:infoTable", namespaces=nsmap) if nsmap else []
    if not info_tables:
        info_tables = root.xpath(".//infoTable")

    rows = []
    for node in info_tables:
        def xp(paths: List[str]) -> str:
            for p in paths:
                res = node.xpath(p, namespaces=nsmap) if nsmap else node.xpath(p)
                if res:
                    first = res[0]
                    if isinstance(first, etree._Element):
                        return _clean_text(first.text)
                    return _clean_text(first)
            return ""

        issuer = xp(["./ns:nameOfIssuer/text()", "./nameOfIssuer/text()"])
        title = xp(["./ns:titleOfClass/text()", "./titleOfClass/text()"])
        cusip = xp(["./ns:cusip/text()", "./cusip/text()"])
        value_k = xp(["./ns:value/text()", "./value/text()"])  # in thousands
        shares = xp([
            "./ns:shrsOrPrnAmt/ns:sshPrnamt/text()",
            "./shrsOrPrnAmt/sshPrnamt/text()",
        ])
        put_call = xp(["./ns:putCall/text()", "./putCall/text()"])
        discretion = xp(["./ns:investmentDiscretion/text()", "./investmentDiscretion/text()"])
        voting_sole = xp([
            "./ns:votingAuthority/ns:Sole/text()",
            "./votingAuthority/Sole/text()",
        ])

        rows.append(
            {
                "issuer": issuer,
                "class": title,
                "cusip": cusip,
                "value_kusd": pd.to_numeric(value_k, errors="coerce"),
                "shares": pd.to_numeric(shares, errors="coerce"),
                "put_call": put_call,
                "discretion": discretion,
                "voting_sole": pd.to_numeric(voting_sole, errors="coerce"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["value_usd"] = df["value_kusd"] * 1000
    df["issuer"] = df["issuer"].str.strip()
    df["class"] = df["class"].str.strip()
    df["cusip"] = df["cusip"].str.strip()

    # 13F may include options; keep only common stock-ish holdings by default later if needed
    return df.sort_values("value_usd", ascending=False).reset_index(drop=True)


def build_comparison(current_df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
    cur = current_df.copy()
    prv = prev_df.copy()

    # Use issuer + cusip as a pragmatic key
    cur["key"] = cur["issuer"].fillna("") + "|" + cur["cusip"].fillna("")
    prv["key"] = prv["issuer"].fillna("") + "|" + prv["cusip"].fillna("")

    merged = cur.merge(
        prv[["key", "shares", "value_usd"]],
        on="key",
        how="left",
        suffixes=("", "_prev"),
    )

    merged["shares_change"] = merged["shares"] - merged["shares_prev"]
    merged["shares_change_pct"] = merged["shares_change"] / merged["shares_prev"].replace(0, pd.NA) * 100
    total_value = merged["value_usd"].sum()
    merged["portfolio_weight_pct"] = merged["value_usd"] / total_value * 100 if total_value else 0

    return merged.sort_values("value_usd", ascending=False).reset_index(drop=True)


def try_guess_ticker(issuer: str) -> Optional[str]:
    """
    Simple free heuristic for major Berkshire names.
    Extend this dictionary as needed.
    """
    mapping = {
        "APPLE INC": "AAPL",
        "AMERICAN EXPRESS CO": "AXP",
        "BANK OF AMERICA CORP": "BAC",
        "COCA COLA CO": "KO",
        "CHEVRON CORP NEW": "CVX",
        "OCCIDENTAL PETE CORP DEL": "OXY",
        "KRAFT HEINZ CO": "KHC",
        "MOODYS CORP": "MCO",
        "DAVITA INC": "DVA",
        "CHUBB LIMITED": "CB",
        "VERISIGN INC": "VRSN",
        "SIRIUS XM HLDGS INC": "SIRI",
        "CITIGROUP INC": "C",
        "ALLY FINL INC": "ALLY",
        "AON PLC": "AON",
        "LIBERTY MEDIA CORP DEL": None,  # share classes vary
        "T MOBILE US INC": "TMUS",
        "DOMINOS PIZZA INC": "DPZ",
        "POOL CORP": "POOL",
        "HEICO CORP NEW": "HEI",
        "VISA INC": "V",
        "MASTERCARD INC": "MA",
        "AMAZON COM INC": "AMZN",
        "NU HLDGS LTD": "NU",
        "CAPITAL ONE FINL CORP": "COF",
    }
    key = issuer.upper().strip()
    return mapping.get(key)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_price_history(ticker: str, period: str = "3y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df


# ============================================================
# Load SEC filing data
# ============================================================
with st.spinner("Loading Berkshire Hathaway 13F filings from SEC..."):
    submissions = load_submissions_json()
    filings = get_recent_13f_filings(submissions, max_count=6)

if len(filings) < 2:
    st.error("Could not find enough Berkshire 13F filings.")
    st.stop()

latest = filings[0]
previous = filings[1]

st.subheader("Latest Berkshire 13F")
c1, c2, c3 = st.columns(3)
c1.metric("Filing date", latest.filing_date)
c2.metric("Report date", latest.report_date)
c3.metric("Previous report date", previous.report_date)

with st.spinner("Parsing latest and previous information tables..."):
    latest_xml = discover_information_table_xml(latest)
    previous_xml = discover_information_table_xml(previous)

    if not latest_xml or not previous_xml:
        st.error("Could not locate the SEC information table XML.")
        st.stop()

    latest_df = parse_13f_information_table(latest_xml)
    previous_df = parse_13f_information_table(previous_xml)

if latest_df.empty:
    st.error("Latest 13F table appears empty.")
    st.stop()

comparison_df = build_comparison(latest_df, previous_df)

# Filter out obvious non-long-equity artifacts if desired
show_options = st.sidebar.checkbox("Include options / special lines", value=False)
if not show_options:
    comparison_df = comparison_df[
        comparison_df["put_call"].fillna("").eq("")
    ].copy()

# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Controls")

top_n = st.sidebar.slider("Top holdings to show", 5, 30, 15)
min_weight = st.sidebar.slider("Minimum weight (%)", 0.0, 10.0, 0.0, 0.1)
chart_period = st.sidebar.selectbox("Price chart period", ["6mo", "1y", "2y", "3y", "5y"], index=3)

view_df = comparison_df[comparison_df["portfolio_weight_pct"] >= min_weight].copy()

# ============================================================
# Summary metrics
# ============================================================
total_value = view_df["value_usd"].sum()
num_positions = len(view_df)
new_positions = int(view_df["shares_prev"].isna().sum())
increased_positions = int((view_df["shares_change"] > 0).sum())
reduced_positions = int((view_df["shares_change"] < 0).sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Reported value", f"${total_value/1e9:,.1f}B")
m2.metric("Positions", f"{num_positions}")
m3.metric("New vs prior filing", f"{new_positions}")
m4.metric("Raised / Cut", f"{increased_positions} / {reduced_positions}")

# ============================================================
# Top holdings bar chart
# ============================================================
st.subheader("Top Holdings by Reported Value")

top_df = view_df.head(top_n).copy()
fig_bar = px.bar(
    top_df.sort_values("value_usd"),
    x="value_usd",
    y="issuer",
    orientation="h",
    text="portfolio_weight_pct",
)
fig_bar.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside",
)
fig_bar.update_layout(
    xaxis_title="Reported Value (USD)",
    yaxis_title="Issuer",
    height=max(450, 25 * len(top_df)),
)
st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================
# Holdings table
# ============================================================
st.subheader("Holdings Table")

display_df = view_df.copy()
display_df["value_usd_bn"] = display_df["value_usd"] / 1e9
display_df["shares_mn"] = display_df["shares"] / 1e6
display_df["shares_change_mn"] = display_df["shares_change"] / 1e6

st.dataframe(
    display_df[
        [
            "issuer",
            "class",
            "cusip",
            "value_usd_bn",
            "portfolio_weight_pct",
            "shares_mn",
            "shares_change_mn",
            "shares_change_pct",
        ]
    ].rename(
        columns={
            "issuer": "Issuer",
            "class": "Class",
            "cusip": "CUSIP",
            "value_usd_bn": "Value ($Bn)",
            "portfolio_weight_pct": "Weight (%)",
            "shares_mn": "Shares (Mn)",
            "shares_change_mn": "Share Change (Mn)",
            "shares_change_pct": "Share Change (%)",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# Single holding price chart
# ============================================================
st.subheader("Price Chart for a Selected Holding")

issuer_list = view_df["issuer"].tolist()
selected_issuer = st.selectbox("Select issuer", issuer_list, index=0)

selected_row = view_df[view_df["issuer"] == selected_issuer].iloc[0]
guessed_ticker = try_guess_ticker(selected_issuer)

manual_ticker = st.text_input(
    "Ticker override (optional)",
    value=guessed_ticker or "",
    help="Some issuers need manual ticker input because 13F does not include ticker symbols.",
)

ticker_to_use = manual_ticker.strip().upper()

if ticker_to_use:
    price_df = load_price_history(ticker_to_use, period=chart_period)

    if price_df.empty:
        st.warning(f"Could not load price history for {ticker_to_use}. Try another ticker.")
    else:
        if isinstance(price_df.columns, pd.MultiIndex):
            # yfinance can occasionally return multiindex
            if ("Close" in price_df.columns.get_level_values(0)):
                close = price_df["Close"].copy()
            else:
                close = price_df.iloc[:, 0].copy()
        else:
            close = price_df["Close"] if "Close" in price_df.columns else price_df.iloc[:, 0]

        fig_price = go.Figure()
        fig_price.add_trace(
            go.Scatter(
                x=close.index,
                y=close.values,
                mode="lines",
                name=ticker_to_use,
            )
        )
        fig_price.update_layout(
            title=f"{selected_issuer} ({ticker_to_use}) Price",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # simple moving averages
        ma_df = pd.DataFrame({"Close": close})
        ma_df["MA50"] = ma_df["Close"].rolling(50).mean()
        ma_df["MA200"] = ma_df["Close"].rolling(200).mean()

        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=ma_df.index, y=ma_df["Close"], mode="lines", name="Close"))
        fig_ma.add_trace(go.Scatter(x=ma_df.index, y=ma_df["MA50"], mode="lines", name="MA50"))
        fig_ma.add_trace(go.Scatter(x=ma_df.index, y=ma_df["MA200"], mode="lines", name="MA200"))
        fig_ma.update_layout(
            title=f"{selected_issuer} ({ticker_to_use}) with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
        )
        st.plotly_chart(fig_ma, use_container_width=True)
else:
    st.info("Enter or confirm a ticker to show the price chart.")

# ============================================================
# Filing history table
# ============================================================
st.subheader("Recent Berkshire 13F Filings")

filings_df = pd.DataFrame(
    [
        {
            "filing_date": f.filing_date,
            "report_date": f.report_date,
            "accession_number": f.accession_number,
            "folder_url": filing_index_url(f),
        }
        for f in filings
    ]
)

st.dataframe(filings_df, use_container_width=True, hide_index=True)

st.caption(
    "Note: 13F data is quarterly and delayed. It shows Berkshire’s reported U.S. long holdings, not real-time trades."
)
