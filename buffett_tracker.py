from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional, List, Dict

import pandas as pd
import requests
import streamlit as st
import xml.etree.ElementTree as ET

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Buffett Top 10 Compare",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Buffett Top 10 Holdings — 3 Period Comparison")
st.caption("Simple Berkshire Hathaway 13F top-10 table comparison by report date")

# ============================================================
# Constants
# ============================================================
BERKSHIRE_CIK = "0001067983"
SEC_SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{BERKSHIRE_CIK}.json"

SEC_HEADERS = {
    "User-Agent": "BuffettTop10Tracker/1.0 your_email@example.com",
    "Accept-Encoding": "gzip, deflate",
}

REQUEST_SLEEP = 0.25


# ============================================================
# Data class
# ============================================================
@dataclass
class FilingInfo:
    accession_number: str
    filing_date: str
    report_date: str
    primary_doc: str
    accession_nodash: str


# ============================================================
# HTTP helper
# ============================================================
def safe_get(url: str, headers: dict, timeout: int = 30) -> requests.Response:
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    time.sleep(REQUEST_SLEEP)
    return r


# ============================================================
# SEC loaders
# ============================================================
@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_submissions_json() -> dict:
    r = safe_get(SEC_SUBMISSIONS_URL, SEC_HEADERS)
    return r.json()


def get_recent_13f_filings(submissions: dict, max_count: int = 12) -> List[FilingInfo]:
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
    return f"https://www.sec.gov/Archives/edgar/data/{int(BERKSHIRE_CIK)}/{filing.accession_nodash}/"


@st.cache_data(show_spinner=False, ttl=60 * 60)
def discover_information_table_xml(filing: FilingInfo) -> Optional[str]:
    index_json_url = filing_index_url(filing) + "index.json"
    r = safe_get(index_json_url, SEC_HEADERS)
    data = r.json()

    items = data.get("directory", {}).get("item", [])
    names = [item.get("name", "") for item in items]

    preferred_patterns = [
        r"infotable.*\.xml$",
        r"informationtable.*\.xml$",
        r".*form13f.*\.xml$",
        r".*\.xml$",
    ]

    for pattern in preferred_patterns:
        for name in names:
            if re.search(pattern, name, flags=re.IGNORECASE):
                if "primary_doc" in name.lower() and pattern != preferred_patterns[-1]:
                    continue
                return filing_index_url(filing) + name

    return None


# ============================================================
# XML parsing helpers
# ============================================================
def strip_namespace(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def find_child_text(parent: ET.Element, child_name: str) -> Optional[str]:
    for child in parent:
        if strip_namespace(child.tag) == child_name:
            return child.text.strip() if child.text else None
    return None


@st.cache_data(show_spinner=False, ttl=60 * 60)
def parse_13f_information_table(xml_url: str) -> pd.DataFrame:
    r = safe_get(xml_url, SEC_HEADERS)
    root = ET.fromstring(r.content)

    rows = []

    for elem in root.iter():
        if strip_namespace(elem.tag) != "infoTable":
            continue

        issuer = find_child_text(elem, "nameOfIssuer")
        title = find_child_text(elem, "titleOfClass")
        cusip = find_child_text(elem, "cusip")
        value_k = find_child_text(elem, "value")
        put_call = find_child_text(elem, "putCall")

        shares = None
        for child in elem:
            if strip_namespace(child.tag) == "shrsOrPrnAmt":
                for sub in child:
                    if strip_namespace(sub.tag) == "sshPrnamt":
                        shares = sub.text.strip() if sub.text else None
                        break

        rows.append(
            {
                "issuer": issuer,
                "class": title,
                "cusip": cusip,
                "value_kusd": pd.to_numeric(value_k, errors="coerce"),
                "shares": pd.to_numeric(shares, errors="coerce"),
                "put_call": put_call,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["issuer"] = df["issuer"].fillna("").str.strip()
    df["class"] = df["class"].fillna("").str.strip()
    df["cusip"] = df["cusip"].fillna("").str.strip()
    df["value_usd"] = df["value_kusd"] * 1000

    return df.sort_values("value_usd", ascending=False).reset_index(drop=True)


# ============================================================
# Build top 10 table
# ============================================================
def build_top10_table(df: pd.DataFrame, top_n: int = 10, include_options: bool = False) -> pd.DataFrame:
    x = df.copy()

    if not include_options:
        x = x[x["put_call"].fillna("").eq("")].copy()

    total_value = x["value_usd"].sum()

    x = x.sort_values("value_usd", ascending=False).head(top_n).copy()
    x["weight_pct"] = x["value_usd"] / total_value * 100 if total_value else 0
    x["value_usd_bn"] = x["value_usd"] / 1e9
    x["shares_mn"] = x["shares"] / 1e6

    x = x[
        ["issuer", "class", "value_usd_bn", "weight_pct", "shares_mn"]
    ].rename(
        columns={
            "issuer": "Issuer",
            "class": "Class",
            "value_usd_bn": "Value ($Bn)",
            "weight_pct": "Weight (%)",
            "shares_mn": "Shares (Mn)",
        }
    )

    x["Value ($Bn)"] = x["Value ($Bn)"].round(2)
    x["Weight (%)"] = x["Weight (%)"].round(2)
    x["Shares (Mn)"] = x["Shares (Mn)"].round(2)

    return x.reset_index(drop=True)


# ============================================================
# Load filings
# ============================================================
with st.spinner("Loading Berkshire 13F filing list..."):
    submissions = load_submissions_json()
    filings = get_recent_13f_filings(submissions, max_count=10)

if not filings:
    st.error("No Berkshire 13F filings found.")
    st.stop()

filing_map: Dict[str, FilingInfo] = {
    f"{f.report_date} | filed {f.filing_date}": f for f in filings
}

report_labels = list(filing_map.keys())

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Settings")

include_options = st.sidebar.checkbox("Include options / special lines", value=False)
top_n = st.sidebar.slider("Top N holdings", 5, 15, 10)

default_labels = report_labels[:3] if len(report_labels) >= 3 else report_labels

selected_labels = st.multiselect(
    "Select 3 report dates",
    options=report_labels,
    default=default_labels,
    max_selections=3,
)

if len(selected_labels) == 0:
    st.warning("Please select at least one report date.")
    st.stop()

selected_filings = [filing_map[label] for label in selected_labels]

# ============================================================
# Parse selected periods
# ============================================================
tables = []

with st.spinner("Loading selected 13F top holdings..."):
    for filing in selected_filings:
        xml_url = discover_information_table_xml(filing)
        if not xml_url:
            tables.append((filing, pd.DataFrame()))
            continue

        df = parse_13f_information_table(xml_url)
        top_table = build_top10_table(df, top_n=top_n, include_options=include_options)
        tables.append((filing, top_table))

# ============================================================
# Summary header
# ============================================================
st.subheader("Selected Periods")

summary_cols = st.columns(len(tables))
for col, (filing, table) in zip(summary_cols, tables):
    with col:
        st.metric("Report Date", filing.report_date)
        st.caption(f"Filed: {filing.filing_date}")
        st.caption(f"Top rows: {len(table)}")

# ============================================================
# Side-by-side tables
# ============================================================
st.subheader(f"Top {top_n} Holdings Comparison")

table_cols = st.columns(len(tables))

for col, (filing, table) in zip(table_cols, tables):
    with col:
        st.markdown(f"### {filing.report_date}")
        st.caption(f"Filed: {filing.filing_date}")

        if table.empty:
            st.warning("Could not load this filing.")
        else:
            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
                height=420,
            )

# ============================================================
# Optional merged compare table
# ============================================================
st.subheader("Merged View by Rank")

merged_blocks = []
for filing, table in tables:
    if table.empty:
        continue
    t = table.copy()
    t.insert(0, "Rank", range(1, len(t) + 1))
    t.columns = ["Rank"] + [f"{c} ({filing.report_date})" for c in t.columns[1:]]
    merged_blocks.append(t)

if merged_blocks:
    merged_df = merged_blocks[0]
    for nxt in merged_blocks[1:]:
        merged_df = merged_df.merge(nxt, on="Rank", how="outer")

    st.dataframe(merged_df, use_container_width=True, hide_index=True)
else:
    st.info("No merged table available.")

# ============================================================
# Filing history
# ============================================================
st.subheader("Recent Berkshire 13F Filings")

filings_df = pd.DataFrame(
    [
        {
            "Report Date": f.report_date,
            "Filing Date": f.filing_date,
            "Accession Number": f.accession_number,
        }
        for f in filings
    ]
)

st.dataframe(filings_df, use_container_width=True, hide_index=True)

st.caption("13F is quarterly and delayed. It shows reported U.S. long holdings, not real-time trades.")
