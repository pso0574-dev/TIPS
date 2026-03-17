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
# Page
# ============================================================
st.set_page_config(
    page_title="Buffett Top 10 Compare",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Buffett Top 10 Holdings — 3 Period Comparison")
st.caption("Stable version: Berkshire Hathaway 13F top-10 tables")

# ============================================================
# Constants
# ============================================================
BERKSHIRE_CIK = "0001067983"
SEC_SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{BERKSHIRE_CIK}.json"

USER_AGENT = "Mozilla/5.0 BuffettTop10Tracker/1.0 contact: pso0574@gmail.com"

HEADERS_DATA = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
}

HEADERS_ARCHIVE = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
}

REQUEST_SLEEP = 0.2
REQUEST_TIMEOUT = 20


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
# HTTP
# ============================================================
def safe_get(url: str, headers: dict, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    time.sleep(REQUEST_SLEEP)
    return r


# ============================================================
# SEC loaders
# ============================================================
@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_submissions_json() -> dict:
    r = safe_get(SEC_SUBMISSIONS_URL, HEADERS_DATA)
    return r.json()


def get_recent_13f_filings(submissions: dict, max_count: int = 10) -> List[FilingInfo]:
    recent = submissions.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_docs = recent.get("primaryDocument", [])

    filings = []
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


def filing_folder_url(filing: FilingInfo) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(BERKSHIRE_CIK)}/{filing.accession_nodash}/"


@st.cache_data(show_spinner=False, ttl=60 * 60)
def discover_information_table_xml(filing: FilingInfo) -> Optional[str]:
    folder = filing_folder_url(filing)
    index_json_url = folder + "index.json"

    r = safe_get(index_json_url, HEADERS_ARCHIVE)
    data = r.json()

    items = data.get("directory", {}).get("item", [])
    names = [item.get("name", "") for item in items]

    candidates = []
    for name in names:
        low = name.lower()
        if low.endswith(".xml"):
            candidates.append(name)

    preferred_order = [
        "infotable",
        "informationtable",
        "form13f",
    ]

    for keyword in preferred_order:
        for name in candidates:
            if keyword in name.lower():
                return folder + name

    if candidates:
        return folder + candidates[0]

    return None


# ============================================================
# XML parse
# ============================================================
def strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def child_text(elem: ET.Element, name: str) -> Optional[str]:
    for c in elem:
        if strip_ns(c.tag) == name:
            return c.text.strip() if c.text else None
    return None


@st.cache_data(show_spinner=False, ttl=60 * 60)
def parse_13f_information_table(xml_url: str) -> pd.DataFrame:
    r = safe_get(xml_url, HEADERS_ARCHIVE)
    root = ET.fromstring(r.content)

    rows = []

    for elem in root.iter():
        if strip_ns(elem.tag) != "infoTable":
            continue

        issuer = child_text(elem, "nameOfIssuer")
        title = child_text(elem, "titleOfClass")
        cusip = child_text(elem, "cusip")
        value_k = child_text(elem, "value")
        put_call = child_text(elem, "putCall")

        shares = None
        for c in elem:
            if strip_ns(c.tag) == "shrsOrPrnAmt":
                for sub in c:
                    if strip_ns(sub.tag) == "sshPrnamt":
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
# Build top table
# ============================================================
def build_top_table(df: pd.DataFrame, top_n: int = 10, include_options: bool = False) -> pd.DataFrame:
    x = df.copy()

    if not include_options and "put_call" in x.columns:
        x = x[x["put_call"].fillna("").eq("")].copy()

    if x.empty:
        return pd.DataFrame()

    total_value = x["value_usd"].sum()

    x = x.sort_values("value_usd", ascending=False).head(top_n).copy()
    x["weight_pct"] = (x["value_usd"] / total_value * 100).round(2)
    x["value_bn"] = (x["value_usd"] / 1e9).round(2)
    x["shares_mn"] = (x["shares"] / 1e6).round(2)

    x = x[["issuer", "class", "value_bn", "weight_pct", "shares_mn"]].rename(
        columns={
            "issuer": "Issuer",
            "class": "Class",
            "value_bn": "Value ($Bn)",
            "weight_pct": "Weight (%)",
            "shares_mn": "Shares (Mn)",
        }
    )

    x.insert(0, "Rank", range(1, len(x) + 1))
    return x.reset_index(drop=True)


# ============================================================
# Main load
# ============================================================
status_box = st.empty()

try:
    status_box.info("Loading Berkshire 13F filing list...")
    submissions = load_submissions_json()
    filings = get_recent_13f_filings(submissions, max_count=8)
except Exception as e:
    st.error(f"Failed to load SEC filing list: {e}")
    st.stop()

if not filings:
    st.error("No Berkshire 13F filings found.")
    st.stop()

filing_map: Dict[str, FilingInfo] = {
    f"{f.report_date} | filed {f.filing_date}": f for f in filings
}
labels = list(filing_map.keys())

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Settings")
include_options = st.sidebar.checkbox("Include options / special lines", value=False)
top_n = st.sidebar.slider("Top N holdings", 5, 15, 10)

default_selected = labels[:3] if len(labels) >= 3 else labels

selected_labels = st.sidebar.multiselect(
    "Select up to 3 periods",
    options=labels,
    default=default_selected,
    max_selections=3,
)

if len(selected_labels) == 0:
    st.warning("Please select at least one period.")
    st.stop()

selected_filings = [filing_map[x] for x in selected_labels]

# ============================================================
# Load selected tables
# ============================================================
tables = []

for filing in selected_filings:
    try:
        status_box.info(f"Loading {filing.report_date} ...")
        xml_url = discover_information_table_xml(filing)

        if not xml_url:
            tables.append((filing, pd.DataFrame(), "XML not found"))
            continue

        df = parse_13f_information_table(xml_url)

        if df.empty:
            tables.append((filing, pd.DataFrame(), "Parsed table is empty"))
            continue

        top_table = build_top_table(df, top_n=top_n, include_options=include_options)
        tables.append((filing, top_table, None))

    except Exception as e:
        tables.append((filing, pd.DataFrame(), str(e)))

status_box.success("Done.")

# ============================================================
# Summary
# ============================================================
st.subheader("Selected Periods")
cols = st.columns(len(tables))

for col, (filing, table, err) in zip(cols, tables):
    with col:
        st.metric("Report Date", filing.report_date)
        st.caption(f"Filed: {filing.filing_date}")
        if err:
            st.error(err)
        else:
            st.success(f"{len(table)} rows loaded")

# ============================================================
# 3 tables side by side
# ============================================================
st.subheader(f"Top {top_n} Holdings Comparison")
cols = st.columns(len(tables))

for col, (filing, table, err) in zip(cols, tables):
    with col:
        st.markdown(f"### {filing.report_date}")
        if err:
            st.warning(err)
        elif table.empty:
            st.warning("No data")
        else:
            st.dataframe(table, use_container_width=True, hide_index=True, height=420)

# ============================================================
# Filing list
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

st.caption("If SEC access is blocked, replace the User-Agent email with your real email.")
