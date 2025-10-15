import io
import os
from pathlib import Path
from datetime import datetime
from PIL import Image

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------- Page Setup ----------------------
st.set_page_config(
    page_title="FBA Lost Inventory – Reimbursement Tool",
    page_icon="🧾",
    layout="wide",
)

# ---------------------- Admin Login ----------------------
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "12345"  # baad mai env var / secrets.toml mai rakhna secure ke liye

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔒 Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state.logged_in = True
            st.success("✅ Login successful")
        else:
            st.error("❌ Invalid credentials")
    st.stop()

# ---------------------- Sidebar Logo ----------------------
logo = Image.open("images.png")   # make sure images.png is present
st.sidebar.image(logo, use_container_width=True)
st.sidebar.title("⚙️ Controls")

# ---------------------- Custom Orange Theme ----------------------
st.markdown("""
<style>
/* Background */
.main { background-color: #fff8f0; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #fff4e6; }

/* Cards */
.metric-card {
  background: #ffffff;
  border: 1px solid #fca311;
  border-radius: 14px;
  padding: 18px 16px;
  box-shadow: 0 4px 14px rgba(252,163,17,0.2);
}
.metric-title {
  font-size: 0.90rem;
  color: #f77f00;
  font-weight: 600;
}
.metric-value {
  font-size: 1.6rem;
  font-weight: 800;
  color: #d62828;
}
.metric-sub {
  font-size: 0.85rem;
  color: #6b7280;
}
.section { padding: 10px 0 4px 0; }

/* Buttons */
div.stButton > button {
    background-color: #fca311;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #e85d04;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- Helpers ----------------------
def _find_dir(*names: str) -> Path:
    here = Path(__file__).resolve().parent
    root = here.parent
    for n in names:
        p = root / n
        if p.exists():
            return p
    p = root / names[0]
    p.mkdir(parents=True, exist_ok=True)
    return p

DATA_DIR = _find_dir("Data", "data")
OUTPUT_DIR = _find_dir("Outputs", "output")

ALLOWED_EXTS = {".csv", ".xlsx", ".xlsm", ".xls"}

def read_csv_robust(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in [None, "utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def read_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".xlsx", ".xlsm", ".xls"]:
        try:
            return pd.read_excel(path, sheet_name=0)
        except Exception:
            return pd.read_excel(path)
    elif ext == ".csv":
        return read_csv_robust(path)
    else:
        try:
            return read_csv_robust(path)
        except Exception:
            return pd.read_excel(path)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .map(lambda c: str(c).strip().lower().replace(" ", "_"))
        .str.replace(r"[^\w_]+", "", regex=True)
    )
    return df

def numify(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biufc":
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(
        series.astype(str).str.replace(r"[,$% ]", "", regex=True).replace({"": np.nan}),
        errors="coerce",
    )

def to_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    except Exception:
        return pd.to_datetime(series.astype(str), errors="coerce")

def find_latest_file(data_dir: Path) -> Path | None:
    files = [p for p in data_dir.glob("*") if p.suffix.lower() in ALLOWED_EXTS and p.is_file()]
    if not files:
        return None
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)[0]

CANDIDATES = {
    "shipment_name": ["shipment_name"],
    "created": ["created", "creation_date"],
    "last_updated": ["last_updated", "updated_at", "approval_date", "approvaldate", "approval-date"],
    "ship_to": ["ship_to", "destination", "fc", "shipto", "ship_to_code"],
    "units_expected": ["units_expected", "expected_units"],
    "units_located": ["units_located", "located_units", "units_received", "received_units"],
    "discrepancies": ["discrepencies", "discrepancies", "qty_difference", "quantity_diff", "quantity_difference"],
    "status": ["status", "case_status"],
    "reimbursement_amount": ["reimbursement_amount", "amounttotal", "amount_total", "amount-per-unit", "amount_per_unit"],
    "reimbursement_id": ["reimbursement_id", "reimbursementid"],
    "case_status": ["case_status", "status"],
    "asin": ["asin"],
    "sku": ["sku"],
}

def first_present(df: pd.DataFrame, keys: list[str]) -> str | None:
    for k in keys:
        if k in df.columns:
            return k
    return None

def select_columns(df: pd.DataFrame) -> dict:
    cols = {}
    for canon, candidates in CANDIDATES.items():
        hit = first_present(df, candidates)
        cols[canon] = hit
    return cols

def compute_views(df: pd.DataFrame, cols: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # normalize number/date fields
    if cols["reimbursement_amount"]:
        df[cols["reimbursement_amount"]] = numify(df[cols["reimbursement_amount"]])
    if cols["discrepancies"]:
        df[cols["discrepancies"]] = numify(df[cols["discrepancies"]])
    if cols["created"]:
        df[cols["created"]] = to_datetime(df[cols["created"]])
    if cols["last_updated"]:
        df[cols["last_updated"]] = to_datetime(df[cols["last_updated"]])

    # Lost inventory rows (negative discrepancies)
    if cols["discrepancies"]:
        lost = df[df[cols["discrepancies"]] < 0].copy()
    else:
        lost = df.iloc[0:0].copy()

    # Pending reimbursement rows (lost but no $ yet)
    if cols["reimbursement_amount"] and not lost.empty:
        pending = lost[(lost[cols["reimbursement_amount"]].isna()) | (lost[cols["reimbursement_amount"]] == 0)].copy()
    else:
        pending = lost.copy() if cols["discrepancies"] else df.iloc[0:0].copy()

    # Reimbursed cases (sum by reimbursement_id if present; include negatives to reduce total)
    if cols["reimbursement_amount"]:
        reimbursed_raw = df[df[cols["reimbursement_amount"]].notna()].copy()
        if cols["reimbursement_id"]:
            reimbursed = (
                reimbursed_raw.groupby(cols["reimbursement_id"], as_index=False)[cols["reimbursement_amount"]]
                .sum()
            )
        else:
            reimbursed = reimbursed_raw
        reimb_amount_sum = reimbursed[cols["reimbursement_amount"]].sum()
    else:
        reimbursed = df.iloc[0:0].copy()
        reimb_amount_sum = np.nan

    summary = pd.DataFrame([{
        "total_rows": len(df),
        "lost_inventory_rows": len(lost),
        "pending_reimbursement_rows": len(pending),
        "reimbursed_rows": len(reimbursed),
        "total_reimbursed_amount": reimb_amount_sum,
    }])

    return lost, pending, reimbursed, summary

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    return buff.getvalue().encode("utf-8")

def save_outputs(lost: pd.DataFrame, pending: pd.DataFrame, reimb: pd.DataFrame, summary: pd.DataFrame) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {
        "lost": OUTPUT_DIR / f"lost_inventory_{ts}.csv",
        "pending": OUTPUT_DIR / f"pending_reimbursement_{ts}.csv",
        "reimbursed": OUTPUT_DIR / f"reimbursed_cases_{ts}.csv",
        "summary": OUTPUT_DIR / f"summary_{ts}.csv",
    }
    lost.to_csv(paths["lost"], index=False)
    pending.to_csv(paths["pending"], index=False)
    reimb.to_csv(paths["reimbursed"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return paths

# ---------------------- Rule-based Checks ----------------------
def compute_rule_based(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    issues = []

    # Shipped vs Received mismatch
    if cols["units_expected"] and cols["units_located"]:
        mismatch = df[df[cols["units_expected"]] != df[cols["units_located"]]].copy()
        for _, row in mismatch.iterrows():
            issues.append({
                "Issue": "Shipped vs Received mismatch",
                "Shipment": row.get(cols["shipment_name"], ""),
                "Expected": row.get(cols["units_expected"], ""),
                "Received": row.get(cols["units_located"], ""),
            })

    # Return mismatch (use discrepancies if present)
    if cols["discrepancies"]:
        return_mismatch = df[df[cols["discrepancies"]] > 0].copy()
        for _, row in return_mismatch.iterrows():
            issues.append({
                "Issue": "Return mismatch (extra units)",
                "Shipment": row.get(cols["shipment_name"], ""),
                "Discrepancy": row.get(cols["discrepancies"], ""),
            })

    return pd.DataFrame(issues)

# ---------------------- Sidebar ----------------------
page = st.sidebar.radio(
    "📂 Select Page",
    ["Dashboard", "Monthly Graph", "Missing Funds Detection"],
    index=0
)

mode = st.sidebar.radio("Input Mode", ["Use latest from Data/", "Upload file"], index=0)

latest_file = find_latest_file(DATA_DIR)
if mode == "Use latest from Data/":
    if latest_file:
        st.sidebar.success(f"Latest: {latest_file.name}")
    else:
        st.sidebar.warning("No files found in Data/ — upload a file instead.")
else:
    st.sidebar.info("Upload a CSV/XLSX below")

uploaded = None
if mode == "Upload file":
    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xlsm","xls"])

st.sidebar.markdown("---")

# ---------------------- Load Data ----------------------
if mode == "Use latest from Data/":
    if latest_file is None:
        st.stop()
    df_raw = read_any(latest_file)
    source_label = f"Source: Data/{latest_file.name}"
else:
    if uploaded is None:
        st.info("Upload a CSV/Excel file from Seller Central to begin.")
        st.stop()
    suffix = Path(uploaded.name).suffix.lower()
    if suffix in [".xlsx",".xlsm",".xls"]:
        df_raw = pd.read_excel(uploaded, sheet_name=0)
    else:
        up_bytes = uploaded.read()
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                df_raw = pd.read_csv(io.BytesIO(up_bytes), sep=None, engine="python", encoding=enc)
                break
            except Exception:
                continue
        else:
            st.error("Could not read the uploaded file. Please export as CSV/Excel.")
            st.stop()
    source_label = f"Source: Uploaded – {uploaded.name}"

df = standardize_columns(df_raw)
cols_map = select_columns(df)
lost_df, pending_df, reimb_df, summary_df = compute_views(df, cols_map)
rule_df = compute_rule_based(df, cols_map)

# ---------------------- Pages ----------------------
if page == "Dashboard":
    st.title("🧾 FBA Lost Inventory – Reimbursement Dashboard")

    # Summary Cards
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-title">Total Rows</div>'
                    f'<div class="metric-value">{int(summary_df.loc[0,"total_rows"])}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-title">Lost Inventory (rows)</div>'
                    f'<div class="metric-value">{int(summary_df.loc[0,"lost_inventory_rows"])}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-title">Pending Reimbursement (rows)</div>'
                    f'<div class="metric-value">{int(summary_df.loc[0,"pending_reimbursement_rows"])}</div></div>', unsafe_allow_html=True)
    with c4:
        total_amt = summary_df.loc[0,"total_reimbursed_amount"]
        val = "—" if pd.isna(total_amt) else f"${float(total_amt):,.2f}"
        st.markdown('<div class="metric-card"><div class="metric-title">Total Reimbursed Amount</div>'
                    f'<div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

    st.caption(source_label)

    # Tables
    st.subheader("📉 Lost Inventory")
    st.dataframe(lost_df, use_container_width=True, height=260)

    st.subheader("⏳ Pending Reimbursement")
    st.dataframe(pending_df, use_container_width=True, height=240)

    st.subheader("💸 Reimbursed Cases")
    st.dataframe(reimb_df, use_container_width=True, height=240)

    # 🚨 Rule-based checks table
    st.subheader("🚨 Rule-based Issues")
    if rule_df.empty:
        st.info("No issues flagged by rule-based checks ✅")
    else:
        st.dataframe(rule_df, use_container_width=True, height=240)

    st.subheader("🧮 Summary (CSV view)")
    st.dataframe(summary_df, use_container_width=True)

    # Downloads
    st.markdown("### ⬇️ Download Reports")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.download_button("Lost CSV", df_to_csv_bytes(lost_df), file_name="lost_inventory.csv", mime="text/csv", use_container_width=True)
    with colB:
        st.download_button("Pending CSV", df_to_csv_bytes(pending_df), file_name="pending_reimbursement.csv", mime="text/csv", use_container_width=True)
    with colC:
        st.download_button("Reimbursed CSV", df_to_csv_bytes(reimb_df), file_name="reimbursed_cases.csv", mime="text/csv", use_container_width=True)
    with colD:
        st.download_button("Summary CSV", df_to_csv_bytes(summary_df), file_name="summary.csv", mime="text/csv", use_container_width=True)

    st.markdown("---")
    if st.button("💾 Save all to Outputs/ (timestamped)", type="primary", use_container_width=True):
        paths = save_outputs(lost_df, pending_df, reimb_df, summary_df)
        st.success("Saved! Files:")
        for k, p in paths.items():
            st.write(f"- {k}: {p.name}")
        st.balloons()

elif page == "Monthly Graph":
    st.title("📊 Monthly Reimbursement Analysis")

    date_col = cols_map.get("last_updated") or cols_map.get("created")
    amt_col = cols_map.get("reimbursement_amount")

    if date_col and amt_col and date_col in df.columns and amt_col in df.columns:
        df["month"] = pd.to_datetime(df[date_col]).dt.to_period("M")
        monthly = df.groupby("month")[amt_col].sum().reset_index()
        monthly["month"] = monthly["month"].astype(str)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(monthly["month"], monthly[amt_col], color="#fca311")
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Reimbursement Amount")
        ax.set_title("Reimbursement Amount by Month")
        plt.xticks(rotation=45, fontsize=6)
        plt.yticks(fontsize=8)
        st.pyplot(fig)

        st.subheader("📋 Monthly Data Table")
        st.dataframe(monthly, use_container_width=True)
    else:
        st.warning("⚠️ Required columns (date + reimbursement_amount) not found in file.")

# ---------------------- Missing Funds Detection ----------------------
elif page == "Missing Funds Detection":
    st.title("📦 Missing Funds Detection – Lost / Damaged / Unreimbursed Inventory")

    st.markdown("""
    This module analyzes your FBA inventory data to detect:
    - 🟠 Lost or missing inbound shipments  
    - 🔴 Damaged or unreturned items  
    - 🟢 Customer return mismatches  
    - 💰 Potential reimbursement opportunities  
    """)

    # 1) Base detection: Expected vs Located
    if cols_map["units_expected"] and cols_map["units_located"]:
        df["units_expected"] = numify(df[cols_map["units_expected"]])
        df["units_located"]  = numify(df[cols_map["units_located"]])
        df["potential_missing_units"] = (df[cols_map["units_expected"]] - df[cols_map["units_located"]]).clip(lower=0)
        df_missing = df[df["potential_missing_units"] > 0].copy()
    else:
        df_missing = df.iloc[0:0].copy()

    # 2) Estimated refund: use reimbursement_amount avg (if exists) as per-unit proxy
    est_refund = 0.0
    if cols_map["reimbursement_amount"] and not df_missing.empty:
        df_missing["reimbursement_amount"] = numify(df[cols_map["reimbursement_amount"]])
        per_unit_proxy = df_missing["reimbursement_amount"].replace(0, np.nan).mean()
        if pd.isna(per_unit_proxy):
            per_unit_proxy = 0.0
        df_missing["estimated_refund"] = df_missing["potential_missing_units"] * per_unit_proxy
        est_refund = float(df_missing["estimated_refund"].sum())
    elif not df_missing.empty:
        df_missing["estimated_refund"] = 0.0

    # 3) Summary cards + table + chart + download
    if not df_missing.empty:
        total_missing_units = int(df_missing["potential_missing_units"].sum())
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"<div class='metric-card'><div class='metric-title'>Total Missing Units</div>"
                f"<div class='metric-value'>{total_missing_units}</div></div>",
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"<div class='metric-card'><div class='metric-title'>Estimated Recoverable Amount</div>"
                f"<div class='metric-value'>${est_refund:,.2f}</div></div>",
                unsafe_allow_html=True
            )

        st.subheader("📋 Detailed Missing Items Report")
        show_cols = []
        # add safe columns if available
        if cols_map["shipment_name"] and cols_map["shipment_name"] in df_missing.columns:
            show_cols.append(cols_map["shipment_name"])
        if cols_map["sku"] and cols_map["sku"] in df_missing.columns:
            show_cols.append(cols_map["sku"])
        if cols_map["asin"] and cols_map["asin"] in df_missing.columns:
            show_cols.append(cols_map["asin"])
        show_cols += ["potential_missing_units", "estimated_refund"]

        rename_map = {}
        if cols_map["shipment_name"]: rename_map[cols_map["shipment_name"]] = "Shipment"
        if cols_map["sku"]:            rename_map[cols_map["sku"]] = "SKU"
        if cols_map["asin"]:           rename_map[cols_map["asin"]] = "ASIN"
        rename_map["potential_missing_units"] = "Missing Units"
        rename_map["estimated_refund"]        = "Estimated Refund ($)"

        st.dataframe(
            df_missing[show_cols].rename(columns=rename_map),
            use_container_width=True, height=300
        )

        st.download_button(
            "⬇️ Download Missing Funds Report",
            df_to_csv_bytes(df_missing.rename(columns=rename_map)),
            file_name="missing_funds_report.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Chart: Top 5 SKUs by Estimated Refund
        st.subheader("📊 Top 5 SKUs by Estimated Refund")
        if "estimated_refund" in df_missing.columns and cols_map["sku"]:
            top5 = df_missing.copy()
            # make sure SKU exists
            top5 = top5[top5[cols_map["sku"]].notna()]
            top5 = top5.sort_values("estimated_refund", ascending=False).head(5)
            if not top5.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(top5[cols_map["sku"]].astype(str), top5["estimated_refund"], color="#f77f00")
                ax.set_xlabel("SKU")
                ax.set_ylabel("Estimated Refund ($)")
                ax.set_title("Top 5 SKUs by Refund Value")
                plt.xticks(rotation=30, ha="right")
                st.pyplot(fig)
            else:
                st.info("Not enough SKU data to chart.")
        else:
            st.info("Add SKU and reimbursement columns to view the chart.")
    else:
        st.info("✅ No missing or unreimbursed inventory detected.")

# ---------------------- Sidebar Alerts ----------------------
if page == "Dashboard":
    st.sidebar.markdown("### 🚨 Alerts")
    if rule_df.empty:
        st.sidebar.success("No issues flagged ✅")
    else:
        st.sidebar.error(f"Issues flagged: {len(rule_df)}")
        issue_counts = rule_df["Issue"].value_counts()
        for issue, cnt in issue_counts.items():
            st.sidebar.write(f"- {issue}: {cnt}")
