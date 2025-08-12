import re, os, sys, math, json
from datetime import datetime
import numpy as np
import pandas as pd

RAW = os.environ.get("RAW", "data/raw/laptops_dirty.csv")
OUT = os.environ.get("OUT", "data/processed/laptops_clean.csv")
OUT_FEATS = os.environ.get("OUT_FEATS", "data/processed/laptops_features.csv")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

def coerce_date(s):
    if pd.isna(s) or s == "" or s is None:
        return pd.NaT
    s = str(s).strip()
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    # thử nhiều định dạng phổ biến (Windows-friendly)
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d", "%d-%b-%Y"]
    for fmt in fmts:
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    # fallback linh hoạt
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def to_number(x):
    if pd.isna(x) or x == "" or x is None:
        return np.nan
    s = str(x).strip()
    s = s.replace(",", "")
    s = s.replace("—", "")
    s = re.sub(r"[^\d\.\-]", "", s)
    if s in ["", "-", "--"]:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def normalize_vendor(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower().replace(" ", "")
    mapping = {
        "lenovo":"Lenovo", "lenov0":"Lenovo", "lenvo":"Lenovo",
        "dell":"Dell", "delll":"Dell",
        "hp":"HP", "h-p":"HP",
        "apple":"Apple", "ap ple":"Apple",
        "asus":"Asus", "asuss":"Asus",
        "acer":"Acer", "âcer":"Acer",
        "msi":"MSI", "msi":"MSI", "msi":"MSI"
    }
    return mapping.get(s, "unknown")

def cap_outliers(series, lower_q=0.01, upper_q=0.99):
    if series.dtype.kind not in "biufc":
        return series
    q_low, q_hi = series.quantile(lower_q), series.quantile(upper_q)
    return series.clip(q_low, q_hi)

df = pd.read_csv(RAW)

# Deduplicate theo asset_id (giữ lần xuất hiện cuối)
if "asset_id" in df.columns:
    df = df.drop_duplicates(subset=["asset_id"], keep="last")

# Trim tiêu đề cột
df.columns = [c.strip() for c in df.columns]

# Parse date
for c in ["purchase_date", "warranty_end", "retire_date"]:
    if c in df.columns:
        df[c] = df[c].apply(coerce_date)

# Chuẩn hoá vendor/model
if "vendor" in df.columns:
    df["vendor"] = df["vendor"].apply(normalize_vendor)
if "model" in df.columns:
    df["model"] = df["model"].astype(str).str.strip()

# Ép số
num_cols = [
    "ram_gb","storage_gb","ticket_count_last_6m","bsod_cnt_30d",
    "battery_cycle","battery_design_cap","battery_full_cap",
    "cpu_temp_max","gpu_temp_max","thermal_throttle_cnt",
    "smart_realloc","smart_pending","disk_errors_30d",
    "uptime_hours_7d","patch_missing_cnt"
]
for c in num_cols:
    if c in df.columns:
        df[c] = df[c].apply(to_number)

# Sửa bất khả thi
if "battery_cycle" in df.columns:
    df.loc[df["battery_cycle"] < 0, "battery_cycle"] = np.nan
for c in ["cpu_temp_max","gpu_temp_max"]:
    if c in df.columns:
        df.loc[df[c] < 20, c] = np.nan
        df.loc[df[c] > 120, c] = np.nan

# Impute an toàn cho pandas 2.x: transform giữ index
for c in num_cols:
    if c not in df.columns: 
        continue
    med_by_vendor = df.groupby("vendor", group_keys=False)[c].transform("median")
    df[c] = df[c].fillna(med_by_vendor)
    med_global = df[c].median()
    df[c] = df[c].fillna(med_global)

# Impute categorical
for c in ["vendor","model","cpu","storage_type","os_version","location","status"]:
    if c in df.columns:
        df[c] = df[c].replace({"": np.nan})
        df[c] = df[c].fillna("unknown")

# Capping outliers
for c in ["cpu_temp_max","gpu_temp_max","battery_cycle","uptime_hours_7d"]:
    if c in df.columns:
        df[c] = cap_outliers(df[c])

# Feature engineering
today = pd.Timestamp("2025-08-13")
if "purchase_date" in df.columns:
    df["age_days"] = (today - df["purchase_date"]).dt.days
    df["age_days"] = df["age_days"].clip(lower=0)
    df["age_months"] = (df["age_days"]/30.0).round(1)
else:
    df["age_months"] = np.nan

if "warranty_end" in df.columns:
    df["in_warranty"] = (today <= df["warranty_end"]).astype(int)
else:
    df["in_warranty"] = 0

if {"battery_full_cap","battery_design_cap"}.issubset(df.columns):
    with np.errstate(divide='ignore', invalid='ignore'):
        df["battery_health"] = (df["battery_full_cap"] / df["battery_design_cap"]).clip(upper=1.2)
else:
    df["battery_health"] = np.nan

if "storage_type" in df.columns:
    df["is_nvme"] = (df["storage_type"].astype(str).str.upper().str.contains("NVME")).astype(int)
else:
    df["is_nvme"] = 0

if "os_version" in df.columns:
    df["is_mac"] = df["os_version"].astype(str).str.lower().str.contains("macos").astype(int)
else:
    df["is_mac"] = 0

# Lưu dữ liệu sạch
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)

# Feature table tối thiểu
feat_cols = [
    "asset_id","vendor","model","cpu","ram_gb","storage_gb","storage_type",
    "os_version","location","age_months","in_warranty","battery_health",
    "battery_cycle","cpu_temp_max","gpu_temp_max","thermal_throttle_cnt",
    "smart_realloc","smart_pending","disk_errors_30d","uptime_hours_7d",
    "patch_missing_cnt","ticket_count_last_6m","bsod_cnt_30d",
    "label_failure_90d","label_retire_180d"
]
feat_cols = [c for c in feat_cols if c in df.columns]
df[feat_cols].to_csv(OUT_FEATS, index=False)

print(f"Saved clean CSV to {OUT}")
print(f"Saved feature CSV to {OUT_FEATS}")
