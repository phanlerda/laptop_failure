
"""
Optimized EDA script (static + interactive) for the laptop failure dataset.

What this script does:
1) Loads the cleaned CSV produced by 01_preprocess.py
2) Builds BOTH static charts (Matplotlib -> PNG) and interactive charts (Plotly -> single HTML)
3) Writes a rich HTML report with:
   - Overview & data quality
   - Distributions
   - Feature vs label insights
   - Correlations
   - Cohort views (by vendor)
Design choices:
- Keep functions small + reusable
- Comment every key step for clarity
"""

import os
import io
import base64
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Plotly is used for interactivity. If missing, we degrade gracefully. ---
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.offline import plot as plotly_to_html_div
    HAS_PLOTLY = True
except Exception as e:
    HAS_PLOTLY = False
    PLOTLY_IMPORT_ERROR = str(e)

# -------------------- Config & IO --------------------
INP = os.environ.get("INP", "data/processed/laptops_clean.csv")
REPORT_DIR = os.environ.get("REPORT_DIR", "reports")
FIG_DIR = f"{REPORT_DIR}/figures"
HTML_OUT = f"{REPORT_DIR}/eda.html"            # static HTML (PNG embedded)
HTML_OUT_INTERACTIVE = f"{REPORT_DIR}/eda_interactive.html"  # interactive Plotly report

os.makedirs(FIG_DIR, exist_ok=True)

# Load data; parse dates for time-derived features
df = pd.read_csv(INP, parse_dates=["purchase_date","warranty_end","retire_date"])

# -------------------- Helpers --------------------
def save_mpl(fig, name: str) -> str:
    """
    Save a Matplotlib figure to PNG and return path.
    """
    path = f"{FIG_DIR}/{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def img_to_base64(path: str) -> str:
    """
    Convert a PNG into base64 so we can inline it into HTML.
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def toc_block(items: List[Tuple[str,str]]) -> str:
    """
    Create a mini Table of Contents.
    items: list of (anchor, title)
    """
    lis = "".join([f"<li><a href='#{a}'>{t}</a></li>" for a,t in items])
    return f"<ul>{lis}</ul>"

def write_html_file(filename: str, title: str, sections: List[Tuple[str,str]]):
    """
    Write a simple HTML with sections.
    sections: list of (anchor, html_content)
    """
    toc = toc_block([(a, a.replace('_',' ').title()) for a,_ in sections])
    body = "".join([f"<h2 id='{a}'>{a.replace('_',' ').title()}</h2>{html}" for a,html in sections])
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
img {{ border: 1px solid #eee; padding: 6px; max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ddd; padding: 6px; font-size: 12px; }}
code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
.container {{ max-width: 1200px; margin: auto; }}
h1,h2 {{ color: #202630; }}
.note {{ color: #444; font-size: 0.95em; }}
</style>
</head>
<body>
<div class='container'>
<h1>{title}</h1>
<h3>Table of Contents</h3>
{toc}
{body}
</div>
</body>
</html>
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

# -------------------- Data Quality & Overview --------------------
# Compute missing ratio for each column (important for real-life "dirty" data)
missing_pct = (df.isna().mean()*100).sort_values(ascending=False).round(2)
desc = df.describe(include="all").transpose()

# Save CSVs for transparency
missing_pct.to_csv(f"{REPORT_DIR}/missing_percent.csv")
desc.to_csv(f"{REPORT_DIR}/describe.csv")

# Static barh for top missing columns (readable & compact)
top_miss = missing_pct.head(25)
fig = plt.figure(figsize=(8, 6))
plt.barh(top_miss.index[::-1], top_miss.values[::-1])
plt.xlabel("Missing (%)")
plt.title("Top Missing Columns")
missing_bar_path = save_mpl(fig, "barh_missing")

# -------------------- Distributions --------------------
# Histograms are essential to see skew and outliers

# 1) age_months
fig = plt.figure()
plt.hist(df["age_months"].dropna(), bins=30)
plt.title("Age (months) Distribution")
plt.xlabel("age_months"); plt.ylabel("count")
hist_age_path = save_mpl(fig, "hist_age_months")

# 2) battery_health
fig = plt.figure()
plt.hist(df["battery_health"].dropna(), bins=30)
plt.title("Battery Health Distribution")
plt.xlabel("battery_health"); plt.ylabel("count")
hist_bh_path = save_mpl(fig, "hist_battery_health")

# 3) CPU/GPU temperatures: boxplot (helps spot extreme temps)
fig = plt.figure()
# Matplotlib 3.9 deprecates 'labels' -> use 'tick_labels'

try:
    plt.boxplot([df["cpu_temp_max"].dropna(), df["gpu_temp_max"].dropna()], tick_labels=["CPU","GPU"])
except TypeError:
    # Fallback for Matplotlib < 3.9
    plt.boxplot([df["cpu_temp_max"].dropna(), df["gpu_temp_max"].dropna()], labels=["CPU","GPU"])

plt.title("Max Temperatures (Boxplot)")
plt.ylabel("°C")
box_temps_path = save_mpl(fig, "box_temps")

# -------------------- Label-aware insights --------------------
# We examine how features relate to failure/retire labels — crucial for downstream modeling
labels_exist = {"label_failure_90d" in df.columns, "label_retire_180d" in df.columns}

# Bar: failure rate by vendor (which cohorts hurt us most?)
if "label_failure_90d" in df.columns:
    rate_by_vendor = df.groupby("vendor")["label_failure_90d"].mean().sort_values(ascending=False)
    fig = plt.figure(figsize=(8,4))
    plt.bar(rate_by_vendor.index.astype(str), (100*rate_by_vendor.values))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Failure 90d rate (%)")
    plt.title("Failure Rate by Vendor")
    bar_fail_vendor_path = save_mpl(fig, "bar_fail_rate_vendor")
else:
    bar_fail_vendor_path = None

# Histogram overlay: battery_health by failure label (actionable: consider replacing batteries)
if "label_failure_90d" in df.columns:
    fig = plt.figure()
    plt.hist(df[df["label_failure_90d"]==0]["battery_health"].dropna(), bins=30, alpha=0.6, label="No fail")
    plt.hist(df[df["label_failure_90d"]==1]["battery_health"].dropna(), bins=30, alpha=0.6, label="Fail")
    plt.legend()
    plt.xlabel("battery_health"); plt.ylabel("count")
    plt.title("Battery Health vs Failure Label")
    hist_bh_by_label_path = save_mpl(fig, "hist_battery_health_by_label")
else:
    hist_bh_by_label_path = None

# Boxplot: CPU temp by failure label (thermal issues often precede HW tickets)
if "label_failure_90d" in df.columns:
    a = df[df["label_failure_90d"]==0]["cpu_temp_max"].dropna()
    b = df[df["label_failure_90d"]==1]["cpu_temp_max"].dropna()
    fig = plt.figure()
    
try:
    plt.boxplot([a, b], tick_labels=["No fail","Fail"])
except TypeError:
    plt.boxplot([a, b], labels=["No fail","Fail"])

    plt.title("CPU Temp by Failure Label"); plt.ylabel("°C")
    box_cpu_by_label_path = save_mpl(fig, "box_cpu_temp_by_label")
else:
    box_cpu_by_label_path = None

# -------------------- Correlations --------------------
# Heatmap on numeric features to quickly identify relationships / leakage risks
num_cols = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()

fig = plt.figure(figsize=(8,6))
plt.imshow(corr.values, aspect='auto')
plt.xticks(range(len(num_cols)), num_cols, rotation=90)
plt.yticks(range(len(num_cols)), num_cols)
plt.title("Correlation Heatmap (Numeric Features)")
plt.colorbar()
corr_path = save_mpl(fig, "corr_heatmap")

# -------------------- Interactive (Plotly) --------------------
# If Plotly is available, produce an interactive HTML with key charts.
interactive_sections = []
if HAS_PLOTLY:
    # Vendor counts (bar)
    vendor_counts = df["vendor"].value_counts().reset_index()
    vendor_counts.columns = ["vendor", "count"]  # đặt tên cột rõ ràng
    fig = px.bar(
        vendor_counts,
        x="vendor", y="count",
        title="Top Vendors"
)

    interactive_sections.append(("vendors", plotly_to_html_div(fig, include_plotlyjs=False, output_type="div")))

    # Battery health vs CPU temp (scatter, colored by label if exists)
    color_col = "label_failure_90d" if "label_failure_90d" in df.columns else None
    fig = px.scatter(
        df, x="battery_health", y="cpu_temp_max",
        color=color_col, opacity=0.6,
        hover_data=["asset_id","vendor","model","age_months"],
        title="Battery Health vs CPU Temp"
    )
    interactive_sections.append(("battery_vs_cpu", plotly_to_html_div(fig, include_plotlyjs=False, output_type="div")))

    # Failure rate by vendor (bar)
    if "label_failure_90d" in df.columns:
        fr = df.groupby("vendor")["label_failure_90d"].mean().sort_values(ascending=False).reset_index()
        fr["rate_%"] = (fr["label_failure_90d"]*100).round(2)
        fig = px.bar(fr, x="vendor", y="rate_%", title="Failure 90d Rate by Vendor (Interactive)")
        interactive_sections.append(("fail_rate_vendor", plotly_to_html_div(fig, include_plotlyjs=False, output_type="div")))

    # Age distribution (hist)
    fig = px.histogram(df, x="age_months", nbins=30, title="Age (months) Distribution")
    interactive_sections.append(("age_hist", plotly_to_html_div(fig, include_plotlyjs=False, output_type="div")))

    # Battery health distribution (hist)
    fig = px.histogram(df, x="battery_health", nbins=30, title="Battery Health Distribution")
    interactive_sections.append(("battery_health_hist", plotly_to_html_div(fig, include_plotlyjs=False, output_type="div")))

    # Compose interactive HTML (single file, includes plotly.js once)
    plotly_js = """<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>"""
    body = "".join([f"<h3 id='{a}'>{a.replace('_',' ').title()}</h3>{div}" for a,div in interactive_sections])
    html_interactive = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>EDA Interactive - Laptops</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
.container {{ max-width: 1200px; margin: auto; }}
</style>
{plotly_js}
</head>
<body>
<div class='container'>
<h1>EDA Interactive - Laptop Failure / Early Retirement</h1>
{body}
</div>
</body>
</html>
"""
    with open(HTML_OUT_INTERACTIVE, "w", encoding="utf-8") as f:
        f.write(html_interactive)

# -------------------- Static HTML report (with PNGs) --------------------
sections = []

# Overview
overview_html = f"""
<p class='note'>
Rows: <b>{len(df):,}</b> — Columns: <b>{len(df.columns)}</b><br/>
We prioritize practical views: distributions, label-conditioned charts, correlations, and cohort splits.
</p>
<h3>Missing Percentage (Top 25)</h3>
<img src='data:image/png;base64,{img_to_base64(missing_bar_path)}'/>
<h3>Describe (first 12 rows)</h3>
{desc.head(12).to_html()}
"""
sections.append(("overview", overview_html))

# Distributions
dist_html = f"""
<h3>Age (months)</h3>
<img src='data:image/png;base64,{img_to_base64(hist_age_path)}'/>
<h3>Battery Health</h3>
<img src='data:image/png;base64,{img_to_base64(hist_bh_path)}'/>
<h3>Max Temperatures (CPU/GPU)</h3>
<img src='data:image/png;base64,{img_to_base64(box_temps_path)}'/>
"""
sections.append(("distributions", dist_html))

# Label-aware
label_html_parts = []
if bar_fail_vendor_path:
    label_html_parts.append(f"<h3>Failure 90d Rate by Vendor</h3><img src='data:image/png;base64,{img_to_base64(bar_fail_vendor_path)}'/>")
if hist_bh_by_label_path:
    label_html_parts.append(f"<h3>Battery Health vs Failure Label</h3><img src='data:image/png;base64,{img_to_base64(hist_bh_by_label_path)}'/>")
if box_cpu_by_label_path:
    label_html_parts.append(f"<h3>CPU Temp by Failure Label</h3><img src='data:image/png;base64,{img_to_base64(box_cpu_by_label_path)}'/>")

sections.append(("label_insights", "".join(label_html_parts) if label_html_parts else "<p>No label columns found.</p>"))

# Correlations
corr_html = f"<img src='data:image/png;base64,{img_to_base64(corr_path)}'/>"
sections.append(("correlations", corr_html))

# Links to interactive (if available)
if HAS_PLOTLY:
    sections.append(("interactive_link", f"<p>Interactive version saved to <code>{HTML_OUT_INTERACTIVE}</code>.</p>"))
else:
    sections.append(("interactive_link", f"<p class='note'>Plotly not available: {PLOTLY_IMPORT_ERROR}. Install with <code>pip install plotly</code> to generate the interactive report.</p>"))

# Write final static HTML
write_html_file(HTML_OUT, "EDA Report - Laptop Failure / Early Retirement", sections)

print(f"Saved STATIC report -> {HTML_OUT}")
if HAS_PLOTLY:
    print(f"Saved INTERACTIVE report -> {HTML_OUT_INTERACTIVE}")
else:
    print("Interactive report skipped (Plotly not installed).")
