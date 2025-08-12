
# Laptop Failure Prediction Project – Documentation

## 1. Project Overview
This project aims to **predict whether a laptop is likely to fail or be retired early** based on hardware, software, and usage telemetry data. The work so far covers:

- **Synthetic data generation** mimicking real-world dirty datasets.
- **Data preprocessing** to clean, impute, normalize, and engineer features.
- **Exploratory Data Analysis (EDA)** in both static (matplotlib/seaborn) and interactive (Plotly) formats.

---

## 2. Data Collection

Since no production dataset was available, we **generated ~500 rows of synthetic laptop data** with intentionally “dirty” qualities to simulate reality:

- **Random typos** in categorical fields (e.g., vendor names like `lenov0`).
- **Mixed formats** for dates (`2022-01-10`, `10/01/2022`, etc.).
- **Numeric fields with units** (`65°C`, `1,024 MB`).
- **Missing values** distributed across multiple columns.
- **Outliers** in temperature, battery cycle, and other numeric metrics.

The generated file is:
```
data/raw/laptops_dirty.csv
```

---

## 3. Data Preprocessing (`src/01_preprocess.py`)

### 3.1 Objectives
- Standardize formats (dates, strings, numbers).
- Remove duplicates.
- Handle missing values with meaningful imputations.
- Cap extreme outliers.
- Create engineered features for modeling.

### 3.2 Key Steps

1. **Deduplication**
   - Drop duplicates based on `asset_id` (keeping latest).

2. **Column Cleaning**
   - Strip spaces in headers.
   - Convert date columns with multiple formats → pandas datetime.

3. **Vendor Normalization**
   - Map known typos/variants to a canonical name.

4. **Numeric Conversion**
   - Strip units/symbols → float.
   - Handle negatives and impossible values (e.g., CPU temp < 20°C → NaN).

5. **Missing Value Imputation**
   - **Numerics**: median per `vendor` (via `groupby().transform("median")`), fallback global median.
   - **Categoricals**: fill `"unknown"`.

6. **Outlier Capping**
   - Clip to 1st–99th percentile for selected columns.

7. **Feature Engineering**
   - `age_months`: time since purchase.
   - `in_warranty`: binary flag.
   - `battery_health`: ratio `full_cap/design_cap`.
   - `is_nvme`, `is_mac`: binary indicators.

8. **Outputs**
   - `data/processed/laptops_clean.csv`: fully cleaned dataset.
   - `data/processed/laptops_features.csv`: modeling-ready subset.

---

## 4. Exploratory Data Analysis (`src/02_eda.py`)

### 4.1 Objectives
- Understand data distribution and quality.
- Explore relationships between features and target labels.
- Identify potential data leakage risks.
- Create insights for model feature selection.

### 4.2 Static EDA (Matplotlib/Seaborn)
- **Missing Values**: horizontal bar plot for top 25 columns.
- **Numeric Distributions**: histograms + KDE curves.
- **Boxplots**:
  - CPU vs GPU temperatures.
  - Battery health across failure labels.
- **Correlation Heatmap**: highlight strong feature relationships.

### 4.3 Label-based Analysis
- Failure rate by vendor, battery health, temperature.
- Vendor share and associated risk.

### 4.4 Interactive EDA (Plotly)
- **Bar charts** for vendor distribution.
- **Histograms** for battery health, CPU temp, etc.
- **Hover info** for precise inspection.
- HTML report (`reports/eda_interactive.html`) for dynamic exploration.

### 4.5 Outputs
- `reports/eda.html`: static plots.
- `reports/eda_interactive.html`: interactive dashboard.

---

## 5. Current Status

✅ Synthetic dataset generated and stored.  
✅ Preprocessing pipeline implemented, tested, and made Pandas 2.x compatible.  
✅ EDA implemented with both static and interactive outputs.  
✅ Outlier handling and vendor normalization in place.  

---

## 6. Next Steps

1. **Feature Selection & Modeling**
   - Baseline LightGBM / XGBoost models for 90-day failure & 180-day retirement.
   - Cross-validation and performance metrics (AUC-PR, Precision@K).

2. **Model Explainability**
   - SHAP values for global and local interpretations.

3. **Operational Integration**
   - MLflow tracking.
   - Batch inference for risk prioritization.

4. **Cohort Analysis**
   - Add failure rates by age buckets and warranty status.

---

## 7. Project Structure
```
laptop_failure_mini/
│
├── data/
│   ├── raw/
│   │   └── laptops_dirty.csv
│   └── processed/
│       ├── laptops_clean.csv
│       └── laptops_features.csv
│
├── reports/
│   ├── eda.html
│   └── eda_interactive.html
│
├── src/
│   ├── 00_generate_data.py
│   ├── 01_preprocess.py
│   └── 02_eda.py
│
└── README.md
```

---

## 8. Key Technical Decisions

- **Imputation via transform**: safer with Pandas 2.x, preserves index.
- **Dual EDA outputs**: static for portability, interactive for depth.
- **Synthetic “dirty” data**: ensures pipeline robustness before applying to production data.
- **Feature engineering** chosen for interpretability and immediate model utility.
