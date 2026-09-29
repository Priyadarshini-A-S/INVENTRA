# INVENTRA — Retail Demand & Inventory Intelligence

> **IntelliData 2026 Data Science Hackathon**  
> *A Management-Ready, Enterprise Retail Decision-Support System: Predict → Explain → Act*

---

## 📌 Executive Overview & Business Problem

Retail operations across multi-city store networks face a persistent challenge: balancing inventory levels to eliminate costly stock-outs without over-allocating working capital in safety stock.

**INVENTRA** is an end-to-end retail demand forecasting, stock-out risk classification, and inter-store inventory optimization platform. It bridges the gap between machine learning predictions and manager actions by executing a 3-step core operational principle:

1. **Predict**: Forecast next 7-day demand and estimate stock-out probability for every Store × Product SKU pair.
2. **Explain**: Provide feature importance and deterministic natural-language reasoning behind every risk classification.
3. **Act**: Execute smart inter-store inventory transfer recommendations to fulfill shortages from regional surplus stores before issuing supplier purchase orders.

---

## 🏗️ System Architecture

```
                       RAW DATA (master_cleaned.csv)
                                   │
                                   ▼
                             DATA CLEANING
                                   │
                                   ▼
                       FEATURE ENGINEERING (src/feature_engineering.py)
                       (Lags, Rolling Means, Volatility, Days of Inventory)
                                   │
                                   ▼
                         MASTER FEATURE DATASET
                                   │
          ┌────────────────────────┴────────────────────────┐
          ▼                                                 ▼
 DEMAND FORECAST MODEL                            STOCKOUT RISK MODEL
(RandomForestRegressor)                          (RandomForestClassifier)
          │                                                 │
          ▼                                                 ▼
 7-Day Forecast Demand                             Stockout Probability
          │                                                 │
          └────────────────────────┬────────────────────────┘
                                   ▼
                       RISK LEVEL CLASSIFICATION
                       (HIGH ≥70%, MED 40-70%, LOW <40%)
                                   │
                                   ▼
                      SMART INVENTORY OPTIMIZER
                       (src/reorder_engine.py & src/transfer_engine.py)
                       - Safety Stock Calculation (z=1.65)
                       - Inter-Store Surplus/Shortage Matching
                       - Transfer-Before-Purchase Logic
                                   │
                                   ▼
                       UNIFIED PREDICTION CONTRACT
                     (data/processed/final_recommendations.csv)
                                   │
                                   ▼
                     INVENTRA DASHBOARD SYSTEM (dashboard/app.py)
                    (9 Bloomberg-Style Operations Pages)
```

---

## 📦 Data Pipeline & Feature Engineering

The feature engineering module (`src/feature_engineering.py`) transforms raw transaction records into a machine-learning-ready time-series dataset:

- **Target Variables**:
  - `next_7_day_demand`: Rolling 7-day future demand sum.
  - `stockout_flag`: Binary indicator (1 when stock out occurs, 0 otherwise).
- **Time Features**: `day_of_week`, `month`, `quarter`, `weekend_flag`.
- **Lag Features**: Lags at 1, 2, 3, 7, 14, and 28 days (`lag_1` ... `lag_28`).
- **Rolling Window Statistics**: Moving averages (`rolling_mean_3`, `7`, `14`, `28`), standard deviations (`rolling_std_7`, `14`), and min/max ranges.
- **Demand Trend & Volatility**: `demand_growth_7` (7d mean / 14d mean), `demand_growth_14`.
- **Inventory Metrics**: `days_of_inventory` (closing stock / 7d rolling mean), `reorder_gap` (`reorder_lvl - closing`).
- **External Factors**: `temp_c`, `rain_mm`, `holiday`, `festival`, `promotion_flag`.

---

## 🤖 Machine Learning Models & Validation Strategy

In compliance with hackathon regulations, model architectures strictly utilize non-deep-learning algorithms:

### 1. Demand Forecasting Model
- **Algorithm**: `RandomForestRegressor` (`n_estimators=30`, `max_depth=8`)
- **Baseline**: 7-Day Lag Baseline (`lag_7`)
- **Evaluation Split**: Chronological Train (70%) / Validation (15%) / Test (15%)
- **Validation Metrics**:
  - **Model MAE**: `4.63` units (vs Baseline MAE: `29.81` units — **84.5% error reduction**)
  - **RMSE**: `6.80` units
  - **MAPE**: `14.3%`
  - **R² Score**: `0.9203`

### 2. Stock-out Risk Classification Model
- **Algorithm**: `RandomForestClassifier` (`n_estimators=30`, `max_depth=8`, `class_weight='balanced'`)
- **Risk Thresholds**:
  - **HIGH RISK**: Stockout Probability $\ge 0.70$
  - **MEDIUM RISK**: $0.40 \le \text{Probability} < 0.70$
  - **LOW RISK**: Probability $< 0.40$
- **Validation Metrics**:
  - **Accuracy**: `99.8%`
  - **Precision**: `1.000`
  - **Recall**: `1.000`
  - **ROC-AUC**: `1.0000`

---

## 🧠 Decision Engine & Smart Transfer Logic

The Smart Transfer Engine (`src/transfer_engine.py`) implements a business logic optimizer to reduce supplier procurement costs:

1. **Shortage Identification**: Computes `safety_stock = 1.65 * rolling_std_7 * sqrt(lead_days)` and `recommended_stock = forecast_7d + safety_stock`. Shortage is identified when `current_stock + incoming_stock < recommended_stock`.
2. **Surplus Identification**: Stores with `current_stock + incoming_stock > recommended_stock` are identified as surplus candidates.
3. **Inter-Store Matching**: For every product experiencing a shortage in Store A, the engine searches regional Store B with surplus inventory. Surplus is transferred first to satisfy the deficit.
4. **Supplier Fallback**: Only remaining deficits after all inter-store transfer options are exhausted generate supplier purchase orders.

> **Impact**: In evaluation, out of **1,225.5 total shortage units**, **1,192.5 units (97.3%)** were satisfied via inter-store transfers, reducing supplier orders from 1,225.5 to just 33 units!

---

## 💡 Explainability Framework

INVENTRA provides dual-layer explainability:

1. **Global Model Feature Importance**: Extracted directly from model trees to highlight overall drivers (`closing`, `rolling_mean_7`, `lag_7`, `days_of_inventory`, `demand_growth_7`).
2. **Deterministic Item Diagnosis**: Synthesizes natural-language explanations from actual item feature values (e.g., active promotion, weekend footfall surge, recent demand acceleration, low days of supply).

---

## 🖥️ Dashboard Application Structure

The Streamlit dashboard (`dashboard/app.py`) features an enterprise dark-slate Bloomberg aesthetic:

| # | Page | Key Functionality |
|---|---|---|
| 1 | **Executive Overview** | Command center with 6 KPI cards, 7-day demand outlook curve, risk distribution pie, prioritized actions table, and business impact panel. |
| 2 | **Demand Intelligence** | Time-series demand curves, category & store trends, top growing/declining products, volatility index, and model accuracy metrics. |
| 3 | **Inventory Risk** | Risk matrix table, risk by store heatmap, category risk breakdown, stockout probability distribution, and top 10 interactive inspector. |
| 4 | **Manager Action Center** | Interactive action cards, visual shortage-to-transfer decision flow, WHY explanation bullets, and simulated Transfer / Order workflow buttons. |
| 5 | **Store Analytics** | Store operational benchmarking, revenue breakdown, stock-out rate comparison, high-risk SKU concentration, and store mix table. |
| 6 | **Product Analytics** | Catalog search deep-dive, product profile (margin, shelf life, supplier), demand history, inventory positioning, and store breakdown matrix. |
| 7 | **Model Performance** | Detailed ML diagnostics, actual vs predicted scatter plots, residual distribution, confusion matrix, class distribution, and comparison table. |
| 8 | **Explainability** | "Why is this item at risk?" selection tool, global feature importance charts, and deterministic natural-language risk diagnosis. |
| 9 | **Data Quality** | Comprehensive governance report, issue table with `PASS`/`WARNING`/`FIXED` badges, data coverage statistics, and statistical EDA highlights. |

---

## ⚡ Installation & Execution Guide

### Prerequisites
- Python 3.10+
- Requirements installed (`pandas`, `numpy`, `scikit-learn`, `joblib`, `plotly`, `streamlit`)

### Installation
```bash
pip install -r requirements.txt
```

### Running the End-to-End Pipeline
```bash
# 1. Feature Engineering
python src/feature_engineering.py

# 2. Train ML Models & Save Metadata
python src/train_models.py

# 3. Run Inference & Smart Transfer Engine
python src/inference_pipeline.py

# 4. Generate Quality & EDA Reports
python src/generate_reports.py
```

### Launching the Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📁 Repository Structure

```
INVENTRA/
├── data/
│   ├── raw/
│   │   └── master_cleaned.csv            # Raw dataset
│   └── processed/
│       ├── master_dataset.csv            # Feature engineered master dataset
│       └── final_recommendations.csv     # Unified prediction & recommendation contract
├── models/
│   ├── demand_forecast_model.pkl         # Trained Random Forest Regressor
│   ├── stockout_risk_model.pkl           # Trained Random Forest Classifier
│   └── model_metadata.pkl                # Serialized model metrics & feature importances
├── reports/
│   ├── model_comparison.csv              # Regression comparison table
│   ├── model_comparison.md               # Detailed markdown model report
│   ├── leakage_audit.md                  # Data leakage audit checklist
│   ├── data_quality_report.json          # Automated data quality audit JSON
│   └── eda_summary.json                  # Statistical EDA summary JSON
├── src/
│   ├── feature_engineering.py            # Feature engineering pipeline
│   ├── train_models.py                   # Model training & serialization script
│   ├── predict.py                        # StockSensePredictor class
│   ├── reorder_engine.py                 # Safety stock & shortage calculator
│   ├── transfer_engine.py                # Smart inter-store transfer optimizer
│   ├── inference_pipeline.py             # End-to-end inference execution script
│   └── generate_reports.py               # Data quality & EDA report generator
├── dashboard/
│   ├── app.py                            # Streamlit entry point & global navigation
│   ├── assets/
│   │   └── style.css                     # Custom enterprise CSS stylesheet
│   ├── services/
│   │   └── data_loader.py                # Cached data loading & filtering service
│   ├── components/
│   │   ├── header.py                     # Top header bar component
│   │   └── kpi_cards.py                  # Styled KPI metric cards & badges
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Demand_Intelligence.py
│       ├── 3_Inventory_Risk.py
│       ├── 4_Manager_Action_Center.py
│       ├── 5_Store_Analytics.py
│       ├── 6_Product_Analytics.py
│       ├── 7_Model_Performance.py
│       ├── 8_Explainability.py
│       └── 9_Data_Quality.py
├── requirements.txt
└── README.md                             # Documentation
```

---

## ⚠️ Known Limitations & Future Improvements

1. **Lead Time Variability**: Lead days are currently treated as constant per product; future iterations can integrate stochastic supplier lead-time modeling.
2. **Transportation Cost Matrix**: Inter-store transfers currently assume uniform transfer cost across stores; adding real-world distance matrix routing can optimize logistics expenses.
3. **Multi-Horizon Forecasting**: Expanding from a 7-day rolling window to 14-day and 30-day multi-horizon forecasting will further assist long-lead procurement.
