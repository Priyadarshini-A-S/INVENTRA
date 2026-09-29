import streamlit as st
import pandas as pd

from dashboard.services.data_loader import (
    load_quality_report, load_eda_summary
)
from dashboard.components.kpi_cards import render_kpi_card

st.markdown("## 🔍 Data Quality & Governance Report")
st.markdown("Automated data auditing, data coverage metrics, anomaly checks, and data quality status.")

# Load Quality & EDA Reports
quality_report = load_quality_report()
eda_summary = load_eda_summary()

if not quality_report:
    st.warning("Data quality report artifact not found. Run python src/generate_reports.py to generate.")
    st.stop()

coverage = quality_report.get('coverage', {})
quality_checks = quality_report.get('quality_checks', [])

# 1. Dataset Coverage KPIs
st.markdown("### 📅 Dataset Coverage Statistics")

c_col1, c_col2, c_col3, c_col4, c_col5 = st.columns(5)
c_col1.metric("Date Range", f"{coverage.get('min_date', '')} to {coverage.get('max_date', '')}")
c_col2.metric("Total Rows Processed", f"{coverage.get('total_rows', 0):,}")
c_col3.metric("Stores Covered", f"{coverage.get('num_stores', 0)}")
c_col4.metric("Active SKUs (Products)", f"{coverage.get('num_products', 0)}")
c_col5.metric("Product Categories", f"{coverage.get('num_categories', 0)}")

st.markdown("<br>", unsafe_allow_html=True)

# 2. Data Quality Audit Checks Table
st.markdown("### 🛡️ Data Quality Audit Checks & Issue Resolution")

if quality_checks:
    df_qc = pd.DataFrame(quality_checks)
    
    # Format status badges
    def format_status(val):
        if val == 'PASS':
            return '✅ PASS'
        elif val == 'FIXED':
            return '🛠️ FIXED'
        else:
            return '⚠️ WARNING'
            
    df_qc['status_icon'] = df_qc['status'].apply(format_status)
    
    st.dataframe(
        df_qc[['issue', 'count', 'action_taken', 'status_icon']].rename(columns={
            'issue': 'Data Issue Category',
            'count': 'Detected Count',
            'action_taken': 'Pipeline Action Taken',
            'status_icon': 'Audit Status'
        }),
        use_container_width=True,
        hide_index=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# 3. Statistical EDA Validation Highlights
st.markdown("### 📊 Statistical EDA Validation Highlights")

if eda_summary:
    tab_promo, tab_st = st.tabs(["Promotion vs Demand Impact", "Store Format Performance"])
    
    with tab_promo:
        st.markdown("#### Promotion Impact Analysis")
        promo_list = eda_summary.get('promo_impact', [])
        if promo_list:
            df_promo = pd.DataFrame(promo_list)
            df_promo['promotion_flag'] = df_promo['promotion_flag'].map({0: 'Non-Promotional Days', 1: 'Promotional Days'})
            st.dataframe(df_promo.rename(columns={
                'promotion_flag': 'Day Type',
                'avg_demand': 'Avg Daily Demand (Units)',
                'avg_revenue': 'Avg Daily Revenue (₹)',
                'stockout_rate': 'Stock-out Frequency (%)'
            }), use_container_width=True, hide_index=True)
            
    with tab_st:
        st.markdown("#### Demand Across Store Types")
        st_list = eda_summary.get('store_type_performance', [])
        if st_list:
            df_st = pd.DataFrame(st_list)
            st.dataframe(df_st.rename(columns={
                'store_type': 'Store Format',
                'avg_demand': 'Avg Demand',
                'avg_revenue': 'Avg Revenue (₹)',
                'total_revenue': 'Total Revenue (₹)',
                'stockout_rate': 'Stockout Rate (%)'
            }), use_container_width=True, hide_index=True)
