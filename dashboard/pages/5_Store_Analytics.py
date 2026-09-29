import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.services.data_loader import (
    load_final_recommendations, load_raw_dataset, filter_dataframe
)
from dashboard.components.kpi_cards import render_kpi_card

st.markdown("## 🏪 Store Analytics & Operational Performance")
st.markdown("Multi-store operational benchmarking, demand growth, revenue generation, and risk concentration.")

# Load Data
df_recs = load_final_recommendations()
df_raw = load_raw_dataset()

# Page Level Filters
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    cities = sorted(df_raw['city'].dropna().unique()) if not df_raw.empty and 'city' in df_raw.columns else []
    selected_city = st.multiselect("Filter City", options=cities, default=[])

with col_f2:
    types = sorted(df_raw['store_type'].dropna().unique()) if not df_raw.empty and 'store_type' in df_raw.columns else []
    selected_type = st.multiselect("Filter Store Type", options=types, default=[])

with col_f3:
    regions = sorted(df_raw['region'].dropna().unique()) if not df_raw.empty and 'region' in df_raw.columns else []
    selected_region = st.multiselect("Filter Region", options=regions, default=[])

# Apply filters
filtered_raw = df_raw.copy()
filtered_recs = df_recs.copy()

if selected_city:
    filtered_raw = filtered_raw[filtered_raw['city'].isin(selected_city)]
    filtered_recs = filtered_recs[filtered_recs['city'].isin(selected_city)]

if selected_type:
    filtered_raw = filtered_raw[filtered_raw['store_type'].isin(selected_type)]
    filtered_recs = filtered_recs[filtered_recs['store_type'].isin(selected_type)]

if selected_region:
    filtered_raw = filtered_raw[filtered_raw['region'].isin(selected_region)]
    filtered_recs = filtered_recs[filtered_recs['region'].isin(selected_region)]

if filtered_raw.empty:
    st.warning("No store data matches the selected filters.")
    st.stop()

# Store-level Summaries
store_summary = filtered_raw.groupby('store_id').agg(
    total_revenue=('revenue', 'sum'),
    units_sold=('quantity', 'mean'),
    stockout_rate=('stockout_flag', lambda x: x.mean() * 100),
    avg_daily_cust=('avg_daily_customers', 'mean') if 'avg_daily_customers' in filtered_raw.columns else ('revenue', 'count')
).reset_index()

recs_summary = filtered_recs.groupby('store_id').agg(
    inventory_val=('current_stock', lambda x: (x * filtered_recs.loc[x.index, 'cost_price']).sum()),
    high_risk_skus=('risk_level', lambda x: (x == 'HIGH').sum()),
    avg_days_inv=('days_of_inventory', 'mean'),
    demand_growth=('demand_growth_7', 'mean')
).reset_index()

merged_store = pd.merge(store_summary, recs_summary, on='store_id', how='outer').fillna(0)

# Top Store KPIs
s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns(5)

s_col1.metric("Active Stores", f"{len(merged_store)}")
s_col2.metric("Total Revenue", f"₹{merged_store['total_revenue'].sum():,.0f}")
s_col3.metric("Avg Stock-out Rate", f"{merged_store['stockout_rate'].mean():.1f}%")
s_col4.metric("Avg Days of Inventory", f"{merged_store['avg_days_inv'].mean():.1f} days")
s_col5.metric("High-Risk SKUs", f"{merged_store['high_risk_skus'].sum()}")

st.markdown("<br>", unsafe_allow_html=True)

# Charts Grid
ch_col1, ch_col2 = st.columns(2)

with ch_col1:
    st.markdown("### 📈 Store Demand Trend Over Time")
    store_daily = filtered_raw.groupby(['date', 'store_id'])['quantity'].sum().reset_index()
    fig_sd = px.line(store_daily, x='date', y='quantity', color='store_id', title="Daily Units Demand by Store")
    fig_sd.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=320)
    st.plotly_chart(fig_sd, use_container_width=True)

with ch_col2:
    st.markdown("### ⚠️ Stock-out Rate by Store (%)")
    fig_sr = px.bar(merged_store, x='store_id', y='stockout_rate', color='stockout_rate', color_continuous_scale='Reds', title="Historical Stock-out Rate Comparison")
    fig_sr.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=320)
    st.plotly_chart(fig_sr, use_container_width=True)

ch_col3, ch_col4 = st.columns(2)

with ch_col3:
    st.markdown("### 💰 Revenue Generation by Store")
    fig_rev = px.pie(merged_store, values='total_revenue', names='store_id', hole=0.4, title="Revenue Share by Store")
    fig_rev.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=320)
    st.plotly_chart(fig_rev, use_container_width=True)

with ch_col4:
    st.markdown("### 📦 Risk Concentration & High-Risk SKUs")
    fig_hr = px.bar(merged_store, x='store_id', y='high_risk_skus', color='high_risk_skus', color_continuous_scale='Oranges', title="High-Risk SKUs per Store")
    fig_hr.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=320)
    st.plotly_chart(fig_hr, use_container_width=True)

# Store Benchmarking Table
st.markdown("### 📋 Store Comparison & Operational Benchmarking Table")

display_store = merged_store.copy()
display_store['total_revenue'] = display_store['total_revenue'].round(0)
display_store['stockout_rate'] = display_store['stockout_rate'].round(1).astype(str) + '%'
display_store['avg_days_inv'] = display_store['avg_days_inv'].round(1)

st.dataframe(
    display_store.rename(columns={
        'store_id': 'Store ID', 'total_revenue': 'Total Revenue (₹)',
        'units_sold': 'Avg Daily Units', 'stockout_rate': 'Stockout Rate',
        'inventory_val': 'Inventory Value (₹)', 'high_risk_skus': 'High Risk SKUs',
        'avg_days_inv': 'Avg Days Inv', 'demand_growth': 'Growth Factor'
    }),
    use_container_width=True,
    hide_index=True
)
