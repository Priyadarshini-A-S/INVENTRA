import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dashboard.services.data_loader import (
    load_final_recommendations, load_raw_dataset, load_master_dataset, load_model_metadata, filter_dataframe
)

st.markdown("## 📈 Demand Intelligence Center")
st.markdown("Analyze demand patterns, 7-day forecasts, growth trajectories, and volatility across retail locations.")

# Load Data
df_recs = load_final_recommendations()
df_raw = load_raw_dataset()
df_master = load_master_dataset()
metadata = load_model_metadata()

# Page Specific Filters
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    available_products = sorted(df_raw['product_id'].dropna().unique()) if not df_raw.empty else []
    selected_prod = st.multiselect("Filter Specific Product(s)", options=available_products, default=[])

with col_f2:
    if not df_raw.empty:
        min_d = df_raw['date'].min().date()
        max_d = df_raw['date'].max().date()
        date_range = st.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    else:
        date_range = None

with col_f3:
    categories = sorted(df_raw['category'].dropna().unique()) if not df_raw.empty else []
    selected_cat = st.multiselect("Filter Category", options=categories, default=[])

# Combine global + local filters
cities = st.session_state.get('global_cities', [])
stores = st.session_state.get('global_stores', [])
risks = st.session_state.get('global_risks', [])

cats_to_filter = selected_cat if selected_cat else st.session_state.get('global_categories', [])
prods_to_filter = selected_prod

filtered_raw = filter_dataframe(df_raw, cities=cities, stores=stores, categories=cats_to_filter, products=prods_to_filter)
filtered_recs = filter_dataframe(df_recs, cities=cities, stores=stores, categories=cats_to_filter, products=prods_to_filter, risk_levels=risks)

if filtered_raw.empty:
    st.warning("No historical demand records match the selected filters.")
    st.stop()

# Filter by date range if provided
if date_range and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_raw = filtered_raw[(filtered_raw['date'].dt.date >= start_d) & (filtered_raw['date'].dt.date <= end_d)]

# Model Forecast Error Metrics Panel
metrics = metadata.get('metrics', {})
if metrics:
    st.markdown("### 🎯 Model Forecast Accuracy Metrics")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Baseline MAE (Lag 7)", f"{metrics.get('demand_baseline_mae', 0):.2f}")
    m_col2.metric("Random Forest MAE", f"{metrics.get('demand_mae', 0):.2f}")
    m_col3.metric("RMSE", f"{metrics.get('demand_rmse', 0):.2f}")
    m_col4.metric("MAPE", f"{metrics.get('demand_mape', 0)*100:.1f}%")
    m_col5.metric("R² Score", f"{metrics.get('demand_r2', 0):.4f}")

st.markdown("<br>", unsafe_allow_html=True)

# 1. Historical vs 7-Day Forecast Horizon Chart
st.markdown("### 📊 Historical Demand vs 7-Day Forecast Horizon")

daily_hist = filtered_raw.groupby('date')['quantity'].sum().reset_index().sort_values('date')
last_date = daily_hist['date'].max()

total_forecast_7d = filtered_recs['forecast_7d_demand'].sum()
daily_forecast_avg = total_forecast_7d / 7.0

future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
future_df = pd.DataFrame({'date': future_dates, 'quantity': [daily_forecast_avg]*7})

fig_f = go.Figure()
fig_f.add_trace(go.Scatter(
    x=daily_hist['date'], y=daily_hist['quantity'],
    mode='lines+markers', name='Historical Actual Demand',
    line=dict(color='#38bdf8', width=2)
))
fig_f.add_trace(go.Scatter(
    x=future_df['date'], y=future_df['quantity'],
    mode='lines+markers', name='7-Day Predicted Forecast',
    line=dict(color='#f59e0b', width=3, dash='dash')
))
# Add vertical line for forecast horizon
fig_f.add_vline(x=last_date.timestamp() * 1000, line_width=2, line_dash="dot", line_color="#ef4444")
fig_f.add_annotation(x=last_date, y=daily_hist['quantity'].max(), text="Forecast Horizon Start", showarrow=True, arrowhead=1)

fig_f.update_layout(
    template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
    margin=dict(l=20, r=20, t=40, b=20), height=380,
    title="Time-Series Demand Curve (Historical Actual vs 7-Day Forecast Horizon)"
)
st.plotly_chart(fig_f, use_container_width=True)

# Trends Grid
t_col1, t_col2 = st.columns(2)

with t_col1:
    st.markdown("### 🏷️ Category Demand Trend")
    cat_trend = filtered_raw.groupby(['date', 'category'])['quantity'].sum().reset_index()
    fig_cat = px.line(
        cat_trend, x='date', y='quantity', color='category',
        title="Daily Demand by Product Category"
    )
    fig_cat.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=320)
    st.plotly_chart(fig_cat, use_container_width=True)

with t_col2:
    st.markdown("### 🏪 Store Demand Trend")
    store_trend = filtered_raw.groupby(['date', 'store_id'])['quantity'].sum().reset_index()
    fig_store = px.line(
        store_trend, x='date', y='quantity', color='store_id',
        title="Daily Demand across Retail Stores"
    )
    fig_store.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=320)
    st.plotly_chart(fig_store, use_container_width=True)

# Top Growing & Declining Products & Volatility
g_col1, g_col2, g_col3 = st.columns(3)

if not filtered_master.empty if 'filtered_master' in locals() else True:
    prod_growth = filtered_recs.groupby(['product_id', 'category']).agg(
        forecast_7d=('forecast_7d_demand', 'sum'),
        current_stock=('current_stock', 'sum'),
        growth=('demand_growth_7', 'mean'),
        volatility=('days_of_inventory', 'std')
    ).reset_index()
    
    with g_col1:
        st.markdown("### 🚀 Top Growing Products")
        top_growing = prod_growth.sort_values('growth', ascending=False).head(7)
        fig_grow = px.bar(top_growing, x='growth', y='product_id', orientation='h', color='growth', color_continuous_scale='Greens')
        fig_grow.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=300)
        st.plotly_chart(fig_grow, use_container_width=True)

    with g_col2:
        st.markdown("### 📉 Top Declining Products")
        top_declining = prod_growth.sort_values('growth', ascending=True).head(7)
        fig_dec = px.bar(top_declining, x='growth', y='product_id', orientation='h', color='growth', color_continuous_scale='Reds')
        fig_dec.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=300)
        st.plotly_chart(fig_dec, use_container_width=True)

    with g_col3:
        st.markdown("### ⚡ Highest Demand Volatility")
        prod_vol = filtered_raw.groupby('product_id')['quantity'].std().reset_index().rename(columns={'quantity': 'std_demand'})
        top_vol = prod_vol.sort_values('std_demand', ascending=False).head(7)
        fig_vol = px.bar(top_vol, x='std_demand', y='product_id', orientation='h', color='std_demand', color_continuous_scale='Purples')
        fig_vol.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=300)
        st.plotly_chart(fig_vol, use_container_width=True)

# Detailed Forecast Table
st.markdown("### 📋 7-Day Forecast Data Table")
f_table = filtered_recs[['store_id', 'product_id', 'category', 'current_stock', 'forecast_7d_demand', 'risk_level']].copy()
f_table['forecast_7d_demand'] = f_table['forecast_7d_demand'].round(2)
st.dataframe(f_table, use_container_width=True, hide_index=True)
