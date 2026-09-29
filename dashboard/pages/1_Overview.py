import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dashboard.services.data_loader import (
    load_final_recommendations, load_raw_dataset, load_master_dataset, filter_dataframe
)
from dashboard.components.kpi_cards import render_kpi_card, get_risk_badge_html

# Load Data
df_recs = load_final_recommendations()
df_raw = load_raw_dataset()
df_master = load_master_dataset()

# Read global filters
cities = st.session_state.get('global_cities', [])
stores = st.session_state.get('global_stores', [])
categories = st.session_state.get('global_categories', [])
risks = st.session_state.get('global_risks', [])

filtered_recs = filter_dataframe(df_recs, cities=cities, stores=stores, categories=categories, risk_levels=risks)
filtered_raw = filter_dataframe(df_raw, cities=cities, stores=stores, categories=categories)

st.markdown("## 📊 Executive Command Center")

if filtered_recs.empty or filtered_raw.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# Calculations
total_revenue = filtered_raw['revenue'].sum()
total_units_sold = filtered_raw['quantity'].sum()
stockout_rate = filtered_raw['stockout_flag'].mean() * 100
total_inventory_val = filtered_recs['current_stock'].mul(filtered_recs['cost_price']).sum()

high_risk_df = filtered_recs[filtered_recs['risk_level'] == 'HIGH']
high_risk_count = len(high_risk_df)
high_risk_pct = (high_risk_count / len(filtered_recs) * 100) if len(filtered_recs) > 0 else 0

# Estimated lost sales: High risk items forecast demand * avg price
est_lost_sales = high_risk_df['forecast_7d_demand'].mul(high_risk_df['avg_selling_price']).sum()

# Top 6 KPI Cards
kpi_cols = st.columns(6)
with kpi_cols[0]:
    render_kpi_card("Total Revenue", f"₹{total_revenue:,.0f}", "Historical Total", "#38bdf8")
with kpi_cols[1]:
    render_kpi_card("Units Sold", f"{total_units_sold:,.0f}", "Historical Volume", "#6366f1")
with kpi_cols[2]:
    render_kpi_card("Stock-out Rate", f"{stockout_rate:.1f}%", "Historical Avg", "#ef4444" if stockout_rate > 10 else "#f59e0b")
with kpi_cols[3]:
    render_kpi_card("Inventory Value", f"₹{total_inventory_val:,.0f}", "Current Holding", "#10b981")
with kpi_cols[4]:
    render_kpi_card("High-Risk SKUs", f"{high_risk_count}", f"{high_risk_pct:.1f}% of catalog", "#ef4444")
with kpi_cols[5]:
    render_kpi_card("Lost Sales Exposure", f"₹{est_lost_sales:,.0f}", "Next 7 Days Risk", "#f59e0b")

st.markdown("<br>", unsafe_allow_html=True)

# Main Grid: Demand Outlook + Risk Distribution
col_left, col_right = st.columns([6, 4])

with col_left:
    st.markdown("### 📈 7-Day Demand Outlook")
    # Group historical daily demand
    daily_hist = filtered_raw.groupby('date')['quantity'].sum().reset_index()
    daily_hist = daily_hist.sort_values('date').tail(30) # Last 30 days
    
    last_hist_date = daily_hist['date'].max()
    future_dates = [last_hist_date + pd.Timedelta(days=i) for i in range(1, 8)]
    
    # 7-day predicted total demand
    total_forecast_7d = filtered_recs['forecast_7d_demand'].sum()
    daily_forecast_val = total_forecast_7d / 7.0
    
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'quantity': [daily_forecast_val] * 7,
        'type': ['Predicted Forecast'] * 7
    })
    
    daily_hist['type'] = 'Historical Actual'
    
    combined_demand = pd.concat([daily_hist[['date', 'quantity', 'type']], forecast_df])
    
    fig_demand = px.line(
        combined_demand, x='date', y='quantity', color='type',
        color_discrete_map={'Historical Actual': '#38bdf8', 'Predicted Forecast': '#f59e0b'},
        title="Daily Total Retail Demand (Historical Actual vs 7-Day Predicted Outlook)"
    )
    fig_demand.update_layout(
        template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
        margin=dict(l=20, r=20, t=40, b=20), height=320, legend_title_text=""
    )
    st.plotly_chart(fig_demand, use_container_width=True)

with col_right:
    st.markdown("### 🛡️ Inventory Risk Distribution")
    risk_counts = filtered_recs['risk_level'].value_counts().reset_index()
    risk_counts.columns = ['risk_level', 'count']
    
    fig_risk = px.pie(
        risk_counts, values='count', names='risk_level',
        color='risk_level',
        color_discrete_map={'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'},
        hole=0.5,
        title="Stockout Risk Level Breakup"
    )
    fig_risk.update_layout(
        template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
        margin=dict(l=20, r=20, t=40, b=20), height=320
    )
    st.plotly_chart(fig_risk, use_container_width=True)

# Business Impact Panel
st.markdown("### ⚡ Operational Business Impact")
b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)

items_needing_action = len(filtered_recs[filtered_recs['action'] != 'NO ACTION REQUIRED'])
units_to_replenish = filtered_recs['shortage_before_transfer'].sum()
transfer_opps = filtered_recs['transfer_quantity'].sum()
supplier_orders_req = filtered_recs['supplier_order_quantity'].sum()

b_col1.metric("Items Needing Action", f"{items_needing_action}")
b_col2.metric("Replenishment Units", f"{units_to_replenish:,.0f}")
b_col3.metric("Transfer Opportunities", f"{transfer_opps:,.0f} units")
b_col4.metric("Supplier Orders Req.", f"{supplier_orders_req:,.0f} units")
b_col5.metric("Cost Exposure", f"₹{est_lost_sales:,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# Top Actions Required Table
st.markdown("### 🚨 Prioritized Top Actions Required")

# Sort by urgency: HIGH risk first, highest stockout_probability second, highest shortage third
top_actions = filtered_recs.sort_values(
    by=['stockout_probability', 'shortage_before_transfer'], ascending=[False, False]
).head(15)

display_table = top_actions[[
    'store_id', 'product_id', 'category', 'forecast_7d_demand', 'current_stock',
    'incoming_stock', 'stockout_probability', 'risk_level', 'action', 'transfer_quantity', 'supplier_order_quantity'
]].copy()

display_table['forecast_7d_demand'] = display_table['forecast_7d_demand'].round(1)
display_table['stockout_probability'] = (display_table['stockout_probability'] * 100).round(1).astype(str) + '%'
display_table['transfer_quantity'] = display_table['transfer_quantity'].round(1)
display_table['supplier_order_quantity'] = display_table['supplier_order_quantity'].round(1)

st.dataframe(
    display_table.rename(columns={
        'store_id': 'Store', 'product_id': 'Product', 'category': 'Category',
        'forecast_7d_demand': '7-Day Forecast', 'current_stock': 'Current Stock',
        'incoming_stock': 'Incoming Stock', 'stockout_probability': 'Stockout Prob',
        'risk_level': 'Risk', 'action': 'Recommended Action',
        'transfer_quantity': 'Transfer Qty', 'supplier_order_quantity': 'Supplier Order Qty'
    }),
    use_container_width=True,
    hide_index=True
)
