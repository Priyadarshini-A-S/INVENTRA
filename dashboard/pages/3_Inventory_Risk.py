import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from dashboard.services.data_loader import (
    load_final_recommendations, filter_dataframe
)
from dashboard.components.kpi_cards import render_kpi_card, get_risk_badge_html

st.markdown("## ⚠️ Inventory Risk Monitor")
st.markdown("Real-time risk assessment and stockout probability matrix across stores and SKU catalog.")

# Load Data
df_recs = load_final_recommendations()

# Apply Global Filters
cities = st.session_state.get('global_cities', [])
stores = st.session_state.get('global_stores', [])
categories = st.session_state.get('global_categories', [])
risks = st.session_state.get('global_risks', [])

filtered_recs = filter_dataframe(df_recs, cities=cities, stores=stores, categories=categories, risk_levels=risks)

if filtered_recs.empty:
    st.warning("No inventory records match the selected filters.")
    st.stop()

# Summary Metrics
high_cnt = len(filtered_recs[filtered_recs['risk_level'] == 'HIGH'])
med_cnt = len(filtered_recs[filtered_recs['risk_level'] == 'MEDIUM'])
low_cnt = len(filtered_recs[filtered_recs['risk_level'] == 'LOW'])

r_col1, r_col2, r_col3, r_col4 = st.columns(4)
with r_col1:
    render_kpi_card("Total Assessed SKUs", f"{len(filtered_recs)}", "Active Store-Product Pairs", "#38bdf8")
with r_col2:
    render_kpi_card("High Risk (≥70%)", f"{high_cnt}", "Immediate Action Req.", "#ef4444")
with r_col3:
    render_kpi_card("Medium Risk (40-70%)", f"{med_cnt}", "Requires Monitoring", "#f59e0b")
with r_col4:
    render_kpi_card("Low Risk (<40%)", f"{low_cnt}", "Sufficient Stock", "#10b981")

st.markdown("<br>", unsafe_allow_html=True)

# Risk Heatmap & Charts
c_left, c_right = st.columns([6, 4])

with c_left:
    st.markdown("### 🗺️ Risk Distribution by Store")
    store_risk = filtered_recs.groupby(['store_id', 'risk_level']).size().unstack(fill_value=0)
    for r in ['HIGH', 'MEDIUM', 'LOW']:
        if r not in store_risk.columns:
            store_risk[r] = 0
    store_risk = store_risk[['HIGH', 'MEDIUM', 'LOW']]
    
    fig_store_risk = px.bar(
        store_risk, barmode='stack',
        color_discrete_map={'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'},
        title="Risk Severity Concentration across Stores"
    )
    fig_store_risk.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=340)
    st.plotly_chart(fig_store_risk, use_container_width=True)

with c_right:
    st.markdown("### 📊 Risk by Product Category")
    cat_risk = filtered_recs.groupby(['category', 'risk_level']).size().unstack(fill_value=0)
    for r in ['HIGH', 'MEDIUM', 'LOW']:
        if r not in cat_risk.columns:
            cat_risk[r] = 0
    cat_risk = cat_risk[['HIGH', 'MEDIUM', 'LOW']]
    
    fig_cat_risk = px.bar(
        cat_risk, barmode='group',
        color_discrete_map={'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'},
        title="Risk Breakdown by Category"
    )
    fig_cat_risk.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=340)
    st.plotly_chart(fig_cat_risk, use_container_width=True)

# Stock-out Probability Distribution
st.markdown("### 📉 Stock-out Probability Density Distribution")
fig_prob = px.histogram(
    filtered_recs, x='stockout_probability', nbins=30, color='risk_level',
    color_discrete_map={'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'},
    title="Model Predicted Stock-out Probability Histogram"
)
fig_prob.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=280)
st.plotly_chart(fig_prob, use_container_width=True)

# Top 10 Highest Risk Products & Interactive Inspector
st.markdown("### 🚨 Top 10 Highest-Risk Products")

top10 = filtered_recs.sort_values(by='stockout_probability', ascending=False).head(10)

selected_row_idx = st.selectbox(
    "Select a product from Top 10 for detailed Risk Diagnostics panel:",
    options=top10.index,
    format_func=lambda idx: f"{top10.loc[idx, 'store_id']} | {top10.loc[idx, 'product_id']} ({top10.loc[idx, 'category']}) - Risk Prob: {top10.loc[idx, 'stockout_probability']*100:.1f}%"
)

if selected_row_idx is not None:
    item = top10.loc[selected_row_idx]
    
    st.markdown(f"""
    <div style="background: #1e293b; border-left: 4px solid #ef4444; padding: 1.2rem; border-radius: 8px; margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0; color: #f8fafc;">STORE: {item['store_id']} | PRODUCT: {item['product_id']} ({item['category']})</h4>
            {get_risk_badge_html(item['risk_level'])}
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem; background: #0f172a; padding: 1rem; border-radius: 6px;">
            <div><span style="font-size: 0.75rem; color: #94a3b8;">7-DAY DEMAND:</span><br><strong style="font-size: 1.1rem; color: #f8fafc;">{item['forecast_7d_demand']:.1f} units</strong></div>
            <div><span style="font-size: 0.75rem; color: #94a3b8;">CURRENT STOCK:</span><br><strong style="font-size: 1.1rem; color: #f8fafc;">{item['current_stock']:.1f} units</strong></div>
            <div><span style="font-size: 0.75rem; color: #94a3b8;">INCOMING STOCK:</span><br><strong style="font-size: 1.1rem; color: #f8fafc;">{item['incoming_stock']:.1f} units</strong></div>
            <div><span style="font-size: 0.75rem; color: #94a3b8;">STOCK-OUT PROB:</span><br><strong style="font-size: 1.1rem; color: #ef4444;">{item['stockout_probability']*100:.1f}%</strong></div>
        </div>
        <div style="margin-top: 1rem; color: #cbd5e1; font-size: 0.9rem;">
            <strong>Recommended Action:</strong> <span style="color: #38bdf8; font-weight: 600;">{item['action']}</span><br>
            <strong>Reasoning:</strong> {item['reason']}<br>
            <strong>Inter-Store Transfer Source:</strong> {item['source_store_id']} ({item['transfer_quantity']:.1f} units available)<br>
            <strong>Supplier Order Requirement:</strong> {item['supplier_order_quantity']:.1f} units
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main Inventory Risk Matrix Table
st.markdown("### 📋 Complete Inventory Risk Matrix Table")

matrix_df = filtered_recs[[
    'store_id', 'product_id', 'category', 'current_stock', 'incoming_stock',
    'forecast_7d_demand', 'stockout_probability', 'risk_level', 'days_of_inventory',
    'reorder_gap', 'recommended_stock'
]].copy()

matrix_df['forecast_7d_demand'] = matrix_df['forecast_7d_demand'].round(1)
matrix_df['stockout_probability'] = (matrix_df['stockout_probability'] * 100).round(1).astype(str) + '%'
matrix_df['days_of_inventory'] = matrix_df['days_of_inventory'].round(1)
matrix_df['reorder_gap'] = matrix_df['reorder_gap'].round(1)

st.dataframe(
    matrix_df.rename(columns={
        'store_id': 'Store', 'product_id': 'Product', 'category': 'Category',
        'current_stock': 'Current Stock', 'incoming_stock': 'Incoming Stock',
        'forecast_7d_demand': '7-Day Demand', 'stockout_probability': 'Stockout Prob',
        'risk_level': 'Risk Level', 'days_of_inventory': 'Days of Inv',
        'reorder_gap': 'Reorder Gap', 'recommended_stock': 'Rec Stock'
    }),
    use_container_width=True,
    hide_index=True
)
