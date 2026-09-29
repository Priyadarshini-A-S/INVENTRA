import streamlit as st
import pandas as pd
import numpy as np

from dashboard.services.data_loader import (
    load_final_recommendations, filter_dataframe
)
from dashboard.components.kpi_cards import get_risk_badge_html

st.markdown("## 🎯 Manager Action Center")
st.markdown("Operational decision hub to convert ML stock-out predictions into optimized inventory transfers and supplier orders.")

# Load Data
df_recs = load_final_recommendations()

# Apply Global Filters
cities = st.session_state.get('global_cities', [])
stores = st.session_state.get('global_stores', [])
categories = st.session_state.get('global_categories', [])
risks = st.session_state.get('global_risks', [])

filtered_recs = filter_dataframe(df_recs, cities=cities, stores=stores, categories=categories, risk_levels=risks)

if filtered_recs.empty:
    st.warning("No action items match the selected filters.")
    st.stop()

# Visual Decision Flow Diagram
st.markdown("### 🔄 Inventory Optimization Decision Flow")
st.markdown("""
<div class="decision-flow-container">
    <div class="flow-step">
        <div class="flow-box" style="border-color: #ef4444; color: #fca5a5;">1. Shortage Detected</div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.3rem;">Demand > Current Stock</div>
    </div>
    <div class="flow-arrow">➔</div>
    <div class="flow-step">
        <div class="flow-box" style="border-color: #38bdf8; color: #7dd3fc;">2. Inter-Store Search</div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.3rem;">Check Regional Stores</div>
    </div>
    <div class="flow-arrow">➔</div>
    <div class="flow-step">
        <div class="flow-box" style="border-color: #f59e0b; color: #fde047;">3. Surplus Available?</div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.3rem;">Evaluate Safety Buffer</div>
    </div>
    <div class="flow-arrow">➔</div>
    <div class="flow-step">
        <div class="flow-box" style="border-color: #10b981; color: #86efac;">4. Inter-Store Transfer</div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.3rem;">Zero Supplier Cost</div>
    </div>
    <div class="flow-arrow">➔</div>
    <div class="flow-step">
        <div class="flow-box" style="border-color: #6366f1; color: #a5b4fc;">5. Supplier Order Fallback</div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.3rem;">Order Remaining Deficit</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Filter for items needing action (HIGH or MEDIUM risk, or shortages)
action_items = filtered_recs[
    (filtered_recs['risk_level'].isin(['HIGH', 'MEDIUM'])) | 
    (filtered_recs['shortage_before_transfer'] > 0)
].sort_values(by=['stockout_probability', 'shortage_before_transfer'], ascending=[False, False])

st.markdown(f"### 📋 Recommended Action Queue ({len(action_items)} Items Needing Manager Intervention)")

# Pagination / Filter limit
max_cards = st.slider("Display top N priority action cards:", min_value=5, max_value=50, value=10, step=5)
action_items_display = action_items.head(max_cards)

for idx, row in action_items_display.iterrows():
    card_key = f"card_{row['store_id']}_{row['product_id']}_{idx}"
    
    # Generate deterministic "WHY?" explanation bullets based on available feature values
    why_bullets = []
    if row.get('promotion_flag', 0) == 1:
        why_bullets.append("Active promotion driving elevated consumer demand")
    if row.get('weekend_flag', 0) == 1 or row.get('day_of_week', 0) in [5, 6]:
        why_bullets.append("Weekend shopping surge expected")
    if row.get('demand_growth_7', 1.0) > 1.1:
        why_bullets.append(f"Recent 7-day demand acceleration (+{(row.get('demand_growth_7', 1.0)-1)*100:.0f}%)")
    if row.get('days_of_inventory', 0) < 3.0:
        why_bullets.append(f"Critically low stock coverage ({row.get('days_of_inventory', 0):.1f} days of supply)")
    if not why_bullets:
        why_bullets.append("Forecast 7-day demand exceeds available stock plus incoming shipments")
        why_bullets.append("Reorder threshold trigger reached")

    st.markdown(f"""
    <div class="action-card">
        <div class="action-card-header">
            <div>
                <span style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">STORE {row['store_id']} &nbsp;•&nbsp; PRODUCT {row['product_id']} ({row['category']})</span>
            </div>
            <div>
                {get_risk_badge_html(row['risk_level'])}
            </div>
        </div>
        <div class="action-metrics">
            <div class="action-metric-item">
                <span class="action-metric-label">7-Day Demand</span>
                <span class="action-metric-val">{row['forecast_7d_demand']:.1f} units</span>
            </div>
            <div class="action-metric-item">
                <span class="action-metric-label">Current Stock</span>
                <span class="action-metric-val">{row['current_stock']:.1f} units</span>
            </div>
            <div class="action-metric-item">
                <span class="action-metric-label">Incoming Stock</span>
                <span class="action-metric-val">{row['incoming_stock']:.1f} units</span>
            </div>
            <div class="action-metric-item">
                <span class="action-metric-label">Stockout Probability</span>
                <span class="action-metric-val" style="color: #ef4444;">{row['stockout_probability']*100:.1f}%</span>
            </div>
        </div>
        <div style="margin: 0.8rem 0;">
            <div style="font-weight: 700; color: #38bdf8; font-size: 1rem;">
                RECOMMENDED ACTION: {row['action']}
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 0.4rem;">
                <strong>Source Store:</strong> <span style="color: #f59e0b;">{row['source_store_id']}</span> &nbsp;|&nbsp; 
                <strong>Transfer Quantity:</strong> <span style="color: #10b981;">{row['transfer_quantity']:.1f} units</span> &nbsp;|&nbsp; 
                <strong>Supplier Order Qty:</strong> <span style="color: #a5b4fc;">{row['supplier_order_quantity']:.1f} units</span>
            </div>
        </div>
        <div style="background: #0f172a; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.75rem;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; margin-bottom: 0.3rem;">WHY IS THIS ITEM AT RISK?</div>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.85rem; color: #cbd5e1;">
                {"".join([f"<li>{b}</li>" for b in why_bullets])}
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive Demo Action Buttons
    act_col1, act_col2, act_col3, act_col4 = st.columns(4)
    with act_col1:
        if st.button(f"🚀 Transfer {row['transfer_quantity']:.0f} Units", key=f"t_{card_key}"):
            st.toast(f"✅ Transfer request created: {row['transfer_quantity']:.0f} units from Store {row['source_store_id']} to Store {row['store_id']}", icon="🚚")
    with act_col2:
        if st.button(f"📦 Order {row['supplier_order_quantity']:.0f} Units", key=f"o_{card_key}"):
            st.toast(f"✅ Supplier purchase order generated for {row['supplier_order_quantity']:.0f} units of {row['product_id']}", icon="📝")
    with act_col3:
        if st.button("👁️ View Explanation", key=f"v_{card_key}"):
            st.info(f"Reason: {row['reason']}")
    with act_col4:
        if st.button("✖️ Dismiss Alert", key=f"d_{card_key}"):
            st.toast(f"Dismissed risk alert for Store {row['store_id']} - {row['product_id']}", icon="🔕")
            
    st.markdown("<hr style='border-color: #334155;'>", unsafe_allow_html=True)
