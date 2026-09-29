import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.services.data_loader import (
    load_final_recommendations, load_raw_dataset, filter_dataframe
)
from dashboard.components.kpi_cards import get_risk_badge_html

st.markdown("## 📦 Product Analytics & SKU Deep-Dive")
st.markdown("Granular product intelligence profile, demand forecast history, inventory positioning, risk metrics, and supplier transfer requirements.")

# Load Data
df_recs = load_final_recommendations()
df_raw = load_raw_dataset()

if df_raw.empty or df_recs.empty:
    st.warning("No data available for product analytics.")
    st.stop()

# Product Search & Selection Header
p_col1, p_col2 = st.columns([6, 4])

with p_col1:
    available_products = sorted(df_raw['product_id'].dropna().unique())
    selected_prod = st.selectbox("🔍 Search & Select Product ID:", options=available_products)

with p_col2:
    available_categories = sorted(df_raw['category'].dropna().unique())
    selected_cat_filter = st.multiselect("Filter Catalog Category:", options=available_categories)

# Filter product records
prod_raw = df_raw[df_raw['product_id'] == selected_prod]
prod_recs = df_recs[df_recs['product_id'] == selected_prod]

if prod_raw.empty:
    st.warning("Selected product not found in raw dataset.")
    st.stop()

sample_raw = prod_raw.iloc[0]
sample_rec = prod_recs.iloc[0] if not prod_recs.empty else {}

# Section 1: Product Master Profile
st.markdown("---")
st.markdown(f"### 🏷️ Product Profile: <span style='color: #38bdf8;'>{selected_prod}</span>", unsafe_allow_html=True)

prof_col1, prof_col2, prof_col3, prof_col4, prof_col5, prof_col6 = st.columns(6)

prof_col1.metric("Category", str(sample_raw.get('category', 'N/A')))
prof_col2.metric("Sub-Category", str(sample_raw.get('sub_category', 'N/A')))
prof_col3.metric("Brand", str(sample_raw.get('brand', 'N/A')))
prof_col4.metric("Selling Price", f"₹{sample_raw.get('avg_selling_price', 0):.2f}")
prof_col5.metric("Cost Price", f"₹{sample_raw.get('cost_price', 0):.2f}")
prof_col6.metric("Shelf Life", f"{sample_raw.get('shelf_life_days', 0)} days")

st.markdown("<br>", unsafe_allow_html=True)

# Section 2: Demand, Inventory, Risk & Action Grid
d_col, i_col, r_col = st.columns(3)

with d_col:
    st.markdown("#### 📈 Demand Intelligence")
    recent_demand = prod_raw.tail(7)['quantity'].sum()
    forecast_demand = prod_recs['forecast_7d_demand'].sum() if not prod_recs.empty else 0
    volatility = prod_raw['quantity'].std()
    
    st.write(f"**Recent 7-Day Demand:** {recent_demand:.1f} units")
    st.write(f"**Predicted 7-Day Forecast:** {forecast_demand:.1f} units")
    st.write(f"**Demand Volatility (Std):** {volatility:.2f}")
    st.write(f"**Supplier ID:** {sample_raw.get('supplier_id', 'N/A')}")

with i_col:
    st.markdown("#### 📦 Inventory Positioning")
    total_curr_stock = prod_recs['current_stock'].sum() if not prod_recs.empty else sample_raw.get('closing', 0)
    total_inc_stock = prod_recs['incoming_stock'].sum() if not prod_recs.empty else sample_raw.get('received', 0)
    avg_days_inv = prod_recs['days_of_inventory'].mean() if not prod_recs.empty else 0
    reorder_lvl = sample_raw.get('reorder_lvl', 0)
    
    st.write(f"**Total Current Stock:** {total_curr_stock:.1f} units")
    st.write(f"**Total Incoming Stock:** {total_inc_stock:.1f} units")
    st.write(f"**Avg Days of Inventory:** {avg_days_inv:.1f} days")
    st.write(f"**Reorder Level:** {reorder_lvl:.1f} units")

with r_col:
    st.markdown("#### ⚠️ Stock-out Risk & Recommendation")
    avg_stockout_prob = prod_recs['stockout_probability'].mean() if not prod_recs.empty else 0
    max_risk = prod_recs['risk_level'].max() if not prod_recs.empty else 'LOW'
    total_transfer_avail = prod_recs['transfer_quantity'].sum() if not prod_recs.empty else 0
    total_supplier_req = prod_recs['supplier_order_quantity'].sum() if not prod_recs.empty else 0
    
    st.markdown(f"**Stock-out Probability:** <span style='color: #ef4444; font-weight: 700;'>{avg_stockout_prob*100:.1f}%</span>", unsafe_allow_html=True)
    st.markdown(f"**Overall Risk Classification:** {get_risk_badge_html(max_risk)}", unsafe_allow_html=True)
    st.write(f"**Transfer Availability:** {total_transfer_avail:.1f} units")
    st.write(f"**Supplier Order Requirement:** {total_supplier_req:.1f} units")

st.markdown("<br>", unsafe_allow_html=True)

# Section 3: Time Series Chart for Selected Product
st.markdown("### 📊 Product Historical Demand Curve across Stores")

prod_time = prod_raw.groupby(['date', 'store_id'])['quantity'].sum().reset_index()
fig_pt = px.line(prod_time, x='date', y='quantity', color='store_id', title=f"Daily Demand Trend for {selected_prod} by Store")
fig_pt.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=340)
st.plotly_chart(fig_pt, use_container_width=True)

# Section 4: Store Breakup Matrix for Selected Product
st.markdown(f"### 📋 Store Breakdown Matrix for Product {selected_prod}")
if not prod_recs.empty:
    p_matrix = prod_recs[['store_id', 'current_stock', 'incoming_stock', 'forecast_7d_demand', 
                          'stockout_probability', 'risk_level', 'action', 'transfer_quantity', 'supplier_order_quantity']].copy()
    p_matrix['forecast_7d_demand'] = p_matrix['forecast_7d_demand'].round(1)
    p_matrix['stockout_probability'] = (p_matrix['stockout_probability']*100).round(1).astype(str) + '%'
    st.dataframe(p_matrix, use_container_width=True, hide_index=True)
