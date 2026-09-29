import streamlit as st
import os
import sys

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

st.set_page_config(
    page_title="INVENTRA — Retail Inventory Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
if os.path.exists(css_path):
    with open(css_path, 'r') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Imports after sys.path setup
from dashboard.services.data_loader import (
    load_final_recommendations, load_raw_dataset, load_model_metadata, filter_dataframe
)
from dashboard.components.header import render_header

# Load Data
df_recs = load_final_recommendations()
df_raw = load_raw_dataset()

# Sidebar Brand Header & Filters
st.sidebar.markdown("""
<div style="padding-bottom: 1rem; border-bottom: 1px solid #334155; margin-bottom: 1rem;">
    <div style="font-weight: 800; font-size: 1.3rem; color: #38bdf8; letter-spacing: 1px;">INVENTRA</div>
    <div style="font-size: 0.75rem; color: #94a3b8;">Demand & Inventory Intelligence</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🎛️ Operations Controls")

# City Filter
cities = sorted(df_raw['city'].dropna().unique()) if not df_raw.empty and 'city' in df_raw.columns else []
selected_cities = st.sidebar.multiselect("Select City", options=cities, default=[])

# Store Filter
filtered_raw = df_raw[df_raw['city'].isin(selected_cities)] if selected_cities else df_raw
stores = sorted(filtered_raw['store_id'].dropna().unique()) if not filtered_raw.empty and 'store_id' in filtered_raw.columns else []
selected_stores = st.sidebar.multiselect("Select Store", options=stores, default=[])

# Category Filter
categories = sorted(df_raw['category'].dropna().unique()) if not df_raw.empty and 'category' in df_raw.columns else []
selected_categories = st.sidebar.multiselect("Select Category", options=categories, default=[])

# Risk Filter
risk_options = ['HIGH', 'MEDIUM', 'LOW']
selected_risks = st.sidebar.multiselect("Risk Level", options=risk_options, default=[])

# Save global filters in session state
st.session_state['global_cities'] = selected_cities
st.session_state['global_stores'] = selected_stores
st.session_state['global_categories'] = selected_categories
st.session_state['global_risks'] = selected_risks

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reload Data Pipeline", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("""
<div style="font-size: 0.7rem; color: #64748b; margin-top: 2rem;">
    INVENTRA Decision-Support Engine v2.0<br>
    Powered by Random Forest Models & Smart Inter-Store Transfer Engine
</div>
""", unsafe_allow_html=True)

# Define Pages
overview_page = st.Page("pages/1_Overview.py", title="Executive Overview", icon="📊", default=True)
demand_page = st.Page("pages/2_Demand_Intelligence.py", title="Demand Intelligence", icon="📈")
risk_page = st.Page("pages/3_Inventory_Risk.py", title="Inventory Risk Monitor", icon="⚠️")
action_page = st.Page("pages/4_Manager_Action_Center.py", title="Manager Action Center", icon="🎯")
store_page = st.Page("pages/5_Store_Analytics.py", title="Store Analytics", icon="🏪")
product_page = st.Page("pages/6_Product_Analytics.py", title="Product Analytics", icon="📦")
model_page = st.Page("pages/7_Model_Performance.py", title="Model Performance", icon="🤖")
explain_page = st.Page("pages/8_Explainability.py", title="Explainability", icon="💡")
quality_page = st.Page("pages/9_Data_Quality.py", title="Data Quality Report", icon="🔍")

pg = st.navigation({
    "Executive Center": [overview_page, action_page],
    "Analytics & Risk": [demand_page, risk_page, store_page, product_page],
    "ML & Governance": [model_page, explain_page, quality_page]
})

# Render Global Header above pages
render_header()

# Run Navigation
pg.run()
