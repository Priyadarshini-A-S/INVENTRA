import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.services.data_loader import (
    load_final_recommendations, load_model_metadata, filter_dataframe
)
from dashboard.components.kpi_cards import get_risk_badge_html

st.markdown("## 💡 Explainability & Risk Driver Analysis")
st.markdown("Transparent feature importance and deterministic natural-language reasoning behind stockout risk classifications.")

# Load Data & Metadata
df_recs = load_final_recommendations()
metadata = load_model_metadata()

if df_recs.empty or not metadata:
    st.warning("Data or model metadata not available.")
    st.stop()

# ----------------------------------------------------
# 1. Individual Store x Product Explainability Inspector
# ----------------------------------------------------
st.markdown("### 🔍 Item Risk Diagnostic Inspector: \"Why is this item at risk?\"")

col_sel1, col_sel2 = st.columns(2)

with col_sel1:
    stores = sorted(df_recs['store_id'].unique())
    sel_store = st.selectbox("Select Store ID:", options=stores)

with col_sel2:
    prods_in_store = sorted(df_recs[df_recs['store_id'] == sel_store]['product_id'].unique())
    sel_prod = st.selectbox("Select Product ID:", options=prods_in_store)

# Retrieve selected item record
item_rows = df_recs[(df_recs['store_id'] == sel_store) & (df_recs['product_id'] == sel_prod)]

if not item_rows.empty:
    item = item_rows.iloc[0]
    
    # Display Risk Header & Badges
    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    h_col1.metric("Selected Store & Product", f"{item['store_id']} • {item['product_id']}")
    h_col2.metric("Category", str(item['category']))
    h_col3.metric("Stock-out Probability", f"{item['stockout_probability']*100:.1f}%")
    with h_col4:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(get_risk_badge_html(item['risk_level']), unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # Deterministic Natural Language Explanation
    st.markdown("#### 🗣️ Natural-Language Risk Diagnosis")
    
    diag_reasons = []
    if item.get('days_of_inventory', 0) < 3.0:
        diag_reasons.append(f"severely constrained inventory coverage ({item.get('days_of_inventory', 0):.1f} days of supply remaining)")
    if item.get('promotion_flag', 0) == 1:
        diag_reasons.append("active promotional campaign boosting customer purchasing velocity")
    if item.get('weekend_flag', 0) == 1:
        diag_reasons.append("upcoming weekend demand volume surge")
    if item.get('demand_growth_7', 1.0) > 1.1:
        diag_reasons.append(f"strong recent 7-day demand growth momentum (+{(item.get('demand_growth_7', 1.0)-1)*100:.0f}%)")
    if item['shortage_before_transfer'] > 0:
        diag_reasons.append(f"forecasted 7-day demand ({item['forecast_7d_demand']:.1f} units) exceeding total closing stock ({item['current_stock']:.1f} units)")
        
    if not diag_reasons:
        diag_reasons.append("regular baseline replenishment cycle and stable inventory buffer")
        
    natural_explanation = f"**Executive Diagnosis:** Demand is projected to reach **{item['forecast_7d_demand']:.1f} units** over the next 7 days. The stockout risk level is **{item['risk_level']}** ({item['stockout_probability']*100:.1f}% probability) due to " + ", ".join(diag_reasons) + "."
    
    st.info(natural_explanation)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Global Model Feature Importance
# ----------------------------------------------------
st.markdown("### 📊 Global Model Feature Importance")
st.markdown("Feature importances extracted directly from the fitted Random Forest Regressor & Classifier ensembles.")

fi_tab1, fi_tab2 = st.tabs(["Demand Forecast Feature Importance", "Stock-out Risk Feature Importance"])

# Business interpretations lookup dictionary
interpretations = {
    'closing': 'Current end-of-day stock level available for sales',
    'rolling_mean_7': '7-day moving average demand trajectory',
    'rolling_mean_14': '14-day medium-term demand baseline',
    'lag_7': 'Prior week same-day historical demand',
    'days_of_inventory': 'Estimated days until stock exhaustion',
    'demand_growth_7': 'Recent 7-day demand momentum ratio',
    'reorder_gap': 'Deficit between reorder threshold and stock',
    'avg_selling_price': 'Product unit selling price positioning',
    'promotion_flag': 'Indicates active promotional campaign',
    'temp_c': 'Regional ambient temperature impact',
    'rain_mm': 'Regional rainfall impact on footfall',
    'weekend_flag': 'Weekend footfall surge flag',
    'category': 'Product category demand baseline',
    'store_type': 'Store format operational demand scale'
}

with fi_tab1:
    demand_fi = pd.DataFrame(metadata.get('demand_feature_importance', []))
    if not demand_fi.empty:
        top_df = demand_fi.head(12)
        fig_df = px.bar(top_df, x='importance', y='feature', orientation='h', color='importance', color_continuous_scale='Blues', title="Demand Forecast Model Feature Importance")
        fig_df.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_df, use_container_width=True)
        
        top_df['Business Interpretation'] = top_df['feature'].map(lambda f: interpretations.get(f, 'Engineered time-series/store feature'))
        st.dataframe(top_df, use_container_width=True, hide_index=True)

with fi_tab2:
    stockout_fi = pd.DataFrame(metadata.get('stockout_feature_importance', []))
    if not stockout_fi.empty:
        top_sf = stockout_fi.head(12)
        fig_sf = px.bar(top_sf, x='importance', y='feature', orientation='h', color='importance', color_continuous_scale='Reds', title="Stock-out Classification Model Feature Importance")
        fig_sf.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_sf, use_container_width=True)
        
        top_sf['Business Interpretation'] = top_sf['feature'].map(lambda f: interpretations.get(f, 'Engineered inventory/risk feature'))
        st.dataframe(top_sf, use_container_width=True, hide_index=True)
