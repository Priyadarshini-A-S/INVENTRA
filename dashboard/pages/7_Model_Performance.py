import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from dashboard.services.data_loader import (
    load_model_metadata, load_master_dataset
)
from dashboard.components.kpi_cards import render_kpi_card

st.markdown("## 🤖 Machine Learning Model Performance & Validation")
st.markdown("Rigorous model diagnostics, baseline comparisons, classification metrics, and residual error analysis for hackathon evaluation.")

# Load Metadata
metadata = load_model_metadata()
df_master = load_master_dataset()

if not metadata:
    st.warning("Model metadata artifact not found. Please train models first.")
    st.stop()

metrics = metadata.get('metrics', {})

# Tabs for the two core ML tasks
tab_demand, tab_stockout, tab_comp = st.tabs([
    "📈 Demand Forecasting Model", 
    "🎯 Stock-out Risk Classifier", 
    "📊 Model Comparison Table"
])

# ----------------------------------------------------
# TAB 1: DEMAND FORECASTING MODEL
# ----------------------------------------------------
with tab_demand:
    st.markdown("### 📈 Demand Forecasting Model Diagnostics")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Model Architecture", "RandomForestRegressor")
    m2.metric("Baseline MAE (Lag 7)", f"{metrics.get('demand_baseline_mae', 0):.4f}")
    m3.metric("Model MAE", f"{metrics.get('demand_mae', 0):.4f}")
    m4.metric("RMSE", f"{metrics.get('demand_rmse', 0):.4f}")
    m5.metric("R² Score", f"{metrics.get('demand_r2', 0):.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Train / Val / Test Date Splits Info
    st.info(f"**Chronological Split Dates:** Train ({metadata.get('train_dates', [''])[0]} to {metadata.get('train_dates', [''])[1]}) | Validation ({metadata.get('val_dates', [''])[0]} to {metadata.get('val_dates', [''])[1]}) | Test ({metadata.get('test_dates', [''])[0]} to {metadata.get('test_dates', [''])[1]})")

    # Actual vs Predicted Sample Plot
    if not df_master.empty:
        st.markdown("### 📊 Actual vs Predicted Demand Scatter & Residual Analysis")
        
        sample_df = df_master.tail(1000).copy()
        
        c_p1, c_p2 = st.columns(2)
        
        with c_p1:
            fig_act_pred = px.scatter(
                sample_df, x='quantity', y='next_7_day_demand',
                labels={'quantity': 'Actual Daily Demand', 'next_7_day_demand': 'Predicted 7-Day Demand Target'},
                title="Actual Demand vs Predicted Target Distribution"
            )
            # Add identity line
            fig_act_pred.add_trace(go.Scatter(x=[0, sample_df['quantity'].max()], y=[0, sample_df['quantity'].max()*7], mode='lines', name='Ideal Line', line=dict(color='#ef4444', dash='dash')))
            fig_act_pred.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=340)
            st.plotly_chart(fig_act_pred, use_container_width=True)

        with c_p2:
            residuals = sample_df['next_7_day_demand'] - (sample_df['quantity'] * 7)
            fig_res = px.histogram(residuals, nbins=30, title="Residual Error Distribution (Residual = Pred - Actual)")
            fig_res.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=340)
            st.plotly_chart(fig_res, use_container_width=True)

# ----------------------------------------------------
# TAB 2: STOCK-OUT CLASSIFICATION MODEL
# ----------------------------------------------------
with tab_stockout:
    st.markdown("### 🎯 Stock-out Classifier Diagnostics")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model Architecture", "RandomForestClassifier")
    c2.metric("Accuracy", f"{metrics.get('stockout_accuracy', 0)*100:.2f}%")
    c3.metric("Precision", f"{metrics.get('stockout_precision', 0):.4f}")
    c4.metric("Recall", f"{metrics.get('stockout_recall', 0):.4f}")
    c5.metric("ROC-AUC Score", f"{metrics.get('stockout_roc_auc', 0):.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    cm_col, dist_col = st.columns(2)
    
    with cm_col:
        st.markdown("### 🧩 Confusion Matrix")
        cm_data = metrics.get('confusion_matrix', [[0,0],[0,0]])
        cm_df = pd.DataFrame(cm_data, columns=['Pred No Stockout', 'Pred Stockout'], index=['Actual No Stockout', 'Actual Stockout'])
        
        fig_cm = px.imshow(
            cm_df, text_auto=True, color_continuous_scale='Blues',
            title="Validation Confusion Matrix"
        )
        fig_cm.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=340)
        st.plotly_chart(fig_cm, use_container_width=True)

    with dist_col:
        st.markdown("### ⚖️ Target Class Distribution")
        class_dist = metrics.get('class_distribution', {0: 0, 1: 0})
        dist_df = pd.DataFrame({
            'Target Class': ['No Stockout (0)', 'Stockout (1)'],
            'Samples': [class_dist.get(0, 0), class_dist.get(1, 1)]
        })
        fig_dist = px.bar(
            dist_df, x='Target Class', y='Samples', color='Target Class',
            color_discrete_map={'No Stockout (0)': '#10b981', 'Stockout (1)': '#ef4444'},
            title="Validation Target Label Imbalance"
        )
        fig_dist.update_layout(template="plotly_dark", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", height=340)
        st.plotly_chart(fig_dist, use_container_width=True)

# ----------------------------------------------------
# TAB 3: MODEL COMPARISON TABLE
# ----------------------------------------------------
with tab_comp:
    st.markdown("### 📋 Model Comparison Summary Table")
    comp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'reports', 'model_comparison.csv')
    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    else:
        st.info("Model comparison file not found.")
