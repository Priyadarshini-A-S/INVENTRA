import streamlit as st
import datetime

def render_header(last_run_date=None, dataset_status="ONLINE"):
    if last_run_date is None:
        last_run_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
    st.markdown(f"""
    <div class="inventra-header">
        <div class="inventra-brand">
            <div class="inventra-logo">INVENTRA</div>
            <div>
                <div class="inventra-title">Retail Demand & Inventory Intelligence</div>
                <div class="inventra-subtitle">Multi-City Decision Support System • Predict → Explain → Act</div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div class="inventra-status-badge">
                <span class="status-dot"></span>
                <span>Pipeline Status: <strong>{dataset_status}</strong></span>
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; border-left: 1px solid #334155; padding-left: 1rem;">
                Last Sync: <strong style="color: #f1f5f9;">{last_run_date}</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
