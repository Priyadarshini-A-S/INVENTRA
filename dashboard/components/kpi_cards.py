import streamlit as st

def render_kpi_card(title, value, subtext=None, accent_color=None):
    border_style = f"border-top: 3px solid {accent_color};" if accent_color else ""
    st.markdown(f"""
    <div class="kpi-card" style="{border_style}">
        <div class="kpi-label">{title}</div>
        <div class="kpi-value">{value}</div>
        {f'<div class="kpi-subtext">{subtext}</div>' if subtext else ''}
    </div>
    """, unsafe_allow_html=True)

def get_risk_badge_html(risk_level):
    risk_upper = str(risk_level).upper()
    if risk_upper == 'HIGH':
        return '<span class="badge-high">HIGH RISK</span>'
    elif risk_upper == 'MEDIUM':
        return '<span class="badge-medium">MEDIUM RISK</span>'
    else:
        return '<span class="badge-low">LOW RISK</span>'
