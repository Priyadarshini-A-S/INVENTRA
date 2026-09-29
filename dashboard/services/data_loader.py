import streamlit as st
import pandas as pd
import joblib
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@st.cache_data(ttl=300)
def load_final_recommendations():
    path = os.path.join(BASE_DIR, 'data', 'processed', 'final_recommendations.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Ensure numerical types
        num_cols = ['current_stock', 'incoming_stock', 'forecast_7d_demand', 'safety_stock', 
                    'recommended_stock', 'stockout_probability', 'shortage_before_transfer', 
                    'transfer_quantity', 'remaining_shortage', 'supplier_order_quantity',
                    'cost_price', 'avg_selling_price', 'days_of_inventory', 'reorder_gap']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_master_dataset():
    path = os.path.join(BASE_DIR, 'data', 'processed', 'master_dataset.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_raw_dataset():
    path = os.path.join(BASE_DIR, 'data', 'raw', 'master_cleaned.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame()

@st.cache_resource
def load_model_metadata():
    path = os.path.join(BASE_DIR, 'models', 'model_metadata.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return {}

@st.cache_data(ttl=300)
def load_quality_report():
    path = os.path.join(BASE_DIR, 'reports', 'data_quality_report.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=300)
def load_eda_summary():
    path = os.path.join(BASE_DIR, 'reports', 'eda_summary.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def filter_dataframe(df, cities=None, stores=None, categories=None, products=None, risk_levels=None):
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    if cities and 'city' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['city'].isin(cities)]
        
    if stores and 'store_id' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['store_id'].isin(stores)]
        
    if categories and 'category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
        
    if products and 'product_id' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['product_id'].isin(products)]
        
    if risk_levels and 'risk_level' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_levels)]
        
    return filtered_df
