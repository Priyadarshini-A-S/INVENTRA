import pandas as pd
import numpy as np
import os

def create_features(input_file, output_file):
    print("Loading data...")
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['store_id', 'product_id', 'date']).reset_index(drop=True)
    
    print("Creating target...")
    # Target 1: next_7_day_demand (using 'quantity' which is daily demand)
    df['next_7_day_demand'] = df.groupby(['store_id', 'product_id'])['quantity'].transform(lambda x: x.shift(-1).rolling(window=7, min_periods=1).sum())
    
    # Drop rows where target is NaN (last 7 days of data)
    # Actually, we can keep some, but let's drop rows where we don't have next_7_day_demand at all
    df.dropna(subset=['next_7_day_demand'], inplace=True)
    
    print("Creating time features...")
    # Time features already partially there, let's add quarter
    df['quarter'] = df['date'].dt.quarter
    
    print("Creating lag features...")
    for lag in [1, 2, 3, 7, 14, 28]:
        df[f'lag_{lag}'] = df.groupby(['store_id', 'product_id'])['quantity'].shift(lag)
        
    print("Creating rolling features...")
    for window in [3, 7, 14, 28]:
        df[f'rolling_mean_{window}'] = df.groupby(['store_id', 'product_id'])['quantity'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    
    for window in [7, 14]:
        df[f'rolling_std_{window}'] = df.groupby(['store_id', 'product_id'])['quantity'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std())
        df[f'rolling_min_{window}'] = df.groupby(['store_id', 'product_id'])['quantity'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).min())
        df[f'rolling_max_{window}'] = df.groupby(['store_id', 'product_id'])['quantity'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).max())
    
    print("Creating trend features...")
    df['demand_growth_7'] = df['rolling_mean_7'] / (df['rolling_mean_14'] + 1e-5)
    df['demand_growth_14'] = df['rolling_mean_14'] / (df['rolling_mean_28'] + 1e-5)
    
    print("Creating inventory features...")
    df['days_of_inventory'] = df['closing'] / (df['rolling_mean_7'] + 1e-5)
    df['inventory_to_demand_ratio'] = df['closing'] / (df['rolling_mean_14'] + 1e-5)
    df['reorder_gap'] = df['reorder_lvl'] - df['closing']
    
    print("Imputing sparse features...")
    # For NaNs caused by shift/rolling
    df.fillna(0, inplace=True)
    
    print("Saving processed data...")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == '__main__':
    create_features('../data/raw/master_cleaned.csv', '../data/processed/master_dataset.csv')
