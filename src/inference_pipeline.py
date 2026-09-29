import os
import sys
import pandas as pd

# Support running directly or as module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import StockSensePredictor
from reorder_engine import calculate_safety_stock, calculate_recommended_stock, calculate_shortage, calculate_surplus
from transfer_engine import generate_transfer_recommendations

def run_inference(input_csv, demand_model_path, stockout_model_path, output_csv):
    print("Loading test data...")
    df = pd.read_csv(input_csv)
    
    # Simulate latest day data for inference
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    df_latest = df[df['date'] == latest_date].copy()
    
    print("Running predictions...")
    predictor = StockSensePredictor(demand_model_path, stockout_model_path)
    df_preds = predictor.predict(df_latest)
    
    print("Running business logic engine...")
    df_preds['safety_stock'] = calculate_safety_stock(df_preds['rolling_std_7'], df_preds['lead_days'])
    df_preds['recommended_stock'] = calculate_recommended_stock(df_preds['forecast_7d_demand'], df_preds['safety_stock'])
    
    df_preds['incoming_stock'] = df_preds.get('received', 0)
    df_preds['current_stock'] = df_preds['closing']
    
    df_preds['shortage'] = calculate_shortage(df_preds['recommended_stock'], df_preds['current_stock'], df_preds['incoming_stock'])
    df_preds['surplus'] = calculate_surplus(df_preds['recommended_stock'], df_preds['current_stock'], df_preds['incoming_stock'])
    
    print("Running Smart Transfer Engine...")
    final_recs = generate_transfer_recommendations(df_preds)
    
    print("Saving final recommendations...")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    final_recs.to_csv(output_csv, index=False)
    
    # Calculate summary metrics
    total_shortage = df_preds['shortage'].sum()
    transferred = final_recs['transfer_quantity'].sum()
    print(f"Total Shortage: {total_shortage}, Met by Transfer: {transferred}, Reduced supplier order by: {transferred}")
    print("Done!")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_inference(
        os.path.join(base_dir, 'data', 'processed', 'master_dataset.csv'), 
        os.path.join(base_dir, 'models', 'demand_forecast_model.pkl'),
        os.path.join(base_dir, 'models', 'stockout_risk_model.pkl'),
        os.path.join(base_dir, 'data', 'processed', 'final_recommendations.csv')
    )
