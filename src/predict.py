import pandas as pd
import joblib

class StockSensePredictor:
    def __init__(self, demand_model_path, stockout_model_path):
        self.demand_model = joblib.load(demand_model_path)
        self.stockout_model = joblib.load(stockout_model_path)
        
    def predict(self, df):
        demand_preds = self.demand_model.predict(df)
        stockout_probs = self.stockout_model.predict_proba(df)[:, 1]
        
        df_out = df.copy()
        df_out['forecast_7d_demand'] = demand_preds
        df_out['stockout_probability'] = stockout_probs
        
        # Risk Classification
        df_out['risk_level'] = pd.cut(
            df_out['stockout_probability'], 
            bins=[-1, 0.40, 0.70, 1.1], 
            labels=['LOW', 'MEDIUM', 'HIGH'],
            right=False
        )
        return df_out
