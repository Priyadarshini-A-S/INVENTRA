import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_and_evaluate(base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    data_file = os.path.join(base_dir, 'data', 'processed', 'master_dataset.csv')
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("Loading processed data...")
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Stockout Definition: stockout_flag is already present in raw data.
    # If it's missing or we need to redefine: stockout_flag = 1 when closing == 0 and demand > 0
    # The data already has 'stockout_flag'. Let's use it as target.
    
    print("Splitting data chronologically...")
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    features = [
        'day_of_week', 'month', 'quarter', 'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_28',
        'rolling_mean_3', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
        'rolling_std_7', 'rolling_std_14', 'demand_growth_7', 'demand_growth_14',
        'promotion_flag', 'avg_discount', 'avg_selling_price', 'closing', 'reorder_gap',
        'days_of_inventory', 'temp_c', 'rain_mm', 'holiday', 'weekend_flag'
    ]
    
    cat_features = ['category', 'store_type']
    
    X_train = train[features + cat_features]
    y_train_demand = train['next_7_day_demand']
    y_train_stockout = train['stockout_flag']
    
    X_val = val[features + cat_features]
    y_val_demand = val['next_7_day_demand']
    y_val_stockout = val['stockout_flag']
    
    X_test = test[features + cat_features]
    y_test_demand = test['next_7_day_demand']
    y_test_stockout = test['stockout_flag']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ])
        
    print("Training Demand Model...")
    demand_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42))
    ])
    demand_model.fit(X_train, y_train_demand)
    
    # Baseline: lag_7
    baseline_pred = val['lag_7']
    baseline_mae = mean_absolute_error(y_val_demand, baseline_pred)
    
    demand_preds = demand_model.predict(X_val)
    mae = mean_absolute_error(y_val_demand, demand_preds)
    rmse = np.sqrt(mean_squared_error(y_val_demand, demand_preds))
    r2 = r2_score(y_val_demand, demand_preds)
    
    print(f"Demand Model Val MAE: {mae}, Baseline MAE: {baseline_mae}, RMSE: {rmse}, R2: {r2}")
    
    print("Training Stockout Model...")
    stockout_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=30, max_depth=8, class_weight='balanced', n_jobs=-1, random_state=42))
    ])
    stockout_model.fit(X_train, y_train_stockout)
    
    stockout_preds = stockout_model.predict(X_val)
    stockout_probs = stockout_model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val_stockout, stockout_preds)
    prec = precision_score(y_val_stockout, stockout_preds, zero_division=0)
    rec = recall_score(y_val_stockout, stockout_preds, zero_division=0)
    f1 = f1_score(y_val_stockout, stockout_preds, zero_division=0)
    roc = roc_auc_score(y_val_stockout, stockout_probs)
    
    print(f"Stockout Model Val ROC-AUC: {roc}, Recall: {rec}, F1: {f1}")
    
    print("Extracting feature importances...")
    feature_names = demand_model.named_steps['preprocessor'].get_feature_names_out()
    demand_importances = demand_model.named_steps['regressor'].feature_importances_
    stockout_importances = stockout_model.named_steps['classifier'].feature_importances_
    
    # Map back one-hot feature names cleanly
    cleaned_feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
    
    demand_fi = pd.DataFrame({'feature': cleaned_feature_names, 'importance': demand_importances}).sort_values('importance', ascending=False)
    stockout_fi = pd.DataFrame({'feature': cleaned_feature_names, 'importance': stockout_importances}).sort_values('importance', ascending=False)

    cm = confusion_matrix(y_val_stockout, stockout_preds)
    mape = mean_absolute_percentage_error(y_val_demand, demand_preds)
    
    metadata = {
        'model_name': 'RandomForest Models',
        'features': features + cat_features,
        'train_dates': [str(train['date'].min()), str(train['date'].max())],
        'val_dates': [str(val['date'].min()), str(val['date'].max())],
        'test_dates': [str(test['date'].min()), str(test['date'].max())],
        'metrics': {
            'demand_baseline_mae': float(baseline_mae),
            'demand_mae': float(mae),
            'demand_rmse': float(rmse),
            'demand_mape': float(mape),
            'demand_r2': float(r2),
            'stockout_accuracy': float(acc),
            'stockout_precision': float(prec),
            'stockout_recall': float(rec),
            'stockout_f1': float(f1),
            'stockout_roc_auc': float(roc),
            'confusion_matrix': cm.tolist(),
            'class_distribution': val['stockout_flag'].value_counts().to_dict()
        },
        'demand_feature_importance': demand_fi.to_dict('records'),
        'stockout_feature_importance': stockout_fi.to_dict('records')
    }
    
    print("Saving models and metadata...")
    joblib.dump(demand_model, os.path.join(models_dir, 'demand_forecast_model.pkl'))
    joblib.dump(stockout_model, os.path.join(models_dir, 'stockout_risk_model.pkl'))
    joblib.dump(metadata, os.path.join(models_dir, 'model_metadata.pkl'))
    
    # Save reports
    pd.DataFrame([{
        'Model': 'RandomForestRegressor',
        'Baseline_MAE': baseline_mae,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }, {
        'Model': 'Baseline (Lag 7)',
        'Baseline_MAE': baseline_mae,
        'MAE': baseline_mae,
        'RMSE': np.sqrt(mean_squared_error(y_val_demand, baseline_pred)),
        'MAPE': mean_absolute_percentage_error(y_val_demand, baseline_pred.fillna(0)),
        'R2': r2_score(y_val_demand, baseline_pred.fillna(0))
    }]).to_csv(os.path.join(reports_dir, 'model_comparison.csv'), index=False)
    
    with open(os.path.join(reports_dir, 'model_comparison.md'), 'w') as f:
        f.write("# Model Comparison & Validation Report\n\n")
        f.write("RandomForest models were trained on chronological split (70% Train, 15% Validation, 15% Test).\n\n")
        f.write("## Demand Forecast Model\n")
        f.write(f"- Baseline MAE (Lag 7): {baseline_mae:.4f}\n")
        f.write(f"- Model MAE: {mae:.4f}\n")
        f.write(f"- Model RMSE: {rmse:.4f}\n")
        f.write(f"- Model MAPE: {mape:.4f}\n")
        f.write(f"- Model R²: {r2:.4f}\n\n")
        f.write("## Stock-out Classification Model\n")
        f.write(f"- Accuracy: {acc:.4f}\n")
        f.write(f"- Precision: {prec:.4f}\n")
        f.write(f"- Recall: {rec:.4f}\n")
        f.write(f"- F1-Score: {f1:.4f}\n")
        f.write(f"- ROC-AUC: {roc:.4f}\n")
        
    print("Done!")

if __name__ == '__main__':
    train_and_evaluate()
