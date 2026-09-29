# IntelliData 2026 - StockSense Hackathon

## Architecture

RAW DATA
   ↓
DATA CLEANING
   ↓
MASTER DATASET
   ↓
FEATURE ENGINEERING
   ↓
 ┌───────────────────────┐
 │ Demand Forecast Model │
 └───────────┬───────────┘
             ↓
      7-Day Demand
             ↓
 ┌───────────────────────┐
 │ Stockout Risk Model   │
 └───────────┬───────────┘
             ↓
     Stockout Probability
             ↓
      Risk Classification
             ↓
 ┌────────────────────────────┐
 │ Smart Inventory Optimizer  │
 │                            │
 │ Transfer before Purchase   │
 └──────────────┬─────────────┘
                ↓
     Transfer + Supplier Order
                ↓
       Manager Recommendation

## Components
1. **Demand Forecasting**: A Random Forest model forecasting the next 7 days of demand.
2. **Stock-out Risk**: A Random Forest Classifier predicting the probability of a stockout.
3. **Smart Transfer Engine**: A business logic engine that evaluates shortage across stores and attempts to satisfy it by transferring surplus inventory from other stores, before resorting to placing supplier orders.

## Scripts
- src/feature_engineering.py: Prepares the dataset.
- src/train_models.py: Trains the ML models and serializes them.
- src/inference_pipeline.py: Loads the models and executes the end-to-end pipeline generating recommendations.

## Usage
1. Place raw data in data/raw/master_cleaned.csv
2. Run python src/feature_engineering.py
3. Run python src/train_models.py
4. Run python src/inference_pipeline.py
