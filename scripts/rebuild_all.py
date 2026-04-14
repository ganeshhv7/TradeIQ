import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Add project root to path so src modules can be imported
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# 1. GENERATE FEATURES
from src.feature_engineering import create_features

ROOT = Path(base_dir)
files = {
    "RELIANCE": "reliance_clean.csv",
    "TCS": "tcs_clean.csv",
    "INFY": "infosys_clean.csv",
    "HDFCBANK": "hdfcbank_clean.csv"
}

os.makedirs(ROOT / "data" / "features", exist_ok=True)
stock_data_dict = {}

print("Step 1: Rebuilding Feature Datasets...")
for stock, file in files.items():
    df = pd.read_csv(ROOT / "data" / "cleaned" / file, index_col="Date", parse_dates=True)
    df_feat = create_features(df, stock_name=stock)
    
    out_name = f"{stock.lower()}_feat.csv"
    if stock == "INFY":
        out_name = "infosys_feat.csv"
    
    df_feat.to_csv(ROOT / "data" / "features" / out_name)
    stock_data_dict[stock] = df_feat
    print(f"[{stock}] Engineered {df_feat.shape[1]} features.")

# 2. TRAIN MODELS
print("\nStep 2: Training Models...")
drop_cols = ["Close", "Ticker", "Adj Close", "Datetime", "Date"]

sample_df = stock_data_dict["RELIANCE"]
feature_cols = [c for c in sample_df.columns if c not in drop_cols and not str(c).startswith("Target_")]

horizons = [1, 7, 15]

for stock_name, df in stock_data_dict.items():
    print(f"\n--- Training for {stock_name} ---")
    best_models_dict = {}
    
    for n in horizons:
        df[f"Target_{n}"] = (df["Close"].shift(-n) - df["Close"]) / df["Close"]
        valid_data = df.dropna(subset=[f"Target_{n}"])
        
        X = valid_data[feature_cols]
        y = valid_data[f"Target_{n}"]
        prices = valid_data["Close"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        _, _, _, prices_test = train_test_split(X, prices, shuffle=False, test_size=0.2)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
        
        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=150, 
                max_depth=10, 
                min_samples_split=5, 
                random_state=42
            ),
            "XGBoost": xgb.XGBRegressor(
                objective="reg:pseudohubererror", # Handles extreme market outliers
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        }
        
        best_rmse = float('inf')
        best_model_obj = None
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            
            curr_price = prices_test.values
            preds_price = curr_price * (1 + preds)
            actual_price = curr_price * (1 + y_test.values)
            rmse = np.sqrt(mean_squared_error(actual_price, preds_price))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_obj = model
                
        print(f"Horizon {n} days -> Best Model saved (RMSE {best_rmse:.2f})")
        best_models_dict[n] = best_model_obj
        if n == 1:
            main_scaler = scaler
            
    stock_dir = ROOT / "models" / stock_name
    os.makedirs(stock_dir, exist_ok=True)
    joblib.dump(best_models_dict, stock_dir / "models.pkl")
    joblib.dump(main_scaler, stock_dir / "scaler.pkl")
    joblib.dump(feature_cols, stock_dir / "features.pkl")
    print(f"All saved for {stock_name}.")
    
print("\n DONE! Models perfectly synced with live pipeline.")
