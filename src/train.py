# src/train.py

import sys, os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

from preprocessing import load_and_clean
from feature_engineering import create_features


# ======================
# LOAD DATA
# ======================
DATA_PATH = "data/cleaned/reliance_clean.csv"

df = load_and_clean(DATA_PATH)

# ======================
# FEATURE ENGINEERING (UPDATED 🔥)
# ======================
df = create_features(df, stock_name="Reliance")


# ======================
# MULTI-HORIZON TARGETS
# ======================
horizons = [1, 7, 30]

for h in horizons:
    df[f"target_{h}"] = df["Close"].shift(-h)

df.dropna(inplace=True)


# ======================
# FEATURES
# ======================
X = df.drop(columns=["Close"] + [f"target_{h}" for h in horizons])


# ======================
# TRAIN MODELS
# ======================
models = {}

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

for h in horizons:
    print(f"\n🔄 Training model for {h}-day prediction...")

    y = df[f"target_{h}"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=False,
        test_size=0.2,
    )

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    cv_splits = min(5, max(2, len(X_train) // 200))
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=5,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # ======================
    # EVALUATION
    # ======================
    preds = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n📊 {h}-Day Model Performance")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

    models[h] = best_model


# ======================
# SAVE EVERYTHING
# ======================
os.makedirs("models", exist_ok=True)

joblib.dump(models, "models/models.pkl")   # 🔥 multi-horizon models

feature_names = X.columns.tolist()
joblib.dump(feature_names, "models/features.pkl")

print("\n✅ All Models & Features Saved!")
