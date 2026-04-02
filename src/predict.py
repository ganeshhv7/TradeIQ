# src/predict.py

import sys, os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import joblib

from preprocessing import load_and_clean
from feature_engineering import create_features


MODELS_PATH = "models/models.pkl"
FEATURES_PATH = "models/features.pkl"


def predict_price(data_path: str, horizon: int = 1, stock_name: str = "Reliance"):
    """
    Predict a future stock price for the requested horizon.
    """

    models = joblib.load(MODELS_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    if horizon not in models:
        raise ValueError(f"No trained model found for horizon={horizon}")

    df = load_and_clean(data_path)
    df = create_features(df, stock_name=stock_name)

    latest = df.iloc[-1:]
    X = latest.reindex(columns=feature_names, fill_value=0)

    prediction = models[horizon].predict(X)[0]
    return prediction


if __name__ == "__main__":
    pred = predict_price("data/cleaned/reliance_clean.csv", horizon=1, stock_name="Reliance")
    print(f"Next Day Prediction: {pred:.2f}")
