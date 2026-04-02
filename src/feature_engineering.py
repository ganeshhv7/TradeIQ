# src/feature_engineering.py
import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sentiment import get_news_sentiment


def create_features(df: pd.DataFrame, stock_name: str = "Reliance") -> pd.DataFrame:
    df = df.copy()

    # =========================
    # RETURNS
    # =========================
    df["return"] = df["Close"].pct_change()

    # =========================
    # LAGS
    # =========================
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = df["Close"].shift(lag)

    # =========================
    # MOVING AVERAGE
    # =========================
    for w in [7, 14, 21, 50]:
        df[f"ma_{w}"] = df["Close"].rolling(w).mean()

    # =========================
    # EMA
    # =========================
    for w in [7, 14, 21]:
        df[f"ema_{w}"] = df["Close"].ewm(span=w).mean()

    # =========================
    # RSI
    # =========================
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # =========================
    # MACD
    # =========================
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26

    # =========================
    # VOLATILITY
    # =========================
    df["volatility"] = df["return"].rolling(14).std()

    # =========================
    # OBV
    # =========================
    df["obv"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    # =========================
    # ATR
    # =========================
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # =========================
    # ROC
    # =========================
    df["roc"] = df["Close"].pct_change(periods=14)

    # =========================
    # 📰 NEWS SENTIMENT (NEW 🔥)
    # =========================
    try:
        sentiment_score = get_news_sentiment(stock_name)
    except Exception as e:
        print("Sentiment Error:", e)
        sentiment_score = 0

    df["news_sentiment"] = sentiment_score

    # =========================
    # CLEAN
    # =========================
    df.dropna(inplace=True)

    return df