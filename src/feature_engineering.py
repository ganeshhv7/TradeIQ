import sys, os
import numpy as np
import pandas as pd
from typing import Tuple

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, min_periods=fast).mean()
    ema_slow = prices.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_atr(df, window=14):
    high_low = df["High"] - df["Low"]
    high_pc = (df["High"] - df["Close"].shift(1)).abs()
    low_pc  = (df["Low"]  - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()

def compute_roc(df, period=12):
    return ((df["Close"] - df["Close"].shift(period)) / df["Close"].shift(period)) * 100

def compute_williams_r(df, window=14):
    highest_high = df["High"].rolling(window).max()
    lowest_low   = df["Low"].rolling(window).min()
    return -100 * ((highest_high - df["Close"]) / (highest_high - lowest_low + 1e-10))

def compute_obv(df):
    direction = np.sign(df["Close"].diff())
    df["OBV"] = (direction * df["Volume"]).fillna(0).cumsum()
    return df

def compute_stochastic(df, window=14, smooth_k=3):
    low_min  = df["Low"].rolling(window).min()
    high_max = df["High"].rolling(window).max()
    df["Stoch_%K"] = 100 * ((df["Close"] - low_min) / (high_max - low_min + 1e-10))
    df["Stoch_%D"] = df["Stoch_%K"].rolling(smooth_k).mean()
    return df

def compute_cci(df, window=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_ma = tp.rolling(window).mean()
    mean_dev = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df["CCI"] = (tp - tp_ma) / (0.015 * mean_dev + 1e-10)
    return df

def create_features(df: pd.DataFrame, stock_name: str = "Reliance") -> pd.DataFrame:
    df = df.copy()

    # Returns
    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["Return_30"] = df["Close"].pct_change(30) * 100
    df["Return_60"] = df["Close"].pct_change(60) * 100

    # Lags
    for lag in [1, 7, 14, 30]:
        df[f"Lag_{lag}"] = df["Close"].shift(lag)

    # Moving Averages
    for w in [7, 14, 21, 50, 100, 200]:
        df[f"MA_{w}"] = df["Close"].rolling(w).mean()

    # Exponential Moving Averages
    for w in [7, 14, 21, 50]:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

    # RSI & MACD
    df["RSI_14"] = compute_rsi(df["Close"], 14)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(df["Close"])

    # Volatility
    for w in [7, 14, 21, 30]:
        df[f"Volatility_{w}"] = df["Close"].pct_change().rolling(w).std()

    # Advanced custom features
    df["MA_ratio"] = df["MA_50"] / (df["MA_200"] + 1e-10)
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["Volatility_change"] = df["Volatility_14"].pct_change()
    df["Volume_change"] = df["Volume"].pct_change()

    # Bollinger Bands
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["STD20"] = df["Close"].rolling(window=20).std()
    df["Upper_Band"] = df["MA20"] + (2 * df["STD20"])
    df["Lower_Band"] = df["MA20"] - (2 * df["STD20"])

    # Oscillators
    df["ATR_14"] = compute_atr(df, window=14)
    df["Williams_%R"] = compute_williams_r(df, 14)
    
    for p in [5, 10, 14, 21]:
        df[f"ROC_{p}"] = compute_roc(df, period=p)

    df = compute_obv(df)
    df = compute_stochastic(df, 14, 3)
    df = compute_cci(df, 20)

    # Drop NaNs created by rolling metrics (like MA_200)
    df.dropna(inplace=True)

    return df