import os
import sys

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)

import yfinance as yf
import pandas as pd
from src.feature_engineering import create_features

df_raw = yf.download("RELIANCE.NS", period="2y", interval="1d")
df = df_raw.copy()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [str(v).strip() for v in df.columns.get_level_values(0)]
date_col = None
for candidate in ("Date", "Datetime"):
    if candidate in df.columns:
        date_col = candidate
        break
if date_col is not None:
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
else:
    df.index = pd.to_datetime(df.index)
df.index.name = "Date"

df_feat = create_features(df)
print(list(df_feat.columns))
