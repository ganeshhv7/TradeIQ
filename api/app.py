# api/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys, os
import joblib
import pandas as pd
import yfinance as yf

# Add project root and src to path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "src"))

from src.feature_engineering import create_features

app = FastAPI(title="TradeIQ Prediction API 🚀", version="2.0.0")

# =========================
# LOAD MODELS (BUG 3 FIXED)
# =========================
try:
    models = joblib.load(os.path.join(base_dir, "models", "models.pkl"))
    scaler = joblib.load(os.path.join(base_dir, "models", "scaler.pkl"))
    feature_names = joblib.load(os.path.join(base_dir, "models", "features.pkl"))
except Exception as e:
    print(f"Error loading models: {e}")
    models, scaler, feature_names = None, None, None

# =========================
# SCHEMAS (INPUT VALIDATION)
# =========================
class PredictRequest(BaseModel):
    ticker: str = "RELIANCE.NS"
    
EXPECTED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to handle YFinance formatting quirks."""
    df = df.copy()
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
    return df

@app.get("/")
def home():
    return {"message": "API is running 🚀. Use /docs to test predictions."}

@app.post("/predict")
def predict(request: PredictRequest):
    """
    Input: Ticker symbol (e.g., RELIANCE.NS)
    Output: Multi-horizon predicted prices
    """
    if models is None:
        raise HTTPException(status_code=500, detail="Models not loaded properly.")
    
    try:
        # =========================
        # FETCH HISTORY (BUG 1 FIXED)
        # =========================
        # Need at least 60 days to compute 50-day moving averages safely
        df_raw = yf.download(request.ticker, period="6mo", interval="1d")
        if df_raw.empty:
            raise ValueError(f"No data returned for ticker: {request.ticker}")
        
        df = _normalize_price_frame(df_raw)
        
        # =========================
        # FEATURE ENGINEERING (BUG 4 FIXED)
        # =========================
        stock_name = request.ticker.split('.')[0] # Pass dynamic name for sentiment
        df_feat = create_features(df, stock_name=stock_name)
        
        if df_feat.empty:
            raise ValueError("Feature engineering resulted in an empty dataset. Try fetching more historical data.")

        # =========================
        # PREDICTION ENGINE (BUG 2 FIXED)
        # =========================
        latest = df_feat.iloc[-1:]
        X = latest[feature_names] # Ensures exact column ordering
        
        X_scaled = scaler.transform(X)
        
        pred_1_ret = models[1].predict(X_scaled)[0]
        pred_7_ret = models[7].predict(X_scaled)[0]
        pred_15_ret = models[15].predict(X_scaled)[0]
        
        current_close = float(df['Close'].iloc[-1])
        
        return {
            "status": "success",
            "ticker": request.ticker,
            "predictions": {
                "1_day": current_close * (1 + float(pred_1_ret)),
                "7_days": current_close * (1 + float(pred_7_ret)),
                "15_days": current_close * (1 + float(pred_15_ret))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))