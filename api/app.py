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
from src.sentiment import get_news_sentiment
app = FastAPI(title="TradeIQ Prediction API 🚀", version="2.0.0")

# =========================
# LOAD MODELS (MULTI-STOCK)
# =========================
STOCK_NAMES = ["HDFCBANK", "INFY", "RELIANCE", "TCS"]
model_store = {}

for stock in STOCK_NAMES:
    try:
        stock_dir = os.path.join(base_dir, "models", stock)
        model_store[stock] = {
            "models": joblib.load(os.path.join(stock_dir, "models.pkl")),
            "scaler": joblib.load(os.path.join(stock_dir, "scaler.pkl")),
            "features": joblib.load(os.path.join(stock_dir, "features.pkl"))
        }
        print(f"Loaded models for {stock}")
    except Exception as e:
        print(f"Error loading models for {stock}: {e}")

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
    if not model_store:
        raise HTTPException(status_code=500, detail="Models not loaded properly.")
    
    stock_name = request.ticker.split('.')[0]
    if stock_name == "INFOSYS": stock_name = "INFY"
    if stock_name not in model_store:
        raise HTTPException(status_code=400, detail=f"Stock {stock_name} not supported or models failed to load.")
        
    store = model_store[stock_name]
    models_for_stock = store["models"]
    scaler_for_stock = store["scaler"]
    features_for_stock = store["features"]

    try:
        # =========================
        # FETCH HISTORY (BUG 1 FIXED)
        # =========================
        # Need at least 200 days to compute 200-day moving averages safely
        df_raw = yf.download(request.ticker, period="2y", interval="1d")
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
        # PREDICTION ENGINE (MULTI-MODEL)
        # =========================
        latest = df_feat.iloc[-1:]
        X = latest[features_for_stock] # Ensures exact column ordering
        
        X_scaled = scaler_for_stock.transform(X)
        
        pred_1_ret = models_for_stock[1].predict(X_scaled)[0]
        pred_7_ret = models_for_stock[7].predict(X_scaled)[0]
        pred_15_ret = models_for_stock[15].predict(X_scaled)[0]

        
        current_close = float(df['Close'].iloc[-1])
        
        pred_1 = current_close * (1 + float(pred_1_ret))
        pred_7 = current_close * (1 + float(pred_7_ret))
        pred_15 = current_close * (1 + float(pred_15_ret))

        # TradeIQ Decision Logic
        sentiment_score, _ = get_news_sentiment(stock_name)
        change_pct = float(pred_1_ret) * 100
        rsi_val = float(df_feat['RSI_14'].iloc[-1])
        
        if change_pct > 1.5 and rsi_val < 65 and sentiment_score > 0.05:
            signal = "STRONG BUY"
        elif change_pct > 0 and rsi_val < 70:
            signal = "BUY"
        elif change_pct < -1.5 and rsi_val > 35 and sentiment_score < -0.05:
            signal = "STRONG SELL"
        else:
            signal = "HOLD"

        # Extract Feature Importances
        feature_importances = {}
        try:
            model = models_for_stock[1]
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feat_df = pd.DataFrame({"Feature": features_for_stock, "Importance": importances})
                feat_df = feat_df.sort_values("Importance", ascending=False).head(10)
                feature_importances = {k: float(v) for k, v in zip(feat_df["Feature"], feat_df["Importance"])}
            elif hasattr(model, 'coef_'):
                importances = abs(model.coef_)
                feat_df = pd.DataFrame({"Feature": features_for_stock, "Importance": importances})
                feat_df = feat_df.sort_values("Importance", ascending=False).head(10)
                feature_importances = {k: float(v) for k, v in zip(feat_df["Feature"], feat_df["Importance"])}
        except Exception as e:
            pass

        return {
            "status": "success",
            "ticker": request.ticker,
            "signal": signal,
            "metrics": {
                "sentiment_score": float(sentiment_score),
                "rsi": rsi_val,
                "projected_change_pct": float(change_pct)
            },
            "predictions": {
                "1_day": pred_1,
                "7_days": pred_7,
                "15_days": pred_15
            },
            "feature_importance": feature_importances
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))