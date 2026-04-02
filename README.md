# 📈 TradeIQ: Predictive Model for Stock Market Analysis

TradeIQ is an end-to-end Machine Learning stock prediction and decision support system. It not only predicts multi-horizon future prices but also translates those technical predictions into actionable **Buy 📈**, **Sell 📉**, and **Hold 🤝** signals. 

## 🚀 Key Features

*   **Multi-Stock "Global" Training:** Uses a consolidated dataset of major Indian stocks (Reliance, TCS, Infosys, and HDFC Bank) to train a robust 'Maha-Model' capable of uncovering generalized market movements.
*   **Multi-Horizon Predictions:** Forecasts stock prices and percentage returns exactly 1-Day, 7-Days, and 15-Days into the future.
*   **Dynamic Decision Support System:** Translates price momentum, RSI values, and news sentiment into a clear risk-managed signal (e.g. `🟢 STRONG BUY` or `🔴 STRONG SELL`).
*   **Comprehensive Data Processing:** Combines Historical Prices, over 40+ Technical Indicators (RSI, MACD, ATR, Bollinger Bands), and Fundamental Metrics (P/E ratio, Market Cap) using `yfinance`.
*   **Explainable AI:** Diagnoses which features drive market prices via an internal Feature Importance dashboard in the UI.

## 📂 Project Architecture (Modular Design)

The project leverages a decoupled architecture, separating the core Data Science exploration from the Production codebase:

```text
TradeIQ/
├── data/              # Stores raw and feature-engineered stock datasets
├── notebooks/         # Full Data Science Pipeline
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb (Global Model Training Pipeline)
├── src/               # Modular logic scripts utilized by APIs & Scripts
│   ├── feature_engineering.py
│   └── sentiment.py
├── api/               # FastAPI Backend
│   └── app.py         # Prediction microservice
├── app/               # Streamlit Frontend UI
│   └── streamlit_app.py 
└── models/            # Serialized XGBoost models & target scalers (.pkl)
```

## 🛠️ Technology Stack
- **Data Engineering:** `pandas`, `numpy`, `yfinance`
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Backend API:** `FastAPI`, `Uvicorn`
- **Frontend Dashboard:** `Streamlit`, `Plotly`, `Matplotlib`
- **NLP / Media:** `vaderSentiment`, `newsapi-python` (Abstracted via src)

## 🏁 How to Run

1. **Setup Environment:**
   Ensure you have configured your virtual environment (`venv`) and installed all standard `requirements.txt` dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the FastAPI Prediction Backend Server:**
   Navigate to the project root and run Uvicorn to host the prediction API endpoints.
   ```bash
   uvicorn api.app:app --reload
   ```

3. **Launch the Interactive Streamlit Dashboard:**
   In a separate terminal window, launch the UI app which interfaces with the local codebase and live data.
   ```bash
   streamlit run app/streamlit_app.py
   ```

## 🛡️ Risk Disclaimer
*TradeIQ is purely an analytical project providing simulated technical insights. It is not financial advice. Always consult a certified financial planner before undertaking any trades in the real market.*
