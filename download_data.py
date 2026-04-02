import yfinance as yf
import os

tickers = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS"
]

os.makedirs("data/raw", exist_ok=True)

for ticker in tickers:
    print(f"\nDownloading {ticker} ...")
    df = yf.download(
        ticker,
        start="2015-01-01",
        end="2025-01-01",
        interval="1d"
    )
    
    csv_path = f"data/raw/{ticker.replace('.', '_')}.csv"
    df.to_csv(csv_path)
    
    print(f"Saved {ticker} to {csv_path}")

print("\nDownload Complete!")
