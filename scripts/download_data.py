import yfinance as yf
import os

tickers = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS"
]

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(os.path.join(base_dir, "data", "raw"), exist_ok=True)

for ticker in tickers:
    print(f"\nDownloading {ticker} ...")
    df = yf.download(
        ticker,
        start="2015-01-01",
        end="2025-01-01",
        interval="1d"
    )
    
    csv_path = os.path.join(base_dir, "data", "raw", f"{ticker.replace('.', '_')}.csv")
    df.to_csv(csv_path)
    
    print(f"Saved {ticker} to {csv_path}")

print("\nDownload Complete!")
