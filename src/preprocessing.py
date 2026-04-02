# src/preprocessing.py

import pandas as pd

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load stock CSV and perform cleaning.
    """

    df = pd.read_csv(path)

    # Date handling
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # Convert numeric safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df