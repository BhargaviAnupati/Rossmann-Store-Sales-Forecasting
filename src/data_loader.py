"""
Data loading and cleaning for the Rossmann Store Sales dataset.

Source: Kaggle "Rossmann Store Sales" competition
https://www.kaggle.com/c/rossmann-store-sales/data

Corresponds to notebook 01_data_loading_and_eda.ipynb, sections 1-3.
"""

from __future__ import annotations

import pandas as pd

RAW_TRAIN_PATH = "data/raw/train.csv"
RAW_STORE_PATH = "data/raw/store.csv"


def load_raw_data(train_path: str = RAW_TRAIN_PATH, store_path: str = RAW_STORE_PATH):
    """Load the two raw Kaggle CSVs (train.csv, store.csv) as-is.

    Returns (train, store) DataFrames. Requires the files to already be
    downloaded into data/raw/ -- see data/README.md.
    """
    train = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(store_path)
    return train, store


def merge_and_clean(train: pd.DataFrame, store: pd.DataFrame, drop_closed: bool = True) -> pd.DataFrame:
    """Merge train + store on Store id, sort chronologically per store, and
    optionally drop closed-store days.

    Sales is always 0 when a store is closed (Open == 0); those rows aren't
    informative about demand and are excluded from modeling and from most
    EDA about sales *levels* (17.0% of raw rows).
    """
    df = train.merge(store, on="Store", how="left")
    if drop_closed:
        df = df[df["Open"] == 1].copy()
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


def load_clean_data(
    train_path: str = RAW_TRAIN_PATH,
    store_path: str = RAW_STORE_PATH,
    drop_closed: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper: load raw CSVs, merge, and clean in one call.

    Returns a DataFrame of shape (844392, 18) with the default drop_closed=True
    (1,017,209 raw rows -> 844,392 after removing the 17.0% closed-store days).
    """
    train, store = load_raw_data(train_path, store_path)
    return merge_and_clean(train, store, drop_closed=drop_closed)


def data_quality_summary(df: pd.DataFrame) -> dict:
    """Quick data-quality snapshot, matching the checks run in notebook 01."""
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_stores": int(df["Store"].nunique()) if "Store" in df.columns else None,
        "date_min": str(df["Date"].min()) if "Date" in df.columns else None,
        "date_max": str(df["Date"].max()) if "Date" in df.columns else None,
        "missing_values_total": int(df.isnull().sum().sum()),
    }


if __name__ == "__main__":
    data = load_clean_data()
    print(data_quality_summary(data))
