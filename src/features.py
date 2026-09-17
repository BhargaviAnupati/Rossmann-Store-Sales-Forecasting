"""
Calendar, lag, and rolling features, plus the chronological train/validation/test
split used throughout this project.

Corresponds to notebook 02_feature_engineering_and_baseline_model.ipynb,
sections 2-4.
"""

from __future__ import annotations

import pandas as pd

RANDOM_STATE = 42

# Chronological split cutoffs, used consistently across notebooks 02-04.
# Train: everything up to and including TRAIN_CUTOFF.
# Validation: the following ~6 weeks, up to and including VALID_CUTOFF.
# Test: the final ~5 weeks of the dataset.
TRAIN_CUTOFF = pd.Timestamp("2015-05-14")
VALID_CUTOFF = pd.Timestamp("2015-06-25")

LAG_DAYS = (1, 7, 14)
ROLLING_WINDOWS = (7, 30)

CATEGORICAL_CODE_COLUMNS = ("StoreType", "Assortment", "StateHoliday")

FEATURE_COLUMNS = [
    "Store", "DayOfWeek", "Month", "Year", "WeekOfYear", "IsWeekend",
    "Promo", "SchoolHoliday", "StateHoliday", "StoreType", "Assortment",
    "CompetitionDistance", "Promo2",
    "Sales_lag_1", "Sales_lag_7", "Sales_lag_14",
    "Sales_rollmean_7", "Sales_rollmean_30",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap date-derived features: day of week, month, year, week of year,
    weekend flag."""
    df = df.copy()
    df["DayOfWeek"] = df["Date"].dt.dayofweek  # 0 = Monday
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    lag_days: tuple[int, ...] = LAG_DAYS,
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS,
    dropna: bool = True,
) -> pd.DataFrame:
    """Per-store lag and rolling-average sales features.

    Computed with groupby('Store') -- these MUST be per-store, never across
    the whole panel at once, or one store's history leaks into another
    store's features.

    Rolling windows are shifted by 1 day first so the window never includes
    the current day (which would leak the target). Rows with insufficient
    history (the first N days of each store, ~4.0% of rows) are dropped
    rather than filled, since filling would fabricate history that doesn't
    exist.
    """
    df = df.copy()

    for lag in lag_days:
        df[f"Sales_lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)

    for window in rolling_windows:
        df[f"Sales_rollmean_{window}"] = (
            df.groupby("Store")["Sales"]
            .transform(lambda s: s.shift(1).rolling(window).mean())
        )

    if dropna:
        required = [f"Sales_lag_{max(lag_days)}", f"Sales_rollmean_{max(rolling_windows)}"]
        df = df.dropna(subset=required).reset_index(drop=True)

    return df


def encode_categoricals(df: pd.DataFrame, columns: tuple[str, ...] = CATEGORICAL_CODE_COLUMNS) -> pd.DataFrame:
    """Integer-code a few categorical columns XGBoost needs as numeric."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


def build_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pipeline: calendar -> lag/rolling ->
    categorical encoding. Mirrors the pipeline rebuilt independently in
    notebooks 02-04."""
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df = encode_categoricals(df)
    return df


def chronological_split(
    df: pd.DataFrame,
    train_cutoff: pd.Timestamp = TRAIN_CUTOFF,
    valid_cutoff: pd.Timestamp = VALID_CUTOFF,
):
    """Split by date rather than randomly, since a random split would let the
    model 'see the future.' Train on the earliest period, validate on the
    next ~6 weeks, test on the final ~5 weeks -- mirroring how the model
    would actually be used (forecasting forward from the past).

    With the default cutoffs: Train (737,707 rows) / Valid (38,555 rows) /
    Test (34,680 rows).
    """
    train_df = df[df["Date"] <= train_cutoff].copy()
    valid_df = df[(df["Date"] > train_cutoff) & (df["Date"] <= valid_cutoff)].copy()
    test_df = df[df["Date"] > valid_cutoff].copy()
    return train_df, valid_df, test_df


def get_feature_columns(df: pd.DataFrame, columns: list[str] = FEATURE_COLUMNS) -> list[str]:
    """Feature columns actually present in df, excluding identifiers, the
    target, and raw date fields the model can't use directly (calendar
    features already capture Date's useful signal)."""
    return [c for c in columns if c in df.columns]
