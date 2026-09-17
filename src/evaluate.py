"""
Evaluation: RMSE / MAE / RMSPE, the two naive baselines, and the model
comparison table.

Corresponds to notebook 02_feature_engineering_and_baseline_model.ipynb
(section 5-6) and notebook 03_forecasting_models_and_evaluation.ipynb
(sections 3, 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmspe(y_true, y_pred) -> float:
    """Root Mean Squared Percentage Error, ignoring rows where actual sales
    are 0 (can't compute a meaningful % error against zero).

    This is the original Kaggle competition metric, and the easiest of the
    three to explain to a non-technical audience as "average % off."
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def evaluate(name: str, y_true, y_pred) -> dict:
    """RMSE, MAE, and RMSPE for one model/baseline."""
    return {
        "Model": name,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSPE": rmspe(y_true, y_pred),
    }


def naive_baselines(valid_df: pd.DataFrame, target_col: str = "Sales") -> pd.DataFrame:
    """The two naive forecasts used as a floor for the real model:

    - 'same day last week': predict this day's sales = Sales_lag_7
    - '7-day rolling average': predict using the trailing 7-day average
      (Sales_rollmean_7)

    Requires Sales_lag_7 and Sales_rollmean_7 to already be present (see
    src/features.py). If a sophisticated model can't beat these, it isn't
    adding value.
    """
    rows = [
        evaluate("Naive: same day last week", valid_df[target_col], valid_df["Sales_lag_7"]),
        evaluate("Naive: 7-day rolling avg", valid_df[target_col], valid_df["Sales_rollmean_7"]),
    ]
    return pd.DataFrame(rows).set_index("Model")


def compare_models(y_true, predictions: dict) -> pd.DataFrame:
    """Build a comparison table from {model_name: y_pred array} plus the
    two naive baselines already computed via naive_baselines().

    `predictions` should include a 'XGBoost' (or similarly named) entry;
    naive baselines are usually merged in via pd.concat with the result of
    naive_baselines() -- see notebook 03, section 6.
    """
    rows = [evaluate(name, y_true, y_pred) for name, y_pred in predictions.items()]
    return pd.DataFrame(rows).set_index("Model").round(1)
