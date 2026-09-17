"""
Model training: the final XGBoost forecaster.

Corresponds to notebook 03_forecasting_models_and_evaluation.ipynb, section 2.
"""

from __future__ import annotations

import pandas as pd
from xgboost import XGBRegressor

from src.features import RANDOM_STATE


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
) -> XGBRegressor:
    """Train the final forecasting model.

    Hyperparameters as used throughout the project: n_estimators=500,
    max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8.
    Passing X_valid/y_valid enables early-stopping-style monitoring via
    eval_set (no early_stopping_rounds is set by default, matching the
    notebook, which trains the full 500 rounds and evaluates after the fact).
    """
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    fit_kwargs = {}
    if X_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["verbose"] = False

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def feature_importance(model: XGBRegressor, feature_cols: list[str]) -> pd.DataFrame:
    """Sorted feature-importance table for the trained model."""
    return (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
