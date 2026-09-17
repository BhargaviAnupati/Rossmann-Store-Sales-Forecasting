"""
Model interpretation via SHAP, plus the model-based counterfactual promotion
lift analysis.

Corresponds to notebook 04_interpretation_and_promo_impact.ipynb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SAMPLE_SIZE = 3000
RANDOM_STATE = 42


def compute_shap_values(model, X_valid: pd.DataFrame, sample_size: int = DEFAULT_SAMPLE_SIZE):
    """Sample the validation set and compute SHAP values via TreeExplainer.

    Returns (X_sample, shap_values).
    """
    import shap

    X_sample = X_valid.sample(n=min(sample_size, len(X_valid)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return X_sample, shap_values


def plot_global_importance(shap_values, max_display: int = 15):
    import shap

    shap.plots.bar(shap_values, max_display=max_display)


def plot_beeswarm(shap_values, max_display: int = 15):
    import shap

    shap.plots.beeswarm(shap_values, max_display=max_display)


def plot_dependence(shap_values, feature: str):
    """Scatter (dependence) plot for one feature, colored by SHAP
    interaction value -- used for Promo and StateHoliday in notebook 04."""
    import shap

    shap.plots.scatter(shap_values[:, feature], color=shap_values)


def counterfactual_promo_lift(model, X_valid: pd.DataFrame) -> dict:
    """Model-based counterfactual estimate of the promotion effect.

    Rather than comparing raw group averages (which mixes in whatever
    stores/seasons happen to run more promotions -- a naive comparison that
    overstates the effect at roughly +38.8% in this dataset), this takes the
    SAME rows, forces Promo=0 for all of them and predicts, then forces
    Promo=1 for all of them and predicts, and compares the two. This isolates
    the promo effect while holding every other factor (store, day of week,
    season, lag features) fixed.

    Caveat: this is a model-based counterfactual, not a randomized
    experiment. It assumes the model has correctly learned how Promo
    interacts with other features -- a reasonable and commonly used
    approach, but an assumption worth stating rather than presenting as
    definitive causal proof.
    """
    X_no_promo = X_valid.copy()
    X_no_promo["Promo"] = 0
    pred_no_promo = model.predict(X_no_promo)

    X_with_promo = X_valid.copy()
    X_with_promo["Promo"] = 1
    pred_with_promo = model.predict(X_with_promo)

    avg_no_promo = float(pred_no_promo.mean())
    avg_with_promo = float(pred_with_promo.mean())

    return {
        "avg_predicted_sales_without_promo": avg_no_promo,
        "avg_predicted_sales_with_promo": avg_with_promo,
        "relative_lift": avg_with_promo / avg_no_promo - 1,
        "absolute_lift_per_store_day": avg_with_promo - avg_no_promo,
        "lift_distribution": pred_with_promo - pred_no_promo,
    }


def explain_prediction(shap_values, index: int, max_display: int = 15):
    """Waterfall plot explaining a single prediction."""
    import shap

    shap.plots.waterfall(shap_values[index], max_display=max_display)
