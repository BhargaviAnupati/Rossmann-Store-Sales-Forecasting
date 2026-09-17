"""
Rossmann Store Sales Forecasting
=================================

Reusable, script-friendly versions of the logic developed in the project
notebooks (see ../notebooks). Import these modules from a notebook or a
Python script instead of copy-pasting cells:

    from src.data_loader import load_clean_data
    from src.features import build_feature_pipeline, chronological_split, get_feature_columns
    from src.train import train_xgboost, feature_importance
    from src.evaluate import naive_baselines, compare_models
    from src.interpret import compute_shap_values, counterfactual_promo_lift

Every function here mirrors what actually ran in notebooks 01-04 -- same
chronological split cutoffs, same feature set, same random seed (42) -- so
results computed through src/ match the reported notebook and report
results.
"""
