import pandas as pd

from src.evaluation import time_based_split 
from src.feature_engineering import add_advanced_features
from src.model_utils import get_feature_columns, evaluate_ensemble
from src.config import TEST_SIZE

from training.train_xgboost import train_gradient_boosting_models
from training.train_random_forest import train_random_forest_models


if __name__ == "__main__":
    # Load your existing engineered features, then layer Round 2 features on top
    features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])
    features_df = add_advanced_features(features_df)
    features_df = features_df.dropna().reset_index(drop=True)  # trims new rolling/diff edge NaNs

    print(f"Rows after Round 2 feature engineering: {len(features_df)}")

    train_df, test_df = time_based_split(features_df, test_size=TEST_SIZE)

    print("\n--- XGBoost (v2 features, pruned) ---")
    xgb_models, xgb_results, _ = train_gradient_boosting_models(train_df, test_df, prune_weak=False)

    print("\n--- Random Forest (v2 features, pruned) ---")
    rf_models, rf_results, _ = train_random_forest_models(train_df, test_df, prune_weak=False)

    print("\n--- Ensemble (XGBoost + Random Forest) ---")
    feature_cols = get_feature_columns(train_df, prune_weak=False)
    ensemble_results = evaluate_ensemble(xgb_models, rf_models, test_df, feature_cols, xgb_weight=0.5)