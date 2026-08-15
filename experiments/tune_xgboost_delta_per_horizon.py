"""
experiments/tune_xgboost_delta_per_horizon.py

Tunes XGBoost hyperparameters SEPARATELY for each horizon's delta target,
then retrains and evaluates each on the real held-out test set (reconstructed
to absolute AQI scale) — not just the CV score, consistent with this
project's established evaluation discipline.
"""

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from scipy.stats import randint, uniform

from src.config import TEST_SIZE, RANDOM_STATE
from src.evaluation import time_based_split, evaluate_predictions
from training.train_xgboost_delta import add_delta_targets, get_delta_feature_columns


def tune_xgboost_delta(X_train, y_train, n_iter=30):
    param_dist = {
        "n_estimators": randint(100, 600),
        "learning_rate": uniform(0.01, 0.29),
        "max_depth": randint(3, 10),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 5),
        "reg_alpha": [0, 0.001, 0.01, 0.1, 1],
        "reg_lambda": [0.1, 1, 5, 10],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_params_, search.best_score_


def train_with_params(train_df, test_df, feature_cols, horizon, params):
    delta_col = f"target_delta_day{horizon}"
    actual_col = f"target_aqi_day{horizon}"

    X_train_full = train_df[feature_cols]
    X_test = test_df[feature_cols]

    val_split_idx = int(len(train_df) * 0.9)
    X_train = X_train_full.iloc[:val_split_idx]
    X_val = X_train_full.iloc[val_split_idx:]
    y_train_full = train_df[delta_col]
    y_train = y_train_full.iloc[:val_split_idx]
    y_val = y_train_full.iloc[val_split_idx:]

    model = XGBRegressor(
        **params,
        n_estimators=1000,  # ceiling — early stopping decides real count
        random_state=RANDOM_STATE,
        early_stopping_rounds=30,
        eval_metric="rmse",
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    predicted_delta = model.predict(X_test)
    predicted_aqi = test_df["us_aqi"].values + predicted_delta
    return evaluate_predictions(test_df[actual_col], predicted_aqi, label=f"Delta Per-Horizon (Day {horizon})")


if __name__ == "__main__":
    features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])
    features_df = add_delta_targets(features_df)
    train_df, test_df = time_based_split(features_df, test_size=TEST_SIZE)
    feature_cols = get_delta_feature_columns(train_df)

    champion_r2 = {"day1": 0.546, "day2": 0.223, "day3": 0.066}
    results = {}

    for horizon in [1, 2, 3]:
        print(f"\nTuning Day {horizon}...")
        X_train = train_df[feature_cols]
        y_train = train_df[f"target_delta_day{horizon}"]

        best_params, best_cv_score = tune_xgboost_delta(X_train, y_train)
        # n_estimators from the search is dropped — early stopping picks the real count instead
        best_params.pop("n_estimators", None)
        print(f"Day {horizon} best params: {best_params}")

        result = train_with_params(train_df, test_df, feature_cols, horizon, best_params)
        results[f"day{horizon}"] = result

    print("\n--- Comparison: shared delta champion vs. per-horizon tuned ---")
    for horizon_key, champ_r2 in champion_r2.items():
        new_r2 = results[horizon_key]["r2"]
        print(f"{horizon_key}: champion={champ_r2:.3f}   per-horizon-tuned={new_r2:.3f}   gap={new_r2 - champ_r2:+.3f}")