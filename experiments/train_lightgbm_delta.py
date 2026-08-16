import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from src.evaluation import time_based_split, evaluate_predictions
from src.config import TEST_SIZE, RANDOM_STATE
from training.train_xgboost_delta import add_delta_targets, get_delta_feature_columns


def train_lightgbm_delta_models(train_df, test_df, n_estimators=1000, learning_rate=0.05):
    feature_cols = get_delta_feature_columns(train_df)

    X_train_full = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # Same chronological validation carve as XGBoost — for early stopping only
    val_split_idx = int(len(train_df) * 0.9)
    X_train = X_train_full.iloc[:val_split_idx]
    X_val = X_train_full.iloc[val_split_idx:]

    models = {}
    results = {}
    importances = {}

    for horizon in [1, 2, 3]:
        delta_col = f"target_delta_day{horizon}"
        actual_col = f"target_aqi_day{horizon}"

        y_train_full = train_df[delta_col]
        y_train = y_train_full.iloc[:val_split_idx]
        y_val = y_train_full.iloc[val_split_idx:]

        model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            subsample=1.0,          # matches the champion's confirmed-helpful setting
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[early_stopping(stopping_rounds=20, verbose=False), log_evaluation(period=0)],
        )

        predicted_delta = model.predict(X_test)
        predicted_aqi = test_df["us_aqi"].values + predicted_delta

        y_true_actual = test_df[actual_col]
        results[f"day{horizon}"] = evaluate_predictions(
            y_true_actual, predicted_aqi, label=f"LightGBM-Delta (Day {horizon})"
        )
        models[f"day{horizon}"] = model

        importances[f"day{horizon}"] = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)

        print(f"  -> stopped at {model.best_iteration_} trees (of {n_estimators} max)")

    return models, results, importances


if __name__ == "__main__":
    features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])
    features_df = add_delta_targets(features_df)
    train_df, test_df = time_based_split(features_df, test_size=TEST_SIZE)

    lgbm_models, lgbm_results, lgbm_importances = train_lightgbm_delta_models(train_df, test_df)

    print("\n--- Comparison vs. current XGBoost-Delta champion ---")
    champion_r2 = {"day1": 0.586, "day2": 0.221, "day3": 0.073}  # your current, real numbers
    for horizon_key, champ_r2 in champion_r2.items():
        lgbm_r2 = lgbm_results[horizon_key]["r2"]
        print(f"{horizon_key}: XGBoost={champ_r2:.3f}   LightGBM={lgbm_r2:.3f}   gap={lgbm_r2 - champ_r2:+.3f}")