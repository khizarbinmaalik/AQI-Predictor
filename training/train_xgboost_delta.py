
import pandas as pd
from xgboost import XGBRegressor

from src.evaluation import time_based_split, evaluate_predictions
from src.config import TEST_SIZE, RANDOM_STATE, DROP_COLS

DELTA_TARGET_COLS = ["target_delta_day1", "target_delta_day2", "target_delta_day3"]


def add_delta_targets(df, horizons=[1, 2, 3]):
    df = df.copy()
    for h in horizons:
        df[f"target_delta_day{h}"] = df[f"target_aqi_day{h}"] - df["us_aqi"]
    return df


def get_delta_feature_columns(df):
    excluded = set(DROP_COLS) | set(DELTA_TARGET_COLS)
    return [c for c in df.columns if c not in excluded]


def train_xgboost_delta_models(train_df, test_df, n_estimators=1000, learning_rate=0.05):
    feature_cols = get_delta_feature_columns(train_df)

    X_train_full = train_df[feature_cols]
    X_test = test_df[feature_cols]

    val_split_idx = int(len(train_df) * 0.9)
    X_train = X_train_full.iloc[:val_split_idx]
    X_val = X_train_full.iloc[val_split_idx:]

    models = {}
    results = {}

    for horizon in [1, 2, 3]:
        delta_col = f"target_delta_day{horizon}"
        actual_col = f"target_aqi_day{horizon}"

        y_train_full = train_df[delta_col]
        y_train = y_train_full.iloc[:val_split_idx]
        y_val = y_train_full.iloc[val_split_idx:]

        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            early_stopping_rounds=30,
            eval_metric="rmse",
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Predict the DELTA, then reconstruct the actual AQI prediction
        predicted_delta = model.predict(X_test)
        predicted_aqi = test_df["us_aqi"].values + predicted_delta

        y_true_actual = test_df[actual_col]
        results[f"day{horizon}"] = evaluate_predictions(
            y_true_actual, predicted_aqi, label=f"XGBoost-Delta (Day {horizon})"
        )
        models[f"day{horizon}"] = model

    return models, results


if __name__ == "__main__":
    features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])
    features_df = add_delta_targets(features_df)

    train_df, test_df = time_based_split(features_df, test_size=TEST_SIZE)

    delta_models, delta_results = train_xgboost_delta_models(train_df, test_df)

    print("\n--- Comparison vs. champion (absolute target) ---")
    champion_r2 = {"day1": 0.523, "day2": 0.169, "day3": 0.020}
    for horizon_key, champ_r2 in champion_r2.items():
        delta_r2 = delta_results[horizon_key]["r2"]
        print(f"{horizon_key}: champion R²={champ_r2:.3f}   delta-model R²={delta_r2:.3f}   gap={delta_r2 - champ_r2:+.3f}")