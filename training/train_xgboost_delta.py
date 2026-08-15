
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
    importances = {}

    for horizon in [1, 2, 3]:
        delta_col = f"target_delta_day{horizon}"
        actual_col = f"target_aqi_day{horizon}"

        y_train_full = train_df[delta_col]
        y_train = y_train_full.iloc[:val_split_idx]
        y_val = y_train_full.iloc[val_split_idx:]

        model = XGBRegressor(
            n_estimators=n_estimators,          
            learning_rate= learning_rate,
            max_depth=5,
            subsample=1,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            early_stopping_rounds=20,
            eval_metric="rmse",
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        predicted_delta = model.predict(X_test)
        predicted_aqi = test_df["us_aqi"].values + predicted_delta

        y_true_actual = test_df[actual_col]
        results[f"day{horizon}"] = evaluate_predictions(
            y_true_actual, predicted_aqi, label=f"XGBoost-Delta (Day {horizon})"
        )
        models[f"day{horizon}"] = model

        importances[f"day{horizon}"] = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)

        print(f"  -> stopped at {model.best_iteration} trees (of 1000 max)")

    return models, results, importances

if __name__ == "__main__":
    features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])
    features_df = add_delta_targets(features_df)
    train_df, test_df = time_based_split(features_df, test_size=TEST_SIZE)

    delta_models, delta_results, delta_importances = train_xgboost_delta_models(train_df, test_df)
