import pandas as pd
from xgboost import XGBRegressor

from src.evaluation import time_based_split, evaluate_predictions 
from src.config import TEST_SIZE, RANDOM_STATE, DROP_COLS

def get_feature_columns(df):
    """All columns except identifiers, constants, and targets."""
    return [c for c in df.columns if c not in DROP_COLS]

def train_gradient_boosting_models(train_df, test_df, n_estimators=1000, learning_rate=0.05):
    feature_cols = get_feature_columns(train_df)

    X_train_full = train_df[feature_cols]
    X_test = test_df[feature_cols]

    val_split_idx = int(len(train_df) * 0.9)
    X_train = X_train_full.iloc[:val_split_idx]
    X_val = X_train_full.iloc[val_split_idx:]

    models = {}
    results = {}
    importances = {}

    for horizon in [1, 2, 3]:
        target_col = f"target_aqi_day{horizon}"
        y_train_full = train_df[target_col]
        y_train = y_train_full.iloc[:val_split_idx]
        y_val = y_train_full.iloc[val_split_idx:]
        y_test = test_df[target_col]

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

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        results[f"day{horizon}"] = evaluate_predictions(y_test, y_pred, label=f"XGBoost (Day {horizon})")
        models[f"day{horizon}"] = model

        importances[f"day{horizon}"] = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)

        print(f"  -> stopped at {model.best_iteration} trees (of {n_estimators} max)")

    return models, results, importances


if __name__ == "__main__":
    features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])
    train_df, test_df = time_based_split(features_df, test_size=TEST_SIZE)

    xgb_models, xgb_results, xgb_importances = train_gradient_boosting_models(train_df, test_df)
    print(xgb_results['day3'])