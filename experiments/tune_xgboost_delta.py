import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from scipy.stats import randint, uniform

from src.config import TEST_SIZE, RANDOM_STATE
from src.evaluation import time_based_split
from training.train_xgboost_delta import add_delta_targets, get_delta_feature_columns


def tune_xgboost_delta(X_train, y_train, n_iter=50):
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


if __name__ == "__main__":
    features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])
    features_df = add_delta_targets(features_df)
    train_df, test_df = time_based_split(features_df, test_size=TEST_SIZE)

    feature_cols = get_delta_feature_columns(train_df)
    X_train = train_df[feature_cols]
    y_train_day1_delta = train_df["target_delta_day1"]

    best_params, best_score = tune_xgboost_delta(X_train, y_train_day1_delta)
    print("Best params:", best_params)
    print("Best CV R² (on delta scale, NOT comparable to reconstructed-AQI R²):", best_score)