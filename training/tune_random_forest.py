import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from src.config import TEST_SIZE, RANDOM_STATE
from src.evaluation import time_based_split
from training.train_random_forest import get_feature_columns

def tune_random_forest(X_train, y_train, n_iter=20):
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2",0.5, 0.75, None],
        "max_samples": [0.5, 0.75, 1.0, None],
    }

    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print("Best CV R²:", search.best_score_)
    return search.best_estimator_

features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])
train_df, test_df = time_based_split(features_df, test_size=TEST_SIZE)
feature_cols = get_feature_columns(train_df)
X_train = train_df[feature_cols]
y_train_day1 = train_df["target_aqi_day1"]

best_rf_day1 = tune_random_forest(X_train, y_train_day1)   