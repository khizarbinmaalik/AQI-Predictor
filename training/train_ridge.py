import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.evaluation import time_based_split, evaluate_predictions
from src.config import TEST_SIZE, RANDOM_STATE, DROP_COLS



def get_feature_columns(df):
    """All columns except identifiers, constants, and targets."""
    return [c for c in df.columns if c not in DROP_COLS]


def train_ridge_models(train_df, test_df, alpha=1):
    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    results = {}

    for horizon in [1, 2, 3]:
        target_col = f"target_aqi_day{horizon}"
        y_train = train_df[target_col]
        y_test = test_df[target_col]

        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        results[f"day{horizon}"] = evaluate_predictions(y_test, y_pred, label=f"Ridge (Day {horizon})")
        models[f"day{horizon}"] = model

    return models, results, scaler, feature_cols


if __name__ == "__main__":
    features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])
    train_df, test_df = time_based_split(features_df, test_size=TEST_SIZE)

    ridge_models, ridge_results, scaler, feature_cols = train_ridge_models(train_df, test_df)