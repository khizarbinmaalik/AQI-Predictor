import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.evaluation import time_based_split, evaluate_predictions 
from src.config import TEST_SIZE, RANDOM_STATE, DROP_COLS


def get_feature_columns(df):
    """All columns except identifiers, constants, and targets."""
    return [c for c in df.columns if c not in DROP_COLS]



def train_random_forest_models(train_df, test_df, n_estimators=200, max_depth=None):
    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    models = {}
    results = {}
    importances = {}

    for horizon in [1, 2, 3]:
        target_col = f"target_aqi_day{horizon}"
        y_train = train_df[target_col]
        y_test = test_df[target_col]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            max_features= 'sqrt',
            min_samples_split= 20,
            min_samples_leaf= 10,
            max_samples= None,
            bootstrap= True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        results[f"day{horizon}"] = evaluate_predictions(y_test, y_pred, label=f"Random Forest (Day {horizon})")
        models[f"day{horizon}"] = model

        importances[f"day{horizon}"] = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)

    return models, results, importances


def compare_importances_across_horizons(importances_dict):
    combined = pd.DataFrame({
        "day1": importances_dict["day1"],
        "day2": importances_dict["day2"],
        "day3": importances_dict["day3"],
    })
    combined["avg"] = combined.mean(axis=1)
    combined = combined.sort_values("avg", ascending=False)
    return combined


if __name__ == "__main__":
    features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])
    train_df, test_df = time_based_split(features_df, test_size=TEST_SIZE)

    rf_models, rf_results, rf_importances = train_random_forest_models(train_df, test_df)


        