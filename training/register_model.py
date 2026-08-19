
import os
import json
import pandas as pd
import hopsworks

from src.evaluation import time_based_split
from src.config import TEST_SIZE, HOPSWORKS_API_KEY, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
from src.hopsworks_setup import get_feature_store
from training.train_xgboost_delta import add_delta_targets, get_delta_feature_columns, train_xgboost_delta_models

MODEL_DIR = "model_artifacts"
MODEL_NAME = "aqi_xgboost_delta"

def load_training_data():
    """Pulls the current feature set from Hopsworks — the actual source of
    truth, updated hourly by feature_pipeline.py. NEVER read from a local
    CSV here: it won't exist in a fresh CI checkout, and even locally it's
    a disposable cache that can drift out of sync."""
    fs = get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    df = fg.read()
    df = df.sort_values("time").reset_index(drop=True)  # never trust storage order

    print(f"Loaded {len(df)} rows from Hopsworks ({FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION})")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")

    return df

def save_model_artifacts(models, feature_cols, results):
    os.makedirs(MODEL_DIR, exist_ok=True)

    for horizon in [1, 2, 3]:
        models[f"day{horizon}"].save_model(os.path.join(MODEL_DIR, f"model_day{horizon}.json"))

    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    readme = f"""# AQI XGBoost Delta Model

## IMPORTANT: predicts a DELTA, not the absolute AQI value.

Each model file predicts: target_delta_dayH = target_aqi_dayH - us_aqi

To get an actual forecast:
    predicted_aqi = current_us_aqi + model.predict(X)

Skipping this step produces meaningless small numbers near zero, not AQI.

## Files
- model_day1.json / model_day2.json / model_day3.json — one XGBoost
  model per horizon. Load via XGBRegressor().load_model(path).
- feature_columns.json — exact ordered feature list. Input data MUST
  match this column set and order.

## Test-set performance (reconstructed, absolute AQI scale)
- Day 1: RMSE={results['day1']['rmse']:.2f}, R²={results['day1']['r2']:.3f}
- Day 2: RMSE={results['day2']['rmse']:.2f}, R²={results['day2']['r2']:.3f}
- Day 3: RMSE={results['day3']['rmse']:.2f}, R²={results['day3']['r2']:.3f}

Full methodology: docs/EDA_Findings.md
"""
    with open(os.path.join(MODEL_DIR, "README.md"), "w") as f:
        f.write(readme)

    print(f"Saved model artifacts to {MODEL_DIR}/")


def register_model(results, feature_cols, train_df):
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    mr = project.get_model_registry()

    input_example = train_df[feature_cols].iloc[0:1]

    metrics = {
        "day1_r2": float(results["day1"]["r2"]), "day1_rmse": float(results["day1"]["rmse"]),
        "day2_r2": float(results["day2"]["r2"]), "day2_rmse": float(results["day2"]["rmse"]),
        "day3_r2": float(results["day3"]["r2"]), "day3_rmse": float(results["day3"]["rmse"]),
    }

    model = mr.python.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        description=(
            "XGBoost delta/residual AQI forecasting model (3 horizons). "
            "Predicts target_aqi_dayH - us_aqi; reconstruct with "
            "predicted_aqi = us_aqi + predicted_delta. See README.md."
        ),
        input_example=input_example,
    )
    model.save(MODEL_DIR)
    print(f"Registered '{MODEL_NAME}' version {model.version} in Hopsworks Model Registry.")
    return model


if __name__ == "__main__":
    features_df = load_training_data()
    features_df = add_delta_targets(features_df)
    train_df, test_df = time_based_split(features_df, test_size=TEST_SIZE)

    models, results, importances = train_xgboost_delta_models(train_df, test_df)
    feature_cols = get_delta_feature_columns(train_df)

    save_model_artifacts(models, feature_cols, results)
    register_model(results, feature_cols, train_df)