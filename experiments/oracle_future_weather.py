"""
experiments/oracle_future_weather.py

DIAGNOSTIC EXPERIMENT — NOT for deployment.

Tests the CEILING on how much genuine future weather information (rather
than today's snapshot) could improve forecasting accuracy, by using the
ACTUAL historical weather that occurred on each target day as a feature.

This is an "oracle" test: it answers "if we had a perfect weather forecast,
how much would it help?" — not what's achievable in production, since a real
deployed app only has access to Open-Meteo's own (imperfect) multi-day
forecast, not the true future. If this shows a meaningful gain, the next
step is rebuilding the same feature using Open-Meteo's actual forecast
endpoint for real deployment. If it shows little gain, the idea isn't worth
pursuing further.
"""

import pandas as pd

from src.evaluation import time_based_split
from src.config import TEST_SIZE
from training.train_xgboost import train_gradient_boosting_models


def add_future_weather_oracle_features(df, horizons=[1, 2, 3]):
    """
    Adds the ACTUAL daily-average weather that occurred on each future
    target date, as new features — an oracle test of a perfect forecast.
    """
    df = df.sort_values("time").reset_index(drop=True)
    df["date"] = df["time"].dt.date

    daily_weather = df.groupby("date").agg(
        temperature_2m=("temperature_2m", "mean"),
        relative_humidity_2m=("relative_humidity_2m", "mean"),
        wind_speed_10m=("wind_speed_10m", "mean"),
        surface_pressure=("surface_pressure", "mean"),
        precipitation=("precipitation", "sum"),
    )

    for h in horizons:
        for col in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure", "precipitation"]:
            df[f"future_{col}_day{h}"] = df["date"].apply(
                lambda d, col=col, h=h: daily_weather[col].get(d + pd.Timedelta(days=h), None)
            )

    return df.drop(columns=["date"])


if __name__ == "__main__":
    features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])

    print("Adding oracle future-weather features...")
    oracle_df = add_future_weather_oracle_features(features_df)
    oracle_df = oracle_df.dropna().reset_index(drop=True)
    print(f"Rows after adding oracle features: {len(oracle_df)}")

    train_df, test_df = time_based_split(oracle_df, test_size=TEST_SIZE)

    print("\n--- XGBoost WITH oracle future weather ---")
    oracle_models, oracle_results, oracle_importances = train_gradient_boosting_models(train_df, test_df)

    print("\n--- Comparison vs. champion (no future weather) ---")
    champion_r2 = {"day1": 0.523, "day2": 0.169, "day3": 0.020}
    for horizon_key, champ_r2 in champion_r2.items():
        oracle_r2 = oracle_results[horizon_key]["r2"]
        print(f"{horizon_key}: champion R²={champ_r2:.3f}   oracle R²={oracle_r2:.3f}   gap={oracle_r2 - champ_r2:+.3f}")