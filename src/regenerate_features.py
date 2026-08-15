"""
regenerate_features.py — one-time, run whenever LAG_HOURS (or similar
feature engineering config) changes, to refresh the local CSV cache
without re-fetching from any API.
"""

import pandas as pd
from src.feature_engineering import engineer_features, add_daily_targets

df = pd.read_csv("aqi_features.csv", parse_dates=["time"])

print(f"Rows before regeneration: {len(df)}")

df = engineer_features(df)

df = add_daily_targets(df)

df = df.dropna().reset_index(drop=True)


df.to_csv("aqi_features.csv", index=False)