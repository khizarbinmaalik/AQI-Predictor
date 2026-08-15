import pandas as pd
import numpy as np
from src.config import LAG_HOURS, TARGET_HORIZONS_DAYS

def merge_data(aqi_df, weather_df):
    merged = pd.merge(aqi_df, weather_df, on="time", how="inner")
    return merged


def merge_fire_features(features_df, daily_fire_df):
    
    features_df = features_df.copy()
    features_df["date"] = features_df["time"].dt.date

    merged = features_df.merge(daily_fire_df, on="date", how="left")

    merged["fire_count"] = merged["fire_count"].fillna(0)
    merged["fire_frp_sum"] = merged["fire_frp_sum"].fillna(0)

    return merged.drop(columns=["date"])

def engineer_features(df, lag_hours= LAG_HOURS):

    df = df.sort_values("time").reset_index(drop=True)

    # Time-based features 
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek       # 0=Monday, 6=Sunday
    df["month"] = df["time"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Lag features (past AQI values as inputs) 
    for lag in lag_hours:
        df[f"aqi_lag_{lag}h"] = df["us_aqi"].shift(lag)

    df["aqi_change_rate_1h"] = df["us_aqi"].diff(1)

    return df


def add_daily_targets(df):

    days= TARGET_HORIZONS_DAYS

    df = df.sort_values("time").reset_index(drop=True)
    df["date"] = df["time"].dt.date

    # Daily average AQI
    daily_avg = df.groupby("date")["us_aqi"].mean()

    for h in days:
        # For each row, look up the daily average 'h' days after its own date
        df[f"target_aqi_day{h}"] = df["date"].apply(
            lambda d: daily_avg.get(d + pd.Timedelta(days=h), None)
        )

    df = df.drop(columns=["date"])
    return df
