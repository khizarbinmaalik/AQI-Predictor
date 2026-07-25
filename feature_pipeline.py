import pandas as pd
import requests
from datetime import datetime

def fetch_aqi_data(latitude, longitude, timezone="auto", past_days=0, forecast_days=1):

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "timezone": timezone,
        "past_days": past_days,
        "forecast_days": forecast_days,
        }


    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code} - {response.text}")

    data = response.json()
    df = pd.DataFrame(data['hourly'])
    df["time"] = pd.to_datetime(df["time"])

    df["latitude"] = data["latitude"]
    df["longitude"] = data["longitude"]

    return df

def fetch_weather_data(latitude, longitude, timezone="auto", past_days=0, forecast_days=1):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation",
        "timezone": timezone,
        "past_days": past_days,
        "forecast_days": forecast_days,
    }

    response = requests.get(url, params=params, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Weather API request failed: {response.status_code} - {response.text}")

    data = response.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df

def merge_data(aqi_df, weather_df):
    merged = pd.merge(aqi_df, weather_df, on="time", how="inner")
    return merged

def engineer_features(df, lag_hours=[1, 3, 24], target_horizon_hours=72):

    df = df.sort_values("time").reset_index(drop=True)

    # Time-based features 
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek       # 0=Monday, 6=Sunday
    df["month"] = df["time"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Lag features (past AQI values as inputs) 
    for lag in lag_hours:
        df[f"aqi_lag_{lag}h"] = df["us_aqi"].shift(lag)

    #  Rate of change (is AQI rising or falling right now?) 
    df["aqi_change_rate_1h"] = df["us_aqi"].diff(1)

    # target: aqi 'target_horizon_hours' into the future
    df["target_aqi"] = df["us_aqi"].shift(-target_horizon_hours)

    # Drop rows where we don't have enough history or future data ---
    df = df.dropna().reset_index(drop=True)

    return df


def add_daily_targets(df):

    days=[1, 2, 3]

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

aqi_df = fetch_aqi_data(latitude=27.70, longitude=68.86, past_days=10, forecast_days=0)
weather_df = fetch_weather_data(latitude=27.70, longitude=68.86, past_days=10, forecast_days=0)
merged_df = merge_data(aqi_df, weather_df)

features_df = engineer_features(merged_df)   # hour, lag, rate-of-change (unchanged from before)
features_df = add_daily_targets(features_df) # new target columns
features_df = features_df.dropna().reset_index(drop=True)

print(features_df[["time", "us_aqi", "temperature_2m", "wind_speed_10m",
                    "target_aqi_day1", "target_aqi_day2", "target_aqi_day3"]].head(10))
