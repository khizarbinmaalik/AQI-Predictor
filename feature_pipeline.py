import pandas as pd
import requests
from datetime import datetime
from hopsworks_setup import get_feature_store, get_or_create_aqi_feature_group, insert_features 

# Using the Start and end date parameters to fetch historical data from the API.
def fetch_aqi_data(latitude, longitude, timezone="auto", past_days=None, forecast_days=None, start_date=None, end_date=None):

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "timezone": timezone
        }

    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    else:
        params["past_days"] = past_days
        params["forecast_days"] = forecast_days

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code} - {response.text}")

    data = response.json()
    df = pd.DataFrame(data['hourly'])
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = data["latitude"]
    df["longitude"] = data["longitude"]

    return df

def fetch_weather_data(latitude, longitude, timezone="auto", past_days=None, forecast_days=None,
                        start_date=None, end_date=None):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation",
        "timezone": timezone,
    }

    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    else:
        params["past_days"] = past_days
        params["forecast_days"] = forecast_days

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

def engineer_features(df, lag_hours=[1, 3, 24]):

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


def run_feature_pipeline(latitude, longitude, past_days=10, forecast_days=0):
    """Orchestrates the full pipeline: fetch -> merge -> engineer -> target -> clean."""
    aqi_df = fetch_aqi_data(latitude, longitude, past_days=past_days, forecast_days=forecast_days)
    weather_df = fetch_weather_data(latitude, longitude, past_days=past_days, forecast_days=forecast_days)
    merged_df = merge_data(aqi_df, weather_df)
    features_df = engineer_features(merged_df)
    features_df = add_daily_targets(features_df)
    features_df = features_df.dropna().reset_index(drop=True)

    return features_df

if __name__ == "__main__":
    df = run_feature_pipeline(latitude=27.70, longitude=68.86, past_days=10)
    fs = get_feature_store()
    fg = get_or_create_aqi_feature_group(fs)
    insert_features(fg, df)
    print(f"Inserted {len(df)} rows into Hopsworks feature group 'aqi_features'.")