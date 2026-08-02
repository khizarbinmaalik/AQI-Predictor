import requests
import pandas as pd
from src.config import AIR_QUALITY_API_URL, AIR_QUALITY_HOURLY_VARS
# Using the Start and end date parameters to fetch historical data from the API.
def fetch_aqi_data(latitude, longitude, timezone="auto", past_days=None, forecast_days=None, start_date=None, end_date=None):

    url = AIR_QUALITY_API_URL
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": AIR_QUALITY_HOURLY_VARS,
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

def fetch_weather_archive_data(latitude, longitude, start_date, end_date, timezone="auto"):
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation",
        "timezone": timezone,
        "start_date": start_date,
        "end_date": end_date,
    }
    response = requests.get(url, params=params, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Weather archive API request failed: {response.status_code} - {response.text}")
    data = response.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
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