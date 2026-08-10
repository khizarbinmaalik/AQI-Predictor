"""
Central configuration for the Pearls AQI Predictor project.
Import constants from here instead of hardcoding them in individual scripts —
changing a value (city, feature group version, split ratio, etc.) should only
ever require editing this one file, not hunting through every script.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Location ---
LATITUDE = 27.70
LONGITUDE = 68.86
TIMEZONE = "auto"

# --- Open-Meteo API endpoints ---
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"

AIR_QUALITY_HOURLY_VARS = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
WEATHER_HOURLY_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation"

# --- Feature engineering ---
LAG_HOURS = [1, 3, 24]
TARGET_HORIZONS_DAYS = [1, 2, 3]

# --- Backfill ---
BACKFILL_DAYS_BACK = 730           # ~2 years
BACKFILL_CHUNK_SIZE = 90           # days per API call, per chunk
BACKFILL_LAG_BUFFER_DAYS = 6       # stay clear of ERA5's reporting lag

# --- Hopsworks ---
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_GROUP_PRIMARY_KEY = ["time"]
FEATURE_GROUP_EVENT_TIME = "time"

# --- Training ---
TEST_SIZE = 0.2
RANDOM_STATE = 42        # for reproducibility across Ridge/RF/LSTM runs
TARGET_COLS = ["target_aqi_day1", "target_aqi_day2", "target_aqi_day3"]
DROP_COLS = ["time", "latitude", "longitude"] + TARGET_COLS

# Confirmed weak/negligible predictors from Phase 6 feature importance analysis
PRUNE_COLS = ["day_of_week", "ozone", "nitrogen_dioxide", "precipitation"]