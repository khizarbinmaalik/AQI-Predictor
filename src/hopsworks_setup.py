import hopsworks
import pandas as pd
from src.config import HOPSWORKS_API_KEY, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

LIVE_FETCH_MAX_DAYS = 90   # safe margin under Open-Meteo's confirmed 93-day live limit

def get_feature_store():
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    return fs



def get_last_feature_store_timestamp(fg):
    """Returns the most recent 'time' in the feature group, or None if
    genuinely empty. Real errors (network/service issues) are NOT caught
    here — they should be handled distinctly by the caller, not silently
    treated as 'empty'."""
    result = fg.read()
    if len(result) == 0:
        return None
    return pd.to_datetime(result["time"]).max()


def calculate_required_past_days(fg, min_days=10, buffer_days=3, max_days=LIVE_FETCH_MAX_DAYS):
    try:
        last_timestamp = get_last_feature_store_timestamp(fg)
    except Exception as e:
        print(f"WARNING: could not check feature store for gaps ({e}). Falling back to {min_days} days.")
        return min_days

    if last_timestamp is None:
        print("WARNING: feature store appears empty. This pipeline is for incremental "
              "updates only — run pipelines.backfill_pipeline for a full historical fill.")
        return min_days

    gap_days = (pd.Timestamp.now() - last_timestamp).days
    required = max(min_days, gap_days + buffer_days)

    if required > max_days:
        print(f"WARNING: gap is {gap_days} days, exceeding the live pipeline's {max_days}-day "
              f"limit. Fetching {max_days} days now; run the historical backfill separately "
              f"to close the remaining gap.")
        return max_days

    return required

def get_or_create_aqi_feature_group(fs):
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI, weather, and engineered features for AQI forecasting",
        primary_key=["time"],
        event_time="time",
        online_enabled=True,
    )
    return fg

def insert_features(fg, df):
    fg.insert(df)


def read_all_features():
    fs = get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=1)
    df = fg.read()
    return df