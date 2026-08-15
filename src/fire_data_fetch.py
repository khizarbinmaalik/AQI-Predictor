import os
import time
import io
import requests
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

MAP_KEY = os.getenv("FIRMS_MAP_KEY")
FIRMS_AREA = "67.4,26.2,70.4,29.2"   
FIRMS_SOURCE = "VIIRS_SNPP_SP"        
FIRMS_MAX_DAY_RANGE = 5               


def fetch_fire_chunk(start_date_str, day_range=FIRMS_MAX_DAY_RANGE):
    """Fetches raw fire detections for a date range starting at start_date_str.
    Returns an empty DataFrame (not an error) if no fires were detected —
    that's a valid, common result, not a failure."""
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{FIRMS_SOURCE}/{FIRMS_AREA}/{day_range}/{start_date_str}"
    resp = requests.get(url, timeout=15)

    if resp.status_code != 200:
        raise Exception(f"FIRMS API request failed: {resp.status_code} - {resp.text}")

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["acq_date", "frp", "type"])

    return df


def backfill_fire_data(days_back=730, day_range=FIRMS_MAX_DAY_RANGE, sleep_seconds=0.5):

    end = date.today()
    start = end - timedelta(days=days_back)

    all_chunks = []
    current = start
    while current < end:
        chunk_str = current.isoformat()
        print(f"Fetching fire data: {chunk_str} (+{day_range} days)...")

        chunk_df = fetch_fire_chunk(chunk_str, day_range)
        if len(chunk_df) > 0:
            all_chunks.append(chunk_df)

        current += timedelta(days=day_range)
        time.sleep(sleep_seconds)

    if not all_chunks:
        return pd.DataFrame(columns=["acq_date", "frp", "type"])

    return pd.concat(all_chunks, ignore_index=True)


def aggregate_daily_fire_features(fire_df, vegetation_only=True):
    """Aggregates raw fire detections into one row per date."""
    df = fire_df.copy()
    if vegetation_only:
        df = df[df["type"] == 0]   

    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date

    daily = df.groupby("acq_date").agg(
        fire_count=("frp", "count"),
        fire_frp_sum=("frp", "sum"),
    ).reset_index()

    return daily.rename(columns={"acq_date": "date"})

def fetch_recent_fire_data(days_back=10):
    end = date.today()
    start = end - timedelta(days=days_back)

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{FIRMS_SOURCE_NRT}/{FIRMS_AREA}/{days_back}/{start.isoformat()}"
    resp = requests.get(url, timeout=15)

    if resp.status_code != 200:
        raise Exception(f"FIRMS NRT API request failed: {resp.status_code} - {resp.text}")

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["acq_date", "frp", "type"])

    return df

if __name__ == "__main__":
    fire_df = backfill_fire_data(days_back=730)
    fire_df.to_csv("fire_data_raw.csv", index=False)