from  datetime import date, timedelta
from src.fetch_data import fetch_aqi_data, fetch_weather_archive_data 
from src.feature_engineering import merge_data, engineer_features, add_daily_targets,merge_fire_features
from src.hopsworks_setup import get_feature_store, get_or_create_aqi_feature_group, insert_features
from src.fire_data_fetch import aggregate_daily_fire_features 
import pandas as pd

def generate_date_chunks(days_back, chunk_size=90, lag_buffer_days=6):
    # Splits a date range into non-overlapping (start, end) chunks.

    end = date.today() - timedelta(days=lag_buffer_days)  # stay clear of ERA5's reporting lag
    start = end - timedelta(days=days_back)

    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_size), end)
        chunks.append((current.isoformat(), chunk_end.isoformat()))
        current = chunk_end + timedelta(days=1)
    return chunks


def run_year_backfill(latitude, longitude, days_back=365, chunk_size=90):
    chunks = generate_date_chunks(days_back, chunk_size)
    all_rows = []

    for start_date, end_date in chunks:
        print(f"Fetching {start_date} to {end_date}...")
        aqi_df = fetch_aqi_data(latitude, longitude, start_date=start_date, end_date=end_date)
        weather_df = fetch_weather_archive_data(latitude, longitude, start_date=start_date, end_date=end_date)
        merged_df = merge_data(aqi_df, weather_df)
        all_rows.append(merged_df)

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df = full_df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)

    # Feature engineering runs ONCE on the full concatenated year, not per chunk
    features_df = engineer_features(full_df)

    print("Merging cached fire data (fire_data_raw.csv)...")
    fire_df = pd.read_csv("fire_data_raw.csv")
    daily_fire_df = aggregate_daily_fire_features(fire_df)
    features_df = merge_fire_features(features_df, daily_fire_df)

    features_df = add_daily_targets(features_df)
    features_df = features_df.dropna().reset_index(drop=True)

    fs = get_feature_store()
    fg = get_or_create_aqi_feature_group(fs)
    insert_features(fg, features_df)

    print(f"Backfilled {len(features_df)} rows spanning {days_back} days into 'aqi_features'.")
    return features_df

if __name__ == "__main__":
    run_year_backfill(latitude=27.70, longitude=68.86, days_back=730, chunk_size=90)