from fire_data_fetch import aggregate_daily_fire_features, fetch_recent_fire_data
from src.hopsworks_setup import get_feature_store, get_or_create_aqi_feature_group, insert_features 
from src.fetch_data import fetch_aqi_data, fetch_weather_data
from src.feature_engineering import merge_data, engineer_features, add_daily_targets, merge_fire_features

def run_feature_pipeline(latitude, longitude, past_days=10, forecast_days=0):
    """Orchestrates the full pipeline: fetch -> merge -> engineer -> target -> clean."""
    aqi_df = fetch_aqi_data(latitude, longitude, past_days=past_days, forecast_days=forecast_days)
    weather_df = fetch_weather_data(latitude, longitude, past_days=past_days, forecast_days=forecast_days)
    merged_df = merge_data(aqi_df, weather_df)
    features_df = engineer_features(merged_df)

    fire_df = fetch_recent_fire_data(days_back=max(past_days, 10))
    daily_fire_df = aggregate_daily_fire_features(fire_df)
    features_df = merge_fire_features(features_df, daily_fire_df)
    
    features_df = add_daily_targets(features_df)
    features_df = features_df.dropna().reset_index(drop=True)

    return features_df

if __name__ == "__main__":
    df = run_feature_pipeline(latitude=27.70, longitude=68.86, past_days=10)
    fs = get_feature_store()
    fg = get_or_create_aqi_feature_group(fs)
    insert_features(fg, df)
    print(f"Inserted {len(df)} rows into Hopsworks feature group 'aqi_features'.")