import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()

def get_feature_store():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    return fs

def get_or_create_aqi_feature_group(fs):
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="Hourly AQI, weather, and engineered features for AQI forecasting",
        primary_key=["time"],
        event_time="time",
        online_enabled=True,
    )
    return fg

def insert_features(fg, df):
    fg.insert(df)
