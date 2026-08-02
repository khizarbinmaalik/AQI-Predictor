import hopsworks
from src.config import HOPSWORKS_API_KEY, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION


def get_feature_store():
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    return fs

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