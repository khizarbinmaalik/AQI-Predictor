import pandas as pd
from src.evaluation import time_based_split, evaluate_predictions
from src.config import TEST_SIZE

def naive_baseline(test_df):
    """
    Predicts each future day's AQI as simply today's current AQI (persistence).
    Same predicted value used for day1/day2/day3 
    """
    results = {}
    for horizon in [1, 2, 3]:
        y_true = test_df[f"target_aqi_day{horizon}"]
        y_pred = test_df["us_aqi"]
        results[f"day{horizon}"] = evaluate_predictions(y_true, y_pred, label=f"Naive Baseline (Day {horizon})")
    return results

features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])
train_df, test_df = time_based_split(features_df, test_size=TEST_SIZE)
baseline_results = naive_baseline(test_df)