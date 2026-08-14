import numpy as np 
from sklearn.preprocessing import StandardScaler
import pandas as pd

from src.model_utils import get_feature_columns

def create_sequences(df, feature_cols, target_cols, window_size=48):
    """
    Converts a flat (rows, features) table into LSTM-ready 3D sequences.

    For each valid position i, X contains the window_size rows BEFORE and
    including row i (as an ordered sequence), and y contains that row's
    target values. Returns numpy arrays, not DataFrames — Keras expects arrays.
    """
    df = df.sort_values("time").reset_index(drop=True)

    X, y = [], []
    for i in range(window_size, len(df)):
        window = df.iloc[i - window_size:i][feature_cols].values  # shape: (window_size, n_features)
        target = df.iloc[i][target_cols].values                    # shape: (n_targets,)
        X.append(window)
        y.append(target)

    return np.array(X), np.array(y, dtype=np.float64)

def split_scale_and_sequence(df, feature_cols, target_cols, window_size=48, test_size=0.2):
    df = df.sort_values("time").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))

    scaler = StandardScaler()
    scaler.fit(df.loc[:split_idx - 1, feature_cols])

    scaled_df = df.copy()
    scaled_df[feature_cols] = scaler.transform(df[feature_cols])

    X, y = create_sequences(scaled_df, feature_cols, target_cols, window_size)

    # Each sequence at position j corresponds to original row (window_size + j).
    # A sequence belongs to test if that original row index falls at/after split_idx.
    seq_split_idx = split_idx - window_size

    X_train, X_test = X[:seq_split_idx], X[seq_split_idx:]
    y_train, y_test = y[:seq_split_idx], y[seq_split_idx:]

    return X_train, X_test, y_train, y_test, scaler


features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])

target_cols = ["target_aqi_day1", "target_aqi_day2", "target_aqi_day3"]

feature_cols = get_feature_columns(features_df)  # reuse your existing function
X_train, X_test, y_train, y_test, scaler = split_scale_and_sequence(
    features_df, feature_cols, target_cols, window_size=48, test_size=0.2
)

print("X_train:", X_train.shape, " X_test:", X_test.shape)
print("y_train:", y_train.shape, " y_test:", y_test.shape)