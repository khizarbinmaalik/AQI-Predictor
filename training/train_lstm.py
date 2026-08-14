import pandas as pd
import numpy as np
from tensorflow import keras # type:ignore
from tensorflow.keras import layers # type:ignore
from tensorflow.keras.callbacks import EarlyStopping  # type:ignoreing

from src.lstm_preprocessing import split_scale_and_sequence
from src.evaluation import evaluate_predictions 
from src.model_utils import get_feature_columns
from src.config import TEST_SIZE

WINDOW_SIZE = 48
TARGET_COLS = ["target_aqi_day1", "target_aqi_day2", "target_aqi_day3"]


def build_lstm_model(input_shape, n_outputs=3):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(n_outputs),  # linear output — regression, no activation
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_lstm(X_train, y_train, epochs=100, batch_size=64):
    val_split_idx = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
    y_tr, y_val = y_train[:val_split_idx], y_train[val_split_idx:]

    model = build_lstm_model(input_shape=X_train.shape[1:], n_outputs=y_train.shape[1])

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )
    return model, history


def evaluate_lstm(model, X_test, y_test):
    """y_test has shape (n_samples, 3) — evaluate each horizon column separately,
    reusing the exact same evaluate_predictions() every other model has used."""
    y_pred = model.predict(X_test)
    results = {}
    for i, horizon in enumerate([1, 2, 3]):
        results[f"day{horizon}"] = evaluate_predictions(
            y_test[:, i], y_pred[:, i], label=f"LSTM (Day {horizon})"
        )
    return results


if __name__ == "__main__":
    features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])
    feature_cols = get_feature_columns(features_df)

    X_train, X_test, y_train, y_test, scaler = split_scale_and_sequence(
        features_df, feature_cols, TARGET_COLS, window_size=WINDOW_SIZE, test_size=TEST_SIZE
    )

    print("X_train dtype:", X_train.dtype)
    print("y_train dtype:", y_train.dtype)
    print(features_df[feature_cols].dtypes)

    model, history = train_lstm(X_train, y_train)
    lstm_results = evaluate_lstm(model, X_test, y_test)