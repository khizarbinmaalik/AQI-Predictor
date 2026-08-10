import numpy as np

def add_advanced_features(df, rolling_windows=[6, 12]):

    df = df.sort_values("time").reset_index(drop=True)

    # --- Cyclical encoding: hour and month wrap around, plain integers don't ---
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Rolling AQI trend/volatility (not just single-point lags) ---
    for window in rolling_windows:
        df[f"aqi_roll_mean_{window}h"] = df["us_aqi"].rolling(window).mean()
        df[f"aqi_roll_std_{window}h"] = df["us_aqi"].rolling(window).std()

    # --- Weather trend: is pressure rising or falling? (inversion early-warning) ---
    df["pressure_change_24h"] = df["surface_pressure"].diff(24)

    return df


def ensemble_predict(xgb_model, rf_model, X_test, xgb_weight=0.5):
    xgb_pred = xgb_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    return xgb_weight * xgb_pred + (1 - xgb_weight) * rf_pred


def evaluate_ensemble(xgb_models, rf_models, test_df, feature_cols, xgb_weight=0.5):
    from src.evaluation import evaluate_predictions

    X_test = test_df[feature_cols]
    results = {}
    for horizon in [1, 2, 3]:
        key = f"day{horizon}"
        y_test = test_df[f"target_aqi_{key}"]
        y_pred = ensemble_predict(xgb_models[key], rf_models[key], X_test, xgb_weight)
        results[key] = evaluate_predictions(y_test, y_pred, label=f"Ensemble ({key})")
    return results