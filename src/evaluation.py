from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def time_based_split(df, test_size=0.2):
    df = df.sort_values("time").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df

def evaluate_predictions(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label}: RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

