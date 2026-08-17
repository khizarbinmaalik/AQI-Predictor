import shap
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation import time_based_split
from src.config import TEST_SIZE
from training.train_xgboost_delta import add_delta_targets, get_delta_feature_columns, train_xgboost_delta_models


def compute_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def plot_summary(shap_values, X, horizon):
    """Global view: which features matter most, AND in which direction."""
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Summary — Day {horizon} (delta prediction)")
    plt.tight_layout()
    plt.savefig(f"shap_summary_day{horizon}.png", dpi=150)
    plt.close()
    print(f"Saved shap_summary_day{horizon}.png")


def explain_single_prediction(model, explainer, shap_values, X, test_df, row_idx, horizon):

    current_aqi = test_df["us_aqi"].iloc[row_idx]
    predicted_delta = shap_values[row_idx].sum() + explainer.expected_value
    predicted_aqi = current_aqi + predicted_delta

    print(f"\n--- Day {horizon} forecast breakdown for one test row ---")
    print(f"Current AQI (starting point):     {current_aqi:.1f}")
    print(f"Model's baseline delta:            {explainer.expected_value:+.2f}")
    print(f"Sum of feature contributions:      {shap_values[row_idx].sum()}")
    print(f"Predicted delta (total):           {predicted_delta:+.2f}")
    print(f"=> Reconstructed AQI forecast:     {predicted_aqi:.1f}")

    single_explanation = shap.Explanation(
            values=shap_values[row_idx],
            base_values=explainer.expected_value,
            data=X.iloc[row_idx].values,
            feature_names=X.columns.tolist(),
        )
    shap.plots.waterfall(single_explanation, show=False)
    plt.title(f"Day {horizon} — why this prediction (delta scale)")
    plt.tight_layout()
    plt.savefig(f"shap_waterfall_day{horizon}_row{row_idx}.png", dpi=150)
    plt.close()
    print(f"Saved shap_waterfall_day{horizon}_row{row_idx}.png")


if __name__ == "__main__":
    features_df = pd.read_csv("aqi_features.csv", parse_dates=["time"])
    features_df = add_delta_targets(features_df)
    train_df, test_df = time_based_split(features_df, test_size=TEST_SIZE)

    models, results, importances = train_xgboost_delta_models(train_df, test_df)
    feature_cols = get_delta_feature_columns(train_df)
    X_test = test_df[feature_cols]

    for horizon in [1, 2, 3]:
        model = models[f"day{horizon}"]
        explainer, shap_values = compute_shap_values(model, X_test)

        plot_summary(shap_values, X_test, horizon)
        explain_single_prediction(model, explainer, shap_values, X_test, test_df, row_idx=0, horizon=horizon)