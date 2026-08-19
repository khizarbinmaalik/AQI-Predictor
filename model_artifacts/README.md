# AQI XGBoost Delta Model

## IMPORTANT: predicts a DELTA, not the absolute AQI value.

Each model file predicts: target_delta_dayH = target_aqi_dayH - us_aqi

To get an actual forecast:
    predicted_aqi = current_us_aqi + model.predict(X)

Skipping this step produces meaningless small numbers near zero, not AQI.

## Files
- model_day1.json / model_day2.json / model_day3.json — one XGBoost
  model per horizon. Load via XGBRegressor().load_model(path).
- feature_columns.json — exact ordered feature list. Input data MUST
  match this column set and order.

## Test-set performance (reconstructed, absolute AQI scale)
- Day 1: RMSE=14.98, R²=0.579
- Day 2: RMSE=19.87, R²=0.250
- Day 3: RMSE=22.67, R²=0.028

Full methodology: docs/EDA_Findings.md
