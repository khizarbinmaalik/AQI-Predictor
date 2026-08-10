# Experiments

Exploratory model/feature variants that did not outperform the production
baseline. Kept for reproducibility and reference — not used by the live
pipeline. See docs/EDA_Findings.md for full results and analysis.

- round2_feature_engineering.py — cyclical encoding, rolling AQI stats,
  season feature, feature pruning (tested in `train_v2.py`)
- train_v2.py — training script that tested the above combinations