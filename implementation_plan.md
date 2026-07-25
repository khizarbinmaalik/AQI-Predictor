# Pearls AQI Predictor — End-to-End Build Roadmap

**Goal:** A 100% serverless system that predicts AQI for your city 3 days out, fully automated, with a live dashboard.

**Architecture recap (from the brief):**
```
Weather/Pollution API → Feature Pipeline → Feature Store
                                                  ↓
                                          Training Pipeline → Model Registry
                                                  ↓                  ↓
                                              Web App  ←――――――――――――
```
Everything is automated via CI/CD (GitHub Actions): feature pipeline runs hourly, training pipeline runs daily.

---

## Phase 0 — Setup & Decisions (Day 1)

Before writing any code, lock in these choices so you don't waffle mid-build:

| Decision | Recommendation | Why |
|---|---|---|
| City | Your own city (e.g. Sukkur/Karachi) | Real data you can sanity-check against what you see outside |
| Data API | **OpenWeather Air Pollution API** (free tier) + **OpenWeather Weather API** | Reliable free tier, gives both pollutants (PM2.5, PM10, NO2, O3, etc.) and weather (temp, humidity, wind) in one ecosystem. AQICN is also fine and often has more historical depth for some cities. |
| Feature Store | **Hopsworks (free tier)** | Purpose-built for this exact "feature store + model registry" pattern, has a generous free serverless tier, and has an official AQI prediction tutorial you can reference structurally (don't copy — build your own) |
| Web app | **Streamlit** | Fastest way to get an interactive dashboard live, minimal boilerplate vs Flask+React |
| Automation | **GitHub Actions** | Free, no infra to manage, cron-based scheduling is trivial to set up |

**Action items:**
- [ ] Sign up for OpenWeather API key
- [ ] Sign up for Hopsworks free account, create a project
- [ ] Create a GitHub repo for this project
- [ ] Set up a Python virtual environment

---

## Phase 1 — Raw Data Exploration (Day 1–2)

Before automating anything, manually pull data once and *look* at it.

```python
import requests
resp = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={key}")
```

- [ ] Understand the raw JSON structure (pollutants, timestamps, units)
- [ ] Pull the **historical** air pollution endpoint too (you'll need this for backfilling)
- [ ] Note: OpenWeather gives raw pollutant concentrations, not "AQI" directly in some regions — you may need to **compute AQI yourself** from PM2.5/PM10 using the EPA or India CPCB breakpoint formula. Understand this formula before coding it — it's a piecewise linear function, not a simple average.

**This step matters:** don't skip straight to feature engineering with data you haven't visually inspected. You've done this discipline before (EDA before modeling) — same principle here.

---

## Phase 2 — Feature Pipeline Script (Day 2–4)

Write `feature_pipeline.py`. This script's job: raw API data → clean feature row(s).

**What it must do:**
1. Fetch current weather + pollution data
2. Compute the target variable (AQI) if not directly provided
3. Engineer features:
   - Time-based: hour, day of week, month, is_weekend
   - Lag features: AQI 1hr ago, 3hr ago, 24hr ago (this is what makes it a *forecasting* problem, not just regression)
   - Rate of change: AQI trend over last few readings
   - Weather features: temp, humidity, wind speed, pressure
4. Return a clean row (or DataFrame) ready to insert into the feature store

**Concept note relevant to you:** this is now a **time series** problem layered on top of regression. You already know OLS/Ridge/Lasso — the new piece is that your "X" now includes *past values of y* (lag features). This is the bridge between regression and time series forecasting. Worth 20 minutes of reading on "lag features for time series forecasting" before you code this.

- [ ] Write and test the script locally on live data
- [ ] Confirm output row shape is consistent every run

---

## Phase 3 — Hopsworks Feature Store Integration (Day 4–5)

- [ ] Create a **Feature Group** in Hopsworks (this is like a table schema: city, timestamp, features, target)
- [ ] Modify `feature_pipeline.py` to `insert()` rows into this feature group instead of just printing them
- [ ] Run it manually a few times, confirm rows appear in the Hopsworks UI

---

## Phase 4 — Backfill Historical Data (Day 5–7)

This is what turns your feature store from "a few rows" into "enough training data."

- [ ] Write `backfill.py` that loops over a date range (aim for 2–3+ months if the historical API allows it) and calls your feature generation logic for each past timestamp
- [ ] Insert all of this into the same Hopsworks feature group
- [ ] **Reality check:** free-tier historical APIs are often limited. If you can't get months of true history, note this constraint in your final report and consider running the hourly pipeline for several weeks to accumulate real data while you build the rest of the system in parallel — don't block on this.

---

## Phase 5 — EDA on the Feature Store Data (Day 7–8)

Pull the accumulated data back out and actually look at it before training:

- [ ] AQI trend over time (line plot)
- [ ] AQI distribution by hour of day / day of week (does rush hour show up?)
- [ ] Correlation between pollutants and weather variables (does wind speed suppress AQI? does humidity correlate with PM2.5?)
- [ ] Missing data / gaps check

This is the same instinct you applied before jumping into Ridge/Lasso — feel the data before modeling it.

---

## Phase 6 — Training Pipeline (Day 8–12)

Write `training_pipeline.py`. Fetch (features, target) from Hopsworks → train → evaluate → save.

**Model progression — build in this order, don't jump to the end:**

1. **Baseline**: predict "tomorrow's AQI = today's AQI" (naive persistence baseline). Always build this first — if your real model can't beat this, something's wrong.
2. **Ridge/Lasso Regression** — you already know this cold. Gets you a real first model fast.
3. **Random Forest** — sklearn, minimal new concept overhead, usually a strong tabular baseline.
4. **LSTM (TensorFlow/Keras)** — this is new for you, but you already have Keras experience from the MobileNetV2 project. An LSTM here is well-suited since you're forecasting a sequence. Expect a real learning curve here — treat it as its own mini-project with intuition → math (recurrence, why LSTM > vanilla RNN) → implementation, same sequence you use for everything else.

**Evaluation:**
- [ ] RMSE, MAE, R² on a **held-out time-based split** (not random shuffle — train on earlier dates, test on later dates, since shuffling leaks future info into training, which is a classic time series mistake)
- [ ] Compare all models against the naive baseline in a single table

**Model Registry:**
- [ ] Save the best model to Hopsworks Model Registry along with its evaluation metrics

---

## Phase 7 — Automate with GitHub Actions (Day 12–14)

- [ ] Write `.github/workflows/feature_pipeline.yml` — cron schedule, runs hourly, calls `feature_pipeline.py`
- [ ] Write `.github/workflows/training_pipeline.yml` — cron schedule, runs daily, calls `training_pipeline.py`
- [ ] Store your API keys and Hopsworks credentials as **GitHub Secrets** (never hardcode them)
- [ ] Let it run for a day or two, confirm it's actually executing on schedule and writing to Hopsworks

---

## Phase 8 — Streamlit Web App (Day 14–17)

- [ ] Load the latest model from the Model Registry and latest features from the Feature Store
- [ ] Compute a 3-day forecast (this likely means recursive prediction: predict hour+1, feed it back in as a lag feature to predict hour+2, etc. — think through this loop carefully, it's a common source of bugs)
- [ ] Build the dashboard:
  - Current AQI + category (Good/Moderate/Unhealthy etc.)
  - 3-day forecast chart
  - Historical trend chart
- [ ] Deploy on Streamlit Community Cloud (free) so it's actually live, not just local

---

## Phase 9 — Explainability + Alerts (Day 17–19)

- [ ] Add **SHAP** to your training pipeline — show which features (e.g. lag AQI, wind speed) drive predictions. Start with `TreeExplainer` if you're using Random Forest, it's the simplest entry point into SHAP.
- [ ] Add a simple threshold-based alert on the dashboard: if predicted AQI crosses into "Unhealthy," show a visible warning banner

---

## Phase 10 — Final Report (Day 19–21)

Document, per the brief's submission requirements:
- [ ] System architecture (reuse/adapt the diagram from the brief)
- [ ] EDA findings
- [ ] Model comparison table (baseline vs Ridge vs RF vs LSTM)
- [ ] What worked, what didn't, and why (e.g. data limitations, model tradeoffs)
- [ ] Screenshots of the live dashboard

---

## Suggested Pace

This is written as a ~3-week plan assuming consistent daily effort. If you're balancing this with your ongoing ML curriculum (Decision Trees, etc.), stretch it to 4–5 weeks and treat Phase 6's LSTM step as a natural "next new concept" to slot in alongside your regular learning — you're already doing regression → classification → next up would be trees, and LSTM is a good parallel deep-learning branch to open once trees feel solid.

## Order-of-operations reminder

Feature pipeline → feature store → backfill → EDA → training → registry → automation → web app → explainability. Resist the urge to build the Streamlit app early just because it's the "fun visible part" — without real accumulated data in the feature store, you'll be dashboarding on noise.
