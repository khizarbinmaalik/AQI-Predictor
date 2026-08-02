# AQI Predictor — EDA Findings Log

Running notes from the exploratory data analysis phase, to be used directly in the final project report. Each entry includes the finding, the evidence, and the mechanism behind it.

---

## Finding 1 — Strong, repeating annual seasonal cycle in AQI

**What we found:** Daily average AQI shows a clear, repeating cycle across both years of data:
- **Trough (~60–100 AQI):** July–September (monsoon season)
- **Peak (~150–210 AQI):** October–January, peaking around December–January

**Mechanism:** Monsoon rain washes pollutants from the air and improves dispersion. Post-monsoon/winter combines post-harvest crop residue burning with temperature inversions (cold, dense surface air trapping pollutants that would normally rise and disperse), producing the year's worst air quality.

**Implication for modeling:** The `month` feature carries strong, physically-grounded predictive signal — this isn't a coincidental correlation, it's a well-understood seasonal mechanism. Confirmed consistent across both years in the dataset (not a one-off).

**Caveat noted:** The most recent partial cycle (mid-2026) is incomplete — data ends July 27, so no claims should be made about how that specific monsoon season compares to prior years.

---

## Finding 2 — `us_aqi` shows almost no hour-of-day pattern (and why)

**What we found:** Mean AQI by hour of day is nearly flat (~120–122 across all 24 hours) with a very large standard deviation band (±30+), swamping any hourly signal.

**Mechanism:** The US AQI is not computed from instantaneous pollutant readings — EPA methodology (which Open-Meteo follows) computes PM2.5's AQI sub-index from a **24-hour rolling average** (or 12-hour NowCast-weighted average), and ozone's sub-index from an **8-hour rolling average**. This smoothing is baked into the AQI value itself before it reaches the dataset, which erases most within-day variation. The large std band reflects day-to-day/seasonal variation (Finding 1), not hour-to-hour noise.

**Implication for modeling:** Don't discard the `hour` feature based on this alone — check actual feature importance during training (Phase 6) rather than assuming irrelevance from a marginal plot. This finding is really about the *target variable's construction*, not about whether time-of-day matters to pollution.

---

## Finding 3 — Raw pollutant concentrations reveal a real diurnal cycle, driven by boundary-layer dynamics and photochemistry (not simple traffic timing)

**What we found:** Unlike smoothed `us_aqi`, raw pollutant concentrations (PM2.5, PM10, NO2, ozone) show a strong, consistent daily cycle:
- **NO2, PM2.5, PM10:** high overnight, decline through morning, bottom out midday (11am–4pm), sharp rise from ~5–9pm
- **Ozone:** inverse pattern — low overnight, peaks in early-mid afternoon (~12–3pm)

**Mechanism (two compounding effects):**
1. **NOx–ozone photochemical cycle:** sunlight breaks down NO2 into NO + free oxygen, which reacts to form ozone — so NO2 is chemically converted into ozone during daylight hours, explaining why the two pollutants are almost perfect mirror images of each other.
2. **Atmospheric boundary layer height:** at night, a shallow, stable air layer traps pollutants near the surface (same inversion mechanism as Finding 1, operating daily instead of seasonally). Daytime solar heating causes convective mixing, expanding this layer and diluting near-surface pollutant concentrations — independent of actual emission levels.

**Note on the original hypothesis:** the initial expectation was a simple traffic-driven double rush-hour bump. The actual data shows a more physically complete story — the evening rise (5–9pm) likely combines both boundary-layer collapse *and* rush-hour traffic reinforcing each other, while the morning traffic signal is likely present but masked by the much stronger effect of the boundary layer still expanding at that time.

**Implication for modeling:** Raw pollutant features (not just `us_aqi`) carry real, complementary diurnal information — worth retaining through training rather than assuming `us_aqi` alone captures everything.

---

## Finding 4 — No meaningful weekday/weekend pattern in `us_aqi` (confirms Finding 2, doesn't extend it)

**What we found:** Mean AQI by day of week is flat (~120–123) across all seven days, well within a wide std band — no visible weekday/weekend separation.

**Mechanism:** This is not a new pattern, but the expected consequence of Finding 2 — `us_aqi`'s 8–24 hour rolling-average construction smooths out shorter-term variation (whether hourly or weekday-driven), so any real weekday/weekend emissions difference (e.g. less industrial or traffic activity on weekends) doesn't survive into the AQI value itself.

**Note on method:** this result was explicitly predicted in advance based on Finding 2's mechanism, then confirmed — a hypothesis-driven check rather than an open-ended one. Worth stating as such in the report rather than presenting it as a standalone discovery.

**Implication for modeling:** `day_of_week`/`is_weekend` features are unlikely to carry strong standalone signal for predicting `us_aqi` directly, though — same caveat as `hour` — final judgment should come from feature importance in Phase 6, not from this plot alone.

---

## Finding 5 — No weekly emissions cycle at all, confirmed at the raw pollutant level (not just an AQI-smoothing artifact)

**What we found:** Unlike the hour-of-day case (Finding 3), where raw pollutants revealed a real cycle hidden by `us_aqi`'s smoothing, raw pollutant concentrations (PM2.5, PM10, NO2, ozone) are **also flat across day of week** — no weekday/weekend difference at the source level.

**Why this is a distinct finding, not a repeat of Finding 4:** Finding 4 only showed that *smoothed AQI* had no weekly pattern, which was ambiguous — it could have meant either "no real weekly cycle" or "a real cycle exists but got smoothed away," analogous to what happened with hour-of-day. Checking raw concentrations directly resolves that ambiguity: there is genuinely no weekly emissions cycle here, at any level of the data.

**Mechanism:** A weekday/weekend emissions dip is typical of cities with a strong Mon–Fri office-commute culture. In this region, transport, informal markets, and industrial activity largely operate on a continuous 7-day schedule rather than dropping off on weekends, so there's no structural reason to expect (or find) a weekly cycle.

**Implication for modeling:** `day_of_week` and `is_weekend` are unlikely to be meaningful predictors for this location — confirmed at both the AQI and raw-pollutant level, not just assumed. Season (Finding 1) and daily boundary-layer/photochemical cycles (Finding 3) are the dominant, confirmed drivers of variation; weekly human-activity rhythm is not a factor here.

---

## Finding 6 — Correlation matrix confirms and refines earlier findings; PM2.5 identified as the dominant pollutant driver

**What we found:**
- **Lag features are the strongest predictors of `us_aqi`** (`aqi_lag_1h`: ~0.99–1.00, `aqi_lag_3h`: 0.99, `aqi_lag_24h`: 0.82) — expected given AQI's hour-to-hour persistence, and good news for forecasting.
- **PM2.5 is the dominant raw pollutant driver** of `us_aqi` (r=0.69), clearly stronger than PM10 (0.29), CO (0.56), or SO2 (0.57).
- **NO2 correlates only moderately with `us_aqi`** (0.33) despite having a strong diurnal cycle (Finding 3) — NO2 has real hourly structure but is rarely the pollutant that dominates the overall AQI calculation; PM2.5 usually is.
- **Weather correlations cross-confirm Finding 1's seasonal/inversion mechanism**, not as new information but as three independent variables agreeing with it: temperature (-0.49, colder = higher AQI), surface pressure (+0.53, stable high-pressure systems trap pollution), wind speed (-0.35, confirms dispersion mechanism).
- **Ozone (-0.04) and precipitation (-0.06) show near-zero linear correlation with `us_aqi`**, despite both having clear mechanistic relevance (Finding 3 for ozone, Finding 1 for precipitation/monsoon). Likely explanations: ozone peaks when PM is typically low (different timing, canceling out linearly); precipitation is heavily skewed (mostly zero, occasional bursts), which Pearson correlation handles poorly. Treated as a correlation-method limitation, not evidence these variables don't matter.
- **Multicollinearity noted** among the three lag features (0.82–0.99 with each other) and within two source-driven pairs (CO/NO2: 0.71; temperature/surface pressure: -0.79) — relevant for Ridge regression interpretability in Phase 6, less of a concern for Random Forest.

**Implication for modeling:** Confirms lag features and PM2.5 as likely top predictors; supports keeping ozone/precipitation/humidity in the feature set despite weak linear correlation, since their relevance is mechanistic and possibly non-linear — final call deferred to feature importance analysis in Phase 6, consistent with the project's stated pruning-timing principle.

---

## Finding 7 — No missing data across the full 2-year span

**What we found:** Zero null values in any column, and zero missing hourly timestamps across the entire dataset (17,544 expected hours = 17,544 actual hours present).

**Why this matters:** Confirms the chunked backfill (Phase 4) executed cleanly with no dropped requests or silent API failures, and that the archive-to-live source boundary (July 21/22, 2026) introduced no gaps despite coming from two different endpoints. A clean, complete dataset with no imputation or gap-filling required going into model training.

---

## EDA Summary (Phase 5 complete)

Seven findings total, established through hypothesis-driven checks (predict → verify → explain mechanism) rather than open-ended pattern-hunting:

1. Strong repeating annual seasonal cycle (monsoon trough, winter peak) — driven by rain/dispersion vs. crop burning + temperature inversions
2. `us_aqi` shows no hour-of-day pattern — an artifact of its 8–24h rolling-average construction, not evidence pollution lacks a daily rhythm
3. Raw pollutants reveal a real diurnal cycle — driven by boundary-layer height (traps pollutants overnight) and NOx–ozone photochemistry (mirror-image relationship confirmed)
4. `us_aqi` shows no weekday/weekend pattern — consistent with Finding 2's smoothing explanation
5. Raw pollutants also show no weekday/weekend pattern — confirms Finding 4 is a real absence of weekly cycle, not just smoothing; consistent with a 7-day-active local economy rather than a Mon–Fri commute culture
6. Correlation matrix — lag features and PM2.5 are the strongest linear predictors of `us_aqi`; weather correlations (temperature, pressure, wind) all cross-confirm Finding 1's seasonal mechanism; ozone/precipitation show weak linear correlation despite mechanistic relevance, likely due to non-linearity/skew rather than true irrelevance
7. No missing data — complete, gap-free 2-year hourly dataset (17,544/17,544 hours), confirming a clean backfill with no source-boundary artifacts

**Feature retention decision:** per the project's stated pruning-timing principle, all engineered features (including those with weak marginal/linear signal — `day_of_week`, `is_weekend`, `ozone`, `precipitation`) are retained through training. Final pruning will be based on feature importance (Random Forest / SHAP) in Phase 6, not on EDA correlation alone, since several findings above show mechanistically relevant variables can still show weak linear correlation.
