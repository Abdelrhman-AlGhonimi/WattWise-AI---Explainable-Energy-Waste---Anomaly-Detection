# Model Card

## Model details

WattWise AI is an academic short-horizon regression and candidate-event triage system. The selected predictor in recorded run `55004ee9-e3d3-433c-9528-6e2227d2b7d8` is an equal-weight average of Tuned HistGradientBoosting, HistGradientBoosting and XGBoost. Ensemble membership was chosen by validation MAE. The feature schema contains 63 past-only/current-origin sensor, consumption-history, change and calendar features.

## Intended use and users

Intended for demonstrating appliance-use forecasting and helping a person review unusual residuals. Not intended for automated energy control, billing, safety decisions, fault diagnosis, proof of waste or savings, or consumer ranking. Forecasts and flags need human interpretation.

## Data

Official UCI Appliances Energy Prediction dataset. It covers a single low-energy home from January to May 2016 with 19,735 ten-minute observations and 29 columns. Target is appliance energy in Wh at the next observed row. The acquired CSV contains no missing values or duplicate timestamps. The source supplies no verified anomaly or waste labels.

Chronological partition: 70% train (13,713), 15% validation (2,938), 15% test (2,939). Preprocessing and derived features use only information at or before each forecast origin. Model selection and threshold calibration use validation. Test is held out and is not used to select a model.

## Model selection and metrics

The selection rule is minimum validation MAE; ties resolve by lower validation RMSE, then model name. Selected Voting validation results: MAE 21.939 Wh, RMSE 56.356 Wh and R² 0.610. Equal-weight components: Tuned HistGradientBoosting, HistGradientBoosting and XGBoost. Selected test results: MAE 22.385 Wh, RMSE 58.723 Wh and R² 0.583 on 2,939 targets. Persistence test MAE/RMSE: 26.723/66.786 Wh.

### All model results on the held-out test period

| Model | MAE (Wh) | RMSE (Wh) | R² | n |
|---|---:|---:|---:|---:|
| Tuned HistGradientBoosting | 22.268 | 58.872 | 0.581 | 2,939 |
| **Voting (top 3), selected** | **22.385** | **58.723** | **0.583** | **2,939** |
| HistGradientBoosting | 22.662 | 59.814 | 0.568 | 2,939 |
| XGBoost | 22.820 | 58.326 | 0.589 | 2,939 |
| Temporal Stacking | 23.722 | 57.941 | 0.594 | 2,939 |
| Extra Trees | 26.159 | 57.324 | 0.603 | 2,939 |
| HistGradientBoosting (squared loss) | 26.644 | 57.693 | 0.598 | 2,939 |
| Persistence | 26.723 | 66.786 | 0.461 | 2,939 |
| Ridge | 27.878 | 59.990 | 0.565 | 2,939 |
| Random Forest | 28.987 | 58.660 | 0.584 | 2,939 |
| Dummy mean | 52.354 | 90.977 | -0.000 | 2,939 |

The retrospective test MAE leader was Tuned HistGradientBoosting, and Extra Trees had the lowest RMSE/highest R². These are descriptive comparisons after the model was selected; neither changes selection. Regression **accuracy** is undefined. MAE/RMSE are Wh per ten-minute interval; R² is unitless and may be negative.

### Expanding temporal cross-validation on training history

Values are fold mean ± population standard deviation over three folds. Tuned HistGradientBoosting uses nested temporal CV with three outer expanding folds and three inner tuning folds within each outer training block. Base candidates use three expanding folds. No training CV is reported for Voting or Stacking because their composition is chosen based on validation.

| Model | CV MAE (Wh) | CV RMSE (Wh) | CV R² |
|---|---:|---:|---:|
| Persistence | 30.387 ± 1.021 | 73.219 ± 1.493 | 0.480 ± 0.027 |
| Dummy mean | 60.099 ± 1.275 | 101.743 ± 4.454 | -0.001 ± 0.001 |
| Ridge | 36.608 ± 8.038 | 69.480 ± 2.314 | 0.527 ± 0.075 |
| Random Forest | 31.720 ± 1.812 | 64.880 ± 0.728 | 0.590 ± 0.046 |
| Extra Trees | 30.113 ± 1.907 | 64.122 ± 0.348 | 0.600 ± 0.040 |
| HistGradientBoosting | 27.049 ± 1.022 | 67.234 ± 1.177 | 0.561 ± 0.027 |
| HistGradientBoosting (squared loss) | 29.386 ± 2.179 | 64.941 ± 1.546 | 0.588 ± 0.057 |
| XGBoost | 27.083 ± 1.461 | 65.273 ± 0.643 | 0.586 ± 0.032 |
| Tuned HistGradientBoosting (nested) | 27.093 ± 1.159 | 67.308 ± 1.296 | 0.560 ± 0.029 |

## Error and anomaly behavior

Selected-model absolute-error quantiles on test: median 7.85 Wh, p90 41.13 Wh, p95 109.36 Wh, p99 297.15 Wh. For targets above 110 Wh per interval, MAE was 86.37 Wh. The final chronological third had MAE 28.56 Wh. The largest miss was a 520.28 Wh underprediction on a valid 580 Wh source reading; it was retained. These results show weakness on rare high demand and temporal variation despite much smaller typical absolute error. See `reports/metrics/residual_diagnostics.json` and `reports/metrics/largest_test_errors.csv`.

The residual cutoff is the validation 99th percentile of absolute residual. Isolation Forest fits training features and uses its training-score 99th percentile as cutoff. Synthetic benchmark: 60 positive two-standard-deviation spikes inserted into a separate test-target copy. Residual detector: precision 0.026, recall 0.017, F1 0.020, PR-AUC 0.263. Isolation Forest: precision 0.035, recall 0.083, F1 0.049, PR-AUC 0.022. These are weak, setup-specific synthetic results; no natural event labels exist, so real-event detection performance is unknown.

## Explainability

Permutation importance measures the change in training MAE for the exact selected predictor. The strongest feature in the current run is `appliances_current`; this reflects predictive reliance and not a cause of consumption. Optional local SHAP uses the loaded selected model and matching feature schema. Explanations do not justify interventions by themselves.

## Risks, limitations and maintenance

Single-home scope, short time span, temporal shift, rare high use, sensor drift and absent ground-truth anomaly labels limit transfer. Do not use as a safety-critical or automated decision system. Validate on additional homes and future periods before broader use. Re-train when data or feature schema changes, generate a new run ID and fingerprint, then inspect errors and integrity reports. Only load trusted joblib bundles; pickle-based artifacts may execute code during loading.
