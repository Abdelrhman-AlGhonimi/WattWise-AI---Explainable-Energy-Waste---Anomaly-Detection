# Methodology

## Forecast definition and data controls

1. Acquire the official UCI archive only when the local raw CSV is absent. Validate required columns, timestamp parsing, target type and nonnegative target values. Keep the source file unchanged and identify generated outputs by run ID and a content fingerprint.
2. Sort by timestamp. When duplicates exist, keep the last timestamp and report the count. Fill numeric gaps forward from past observations only; drop leading rows without prior history. Retain high consumption pending review rather than automatically treating it as corruption. The inspected source contains 19,735 rows, 29 columns, no missing values, no duplicate timestamps and regular ten-minute sampling.
3. At forecast origin *t*, use currently observed sensors and use. The label is Appliances at the next observed row, `y[t] = Appliances[t+1]`; persist origin and target timestamps separately. Lags use prior readings and every rolling statistic shifts by one row before aggregation. Drop the feature warm-up rows and the final row without a next target.

## Features

The recorded experiment has 63 features: current sensors and appliance use; appliance lags at 1, 2, 3, 6, 12 and 144 samples; shifted rolling mean and standard deviation for 3, 6, 12 and 144 samples; one-step differences for appliance use, lights, selected indoor/outdoor temperature, humidity and wind measurements; and calendar terms (hour, weekday, weekend, minute of day, daily and weekly sine/cosine). Feature changes use only the current and past rows. The train-only mutual-information selector and the validation ablation are exploratory; selection does not use test labels.

## Splits, validation and model fitting

Use chronological 70% training, 15% validation and 15% test. For the next-row target, each validation origin can use consumption that has already been observed at that origin. Assert that target timestamps do not overlap across partitions. The recorded run has 13,713 train, 2,938 validation and 2,939 test samples.

Compare Persistence, Dummy mean, Ridge, Random Forest, Extra Trees, HistGradientBoosting with absolute and squared losses, XGBoost with pseudo-Huber loss, tuned HistGradientBoosting, top-three equal-weight Voting and walk-forward Temporal Stacking. Base candidates use three expanding temporal folds within the training period. Tuned HistGradientBoosting uses nested temporal CV: three outer expanding folds, each with tuning on three inner expanding folds. Report each metric's fold mean and population standard deviation. All tuning and feature selection stay inside training history.

Select the model by minimum validation MAE, breaking ties by lower validation RMSE and then ascending model name. Voting components are the three best validation candidates under the same ranking. Stacking trains a Ridge meta-model from chronological walk-forward out-of-fold base predictions. Because ensemble composition is validation-selected, do not report training-CV results for Voting or Stacking. After selection, refit every candidate on the same train-plus-validation history and evaluate each once over the same aligned test targets. Test metrics are descriptive and never influence selection.

## Metrics and diagnostics

MAE and RMSE are in Wh per ten-minute interval; RMSE gives larger misses more weight. R² can be negative and is undefined for constant targets. Metric helpers reject empty, non-finite or misaligned arrays and include `n_obs`. Classification accuracy is not a regression metric. WAPE is reported only as a relative error summary.

Residual means actual minus predicted. Report absolute-error quantiles, errors over chronological blocks, target ranges and the largest misses. Keep valid high-use observations in the evaluation. In the recorded run, the selected test MAE/RMSE/R² are 22.385/58.723/0.583 Wh/Wh/score. The 90th and 99th percentile absolute errors are 41.13 and 297.15 Wh. The final third of the test period has MAE 28.56 Wh; observations above 110 Wh per interval have MAE 86.37 Wh. The largest miss is 520.28 Wh, on a valid recorded 580 Wh target. See `reports/metrics/residual_diagnostics.json` and `reports/metrics/largest_test_errors.csv`.

Feature ablation is evaluated on validation, and permutation importance is calculated for the exact selected predictor against training rows. Neither establishes causal influence. The current validation ablation compares 63 all-features, 49 without consumption history, and 20 train-only mutual-information features; results are saved to `reports/metrics/feature_ablation.csv`.

## Candidate anomaly detection

Calibrate the residual detector's absolute-error threshold at the 99th percentile of validation residuals. Fit Isolation Forest using training features only and derive its cutoff from the training-score 99th percentile. For a reproducible benchmark, add 60 positive spikes at two standard deviations to a separate copy of test targets, record the injected locations and compare continuous anomaly scores with the synthetic labels. Use scores for PR-AUC and thresholded predictions for precision, recall, F1 and confusion matrix. Do not modify the original target or describe these results as natural anomaly accuracy. The UCI dataset has no verified waste/anomaly labels. Current benchmark results and thresholds are recorded in `reports/metrics/anomaly_evaluation.json`.

## Artifact integrity and reproducibility

Write JSON, CSV, model and downloaded data files atomically. Save the model bundle, comparison, predictions, diagnostics, feature reports, anomaly report, tuning details and config under a shared run ID and dataset fingerprint; commit `reports/metrics/artifact_manifest.json` last. The dashboard checks manifest membership and consistency across artifacts before displaying a run. It loads the selected predictor from that bundle and verifies feature schema and finite, aligned predictions. Use only trusted local joblib files.

Recorded run ID: `55004ee9-e3d3-433c-9528-6e2227d2b7d8`. Reproduce with `python -m wattwise.train` from the repository root. Verify with `pytest -q`; see the README for setup and UI commands.
