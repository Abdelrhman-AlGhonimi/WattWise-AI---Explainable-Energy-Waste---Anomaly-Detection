# Project Documentation

## Project scope

WattWise AI forecasts household appliance use for the next observed ten-minute interval and surfaces unusual forecast residuals for human review. It is an academic forecasting and triage demonstration. The available data do not verify waste, faults, causes or savings; a flagged interval is a candidate for review.

## Data and target

The project uses the official UCI Appliances Energy Prediction dataset: one low-energy home, 19,735 observations and 29 columns from January–May 2016. `Appliances` and `lights` are Wh per interval. At time *t*, `target_next` is Appliances at the next observed timestamp. The source has a regular ten-minute interval, no missing cells and no duplicate timestamp. See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for column descriptions and attribution.

The loader validates source columns, timestamps and target values. Cleaning sorts chronologically, reports and deduplicates timestamps by retaining the last reading, forward-fills numeric gaps using past values only, and drops leading rows without prior history. It does not remove high-use observations. Raw data are not modified.

## Feature engineering and evaluation design

The recorded run uses 63 features: current readings, appliance lags and shifted rolling summaries, recent one-step changes in appliance/light/weather and selected room measurements, and daily/weekly cyclic calendar terms. Every feature is available at the forecast origin. Lags and rolling summaries use past observations; rolling windows exclude the current point. Warm-up rows without full history are dropped.

The chronological split is 70/15/15 (13,713 train, 2,938 validation, 2,939 test). Target timestamps are checked for overlap. Validation chooses the model and calibrates the residual cutoff; the test period is held out from both. All candidate models are refit on the same train-plus-validation history and compared over the exact same test target rows.

Candidates include Persistence, Dummy mean, Ridge, Random Forest, Extra Trees, HistGradientBoosting, squared-loss HistGradientBoosting, XGBoost, tuned HistGradientBoosting, equal-weight top-three Voting and walk-forward Temporal Stacking. Base models use three expanding folds. Tuned HistGradientBoosting is evaluated with three outer folds and tuning within three inner expanding folds per outer fold. Voting/Stacking have no training CV score because their components are chosen using validation results. See [METHODOLOGY.md](METHODOLOGY.md) for full protocol and metrics.

## Recorded results

Run `55004ee9-e3d3-433c-9528-6e2227d2b7d8` selected Voting (top 3) by validation MAE 21.939 Wh, with components Tuned HistGradientBoosting, HistGradientBoosting and XGBoost at equal one-third weights. Selected test performance: MAE 22.385 Wh, RMSE 58.723 Wh, R² 0.583 on 2,939 samples. Retrospective test leaders (Tuned HistGradientBoosting for MAE; Extra Trees for RMSE/R²) do not change the pre-test selection.

The tuned model's nested CV results are MAE 27.093 ± 1.159 Wh, RMSE 67.308 ± 1.296 Wh and R² 0.560 ± 0.029. All candidates' test/CV values are in [MODEL_CARD.md](MODEL_CARD.md) and `reports/model_comparison.csv`. Regression has no classification accuracy score. WAPE is a relative error summary, not percent-correct.

Error analysis finds materially harder rare high-use and late-period cases. The 90th/99th percentiles of absolute error are 41.13/297.15 Wh. A recorded target of 580 Wh was underpredicted by 520.28 Wh; data inspection found a valid timestamp and reading, so it was retained. Results and time/range diagnostics are in `reports/metrics/residual_diagnostics.json` and `reports/metrics/largest_test_errors.csv`.

The anomaly benchmark injects 60 positive two-standard-deviation spikes into a copy of test targets. Residual scoring reached precision 0.026, recall 0.017, F1 0.020 and PR-AUC 0.263; Isolation Forest reached precision 0.035, recall 0.083, F1 0.049 and PR-AUC 0.022. These are weak, injection-specific results, not estimates of detection on natural events. No verified anomaly labels exist.

## Application and generated artifacts

`src/wattwise/` contains loading, validation, preprocessing, features, fitting, tuning, ensembles, evaluation, anomaly scoring, explanations and inference. `app.py` provides eight dashboard sections, dark/light themes, a regression model comparison, held-out diagnostics, formulation/parameter details, tuning and ablation results, candidate anomaly views, explainability and a project test/readiness page. Theme styling includes the tables, plots, widgets and metric cards. The project test page runs the suite only when requested and does not retrain the model.

The bundle and generated reports carry a shared run ID and dataset fingerprint. Atomic writes and a manifest-last commit protect against partial runs; the app checks for missing files and mismatched metadata before rendering results. The saved inference bundle contains the selected model/ensemble, its feature schema, selected model metadata and detector artifacts. Do not load untrusted joblib files.

Important outputs:

- `artifacts/wattwise_bundle.joblib`
- `reports/model_comparison.csv`
- `reports/metrics/artifact_manifest.json`
- `reports/metrics/final_metrics.json` and `training_config.json`
- `reports/metrics/test_predictions.csv` and `residual_diagnostics.json`
- `reports/metrics/largest_test_errors.csv`, `feature_ablation.csv`, `selected_features.json`, `feature_importance.csv`
- `reports/metrics/anomaly_evaluation.json` and `best_hyperparameters.json`

## Reproducibility and verification

Use the repository root for commands:

```powershell
pip install -e ".[app,advanced,notebooks,test]"
python -m wattwise.train
streamlit run app.py
pytest -q
```

The latest verification passed 28 pytest tests, Ruff, compileall, execution of all four notebooks and Streamlit AppTest checks for navigation, comparison periods and the in-app test runner. A scoped MyPy run passed on evaluation, inference and training modules. Full-project MyPy remains limited by unavailable dependency stubs and duplicate package module identities.

## Limitations and future work

Only one household and one bounded period are represented. Temporal behavior shifts, rare high-power events, other homes, sensor drift and other climates are not covered. No verified anomaly/waste labels or calibrated uncertainty intervals are available. Permutation importance and SHAP explain model behavior, not causality. Future work: multi-home validation, verified event labels, drift monitoring, time-aware uncertainty methods and carefully checked local explanations.
