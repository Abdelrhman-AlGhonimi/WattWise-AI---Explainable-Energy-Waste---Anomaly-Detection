# WattWise AI — Explainable Energy Waste & Anomaly Detection

An end-to-end Advanced ML project that forecasts household appliance consumption for the next observed 10-minute interval and surfaces unusually large forecast residuals for human review. The UCI dataset has no verified energy-waste labels, so a flag is a candidate event, not a diagnosis.

**Course:** Advanced Machine Learning

| Team member | ID |
| --- | ---: |
| Abdelrhman Mohamed Ahmed Al Ghonimi | 223101228 |
| Denar Reda Roushdy Elsayed | 223101171 |
| Mariem Abd-Elhamied Abo-Shaeshaa | 223101341 |
| Shrouk Ashraf Seria | 223101203 |

## Dataset and scope

The project uses the official [UCI Appliances Energy Prediction dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction). The inspected CSV has 19,735 rows and 29 columns; `Appliances` is Wh consumed per 10-minute interval. The loader downloads and caches the official archive in `data/raw/`, validates it, and keeps the raw file unchanged. See `docs/DATA_DICTIONARY.md` for source and attribution.

At time *t*, features use only currently available sensors, consumption, past lags and shifted rolling history, and calendar values; target is Appliances at the next observed row. Chronological splits use earliest 70% for training, next 15% for validation, newest 15% for final evaluation.

## Capabilities

- UCI data acquisition, timestamp/target validation, cleaning report, reproducible feature engineering (63 features in the recorded run).
- Persistence, dummy, Ridge, Random Forest, Extra Trees, HistGradientBoosting, optional XGBoost, Optuna tuning with randomized search fallback.
- Validation-selected top-three averaging and walk-forward temporal stacking.
- MAE/RMSE/R² and observation counts, nested/expanding temporal CV, residual diagnostics, validation-calibrated residual flags, Isolation Forest artifact, synthetic spike benchmark, feature ablation and selected-model permutation importance.
- Eight-section Streamlit dashboard with dark/light themes, artifact-integrity checks, a project test runner, four executable notebooks, pytest tests, diagrams and methodology/model documentation.

The dashboard's **Model Comparison** page distinguishes validation selection from held-out test performance, shows per-model MAE/RMSE/R² and sample counts, training temporal CV, estimator parameters, tuning history, validation feature ablations and residual diagnostics. WAPE is a relative error summary, not a percentage of predictions that are correct. Since this is regression, classification **accuracy** is not defined. The **Testing and Project Information** page checks artifacts and runs the project test suite on demand; it does not retrain models.

## Setup

Python 3.10 or newer.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[app,advanced,notebooks,test]"
```

Data downloads on first load. Manual recovery: extract the official archive's `energydata_complete.csv` to `data/raw/energydata_complete.csv`.

## Run

```powershell
python -m wattwise.train
streamlit run app.py
pytest -q
```

Training writes `artifacts/wattwise_bundle.joblib`, `reports/model_comparison.csv`, and JSON/CSV evaluation files under `reports/metrics/`. Raw data and model bundles are git-ignored; generated reports remain available locally. No scores are hardcoded: report only values generated in those artifacts.

### Executed experiment in this workspace

Recorded run: `55004ee9-e3d3-433c-9528-6e2227d2b7d8` (seed 42; dataset fingerprint and run ID are persisted with artifacts). The data contained 19,735 rows, 29 columns, no missing values or duplicate timestamps, and a regular 10-minute interval. The 63-feature, one-step-ahead setup produced 13,713 training, 2,938 validation and 2,939 test rows. Forecast origins in the test span 2016-05-07 08:10–2016-05-27 17:50; target timestamps span 08:20–18:00.

**Voting (top 3)** was selected using validation MAE 21.939 Wh (RMSE 56.356 Wh, R² 0.610), with equal weights on Tuned HistGradientBoosting, HistGradientBoosting and XGBoost. On the untouched test it scored **MAE 22.385 Wh, RMSE 58.723 Wh, R² 0.583**. Model selection uses validation MAE, then validation RMSE, then model name. Tuned HistGradientBoosting had the lowest retrospective test MAE (22.268 Wh); Extra Trees had the lowest test RMSE (57.324 Wh) and highest R² (0.603). Those test results do not alter the validation-selected model. The full test and temporal CV tables are in [Model Card](docs/MODEL_CARD.md); raw generated comparisons are in `reports/model_comparison.csv`.

Nested temporal CV for Tuned HistGradientBoosting (3 outer folds, each tuned on 3 inner expanding folds) gave MAE 27.093 ± 1.159 Wh, RMSE 67.308 ± 1.296 Wh and R² 0.560 ± 0.029. Other base candidates use three expanding folds on the training block. Voting and Stacking have no training-CV estimate because their composition is selected from validation results.

Test absolute-error median, 90th, 95th and 99th percentiles were 7.85, 41.13, 109.36 and 297.15 Wh. Cases above 110 Wh per interval had MAE 86.37 Wh; the final third of test had MAE 28.56 Wh. The largest underprediction was 520.28 Wh on a recorded 580 Wh observation. Timestamps and source readings were present, so high-use cases were retained. See `reports/metrics/residual_diagnostics.json` and `reports/metrics/largest_test_errors.csv`.

The synthetic benchmark injected 60 positive spikes into a separate copy of test targets at 2 standard deviations. Residual detection returned precision 0.026, recall 0.017 and PR-AUC 0.263; Isolation Forest returned precision 0.035, recall 0.083 and PR-AUC 0.022. These results are weak and apply only to the deterministic synthetic injection setup. The source contains no natural anomaly labels; these scores do not establish real-world anomaly performance. Exact metrics and thresholds are in `reports/metrics/anomaly_evaluation.json`.

Saved plots: [consumption distribution](reports/figures/appliances_distribution.png), [test forecasts](reports/figures/test_forecast.png), and [residual distribution](reports/figures/residual_distribution.png). The diagnostics JSON and largest-error CSV are under `reports/metrics/`.

## Notebook order

Run from the repository root: `01_data_collection_and_eda.ipynb`, `02_preprocessing_and_feature_engineering.ipynb`, `03_advanced_modeling_and_optimization.ipynb`, then `04_anomaly_detection_and_explainability.ipynb`.

## Evaluation and limitations

MAE/RMSE are error in Wh per 10-minute sample; RMSE weighs large misses more. R² compares variance explained and can be negative. The UCI source represents one home over a limited period, so cross-home generalization is untested. Verified waste and anomaly labels do not exist. Synthetic-spike results apply only to that injection design. Feature importance is model reliance, not cause. See `docs/LIMITATIONS_AND_ETHICS.md`.

## Structure and troubleshooting

`src/wattwise/` contains the shared pipeline; `app.py` is the interface; `notebooks/` documents the workflow; `docs/` contains methodology and diagrams; `tests/` uses small fixtures. If `wattwise` cannot import, run `pip install -e .` from the root. If the dashboard has no model, run `python -m wattwise.train`. Optional XGBoost, Optuna and SHAP are not required for core sklearn operation.

Latest verification: `pytest -q` passes 28 tests; Ruff reports no issues; compileall succeeds; all four notebooks execute; Streamlit AppTest covers navigation, comparison periods and the in-app test runner. A scoped MyPy run succeeds for evaluation, inference and training modules; a full-project MyPy run is not clean because optional/dependency typing stubs are absent and the package is imported under duplicate module names. Future work includes multiple homes, verified anomaly labels, drift monitoring, robust temporal intervals, and validated local SHAP explanations.

