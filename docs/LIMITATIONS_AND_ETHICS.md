# Limitations and Ethics

- **No verified event labels:** The UCI dataset does not label energy waste, faults or genuine anomalies. Dashboard flags are candidates for human review, not diagnoses.
- **Synthetic benchmark only:** Current benchmark adds 60 positive spikes at two standard deviations to a separate copy of test targets. Residual detector PR-AUC is 0.263; Isolation Forest PR-AUC is 0.022, with low precision and recall. These numbers describe that injection process only and cannot be generalized to natural events.
- **Rare high-use and temporal errors:** Selected-model test MAE is 22.385 Wh but RMSE is 58.723 Wh. The largest observed miss is 520.28 Wh on a valid 580 Wh measurement. Targets above 110 Wh per interval have MAE 86.37 Wh, and the final third of test has MAE 28.56 Wh. A low typical error does not guarantee accuracy during high-demand activity.
- **Limited transfer:** Data come from one household during a bounded period in 2016. Results do not establish performance for other homes, appliances, climates, sensor placements or later behavior.
- **Leakage controls:** Features use only information available at the forecast origin, missing values are filled from past readings only, and target timestamps are checked across chronological partitions. Validation controls selection and threshold calibration; the test period is excluded from those decisions.
- **Attribution is not causation:** Permutation importance and SHAP describe model behavior. They do not establish why use occurred or which device should be changed.
- **No guaranteed savings or intervention:** A residual can be ordinary rare activity, sensor error or model misspecification. Review nearby measurements and context before action.
- **Uncertainty limits:** Temporal dependence weakens exchangeability assumptions behind ordinary conformal intervals. This project does not claim calibrated prediction intervals.
- **Artifact security:** Joblib uses pickle-based serialization and can execute code when loading. Load only bundles produced by a trusted local run.

Responsible next steps include evaluation across multiple homes and future periods, verified event annotations, drift monitoring, robust time-aware uncertainty estimates, and human review of proposed interventions.
