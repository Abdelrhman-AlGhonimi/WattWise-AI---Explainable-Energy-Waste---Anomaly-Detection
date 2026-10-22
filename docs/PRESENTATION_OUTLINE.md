# Presentation Outline

Use the current generated reports for every displayed result. State the run ID and keep validation selection distinct from held-out test comparison.

1. **Title and team** — WattWise AI, course and four team members/IDs from the README.
2. **Problem and scope** — next-observed-interval forecast; candidate residual events require human review; no verified waste labels.
3. **Dataset** — UCI Appliances Energy Prediction, one home, date span, units and source attribution.
4. **Data validation** — timestamp order, ten-minute cadence, duplicate/missing report, high-use retention.
5. **Features and leakage controls** — current readings, past lags, shifted rolling windows, one-step changes and cyclic calendar features; show the timeline of what is available at prediction time.
6. **Evaluation design** — chronological 70/15/15 split, expanding CV and nested tuning for the tuned model; clarify that validation selects and test is held out.
7. **Baselines and candidates** — Persistence and mean baseline, linear/tree/boosting models, common train-plus-validation refit and shared test rows.
8. **Selection and ensembles** — validation MAE ranking and tie-break; top-three equal-weight Voting and walk-forward Stacking; no CV claim for validation-composed ensembles.
9. **Complete results** — use test/CV tables from `MODEL_CARD.md`; explain MAE, RMSE, R² and why regression accuracy is undefined. Mention retrospective test leaders without changing the selected model.
10. **Residual analysis** — quantiles, largest valid high-use miss, late test period and demand-range error from the diagnostics report.
11. **Anomaly detection** — validation-calibrated residual threshold and train-fit Isolation Forest; flags are candidates, not diagnoses.
12. **Synthetic benchmark** — report injection setup and weak current precision/recall/PR-AUC; emphasize it does not measure natural anomalies.
13. **Feature ablation and explainability** — validation ablation, exact selected-model permutation importance and optional SHAP; no causal claims.
14. **System and UI** — architecture/data-flow diagram; demo dark/light theme, model detail, comparison, residual diagnostics, anomaly review and project-test page.
15. **Reproducibility and integrity** — seed, run ID, fingerprint, atomic writes, manifest, four executable notebooks and test suite.
16. **Limitations and next work** — one home, limited period, rare high-demand errors, no natural labels; validate across homes and verified events.
