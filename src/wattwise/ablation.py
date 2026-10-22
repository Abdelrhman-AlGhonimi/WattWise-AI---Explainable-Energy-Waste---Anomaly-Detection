from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from .evaluation import regression_metrics


def run_ablation(
    X_train, y_train, X_validation, y_validation, output_dir, seed=42, run_id=None
):
    """Compare feature groups using training-only MI ranking and validation scoring."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mi = mutual_info_regression(X_train, y_train, random_state=seed)
    selected = list(
        pd.Series(mi, index=X_train.columns)
        .sort_values(ascending=False)
        .head(min(20, X_train.shape[1]))
        .index
    )
    groups = {
        "all_features": list(X_train.columns),
        "without_consumption_history": [
            c
            for c in X_train
            if not c.startswith("appliances_lag_") and "appliances_roll_" not in c
        ],
        "mutual_information_top_20": selected,
    }
    rows = []
    for name, cols in groups.items():
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=5,
            early_stopping=False,
            random_state=seed,
        ).fit(X_train[cols], y_train)
        rows.append(
            {
                "feature_set": name,
                "feature_count": len(cols),
                **regression_metrics(y_validation, model.predict(X_validation[cols])),
                "run_id": run_id,
            }
        )
    csv_path = out / "feature_ablation.csv"
    fd, csv_temp = tempfile.mkstemp(prefix=".feature_ablation.", suffix=".csv", dir=out)
    os.close(fd)
    try:
        pd.DataFrame(rows).to_csv(csv_temp, index=False)
        os.replace(csv_temp, csv_path)
    finally:
        Path(csv_temp).unlink(missing_ok=True)
    json_path = out / "selected_features.json"
    fd, json_temp = tempfile.mkstemp(
        prefix=".selected_features.", suffix=".json", dir=out
    )
    os.close(fd)
    try:
        Path(json_temp).write_text(
            json.dumps(
                {
                    "method": "mutual information fitted on train partition only",
                    "selected_features": selected,
                    "seed": seed,
                    "run_id": run_id,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(json_temp, json_path)
    finally:
        Path(json_temp).unlink(missing_ok=True)
    return rows
