from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from .ablation import run_ablation
from .anomaly_detection import (
    fit_isolation_forest,
    inject_spikes,
    residual_threshold,
    synthetic_metrics,
)
from .config import ARTIFACTS, REPORTS, SEED
from .data_loader import load_data
from .ensemble import fit_stack, predict_stack
from .evaluation import (
    regression_metrics,
    residual_diagnostics,
    select_model,
    temporal_cv_summary,
)
from .explainability import permutation_feature_importance
from .feature_engineering import chronological_split, make_features
from .model_factory import model_factory
from .optimization import optimize
from .preprocessing import clean_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _atomic_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(fd)
    return Path(name)


def _save_json(path: Path, data: dict) -> None:
    temporary = _atomic_path(Path(path))
    try:
        temporary.write_text(
            json.dumps(data, indent=2, default=str, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = _atomic_path(Path(path))
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_joblib(path: Path, value) -> None:
    temporary = _atomic_path(Path(path))
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metric_row(name: str, split: str, y_true, y_pred, **extra) -> dict:
    return {
        "model": name,
        "split": split,
        **regression_metrics(y_true, y_pred),
        **extra,
    }


def _persistence_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> dict:
    folds = []
    for fold, (train_idx, valid_idx) in enumerate(
        TimeSeriesSplit(n_splits=n_splits).split(X), start=1
    ):
        scores = regression_metrics(
            y.iloc[valid_idx], X.iloc[valid_idx]["appliances_current"]
        )
        folds.append(
            {
                "fold": fold,
                **scores,
                "train_end_index": int(train_idx[-1]),
                "validation_start_index": int(valid_idx[0]),
                "R2_status": "valid" if scores["R2"] is not None else "undefined",
            }
        )
    table = pd.DataFrame(folds)
    valid_r2 = table.R2.dropna()
    return {
        "CV_MAE": float(table.MAE.mean()),
        "CV_MAE_std": float(table.MAE.std(ddof=0)),
        "CV_RMSE": float(table.RMSE.mean()),
        "CV_RMSE_std": float(table.RMSE.std(ddof=0)),
        "CV_R2": float(valid_r2.mean()) if len(valid_r2) else None,
        "CV_R2_std": float(valid_r2.std(ddof=0)) if len(valid_r2) else None,
        "CV_R2_valid_folds": int(len(valid_r2)),
        "n_folds": int(len(folds)),
        "mean_fold_n_obs": float(table.n_obs.mean()),
        "folds": folds,
    }


def _nested_tuned_cv_summary(X: pd.DataFrame, y: pd.Series, seed: int) -> dict:
    """Outer-fold performance with independent inner tuning for each fold."""
    folds = []
    for fold, (train_idx, valid_idx) in enumerate(
        TimeSeriesSplit(n_splits=3).split(X), start=1
    ):
        fitted, tuning = optimize(
            X.iloc[train_idx], y.iloc[train_idx], trials=4, seed=seed + fold
        )
        scores = regression_metrics(
            y.iloc[valid_idx], fitted.predict(X.iloc[valid_idx])
        )
        folds.append(
            {
                "fold": fold,
                **scores,
                "train_end_index": int(train_idx[-1]),
                "validation_start_index": int(valid_idx[0]),
                "inner_best_params": tuning["best_params"],
                "inner_best_cv_mae": tuning["best_cv_mae"],
            }
        )
    metrics = pd.DataFrame(folds)
    valid_r2 = metrics.R2.dropna()
    return {
        "protocol": "nested temporal CV: 3 outer expanding folds; each outer training block tuned with 3 inner expanding folds",
        "CV_MAE": float(metrics.MAE.mean()),
        "CV_MAE_std": float(metrics.MAE.std(ddof=0)),
        "CV_RMSE": float(metrics.RMSE.mean()),
        "CV_RMSE_std": float(metrics.RMSE.std(ddof=0)),
        "CV_R2": float(valid_r2.mean()) if len(valid_r2) else None,
        "CV_R2_std": float(valid_r2.std(ddof=0)) if len(valid_r2) else None,
        "CV_R2_valid_folds": int(len(valid_r2)),
        "n_folds": len(folds),
        "mean_fold_n_obs": float(metrics.n_obs.mean()),
        "folds": folds,
    }


def _dataset_fingerprint(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def _cached_tuning(fingerprint: str, feature_columns: list[str], train_rows: int):
    """Reuse tuning only when prior result, config, and bundle prove same inputs."""
    metrics_path = REPORTS / "metrics" / "final_metrics.json"
    tuning_path = REPORTS / "metrics" / "best_hyperparameters.json"
    config_path = REPORTS / "metrics" / "training_config.json"
    bundle_path = ARTIFACTS / "wattwise_bundle.joblib"
    if not all(
        path.exists() for path in (metrics_path, tuning_path, config_path, bundle_path)
    ):
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved = json.loads(tuning_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    old_bundle = joblib.load(bundle_path)
    run_ids = {
        metrics.get("run_id"),
        saved.get("run_id"),
        config.get("run_id"),
        old_bundle.get("run_id"),
    }
    optimization = saved.get("optimization", {})
    if (
        len(run_ids) != 1
        or None in run_ids
        or metrics.get("dataset_fingerprint") != fingerprint
        or old_bundle.get("dataset_fingerprint") != fingerprint
        or old_bundle.get("feature_columns") != feature_columns
        or config.get("seed") != SEED
        or config.get("feature_count") != len(feature_columns)
        or metrics.get("train_rows") != train_rows
        or optimization.get("objective")
        != "mean MAE over 3 expanding TimeSeriesSplit folds within training data"
        or optimization.get("seed") != SEED
        or optimization.get("n_splits") != 3
    ):
        return None
    params = optimization.get("best_params")
    required = {
        "max_iter",
        "learning_rate",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
    }
    if not isinstance(params, dict) or set(params) != required:
        return None
    estimator = HistGradientBoostingRegressor(
        loss="absolute_error", **params, early_stopping=False, random_state=SEED
    )
    return estimator, optimization, next(iter(run_ids))


def _assert_target_boundaries(parts, target_dates) -> None:
    target_by_split = {
        name: set(target_dates.loc[values[0].index]) for name, values in parts.items()
    }
    names = list(target_by_split)
    for left, right in zip(names, names[1:]):
        left_dates = pd.DatetimeIndex(sorted(target_by_split[left]))
        right_dates = pd.DatetimeIndex(sorted(target_by_split[right]))
        if (
            len(left_dates)
            and len(right_dates)
            and left_dates.max() >= right_dates.min()
        ):
            raise ValueError(
                f"Target timestamps overlap or are out of order between {left} and {right}"
            )


def train(trials: int = 8):
    ARTIFACTS.mkdir(exist_ok=True, parents=True)
    (REPORTS / "metrics").mkdir(parents=True, exist_ok=True)
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())

    raw = load_data()
    data, cleaning = clean_data(raw)
    fingerprint = _dataset_fingerprint(data)
    X, y, origin_dates = make_features(data)
    parts = chronological_split(X, y, origin_dates)
    _assert_target_boundaries(
        parts, pd.Series(data.date.iloc[1:].to_numpy(), index=np.arange(len(data) - 1))
    )
    X_train, y_train, _ = parts["train"]
    X_valid, y_valid, valid_origins = parts["validation"]
    X_test, y_test, test_origins = parts["test"]
    target_dates = pd.Series(
        data.date.iloc[1:].to_numpy(), index=np.arange(len(data) - 1)
    )
    test_targets = target_dates.loc[y_test.index]
    if not (
        X_train.columns.equals(X_valid.columns)
        and X_train.columns.equals(X_test.columns)
    ):
        raise ValueError("Feature schemas differ across chronological splits")
    if not (
        y_test.index.equals(test_origins.index)
        and y_test.index.equals(test_targets.index)
    ):
        raise ValueError("Test targets and timestamps are not aligned")

    ablation_rows = run_ablation(
        X_train, y_train, X_valid, y_valid, REPORTS / "metrics", SEED, run_id=run_id
    )
    results: list[dict] = []
    cv_summaries: dict = {}
    val_predictions: dict[str, np.ndarray] = {}
    train_models = model_factory(SEED)

    # Persistence uses the latest available Appliances measurement.
    val_predictions["Persistence"] = X_valid["appliances_current"].to_numpy(dtype=float)
    results.append(
        _metric_row(
            "Persistence", "validation", y_valid, val_predictions["Persistence"]
        )
    )
    cv_summaries["Persistence"] = _persistence_cv(X_train, y_train)

    for name, estimator in train_models.items():
        logging.info("Training-only temporal CV for %s", name)
        cv_summaries[name] = temporal_cv_summary(
            estimator, X_train, y_train, n_splits=3
        )
        model = clone(estimator).fit(X_train, y_train)
        val_predictions[name] = np.asarray(model.predict(X_valid), dtype=float)
        results.append(_metric_row(name, "validation", y_valid, val_predictions[name]))

    cache = _cached_tuning(fingerprint, list(X.columns), len(X_train))
    if cache:
        tuned_estimator, cached_optimization, source_run_id = cache
        tuned_model = tuned_estimator.fit(X_train, y_train)
        tuning = {
            **cached_optimization,
            "reused": True,
            "reused_from_run_id": source_run_id,
        }
        logging.info("Reused verified Optuna results from run %s", source_run_id)
    else:
        tuned_model, tuning = optimize(X_train, y_train, trials=trials, seed=SEED)
        tuning = {**tuning, "reused": False}
    tuned_estimator = clone(tuned_model)
    cv_summaries["Tuned HistGradientBoosting"] = _nested_tuned_cv_summary(
        X_train, y_train, seed=SEED
    )
    val_predictions["Tuned HistGradientBoosting"] = np.asarray(
        tuned_model.predict(X_valid), dtype=float
    )
    results.append(
        _metric_row(
            "Tuned HistGradientBoosting",
            "validation",
            y_valid,
            val_predictions["Tuned HistGradientBoosting"],
        )
    )

    cv_rows = []
    for name, summary in cv_summaries.items():
        cv_rows.append(
            {
                "model": name,
                "split": "training_cv",
                "MAE": summary["CV_MAE"],
                "RMSE": summary["CV_RMSE"],
                "R2": summary["CV_R2"],
                "MAE_std": summary["CV_MAE_std"],
                "CV_MAE": summary["CV_MAE"],
                "CV_MAE_std": summary["CV_MAE_std"],
                "CV_RMSE": summary["CV_RMSE"],
                "CV_RMSE_std": summary["CV_RMSE_std"],
                "CV_R2": summary["CV_R2"],
                "CV_R2_std": summary["CV_R2_std"],
                "n_folds": summary["n_folds"],
                "mean_fold_n_obs": summary["mean_fold_n_obs"],
                "CV_protocol": summary.get(
                    "protocol", "3-fold expanding TimeSeriesSplit on training block"
                ),
                "CV_status": "valid"
                if summary["CV_R2"] is not None
                else "R² undefined for every fold",
            }
        )

    # Vote membership is selected on validation MAE only; ties break by RMSE,
    # then model name, so repeated runs select deterministically.
    ranked = sorted(
        (
            (
                regression_metrics(y_valid, pred)["MAE"],
                regression_metrics(y_valid, pred)["RMSE"],
                name,
            )
            for name, pred in val_predictions.items()
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    best_names = [name for _, _, name in ranked[:3]]
    val_predictions["Voting (top 3)"] = np.mean(
        [val_predictions[name] for name in best_names], axis=0
    )
    results.append(
        _metric_row(
            "Voting (top 3)",
            "validation",
            y_valid,
            val_predictions["Voting (top 3)"],
            components=best_names,
        )
    )

    validation_stack = None
    stack_error = None
    stack_base_names = [name for name in best_names if name != "Persistence"][:3]
    try:
        stack = fit_stack(
            {
                name: train_models[name] if name in train_models else tuned_estimator
                for name in stack_base_names
            },
            X_train,
            y_train,
            splits=4,
        )
        validation_stack = predict_stack(stack, X_valid)
        val_predictions["Temporal Stacking"] = np.asarray(validation_stack, dtype=float)
        results.append(
            _metric_row(
                "Temporal Stacking",
                "validation",
                y_valid,
                val_predictions["Temporal Stacking"],
                components=stack_base_names,
            )
        )
    except Exception as exc:
        stack_error = f"{type(exc).__name__}: {exc}"
        logging.exception("Temporal stacking unavailable; preserving other candidates")

    selection_rows = [row for row in results if row["split"] == "validation"]
    selection_metadata = select_model(selection_rows)
    selected = selection_metadata["selected_model"]
    train_valid_X = pd.concat([X_train, X_valid]).sort_index()
    train_valid_y = pd.concat([y_train, y_valid]).sort_index()

    # Refit every candidate on the same permitted history before a single
    # aligned test evaluation. No test metric participates in model selection.
    full_models = {
        name: clone(estimator).fit(train_valid_X, train_valid_y)
        for name, estimator in train_models.items()
    }
    full_models["Tuned HistGradientBoosting"] = clone(tuned_estimator).fit(
        train_valid_X, train_valid_y
    )
    test_predictions: dict[str, np.ndarray] = {
        name: np.asarray(model.predict(X_test), dtype=float)
        for name, model in full_models.items()
    }
    test_predictions["Persistence"] = X_test["appliances_current"].to_numpy(dtype=float)
    test_predictions["Voting (top 3)"] = np.mean(
        [test_predictions[name] for name in best_names], axis=0
    )

    full_stack = None
    if validation_stack is not None:
        full_stack = fit_stack(
            {
                name: train_models[name] if name in train_models else tuned_estimator
                for name in stack_base_names
            },
            train_valid_X,
            train_valid_y,
            splits=4,
        )
        test_predictions["Temporal Stacking"] = np.asarray(
            predict_stack(full_stack, X_test), dtype=float
        )

    if selected not in test_predictions:
        raise RuntimeError(
            f"Validation-selected model {selected!r} has no test predictions"
        )
    selected_test = test_predictions[selected]
    if len({len(values) for values in test_predictions.values()}) != 1 or any(
        not np.isfinite(values).all() for values in test_predictions.values()
    ):
        raise ValueError(
            "Candidate predictions are non-finite or not aligned on the common test timestamps"
        )
    selected_final_model = (
        {
            "kind": "voting",
            "models": {name: full_models[name] for name in best_names},
            "components": best_names,
            "weights": {name: 1 / len(best_names) for name in best_names},
        }
        if selected == "Voting (top 3)"
        else {"kind": "stack", "stack": full_stack, "components": stack_base_names}
        if selected == "Temporal Stacking"
        else None
        if selected == "Persistence"
        else full_models[selected]
    )
    selection_metadata.update(
        {
            "validation_candidates": [row["model"] for row in selection_rows],
            "voting_components": best_names,
            "voting_weights": {name: 1 / len(best_names) for name in best_names},
            "stack_components": stack_base_names if full_stack is not None else [],
            "stack_unavailable_reason": stack_error,
        }
    )

    test_rows = []
    for name, prediction in test_predictions.items():
        test_rows.append(
            _metric_row(
                name,
                "test",
                y_test,
                prediction,
                selected=(name == selected),
                components=best_names
                if name == "Voting (top 3)"
                else stack_base_names
                if name == "Temporal Stacking"
                else None,
            )
        )
    selected_test_metrics = next(row for row in test_rows if row["model"] == selected)
    all_results = pd.DataFrame(cv_rows + results + test_rows)
    all_results["run_id"] = run_id
    all_results["dataset_fingerprint"] = fingerprint
    all_results["target_horizon_rows"] = 1
    _save_csv(REPORTS / "model_comparison.csv", all_results)

    selected_validation = val_predictions[selected]
    threshold = residual_threshold(y_valid.to_numpy() - selected_validation, 0.99)
    isolation_forest = fit_isolation_forest(X_train, seed=SEED)
    bundle = {
        "model": selected_final_model,
        "selected_model": selected,
        "selection": selection_metadata,
        "feature_columns": list(X.columns),
        "threshold": threshold,
        "isolation_forest": isolation_forest,
        "seed": SEED,
        "version": "1.1.0",
        "run_id": run_id,
        "dataset_fingerprint": fingerprint,
        "target_definition": "Appliances at the next observed row (one-step ahead)",
        "training_period": [
            str(origin_dates.loc[X_train.index].iloc[0]),
            str(origin_dates.loc[X_train.index].iloc[-1]),
        ],
        "validation_period": [str(valid_origins.iloc[0]), str(valid_origins.iloc[-1])],
        "test_period": [str(test_origins.iloc[0]), str(test_origins.iloc[-1])],
        "target_test_period": [str(test_targets.iloc[0]), str(test_targets.iloc[-1])],
        "bestnames": best_names,
    }
    _save_joblib(ARTIFACTS / "wattwise_bundle.joblib", bundle)

    # Synthetic anomalies are injected into a copy of test targets after model
    # fitting and threshold calibration. The source test observations stay intact.
    synthetic_y, labels, indices = inject_spikes(y_test.to_numpy(), seed=SEED)
    residual_scores = np.abs(synthetic_y - selected_test)
    isolation_cut = float(np.quantile(-isolation_forest.score_samples(X_train), 0.99))
    synthetic_features = X_test.copy()
    synthetic_features["appliances_current"] = synthetic_y
    isolation_synthetic_scores = -isolation_forest.score_samples(synthetic_features)
    anomaly_report = {
        "run_id": run_id,
        "type": "synthetic benchmark; no real anomaly or waste labels exist",
        "injected_count": int(labels.sum()),
        "injected_indices": indices.tolist(),
        "strength_std": 2.0,
        "residual_detector": synthetic_metrics(labels, residual_scores, threshold),
        "isolation_forest_detector": synthetic_metrics(
            labels, isolation_synthetic_scores, isolation_cut
        ),
        "residual_threshold": threshold,
        "isolation_forest_training_score_quantile_99": isolation_cut,
        "limitations": "Metrics apply only to the deterministic positive-spike injection design, not naturally occurring anomalies.",
    }
    _save_json(REPORTS / "metrics" / "anomaly_evaluation.json", anomaly_report)

    intervals = data.date.diff().dt.total_seconds().dropna() / 60
    target_interval_minutes = float(intervals.median())
    sampling_report = {
        "median_interval_minutes": target_interval_minutes,
        "minimum_interval_minutes": float(intervals.min()),
        "maximum_interval_minutes": float(intervals.max()),
        "unique_interval_count": int(intervals.nunique()),
        "regular_10_minute_sampling": bool(np.allclose(intervals.to_numpy(), 10.0)),
    }
    diagnostics = residual_diagnostics(
        y_test, selected_test, test_targets, test_origins
    )
    diagnostics["run_id"] = run_id
    diagnostics["target_horizon"] = "next observed row"
    diagnostics["median_sampling_interval_minutes"] = target_interval_minutes
    _save_json(REPORTS / "metrics" / "residual_diagnostics.json", diagnostics)
    _save_csv(
        REPORTS / "metrics" / "largest_test_errors.csv",
        pd.DataFrame(diagnostics["largest_errors"]),
    )

    if selected == "Voting (top 3)":

        def importance_predict(frame):
            return np.mean(
                [full_models[name].predict(frame) for name in best_names], axis=0
            )
    elif selected == "Temporal Stacking":

        def importance_predict(frame):
            return predict_stack(full_stack, frame)
    elif selected == "Persistence":

        def importance_predict(frame):
            return frame["appliances_current"].to_numpy()
    else:
        importance_predict = full_models[selected].predict
    importance = permutation_feature_importance(
        importance_predict, X_train, y_train, seed=SEED
    )
    importance["run_id"] = run_id
    importance["reference_split"] = "training"
    importance["selected_model"] = selected
    _save_csv(REPORTS / "metrics" / "feature_importance.csv", importance)

    predictions = pd.DataFrame(
        {
            "origin_date": test_origins.astype(str).to_numpy(),
            "target_date": test_targets.astype(str).to_numpy(),
            "date": test_targets.astype(str).to_numpy(),
            "observed": y_test.to_numpy(),
            "predicted": selected_test,
            "residual": y_test.to_numpy() - selected_test,
            "anomaly": np.abs(y_test.to_numpy() - selected_test) > threshold,
            "isolation_score": -isolation_forest.score_samples(X_test),
            "isolation_anomaly": -isolation_forest.score_samples(X_test)
            > isolation_cut,
            "selected_model": selected,
            "run_id": run_id,
            "dataset_fingerprint": fingerprint,
        }
    )
    _save_csv(REPORTS / "metrics" / "test_predictions.csv", predictions)

    final_metrics = {
        "run_id": run_id,
        "dataset_fingerprint": fingerprint,
        "selected_model": selected,
        "selection": selection_metadata,
        "test": {
            key: selected_test_metrics[key] for key in ("MAE", "RMSE", "R2", "n_obs")
        },
        "validation_selection": selected,
        "threshold": threshold,
        "cleaning": cleaning,
        "feature_count": X.shape[1],
        "train_rows": len(X_train),
        "validation_rows": len(X_valid),
        "test_rows": len(X_test),
        "target_horizon": "next observed row",
        "target_interval_median_minutes": target_interval_minutes,
        "sampling": sampling_report,
        "target_test_period": [str(test_targets.iloc[0]), str(test_targets.iloc[-1])],
        "tuning": tuning,
        "temporal_cv": cv_summaries,
        "validation_metrics": {
            row["model"]: {k: row[k] for k in ("MAE", "RMSE", "R2", "n_obs")}
            for row in selection_rows
        },
        "test_candidate_metrics": {
            row["model"]: {k: row[k] for k in ("MAE", "RMSE", "R2", "n_obs")}
            for row in test_rows
        },
        "residual_diagnostics_file": "residual_diagnostics.json",
    }
    _save_json(REPORTS / "metrics" / "final_metrics.json", final_metrics)
    _save_json(
        REPORTS / "metrics" / "best_hyperparameters.json",
        {
            "run_id": run_id,
            "tuned_model": "HistGradientBoostingRegressor",
            "dataset_fingerprint": fingerprint,
            "feature_columns": list(X.columns),
            "training_rows": len(X_train),
            "seed": SEED,
            "optimization": tuning,
            "selection_metric": "validation MAE; tuning objective is training-only expanding TimeSeriesSplit MAE",
        },
    )
    _save_json(
        REPORTS / "metrics" / "training_config.json",
        {
            "run_id": run_id,
            "dataset_fingerprint": fingerprint,
            "seed": SEED,
            "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "target": "next observed Appliances row",
            "target_horizon_rows": 1,
            "sampling": sampling_report,
            "feature_count": X.shape[1],
            "selected_model": selected,
            "selection": selection_metadata,
            "selected_feature_configuration": "all engineered features, one-step sensor/consumption changes, weekly cyclic time",
            "ablation_validation_results": ablation_rows,
            "temporal_cv": {
                name: {key: value for key, value in summary.items() if key != "folds"}
                for name, summary in cv_summaries.items()
            },
            "train_period": [
                str(origin_dates.loc[X_train.index].iloc[0]),
                str(origin_dates.loc[X_train.index].iloc[-1]),
            ],
            "validation_period": [
                str(valid_origins.iloc[0]),
                str(valid_origins.iloc[-1]),
            ],
            "test_period": [str(test_origins.iloc[0]), str(test_origins.iloc[-1])],
            "target_test_period": [
                str(test_targets.iloc[0]),
                str(test_targets.iloc[-1]),
            ],
        },
    )

    # The manifest is committed last; readers can reject any mixed/incomplete run.
    _save_json(
        REPORTS / "metrics" / "artifact_manifest.json",
        {
            "run_id": run_id,
            "dataset_fingerprint": fingerprint,
            "model_version": bundle["version"],
            "files": [
                "wattwise_bundle.joblib",
                "model_comparison.csv",
                "final_metrics.json",
                "training_config.json",
                "test_predictions.csv",
                "residual_diagnostics.json",
                "largest_test_errors.csv",
                "feature_ablation.csv",
                "selected_features.json",
                "feature_importance.csv",
                "anomaly_evaluation.json",
                "best_hyperparameters.json",
            ],
        },
    )
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.hist(data["Appliances"], bins=45, color="#2a9d8f")
        ax.set(
            title="Appliances consumption distribution",
            xlabel="Wh per observed interval",
            ylabel="Count",
        )
        fig.tight_layout()
        fig.savefig(REPORTS / "figures" / "appliances_distribution.png", dpi=150)
        plt.close(fig)
        residual = y_test.to_numpy() - selected_test
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(test_targets, y_test, label="Observed", lw=0.8)
        ax.plot(test_targets, selected_test, label="Forecast", lw=0.8)
        ax.set(
            title="Held-out observed and forecast consumption",
            xlabel="Target timestamp",
            ylabel="Wh per interval",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPORTS / "figures" / "test_forecast.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.hist(residual, bins=45, color="#e76f51")
        ax.axvline(
            threshold, color="black", ls="--", label="± validation residual threshold"
        )
        ax.axvline(-threshold, color="black", ls="--")
        ax.set(
            title="Held-out forecast residual distribution",
            xlabel="Observed − predicted (Wh)",
            ylabel="Count",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPORTS / "figures" / "residual_distribution.png", dpi=150)
        plt.close(fig)
    except ImportError:
        logging.info("Matplotlib is not installed; skipping optional static figures")

    logging.info(
        "Selected %s by validation MAE; final test metrics %s",
        selected,
        {key: selected_test_metrics[key] for key in ("MAE", "RMSE", "R2", "n_obs")},
    )
    return bundle


if __name__ == "__main__":
    train()
