import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit


def regression_metrics(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.size == 0:
        raise ValueError("Regression metrics require at least one aligned observation")
    if actual.shape != predicted.shape:
        raise ValueError(
            f"Target/prediction length mismatch: {actual.size} != {predicted.size}"
        )
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Regression metrics require finite targets and predictions")
    r2 = (
        None
        if actual.size < 2 or np.all(actual == actual[0])
        else float(r2_score(actual, predicted))
    )
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": r2,
        "n_obs": int(actual.size),
    }


def temporal_cv_summary(estimator, X, y, n_splits=3):
    """Expanding-window scores fitted only within the supplied training period."""
    folds = []
    for fold, (train_idx, val_idx) in enumerate(
        TimeSeriesSplit(n_splits=n_splits).split(X), start=1
    ):
        fitted = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        scores = regression_metrics(y.iloc[val_idx], fitted.predict(X.iloc[val_idx]))
        folds.append(
            {
                "fold": fold,
                **scores,
                "train_end_index": int(train_idx[-1]),
                "validation_start_index": int(val_idx[0]),
                "R2_status": "undefined: fewer than two observations or constant target"
                if scores["R2"] is None
                else "valid",
            }
        )
    metrics = pd.DataFrame(folds)
    valid_r2 = metrics.R2.dropna()
    return {
        "CV_MAE": float(metrics.MAE.mean()),
        "CV_MAE_std": float(metrics.MAE.std(ddof=0)),
        "CV_RMSE": float(metrics.RMSE.mean()),
        "CV_RMSE_std": float(metrics.RMSE.std(ddof=0)),
        "CV_R2": float(valid_r2.mean()) if not valid_r2.empty else None,
        "CV_R2_std": float(valid_r2.std(ddof=0)) if not valid_r2.empty else None,
        "CV_R2_valid_folds": int(valid_r2.size),
        "n_folds": int(len(folds)),
        "mean_fold_n_obs": float(metrics.n_obs.mean()),
        "folds": folds,
    }


def select_model(validation_rows):
    """Select deterministically using validation only; test rows are rejected."""
    rows = [row for row in validation_rows if row.get("split") == "validation"]
    if not rows:
        raise ValueError("Model selection requires at least one validation result")
    if any(
        not np.isfinite([row.get("MAE", np.nan), row.get("RMSE", np.nan)]).all()
        for row in rows
    ):
        raise ValueError("Validation selection metrics must be finite")
    winner = min(rows, key=lambda row: (row["MAE"], row["RMSE"], row["model"]))
    return {
        "selected_model": winner["model"],
        "metric": "validation MAE",
        "score": float(winner["MAE"]),
        "tie_breaking": [
            "lower validation MAE",
            "then lower validation RMSE",
            "then model name ascending",
        ],
        "rationale": "Minimum validation MAE among the declared candidates; the held-out test was not used for selection.",
    }


def residual_diagnostics(y_true, y_pred, target_dates, origin_dates=None, top_n=20):
    """Summarize residual tails, chronology, target ranges and worst misses."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    dates = pd.to_datetime(pd.Series(target_dates)).reset_index(drop=True)
    if actual.shape != predicted.shape or len(actual) != len(dates):
        raise ValueError(
            "Residual diagnostics require aligned targets, predictions, and timestamps"
        )
    residual = actual - predicted
    absolute = np.abs(residual)
    quantiles = {
        f"absolute_error_p{q}": float(np.quantile(absolute, q / 100))
        for q in (50, 90, 95, 99)
    }
    quantiles.update(
        {
            f"signed_residual_p{q}": float(np.quantile(residual, q / 100))
            for q in (50, 90, 95, 99)
        }
    )
    frame = pd.DataFrame(
        {
            "target_date": dates,
            "observed": actual,
            "predicted": predicted,
            "residual": residual,
            "absolute_error": absolute,
        }
    )
    if origin_dates is not None:
        origins = pd.to_datetime(pd.Series(origin_dates)).reset_index(drop=True)
        if len(origins) != len(frame):
            raise ValueError("Origin and target timestamp lengths differ")
        frame.insert(0, "origin_date", origins)
    largest = frame.nlargest(min(top_n, len(frame)), "absolute_error")
    frame["period"] = "period_1"
    frame.loc[frame.index >= len(frame) // 3, "period"] = "period_2"
    frame.loc[frame.index >= 2 * len(frame) // 3, "period"] = "period_3"
    by_period = []
    for name, group in frame.groupby("period", sort=True):
        by_period.append(
            {"period": name, **regression_metrics(group.observed, group.predicted)}
        )
    try:
        frame["target_range"] = pd.qcut(
            frame.observed, q=min(5, frame.observed.nunique()), duplicates="drop"
        )
        by_target = [
            {
                "target_range": str(name),
                **regression_metrics(group.observed, group.predicted),
            }
            for name, group in frame.groupby("target_range", observed=True)
        ]
    except ValueError:
        by_target = []
    return {
        "n_obs": int(len(frame)),
        "quantiles": quantiles,
        "by_time_block": by_period,
        "by_target_range": by_target,
        "largest_errors": largest.to_dict(orient="records"),
    }
