from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score,
)


def residual_threshold(residuals, quantile=0.99):
    vals = np.abs(np.asarray(residuals, dtype=float))
    return float(np.quantile(vals, quantile))


def residual_flags(y, pred, threshold):
    residual = np.asarray(y) - np.asarray(pred)
    return {
        "residual": residual,
        "absolute_residual": np.abs(residual),
        "flag": np.abs(residual) > threshold,
        "high_consumption": residual > threshold,
    }


def fit_isolation_forest(X, contamination=0.01, seed=42):
    model = IsolationForest(
        contamination=contamination, random_state=seed, n_estimators=200
    ).fit(X)
    return model


def inject_spikes(y, count=60, strength=2.0, seed=42):
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    count = min(count, len(y))
    idx = np.sort(rng.choice(len(y), count, replace=False))
    altered = y.copy()
    delta = max(float(np.std(y)), 1.0) * strength
    altered[idx] = np.maximum(0, altered[idx] + delta)
    labels = np.zeros(len(y), dtype=int)
    labels[idx] = 1
    return altered, labels, idx


def synthetic_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError(
            "Synthetic anomaly labels and continuous scores must be aligned and finite"
        )
    pred = scores > threshold
    p, r, f, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0
    )
    out = {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "confusion_matrix": confusion_matrix(labels, pred).tolist(),
    }
    if len(np.unique(labels)) == 2:
        out["pr_auc"] = float(average_precision_score(labels, scores))
    out["threshold"] = float(threshold)
    return out
