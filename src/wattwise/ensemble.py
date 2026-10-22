from __future__ import annotations
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge


def walk_forward_stack(base_models, X, y, splits=4):
    """Temporal out-of-fold predictions; each meta row predicted only by past-trained estimators."""
    n = len(X)
    chunks = np.array_split(np.arange(n), splits + 1)
    meta = np.full((n, len(base_models)), np.nan)
    for i in range(1, len(chunks)):
        train_idx = np.concatenate(chunks[:i])
        val_idx = chunks[i]
        if len(train_idx) < 10 or not len(val_idx):
            continue
        for j, (_, est) in enumerate(base_models.items()):
            m = clone(est).fit(X.iloc[train_idx], y.iloc[train_idx])
            meta[val_idx, j] = m.predict(X.iloc[val_idx])
    valid = np.isfinite(meta).all(axis=1)
    return meta[valid], y.iloc[np.where(valid)[0]], valid


def fit_stack(base_models, X, y, splits=4):
    meta, target, mask = walk_forward_stack(base_models, X, y, splits)
    if len(meta) < 10:
        raise ValueError("Insufficient walk-forward predictions for stacking")
    learner = Ridge(alpha=1.0).fit(meta, target)
    fitted = {k: clone(v).fit(X, y) for k, v in base_models.items()}
    return {"base_models": fitted, "meta_model": learner}


def predict_stack(stack, X):
    return stack["meta_model"].predict(
        np.column_stack([m.predict(X) for m in stack["base_models"].values()])
    )
