from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error


def feature_importance(model, X, y=None, seed=42):
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif y is not None:
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=5,
            random_state=seed,
            n_jobs=-1,
            scoring="neg_mean_absolute_error",
        )
        vals = result.importances_mean
    else:
        raise ValueError(
            "Model has no built-in feature importances; provide y for permutation importance"
        )
    return pd.DataFrame({"feature": X.columns, "importance": vals}).sort_values(
        "importance", ascending=False
    )


def local_shap_values(predict_fn, background, instance):
    """Return a model-agnostic local SHAP explanation, or an actionable fallback message."""
    try:
        import shap

        background = background.sample(min(20, len(background)), random_state=42)
        explainer = shap.Explainer(predict_fn, background, algorithm="permutation")
        explanation = explainer(instance, max_evals=2 * instance.shape[1] + 1)
        values = explanation.values[0]
        return pd.DataFrame(
            {"feature": instance.columns, "contribution": values}
        ).assign(abs_contribution=lambda d: d.contribution.abs()).sort_values(
            "abs_contribution", ascending=False
        ), None
    except Exception as exc:
        return (
            None,
            f"SHAP is unavailable for this estimator in the current environment: {exc}",
        )


def permutation_feature_importance(predict_fn, X, y, n_repeats=5, seed=42):
    """Permutation MAE increase for the exact predictor, using validation only."""
    if len(X) != len(y) or len(X) < 2:
        raise ValueError("Permutation importance requires aligned validation rows")
    baseline = float(mean_absolute_error(y, predict_fn(X)))
    rng = np.random.default_rng(seed)
    rows = []
    for column in X.columns:
        increases = []
        for _ in range(n_repeats):
            permuted = X.copy()
            permuted[column] = rng.permutation(permuted[column].to_numpy())
            increases.append(
                float(mean_absolute_error(y, predict_fn(permuted))) - baseline
            )
        rows.append(
            {
                "feature": column,
                "importance": float(np.mean(increases)),
                "importance_std": float(np.std(increases)),
                "baseline_mae": baseline,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
