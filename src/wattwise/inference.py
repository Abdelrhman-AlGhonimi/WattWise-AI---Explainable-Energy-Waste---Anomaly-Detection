from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from .ensemble import predict_stack


def load_bundle(path):
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load WattWise model bundle at {path}: {exc}"
        ) from exc


def predict_next(bundle, features: pd.DataFrame):
    expected = bundle["feature_columns"]
    missing = set(expected) - set(features.columns)
    if missing:
        raise ValueError(f"Missing required feature(s): {sorted(missing)}")
    if not features.columns.is_unique:
        raise ValueError("Input features contain duplicate column names")
    ordered = features[expected]
    values = ordered.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Input features must contain only finite numeric values")
    model = bundle.get("model")
    if model is None:
        if "appliances_current" not in ordered:
            raise ValueError("Persistence artifact requires appliances_current")
        predictions = ordered["appliances_current"].to_numpy()
    elif isinstance(model, dict) and model.get("kind") == "voting":
        components = model.get("models", {})
        if not components:
            raise ValueError("Voting artifact has no component models")
        weights = model.get(
            "weights", {name: 1 / len(components) for name in components}
        )
        if set(weights) != set(components) or any(
            not np.isfinite(value) or value < 0 for value in weights.values()
        ):
            raise ValueError(
                "Voting artifact weights must be finite, nonnegative, and match its models"
            )
        total_weight = float(sum(weights.values()))
        if total_weight <= 0:
            raise ValueError("Voting artifact weights must sum to a positive value")
        predictions = (
            sum(
                float(weights[name])
                * np.asarray(component.predict(ordered), dtype=float).reshape(-1)
                for name, component in components.items()
            )
            / total_weight
        )
    elif isinstance(model, dict) and model.get("kind") == "stack":
        predictions = predict_stack(model["stack"], ordered)
    else:
        predictions = model.predict(ordered)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if len(predictions) != len(ordered) or not np.isfinite(predictions).all():
        raise ValueError("Model returned invalid or misaligned predictions")
    return predictions
