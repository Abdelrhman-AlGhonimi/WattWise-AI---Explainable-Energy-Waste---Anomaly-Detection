import json
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from wattwise.evaluation import (
    regression_metrics,
    residual_diagnostics,
    select_model,
    temporal_cv_summary,
)
from wattwise.train import _save_json
from wattwise.explainability import permutation_feature_importance


def test_metrics():
    m = regression_metrics([1, 2, 3], [1, 2, 4])
    assert m["MAE"] == 1 / 3 and round(m["RMSE"], 4) == round((1 / 3) ** 0.5, 4)
    assert m["n_obs"] == 3 and m["R2"] == 0.5


def test_metrics_validate_alignment_and_nonfinite_values():
    with pytest.raises(ValueError, match="length mismatch"):
        regression_metrics([1, 2], [1])
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([1, np.nan], [1, 2])


def test_constant_target_has_explicit_undefined_r2():
    result = regression_metrics([4, 4, 4], [3, 4, 5])
    assert result["MAE"] == pytest.approx(2 / 3)
    assert result["R2"] is None


def test_temporal_cv_aggregation_and_serialization():
    X = pd.DataFrame({"x": np.arange(30, dtype=float)})
    y = pd.Series(np.arange(30, dtype=float))
    result = temporal_cv_summary(DummyRegressor(strategy="mean"), X, y, n_splits=3)
    assert result["n_folds"] == 3 and result["CV_MAE"] > 0
    assert result["CV_MAE_std"] >= 0 and result["CV_R2_valid_folds"] == 3
    decoded = json.loads(json.dumps(result, allow_nan=False))
    assert decoded["folds"][0]["fold"] == 1
    assert result["CV_MAE"] == pytest.approx(
        np.mean([fold["MAE"] for fold in result["folds"]])
    )
    assert result["CV_RMSE_std"] == pytest.approx(
        np.std([fold["RMSE"] for fold in result["folds"]])
    )


def test_atomic_json_artifact_persistence(tmp_path):
    path = tmp_path / "metrics.json"
    _save_json(path, {"run_id": "test-run", "score": 1.25})
    assert json.loads(path.read_text()) == {"run_id": "test-run", "score": 1.25}
    assert not list(tmp_path.glob(".*.json"))


def test_selection_is_validation_only_and_deterministic():
    rows = [
        {"model": "z", "split": "validation", "MAE": 2.0, "RMSE": 3.0},
        {"model": "a", "split": "validation", "MAE": 2.0, "RMSE": 2.0},
        {"model": "test-leak", "split": "test", "MAE": 0.0, "RMSE": 0.0},
    ]
    assert select_model(rows)["selected_model"] == "a"
    with pytest.raises(ValueError, match="validation result"):
        select_model([{"model": "test-only", "split": "test", "MAE": 0.0, "RMSE": 0.0}])


def test_residual_diagnostics_quantiles_and_timestamp_alignment():
    dates = pd.date_range("2020-01-01", periods=12, freq="10min")
    result = residual_diagnostics(
        np.arange(12),
        np.arange(12) + np.tile([0, 1], 6),
        dates,
        dates - pd.Timedelta(minutes=10),
        top_n=3,
    )
    assert result["n_obs"] == 12
    assert result["quantiles"]["absolute_error_p99"] == 1
    assert len(result["largest_errors"]) == 3
    with pytest.raises(ValueError, match="aligned"):
        residual_diagnostics([1, 2], [1], [dates[0], dates[1]])


def test_permutation_importance_preserves_selected_predictor_feature_names():
    X = pd.DataFrame({"signal": np.arange(20, dtype=float), "noise": np.ones(20)})
    y = 2 * X.signal
    importance = permutation_feature_importance(
        lambda rows: 2 * rows["signal"].to_numpy(), X, y, n_repeats=4, seed=3
    )
    assert importance.feature.iloc[0] == "signal"
    assert set(importance.feature) == set(X.columns)
