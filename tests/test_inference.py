import pandas as pd
import pytest
import joblib
import numpy as np
from wattwise.inference import predict_next, load_bundle


class Model:
    def predict(self, X):
        return [42.0] * len(X)


def test_inference_schema_validation():
    b = {"model": Model(), "feature_columns": ["a"]}
    assert predict_next(b, pd.DataFrame({"a": [1]}))[0] == 42
    with pytest.raises(ValueError):
        predict_next(b, pd.DataFrame({"b": [1]}))


def test_bad_artifact_error(tmp_path):
    with pytest.raises(RuntimeError):
        load_bundle(tmp_path / "missing.joblib")


def test_valid_model_bundle_round_trip(tmp_path):
    path = tmp_path / "bundle.joblib"
    original = {"model": Model(), "feature_columns": ["a"], "selected_model": "test"}
    joblib.dump(original, path)
    loaded = load_bundle(path)
    assert predict_next(loaded, pd.DataFrame({"a": [0, 1]})).tolist() == [42.0, 42.0]


def test_inference_rejects_nonfinite_predictions():
    class BadModel:
        def predict(self, X):
            return np.full(len(X), np.inf)

    with pytest.raises(ValueError, match="invalid or misaligned"):
        predict_next(
            {"model": BadModel(), "feature_columns": ["a"]}, pd.DataFrame({"a": [1.0]})
        )


def test_inference_rejects_nonfinite_input_and_prediction():
    b = {"model": Model(), "feature_columns": ["a"]}
    with pytest.raises(ValueError, match="finite"):
        predict_next(b, pd.DataFrame({"a": [float("nan")]}))


def test_persistence_and_voting_artifacts_are_supported():
    features = pd.DataFrame({"a": [1.0, 2.0], "appliances_current": [7.0, 9.0]})
    persistence = {"model": None, "feature_columns": ["a", "appliances_current"]}
    assert predict_next(persistence, features).tolist() == [7.0, 9.0]
    voting = {
        "model": {"kind": "voting", "models": {"one": Model(), "two": Model()}},
        "feature_columns": ["a", "appliances_current"],
    }
    assert predict_next(voting, features).tolist() == [42.0, 42.0]
