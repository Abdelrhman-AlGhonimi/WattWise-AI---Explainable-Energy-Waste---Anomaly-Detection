import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from wattwise.ensemble import fit_stack, predict_stack, walk_forward_stack


def test_walk_forward_meta_rows_only_follow_training_history():
    X = pd.DataFrame({"x": np.arange(60, dtype=float)})
    y = pd.Series(np.arange(60, dtype=float) * 2)
    meta, target, valid = walk_forward_stack({"mean": DummyRegressor()}, X, y, splits=4)
    assert valid.shape == (60,)
    assert not valid[:12].any()
    assert valid[12:].all()
    assert len(meta) == len(target) == int(valid.sum())
    assert np.array_equal(target.to_numpy(), y.to_numpy()[valid])


def test_fitted_stack_predicts_ordered_rows():
    X = pd.DataFrame({"x": np.arange(80, dtype=float)})
    y = pd.Series(np.arange(80, dtype=float) * 2 + 1)
    model = fit_stack({"mean": DummyRegressor()}, X.iloc[:60], y.iloc[:60], splits=4)
    prediction = predict_stack(model, X.iloc[60:])
    assert prediction.shape == (20,)
    assert np.isfinite(prediction).all()
