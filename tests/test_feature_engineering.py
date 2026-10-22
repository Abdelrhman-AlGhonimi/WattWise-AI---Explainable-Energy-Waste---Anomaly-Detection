import pandas as pd
import numpy as np
import pytest
from wattwise.feature_engineering import make_features, chronological_split
from wattwise.train import _assert_target_boundaries


def test_next_target_and_past_only_lag():
    d = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="10min"),
            "Appliances": range(10),
            "lights": [0] * 10,
        }
    )
    X, y, dates = make_features(d, lags=(1,), rolling=(2,))
    assert y.iloc[0] == 3 and X.iloc[0].appliances_current == 2
    assert X.iloc[0].appliances_lag_1 == 1
    assert dates.iloc[-1] < d.date.iloc[-1]


def test_temporal_split_order():
    d = pd.DataFrame(
        {
            "date": pd.date_range("2020", periods=30, freq="10min"),
            "Appliances": range(30),
            "lights": [0] * 30,
        }
    )
    X, y, t = make_features(d, lags=(1,), rolling=(2,))
    s = chronological_split(X, y, t)
    assert s["train"][2].max() < s["validation"][2].min() < s["test"][2].min()


def test_future_observation_cannot_change_earlier_features():
    dates = pd.date_range("2020-01-01", periods=8, freq="10min")
    base = pd.DataFrame(
        {
            "date": dates,
            "Appliances": np.arange(8, dtype=float),
            "lights": np.arange(8, dtype=float),
        }
    )
    altered = base.copy()
    altered.loc[7, "Appliances"] = 700.0
    altered.loc[7, "lights"] = -100.0
    X1, y1, _ = make_features(base, lags=(1,), rolling=(2,))
    X2, y2, _ = make_features(altered, lags=(1,), rolling=(2,))
    assert X1.iloc[:-1].equals(X2.iloc[:-1])
    assert y1.iloc[-1] != y2.iloc[-1]


def test_target_timestamps_are_disjoint_across_splits():
    d = pd.DataFrame(
        {
            "date": pd.date_range("2020", periods=40, freq="10min"),
            "Appliances": range(40),
            "lights": [0] * 40,
        }
    )
    X, y, dates = make_features(d, lags=(1,), rolling=(2,))
    parts = chronological_split(X, y, dates)
    targets = pd.Series(d.date.iloc[1:].to_numpy(), index=np.arange(len(d) - 1))
    _assert_target_boundaries(parts, targets)
    overlap_index = parts["train"][0].index[-1]
    broken = {
        "train": parts["train"],
        "validation": (
            X.loc[[overlap_index]],
            y.loc[[overlap_index]],
            dates.loc[[overlap_index]],
        ),
        "test": parts["test"],
    }
    with pytest.raises(ValueError, match="overlap"):
        _assert_target_boundaries(broken, targets)
