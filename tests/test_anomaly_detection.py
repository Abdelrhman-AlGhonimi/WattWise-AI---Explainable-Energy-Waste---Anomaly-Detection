import numpy as np
import pytest
from wattwise.anomaly_detection import (
    residual_threshold,
    residual_flags,
    inject_spikes,
    synthetic_metrics,
)


def test_residual_threshold_flags_signs():
    t = residual_threshold([1, -1, 2, -2], 0.75)
    f = residual_flags([15, 5], [10, 10], t)
    assert f["high_consumption"][0] and f["flag"][1]


def test_synthetic_labels_reproducible():
    source = np.ones(100)
    a, labels, idx = inject_spikes(source, count=5, seed=7)
    b, l2, idx2 = inject_spikes(np.ones(100), count=5, seed=7)
    assert (
        np.array_equal(idx, idx2)
        and np.array_equal(labels, l2)
        and np.array_equal(a, b)
    )
    assert synthetic_metrics(labels, labels, 0.5)["recall"] == 1
    assert np.all(source == 1), (
        "Synthetic injection must not mutate the original evaluation target"
    )


def test_synthetic_metrics_uses_continuous_scores_for_pr_auc():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    result = synthetic_metrics(labels, scores, 0.5)
    assert result["pr_auc"] == 1.0 and result["recall"] == 1
    with pytest.raises(ValueError, match="aligned"):
        synthetic_metrics(labels, scores[:2], 0.5)
