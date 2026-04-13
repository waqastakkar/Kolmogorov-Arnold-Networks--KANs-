"""Tests for train/metrics.py — regression + derived classification metrics."""

from __future__ import annotations

import numpy as np
import pytest

from brd4kan.train.metrics import regression_metrics


def test_perfect_predictions() -> None:
    y = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    m = regression_metrics(y, y, active_threshold=6.5)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["spearman_rho"] == pytest.approx(1.0)
    assert m["pearson_r"] == pytest.approx(1.0)


def test_imperfect_predictions_have_positive_rmse() -> None:
    y_true = np.array([5.0, 6.0, 7.0, 8.0])
    y_pred = np.array([5.5, 5.5, 7.5, 7.5])
    m = regression_metrics(y_true, y_pred, active_threshold=6.5)
    assert m["rmse"] > 0.0
    assert 0 < m["r2"] < 1.0
    assert m["ece"] >= 0.0


def test_binary_metrics_are_nan_when_single_class() -> None:
    y_true = np.array([8.0, 9.0, 10.0])  # all active
    y_pred = np.array([8.1, 9.1, 10.1])
    m = regression_metrics(y_true, y_pred, active_threshold=6.5)
    assert np.isnan(m["roc_auc"])


def test_expected_metric_keys() -> None:
    y = np.array([5.0, 6.0, 7.0, 8.0])
    m = regression_metrics(y, y, active_threshold=6.5)
    expected_keys = {
        "rmse", "mae", "r2", "spearman_rho", "pearson_r",
        "roc_auc", "pr_auc", "mcc", "brier", "ece",
    }
    assert expected_keys == set(m.keys())
