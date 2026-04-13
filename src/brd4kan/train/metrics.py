"""Shared regression + classification metrics used by baselines and KAN.

All metric functions accept numpy arrays and return plain Python dicts so
they serialize to JSON / MLflow without conversion.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss,
)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    active_threshold: float = 6.5,
) -> dict[str, float]:
    """Compute the full PLAN.md §2 Stage-8 metric set.

    Returns dict with: rmse, mae, r2, spearman_rho, pearson_r,
    roc_auc, pr_auc, mcc, brier, ece.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    sp = stats.spearmanr(y_true, y_pred)
    spearman_rho = float(sp.statistic) if not np.isnan(sp.statistic) else 0.0
    pr = stats.pearsonr(y_true, y_pred)
    pearson_r = float(pr.statistic) if not np.isnan(pr.statistic) else 0.0

    # Classification metrics derived from the active threshold
    y_cls_true = (y_true >= active_threshold).astype(int)
    y_cls_pred_prob = _sigmoid((y_pred - active_threshold) * 2.0)
    y_cls_pred = (y_pred >= active_threshold).astype(int)

    n_classes = len(np.unique(y_cls_true))
    if n_classes < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_cls_true, y_cls_pred_prob))
        pr_auc = float(average_precision_score(y_cls_true, y_cls_pred_prob))

    mcc = float(matthews_corrcoef(y_cls_true, y_cls_pred))
    brier = float(brier_score_loss(y_cls_true, y_cls_pred_prob))
    ece = _expected_calibration_error(y_cls_true, y_cls_pred_prob)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman_rho": spearman_rho,
        "pearson_r": pearson_r,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": mcc,
        "brier": brier,
        "ece": ece,
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute ECE with equal-width binning."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if not mask.any():
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += mask.sum() * abs(acc - conf)
    return float(ece / max(len(y_true), 1))
