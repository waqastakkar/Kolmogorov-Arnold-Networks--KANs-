"""Bootstrap confidence intervals for regression metrics.

Resamples (y_true, y_pred) pairs ``n_iters`` times and computes the full
metric set per sample, returning per-metric 95% CIs.
"""

from __future__ import annotations

import numpy as np

from brd4kan.train.metrics import regression_metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iters: int = 1000,
    ci: float = 0.95,
    active_threshold: float = 6.5,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Return ``{metric: {mean, lo, hi, std}}`` from bootstrap resampling."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    n = len(y_true)

    all_metrics: dict[str, list[float]] = {}
    for _ in range(n_iters):
        idx = rng.randint(0, n, size=n)
        m = regression_metrics(y_true[idx], y_pred[idx], active_threshold=active_threshold)
        for k, v in m.items():
            all_metrics.setdefault(k, []).append(float(v))

    alpha = (1.0 - ci) / 2.0
    result: dict[str, dict[str, float]] = {}
    for k, vals in all_metrics.items():
        arr = np.array(vals)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            result[k] = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "std": float("nan")}
        else:
            result[k] = {
                "mean": float(np.mean(valid)),
                "lo": float(np.percentile(valid, alpha * 100)),
                "hi": float(np.percentile(valid, (1 - alpha) * 100)),
                "std": float(np.std(valid)),
            }
    return result
