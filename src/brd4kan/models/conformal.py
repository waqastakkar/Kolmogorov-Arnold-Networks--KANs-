"""Mondrian conformal prediction — per-scaffold calibrated intervals.

Given a calibration set with known scaffold labels, computes per-scaffold
nonconformity quantiles. At prediction time, returns [y_pred ± q_alpha]
intervals that satisfy marginal and conditional coverage guarantees.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class MondrianConformalPredictor:
    """Mondrian (group-conditional) split conformal predictor.

    Partition key defaults to scaffold class. Groups unseen at calibration
    time fall back to the global quantile.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._group_quantiles: dict[str, float] = {}
        self._global_quantile: float = 0.0

    def calibrate(
        self,
        residuals: np.ndarray,
        groups: list[str],
    ) -> None:
        """Fit quantiles from absolute residuals |y_true - y_pred| per group."""
        residuals = np.abs(np.asarray(residuals, dtype=np.float64))
        by_group: dict[str, list[float]] = defaultdict(list)
        for r, g in zip(residuals, groups):
            by_group[g].append(float(r))

        q_level = min(1.0, (1.0 - self.alpha) * (1 + 1))  # finite-sample correction n=1
        for g, vals in by_group.items():
            n = len(vals)
            q_level_g = min(1.0, (1.0 - self.alpha) * (n + 1) / n)
            self._group_quantiles[g] = float(np.quantile(vals, q_level_g))
        self._global_quantile = float(
            np.quantile(residuals, min(1.0, (1.0 - self.alpha) * (len(residuals) + 1) / len(residuals)))
        )

    def predict_intervals(
        self,
        y_pred: np.ndarray,
        groups: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) bounds per prediction."""
        y_pred = np.asarray(y_pred, dtype=np.float64)
        lower = np.empty_like(y_pred)
        upper = np.empty_like(y_pred)
        for i, (yp, g) in enumerate(zip(y_pred, groups)):
            q = self._group_quantiles.get(g, self._global_quantile)
            lower[i] = yp - q
            upper[i] = yp + q
        return lower, upper

    def coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: list[str],
    ) -> dict[str, float]:
        """Empirical coverage overall and per group."""
        lower, upper = self.predict_intervals(y_pred, groups)
        y_true = np.asarray(y_true, dtype=np.float64)
        covered = (y_true >= lower) & (y_true <= upper)
        result: dict[str, float] = {"overall": float(covered.mean())}

        by_group: dict[str, list[bool]] = defaultdict(list)
        for c, g in zip(covered, groups):
            by_group[g].append(bool(c))
        for g, vals in by_group.items():
            result[f"group_{g}"] = float(np.mean(vals))
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "group_quantiles": dict(self._group_quantiles),
            "global_quantile": self._global_quantile,
        }

    @classmethod
    def from_state_dict(cls, d: dict[str, Any]) -> "MondrianConformalPredictor":
        obj = cls(alpha=d["alpha"])
        obj._group_quantiles = d["group_quantiles"]
        obj._global_quantile = d["global_quantile"]
        return obj
