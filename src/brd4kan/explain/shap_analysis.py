"""SHAP analysis for KAN and sklearn baselines.

Uses KernelSHAP for the KAN (black-box) and TreeSHAP for RF/XGBoost.
Produces SHAP value matrices that the figure generators consume.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _import_shap() -> Any:
    try:
        import shap  # type: ignore[import-not-found]
        return shap
    except ImportError:
        raise ImportError("shap is required for Stage 8: pip install shap")


def compute_shap_tree(
    model: Any,
    X_test: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """TreeSHAP for tree-based models (RF, XGBoost)."""
    shap = _import_shap()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return {
        "shap_values": np.asarray(shap_values),
        "expected_value": float(explainer.expected_value),
        "feature_names": feature_names,
    }


def compute_shap_kernel(
    predict_fn: Any,
    X_background: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str] | None = None,
    n_background: int = 100,
) -> dict[str, Any]:
    """KernelSHAP for black-box models (KAN, SVR, MLP)."""
    shap = _import_shap()
    if len(X_background) > n_background:
        idx = np.random.RandomState(42).choice(len(X_background), n_background, replace=False)
        X_bg = X_background[idx]
    else:
        X_bg = X_background
    explainer = shap.KernelExplainer(predict_fn, X_bg)
    shap_values = explainer.shap_values(X_test, nsamples=200)
    return {
        "shap_values": np.asarray(shap_values),
        "expected_value": float(explainer.expected_value),
        "feature_names": feature_names,
    }


def save_shap_values(shap_result: dict[str, Any], path: Path) -> None:
    """Save SHAP values as .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        shap_values=shap_result["shap_values"],
        expected_value=np.array([shap_result["expected_value"]]),
    )


def load_shap_values(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {
        "shap_values": data["shap_values"],
        "expected_value": float(data["expected_value"][0]),
    }
