"""Baseline model wrappers: RF, XGBoost, SVR, parameter-matched MLP.

Every wrapper exposes a uniform interface:

    model = create_model(model_name, hparams, seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

Plus ``suggest_hparams(trial)`` for Optuna, and ``save / load`` via joblib.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BaselineModel(Protocol):
    """Protocol that every baseline wrapper satisfies."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Optuna search spaces (one per model type)
# ---------------------------------------------------------------------------


def suggest_rf(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
    }


def suggest_xgboost(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


def suggest_svr(trial: Any) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-2, 100.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


def suggest_mlp(trial: Any) -> dict[str, Any]:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f"layer_{i}_size", 32, 256, step=32))
    return {
        "hidden_layer_sizes": tuple(layers),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
    }


SUGGEST_FNS: dict[str, Any] = {
    "rf": suggest_rf,
    "xgboost": suggest_xgboost,
    "svr": suggest_svr,
    "mlp": suggest_mlp,
}


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------


def create_model(name: str, hparams: dict[str, Any], seed: int) -> Any:
    """Instantiate a scikit-learn-compatible baseline model."""
    if name == "rf":
        return RandomForestRegressor(
            random_state=seed, n_jobs=-1, **hparams
        )
    if name == "xgboost":
        import xgboost as xgb  # type: ignore[import-not-found]

        return xgb.XGBRegressor(
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            **hparams,
        )
    if name == "svr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(**hparams)),
        ])
    if name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    random_state=seed,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    **hparams,
                ),
            ),
        ])
    raise ValueError(f"Unknown baseline model: {name}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    return joblib.load(path)
