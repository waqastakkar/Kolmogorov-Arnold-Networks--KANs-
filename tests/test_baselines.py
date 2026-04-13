"""Tests for Stage 5 baseline model wrappers and Optuna integration.

Uses tiny synthetic arrays so the tests run fast (seconds, not minutes).
Chemprop tests are skipped if the package is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brd4kan.models.baselines import (
    SUGGEST_FNS,
    create_model,
    load_model,
    save_model,
)
from brd4kan.train.metrics import regression_metrics

rdkit = pytest.importorskip("rdkit")

from brd4kan.train.cv import scaffold_cv_indices  # noqa: E402


TRAIN_SMILES = [
    "c1ccccc1", "c1ccc(O)cc1", "c1ccc(N)cc1",
    "c1ccc2ccccc2c1", "c1ccc2cc(O)ccc2c1",
    "c1ccncc1", "c1cc(C)ncc1", "C1CCNCC1",
    "Cc1cc(C)c2c(c1)C(=O)NC2=O",
    "CN1CCN(c2ccccc2)CC1",
]


@pytest.fixture()
def synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    n_train, n_test, n_feat = 50, 15, 20
    X_train = rng.randn(n_train, n_feat).astype(np.float32)
    y_train = X_train[:, 0] * 2.0 + rng.randn(n_train) * 0.1 + 6.0
    X_test = rng.randn(n_test, n_feat).astype(np.float32)
    y_test = X_test[:, 0] * 2.0 + rng.randn(n_test) * 0.1 + 6.0
    return X_train, y_train, X_test, y_test


@pytest.mark.parametrize("model_name", ["rf", "xgboost", "svr", "mlp"])
def test_create_and_predict(
    model_name: str, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> None:
    X_train, y_train, X_test, _ = synthetic_data
    hparams: dict = {}
    if model_name == "rf":
        hparams = {"n_estimators": 10, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"}
    elif model_name == "xgboost":
        hparams = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 1e-5, "reg_lambda": 1.0, "min_child_weight": 1}
    elif model_name == "svr":
        hparams = {"C": 1.0, "epsilon": 0.1, "kernel": "rbf", "gamma": "scale"}
    elif model_name == "mlp":
        hparams = {"hidden_layer_sizes": (32,), "learning_rate_init": 1e-3, "alpha": 1e-4, "batch_size": 32, "activation": "relu"}

    model = create_model(model_name, hparams, seed=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (X_test.shape[0],)
    assert np.isfinite(preds).all()


@pytest.mark.parametrize("model_name", ["rf", "xgboost", "svr", "mlp"])
def test_save_load_round_trip(
    model_name: str,
    synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    X_train, y_train, X_test, _ = synthetic_data
    hparams: dict = {}
    if model_name == "rf":
        hparams = {"n_estimators": 10, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"}
    elif model_name == "xgboost":
        hparams = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 1e-5, "reg_lambda": 1.0, "min_child_weight": 1}
    elif model_name == "svr":
        hparams = {"C": 1.0, "epsilon": 0.1, "kernel": "rbf", "gamma": "scale"}
    elif model_name == "mlp":
        hparams = {"hidden_layer_sizes": (32,), "learning_rate_init": 1e-3, "alpha": 1e-4, "batch_size": 32, "activation": "relu"}

    model = create_model(model_name, hparams, seed=42)
    model.fit(X_train, y_train)
    preds_before = model.predict(X_test)

    path = tmp_path / f"{model_name}.joblib"
    save_model(model, path)
    loaded = load_model(path)
    preds_after = loaded.predict(X_test)
    np.testing.assert_array_almost_equal(preds_before, preds_after)


def test_suggest_fns_cover_all_descriptor_models() -> None:
    assert set(SUGGEST_FNS.keys()) == {"rf", "xgboost", "svr", "mlp"}


def test_scaffold_cv_indices_no_leakage() -> None:
    folds = scaffold_cv_indices(TRAIN_SMILES, n_folds=3)
    assert len(folds) == 3
    from brd4kan.data.split import bemis_murcko_scaffold

    for train_idx, val_idx in folds:
        assert len(set(train_idx) & set(val_idx)) == 0, "leak in CV fold"
        train_scaffolds = {bemis_murcko_scaffold(TRAIN_SMILES[i]) or f"s{i}" for i in train_idx}
        val_scaffolds = {bemis_murcko_scaffold(TRAIN_SMILES[i]) or f"s{i}" for i in val_idx}
        assert not (train_scaffolds & val_scaffolds), "scaffold leak in CV"


def test_create_model_unknown_raises() -> None:
    with pytest.raises(ValueError):
        create_model("nonexistent_model", {}, 42)


def test_regression_metrics_from_model_output(
    synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X_train, y_train, X_test, y_test = synthetic_data
    model = create_model("rf", {
        "n_estimators": 50, "max_depth": 10,
        "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"
    }, seed=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    m = regression_metrics(y_test, preds)
    assert m["rmse"] > 0
    assert -1.0 <= m["r2"] <= 1.0
