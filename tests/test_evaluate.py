"""Tests for Stage 8 components: bootstrap, AD, figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brd4kan.train.bootstrap import bootstrap_ci
from brd4kan.train.applicability import ApplicabilityDomain


# ---- Bootstrap ----

def test_bootstrap_ci_returns_all_metric_keys() -> None:
    y_true = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    y_pred = np.array([5.1, 5.9, 7.2, 7.8, 9.1])
    result = bootstrap_ci(y_true, y_pred, n_iters=50, seed=42)
    expected_keys = {"rmse", "mae", "r2", "spearman_rho", "pearson_r",
                     "roc_auc", "pr_auc", "mcc", "brier", "ece"}
    assert expected_keys == set(result.keys())
    for key in expected_keys:
        assert "mean" in result[key]
        assert "lo" in result[key]
        assert "hi" in result[key]
        assert result[key]["lo"] <= result[key]["hi"]


def test_bootstrap_ci_perfect_predictions_tight() -> None:
    y = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    result = bootstrap_ci(y, y, n_iters=50, seed=42)
    assert result["rmse"]["mean"] == pytest.approx(0.0, abs=1e-6)


def test_bootstrap_ci_seed_determinism() -> None:
    rng = np.random.RandomState(42)
    y_true = rng.randn(30) + 7
    y_pred = y_true + rng.randn(30) * 0.5
    a = bootstrap_ci(y_true, y_pred, n_iters=50, seed=123)
    b = bootstrap_ci(y_true, y_pred, n_iters=50, seed=123)
    assert a["rmse"]["mean"] == b["rmse"]["mean"]


# ---- Applicability Domain ----

def test_ad_fit_and_score() -> None:
    rng = np.random.RandomState(42)
    n_train, n_test, n_bits, n_desc = 50, 10, 64, 20
    train_fps = (rng.rand(n_train, n_bits) > 0.7).astype(np.uint8)
    train_desc = rng.randn(n_train, n_desc).astype(np.float32)

    ad = ApplicabilityDomain(pca_components=3)
    ad.fit(train_fps, train_desc)

    test_fps = (rng.rand(n_test, n_bits) > 0.7).astype(np.uint8)
    test_desc = rng.randn(n_test, n_desc).astype(np.float32)
    scores = ad.score(test_fps, test_desc)

    assert "tanimoto_nn" in scores
    assert "in_domain" in scores
    assert scores["tanimoto_nn"].shape == (n_test,)
    assert scores["in_domain"].shape == (n_test,)
    assert scores["tanimoto_nn"].min() >= 0.0
    assert scores["tanimoto_nn"].max() <= 1.0


def test_ad_training_compounds_mostly_in_domain() -> None:
    rng = np.random.RandomState(42)
    n = 100
    fps = (rng.rand(n, 64) > 0.5).astype(np.uint8)
    desc = rng.randn(n, 10).astype(np.float32)

    ad = ApplicabilityDomain(pca_components=3)
    ad.fit(fps, desc)
    scores = ad.score(fps, desc)
    # Training compounds should mostly be in-domain
    coverage = scores["in_domain"].mean()
    assert coverage >= 0.5, f"Expected ≥50% in-domain, got {coverage:.2f}"


# ---- Figures (SVG format check) ----

import matplotlib
matplotlib.use("Agg")

from brd4kan.viz.figures import (  # noqa: E402
    fig_dataset_overview,
    fig_benchmark_bars,
    fig_parity_residual,
    fig_kan_splines,
    fig_ad_map,
)


def test_fig_dataset_overview_produces_svg(tmp_path: Path) -> None:
    out = fig_dataset_overview(
        pchembl_values=np.random.randn(100) + 7,
        n_compounds=100,
        n_active=60,
        out_path=tmp_path / "overview.svg",
    )
    assert out.exists()
    assert out.read_bytes()[:100].startswith(b"<?xml") or b"<svg" in out.read_bytes()[:500]


def test_fig_benchmark_bars_produces_svg(tmp_path: Path) -> None:
    out = fig_benchmark_bars(
        model_names=["RF", "KAN"],
        scaffold_metrics={
            "RF": {"rmse_median": 0.8, "rmse_std": 0.1},
            "KAN": {"rmse_median": 0.6, "rmse_std": 0.05},
        },
        out_path=tmp_path / "benchmark.svg",
    )
    assert out.exists()


def test_fig_parity_produces_svg(tmp_path: Path) -> None:
    y = np.linspace(4, 10, 50)
    out = fig_parity_residual(y, y + np.random.randn(50) * 0.3, "test", tmp_path / "parity.svg")
    assert out.exists()


def test_fig_kan_splines_produces_svg(tmp_path: Path) -> None:
    out = fig_kan_splines(
        descriptor_names=[f"d{i}" for i in range(10)],
        importances=[1.0 - i * 0.08 for i in range(10)],
        out_path=tmp_path / "splines.svg",
    )
    assert out.exists()


def test_fig_ad_map_produces_svg(tmp_path: Path) -> None:
    out = fig_ad_map(
        pca_coords=np.random.randn(50, 2),
        in_domain=np.random.rand(50) > 0.3,
        out_path=tmp_path / "ad.svg",
    )
    assert out.exists()
