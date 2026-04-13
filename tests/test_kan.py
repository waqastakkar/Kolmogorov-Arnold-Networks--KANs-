"""KAN forward/backward shape tests + ensemble + conformal coverage.

These tests use a mock KANLinear when efficient-kan is not installed,
ensuring core logic is always verified.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Provide a stub efficient_kan so tests run without the real package.
# ---------------------------------------------------------------------------


class _StubKANLinear(nn.Module):
    """Minimal stand-in for efficient_kan.KANLinear."""

    def __init__(
        self, in_features: int, out_features: int,
        grid_size: int = 5, spline_order: int = 3,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        # Mimic spline weight for regularization_loss
        self.scaled_spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


if "efficient_kan" not in sys.modules:
    _mod = ModuleType("efficient_kan")
    _mod.KANLinear = _StubKANLinear  # type: ignore[attr-defined]
    sys.modules["efficient_kan"] = _mod


from brd4kan.models.kan_model import BRD4KANModel, EnsembleKAN, MultiplicativeLayer  # noqa: E402
from brd4kan.models.conformal import MondrianConformalPredictor  # noqa: E402


# ---------------------------------------------------------------------------
# BRD4KANModel tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def kan_model() -> BRD4KANModel:
    return BRD4KANModel(
        input_dim=64,
        layer_widths=[32, 16],
        grid_size=3,
        spline_order=3,
        dropout=0.1,
        use_mult_layer=True,
        aux_head=True,
    )


def test_forward_output_shapes(kan_model: BRD4KANModel) -> None:
    x = torch.randn(8, 64)
    reg, aux = kan_model(x)
    assert reg.shape == (8,), f"Expected (8,), got {reg.shape}"
    assert aux is not None
    assert aux.shape == (8,), f"Expected (8,), got {aux.shape}"


def test_forward_no_aux_head() -> None:
    model = BRD4KANModel(
        input_dim=32, layer_widths=[16], grid_size=3, spline_order=3,
        dropout=0.0, use_mult_layer=False, aux_head=False,
    )
    x = torch.randn(4, 32)
    reg, aux = model(x)
    assert reg.shape == (4,)
    assert aux is None


def test_backward_gradients(kan_model: BRD4KANModel) -> None:
    x = torch.randn(4, 64, requires_grad=True)
    reg, _ = kan_model(x)
    loss = reg.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (4, 64)
    for p in kan_model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_regularization_loss(kan_model: BRD4KANModel) -> None:
    l1, ent = kan_model.regularization_loss()
    assert l1.item() >= 0
    assert ent.item() != 0  # entropy should be non-zero for random weights


def test_sparsity_in_valid_range(kan_model: BRD4KANModel) -> None:
    s = kan_model.sparsity()
    assert 0.0 <= s <= 1.0


def test_update_grid(kan_model: BRD4KANModel) -> None:
    old_grid = kan_model.grid_size
    kan_model.update_grid(10)
    assert kan_model.grid_size == 10
    # Forward still works
    x = torch.randn(4, 64)
    reg, _ = kan_model(x)
    assert reg.shape == (4,)


# ---------------------------------------------------------------------------
# MultiplicativeLayer tests
# ---------------------------------------------------------------------------


def test_multiplicative_layer_preserves_dim() -> None:
    ml = MultiplicativeLayer(32)
    x = torch.randn(8, 32)
    out = ml(x)
    assert out.shape == (8, 32)


# ---------------------------------------------------------------------------
# EnsembleKAN tests
# ---------------------------------------------------------------------------


def test_ensemble_forward_averages() -> None:
    members = [
        BRD4KANModel(input_dim=16, layer_widths=[8], grid_size=3, spline_order=3,
                      dropout=0.0, use_mult_layer=False, aux_head=False)
        for _ in range(3)
    ]
    ens = EnsembleKAN(members)
    x = torch.randn(4, 16)
    out = ens(x)
    assert out.shape == (4,)


def test_ensemble_predict_with_uncertainty() -> None:
    members = [
        BRD4KANModel(input_dim=16, layer_widths=[8], grid_size=3, spline_order=3,
                      dropout=0.1, use_mult_layer=False, aux_head=False)
        for _ in range(3)
    ]
    ens = EnsembleKAN(members)
    x = torch.randn(4, 16)
    mean, epist, aleat = ens.predict_with_uncertainty(x, mc_samples=5)
    assert mean.shape == (4,)
    assert epist.shape == (4,)
    assert aleat.shape == (4,)
    assert (epist >= 0).all()
    assert (aleat >= 0).all()


# ---------------------------------------------------------------------------
# Conformal prediction tests
# ---------------------------------------------------------------------------


def test_conformal_coverage_on_synthetic_data() -> None:
    """Conformal intervals on clean synthetic data should achieve ≥ (1-α) coverage."""
    rng = np.random.RandomState(42)
    n_cal, n_test = 200, 100

    y_cal_true = rng.randn(n_cal) * 2 + 7
    residuals_cal = rng.randn(n_cal) * 0.3  # small noise
    groups_cal = [f"scaffold_{i % 5}" for i in range(n_cal)]

    alpha = 0.1
    cp = MondrianConformalPredictor(alpha=alpha)
    cp.calibrate(residuals_cal, groups_cal)

    y_test_true = rng.randn(n_test) * 2 + 7
    y_test_pred = y_test_true + rng.randn(n_test) * 0.3
    groups_test = [f"scaffold_{i % 5}" for i in range(n_test)]

    cov = cp.coverage(y_test_true, y_test_pred, groups_test)
    # With small noise and well-calibrated quantiles, should be >= 1-α
    assert cov["overall"] >= (1 - alpha) - 0.05, (
        f"Coverage {cov['overall']:.3f} < expected {1 - alpha - 0.05}"
    )


def test_conformal_unknown_group_uses_global() -> None:
    residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    groups = ["a", "a", "b", "b", "b"]
    cp = MondrianConformalPredictor(alpha=0.1)
    cp.calibrate(residuals, groups)

    lower, upper = cp.predict_intervals(np.array([7.0]), ["never_seen"])
    assert lower[0] < 7.0
    assert upper[0] > 7.0


def test_conformal_state_dict_round_trip() -> None:
    cp = MondrianConformalPredictor(alpha=0.2)
    cp.calibrate(np.array([0.1, 0.2, 0.3]), ["a", "b", "a"])
    d = cp.state_dict()
    cp2 = MondrianConformalPredictor.from_state_dict(d)
    assert cp2.alpha == cp.alpha
    assert cp2._global_quantile == cp._global_quantile
    assert cp2._group_quantiles == cp._group_quantiles
