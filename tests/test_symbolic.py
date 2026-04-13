"""Tests for Stage 7 — symbolic edge fitting and equation assembly."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
import sympy as sp
import torch
import torch.nn as nn

# Ensure stub KANLinear is available
if "efficient_kan" not in sys.modules:
    class _StubKANLinear(nn.Module):
        def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.linear = nn.Linear(in_features, out_features)
            self.scaled_spline_weight = nn.Parameter(
                torch.randn(out_features, in_features, grid_size + spline_order)
            )

        def forward(self, x):
            return self.linear(x)

    _mod = ModuleType("efficient_kan")
    _mod.KANLinear = _StubKANLinear  # type: ignore[attr-defined]
    sys.modules["efficient_kan"] = _mod


from brd4kan.explain.symbolic import (  # noqa: E402
    build_symbolic_equation,
    compute_edge_importances,
    fit_symbolic_edge,
)
from brd4kan.models.kan_model import BRD4KANModel  # noqa: E402


def test_fit_symbolic_edge_recovers_quadratic() -> None:
    rng = np.random.RandomState(42)
    x = np.linspace(-2, 2, 100)
    y = 1.5 * x**2 - 0.3 * x + 0.1 + rng.randn(100) * 0.01
    result = fit_symbolic_edge(x, y)
    assert result["function"] in ("poly2", "poly3")
    assert result["rmse"] < 0.5


def test_fit_symbolic_edge_exp() -> None:
    x = np.linspace(0, 3, 100)
    y = 2.0 * np.exp(0.5 * x)
    result = fit_symbolic_edge(x, y)
    assert result["function"] == "exp"
    assert result["rmse"] < 0.5


def test_build_symbolic_equation_produces_latex() -> None:
    edge_fits = [
        {"input_idx": 0, "function": "poly2", "params": (1.0, -0.5, 0.1)},
        {"input_idx": 1, "function": "exp", "params": (2.0, 0.3)},
    ]
    desc_names = ["MolWt", "LogP"]
    latex, expr = build_symbolic_equation(edge_fits, desc_names)
    assert isinstance(latex, str) and len(latex) > 0
    assert isinstance(expr, sp.Basic)
    # Should contain descriptor symbols
    syms = {str(s) for s in expr.free_symbols}
    assert "MolWt" in syms or "LogP" in syms


def test_compute_edge_importances_returns_sorted() -> None:
    model = BRD4KANModel(
        input_dim=16, layer_widths=[8], grid_size=3, spline_order=3,
        dropout=0.0, use_mult_layer=False, aux_head=False,
    )
    X = np.random.randn(20, 16).astype(np.float32)
    edges = compute_edge_importances(model, X)
    assert len(edges) == 16
    importances = [e["importance"] for e in edges]
    assert importances == sorted(importances, reverse=True)


def test_fit_symbolic_edge_handles_constant() -> None:
    x = np.linspace(-1, 1, 50)
    y = np.full(50, 3.0)
    result = fit_symbolic_edge(x, y)
    assert result["rmse"] < 0.1
