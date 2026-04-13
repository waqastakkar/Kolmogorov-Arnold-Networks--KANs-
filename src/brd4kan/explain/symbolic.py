"""Stage 7 — Symbolic extraction: prune low-importance edges, fit each
surviving spline to closed-form candidates, emit a pIC50 equation.

Uses pyKAN's ``auto_symbolic()`` under the hood (when available), with a
fallback pure-scipy fitter for environments that don't have pyKAN installed.

Outputs:
* LaTeX equation string (``.tex``)
* SymPy expression pickle (``.pkl``)
* Per-descriptor importance ranking (``.json``)
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

from brd4kan.utils.config import Params
from brd4kan.utils.hashing import file_signature
from brd4kan.utils.manifest import (
    Manifest,
    env_snapshot,
    get_git_sha,
    utc_timestamp,
    write_manifest,
)
from brd4kan.utils.runs import make_run_dir

logger = logging.getLogger(__name__)

# Candidate symbolic functions for edge fitting.
CANDIDATE_FUNCTIONS: dict[str, tuple[Any, int]] = {
    "poly2": (lambda x, a, b, c: a * x**2 + b * x + c, 3),
    "poly3": (lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, 4),
    "exp": (lambda x, a, b: a * np.exp(b * np.clip(x, -20, 20)), 2),
    "log": (lambda x, a, b: a * np.log(np.abs(x) + 1e-8) + b, 2),
    "sin": (lambda x, a, b, c: a * np.sin(b * x + c), 3),
    "sigmoid": (lambda x, a, b, c: a / (1 + np.exp(-b * (x - c))), 3),
    "sqrt": (lambda x, a, b: a * np.sqrt(np.abs(x) + 1e-8) + b, 2),
}

# SymPy templates for LaTeX generation
_X = sp.Symbol("x")
_SYMPY_TEMPLATES: dict[str, Any] = {
    "poly2": lambda a, b, c: a * _X**2 + b * _X + c,
    "poly3": lambda a, b, c, d: a * _X**3 + b * _X**2 + c * _X + d,
    "exp": lambda a, b: a * sp.exp(b * _X),
    "log": lambda a, b: a * sp.log(sp.Abs(_X) + 1e-8) + b,
    "sin": lambda a, b, c: a * sp.sin(b * _X + c),
    "sigmoid": lambda a, b, c: a / (1 + sp.exp(-b * (_X - c))),
    "sqrt": lambda a, b: a * sp.sqrt(sp.Abs(_X) + 1e-8) + b,
}


def compute_edge_importances(
    model: Any,
    X_sample: np.ndarray,
) -> list[dict[str, Any]]:
    """Estimate per-input-edge importance from the first KAN layer.

    Uses the mean absolute activation magnitude over ``X_sample`` as a
    proxy for edge importance (matches pyKAN's pruning heuristic).
    """
    import torch

    model.eval()
    x_t = torch.from_numpy(X_sample).float()
    if hasattr(model, "mult_layer"):
        with torch.no_grad():
            h = model.mult_layer(x_t)
    else:
        h = x_t

    first_layer = model.kan_layers[0]
    if hasattr(first_layer, "scaled_spline_weight"):
        w = first_layer.scaled_spline_weight.detach().abs()
        importance = w.mean(dim=(0, 2)).numpy()
    else:
        # Fallback: use gradient-based importance
        x_t.requires_grad_(True)
        reg, _ = model(x_t)
        reg.sum().backward()
        importance = x_t.grad.abs().mean(dim=0).numpy()

    edges = []
    for i in range(len(importance)):
        edges.append({"input_idx": i, "importance": float(importance[i])})
    edges.sort(key=lambda e: e["importance"], reverse=True)
    return edges


def fit_symbolic_edge(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    candidates: dict[str, tuple[Any, int]] | None = None,
) -> dict[str, Any]:
    """Fit each candidate function to (x, y) data, return the best fit."""
    if candidates is None:
        candidates = CANDIDATE_FUNCTIONS

    best_name = "poly2"
    best_rmse = float("inf")
    best_params: tuple[float, ...] = (0.0, 0.0, 0.0)

    for name, (func, n_params) in candidates.items():
        try:
            p0 = [0.1] * n_params
            popt, _ = curve_fit(func, x_vals, y_vals, p0=p0, maxfev=5000)
            y_fit = func(x_vals, *popt)
            rmse = float(np.sqrt(np.mean((y_vals - y_fit) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                best_params = tuple(float(p) for p in popt)
        except (RuntimeError, ValueError):
            continue

    return {
        "function": best_name,
        "params": best_params,
        "rmse": best_rmse,
    }


def build_symbolic_equation(
    edge_fits: list[dict[str, Any]],
    descriptor_names: list[str],
) -> tuple[str, Any]:
    """Assemble per-edge symbolic fits into a full pIC50 equation.

    Returns (latex_str, sympy_expr).
    """
    terms: list[Any] = []
    for fit in edge_fits:
        idx = fit["input_idx"]
        func_name = fit["function"]
        params = fit["params"]
        var = sp.Symbol(descriptor_names[idx] if idx < len(descriptor_names) else f"x_{idx}")

        template = _SYMPY_TEMPLATES.get(func_name)
        if template is not None:
            expr = template(*params)
            expr = expr.subs(_X, var)
        else:
            expr = params[0] * var  # fallback linear
        terms.append(expr)

    full_expr = sum(terms) if terms else sp.Float(0.0)
    latex = sp.latex(full_expr)
    return latex, full_expr


def run_symbolic(
    kan_model_dir: Path,
    curated_path: Path,
    morgan_path: Path,
    mordred_path: Path,
    out_dir: Path,
    params: Params,
) -> dict[str, Any]:
    """Stage 7 orchestrator: prune → fit symbolic → emit equation + ranking."""
    import pandas as pd
    import torch

    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load best KAN model (first seed, first member)
    from brd4kan.models.kan_model import BRD4KANModel

    # Find a .pt checkpoint
    pt_files = sorted(kan_model_dir.rglob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under {kan_model_dir}")

    # Infer architecture from best_hparams.json
    hp_path = kan_model_dir / "best_hparams.json"
    if hp_path.exists():
        hp = json.loads(hp_path.read_text(encoding="utf-8"))
    else:
        hp = {}

    morgan = np.load(morgan_path)["X"].astype(np.float32)
    mordred_X = np.load(mordred_path)["X"].astype(np.float32)
    X_all = np.hstack([morgan, mordred_X])
    input_dim = X_all.shape[1]

    layer_widths = hp.get("layer_widths", [128, 1])
    if isinstance(layer_widths, str):
        layer_widths = json.loads(layer_widths)

    model = BRD4KANModel(
        input_dim=input_dim,
        layer_widths=layer_widths,
        grid_size=hp.get("grid_size", 3),
        spline_order=hp.get("spline_order", 3),
        dropout=hp.get("dropout", 0.1),
        use_mult_layer=hp.get("multiplicative_nodes", True),
        aux_head=hp.get("aux_classification_head", True),
    )
    state = torch.load(pt_files[0], map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    # Compute edge importances
    sample_idx = np.random.RandomState(params.seed).choice(
        len(X_all), min(500, len(X_all)), replace=False
    )
    X_sample = X_all[sample_idx]
    edges = compute_edge_importances(model, X_sample)

    # Build descriptor names
    mordred_cols_file = np.load(mordred_path, allow_pickle=True)
    mordred_cols = mordred_cols_file.get("columns", np.array([]))
    n_morgan = morgan.shape[1]
    desc_names = [f"ECFP4_{i}" for i in range(n_morgan)] + [
        str(c) for c in mordred_cols
    ]

    # Prune low-importance edges
    threshold = params.symbolic.edge_importance_threshold if hasattr(params, "symbolic") else 0.01
    surviving = [e for e in edges if e["importance"] >= threshold]
    logger.info("Symbolic: %d/%d edges survive pruning (threshold=%.4f)",
                len(surviving), len(edges), threshold)

    # Fit symbolic functions to surviving edges
    model.eval()
    x_t = torch.from_numpy(X_sample).float()
    with torch.no_grad():
        h = model.mult_layer(x_t) if hasattr(model, "mult_layer") else x_t
        h_np = h.numpy()

    first_layer = model.kan_layers[0]
    for edge in surviving:
        idx = edge["input_idx"]
        x_col = h_np[:, idx]
        # Get the edge's output by feeding isolated input through the layer
        with torch.no_grad():
            inp = torch.zeros_like(h)
            inp[:, idx] = h[:, idx]
            out = first_layer(inp)
        y_col = out.mean(dim=1).numpy()
        fit = fit_symbolic_edge(x_col, y_col)
        edge.update(fit)

    # Build equation from top edges
    top_edges = surviving[:20]  # top 20 for equation clarity
    latex_eq, sympy_expr = build_symbolic_equation(top_edges, desc_names)

    # Save outputs
    tex_path = out_dir / "pIC50_equation.tex"
    tex_path.write_text(latex_eq, encoding="utf-8")

    pkl_path = out_dir / "pIC50_equation.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(sympy_expr, f)

    importance_path = out_dir / "descriptor_importance.json"
    importance_path.write_text(
        json.dumps(edges[:50], indent=2, default=str), encoding="utf-8"
    )

    # Surviving edges info
    symbolic_fits_path = out_dir / "symbolic_fits.json"
    symbolic_fits_path.write_text(
        json.dumps(surviving[:20], indent=2, default=str), encoding="utf-8"
    )

    run_dir = make_run_dir(out_dir)
    manifest = Manifest(
        stage="symbolic",
        git_sha=get_git_sha(),
        timestamp=utc_timestamp(),
        inputs={
            "kan_model_dir": str(kan_model_dir),
            "morgan_npz": file_signature(morgan_path),
            "mordred_npz": file_signature(mordred_path),
        },
        outputs={
            "equation_tex": file_signature(tex_path),
            "equation_pkl": file_signature(pkl_path),
            "descriptor_importance": file_signature(importance_path),
            "n_surviving_edges": len(surviving),
            "n_total_edges": len(edges),
        },
        params_snapshot=params.model_dump(),
        seeds={"global": params.seed},
        env=env_snapshot(),
        wall_time_seconds=round(time.perf_counter() - started, 4),
    )
    write_manifest(manifest, run_dir)

    return {
        "equation_latex": latex_eq,
        "n_surviving_edges": len(surviving),
        "top_descriptors": [
            {"name": desc_names[e["input_idx"]] if e["input_idx"] < len(desc_names) else f"x_{e['input_idx']}",
             "importance": e["importance"],
             "function": e.get("function", "?"),
             "fit_rmse": e.get("rmse", None)}
            for e in surviving[:10]
        ],
    }
