"""Explainability: symbolic extraction, SHAP analysis (Stages 7-8)."""

from brd4kan.explain.shap_analysis import (
    compute_shap_kernel,
    compute_shap_tree,
    load_shap_values,
    save_shap_values,
)
from brd4kan.explain.symbolic import (
    build_symbolic_equation,
    compute_edge_importances,
    fit_symbolic_edge,
    run_symbolic,
)

__all__ = [
    "build_symbolic_equation",
    "compute_edge_importances",
    "compute_shap_kernel",
    "compute_shap_tree",
    "fit_symbolic_edge",
    "load_shap_values",
    "run_symbolic",
    "save_shap_values",
]
