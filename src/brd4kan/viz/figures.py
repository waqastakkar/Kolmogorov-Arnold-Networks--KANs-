"""Nature MI-compliant figure generators — one function per figure type.

All 9 figure types from PLAN.md §3. Every function:
1. Calls ``apply_style()`` before plotting.
2. Saves to SVG only.
3. Uses panel letters (a, b, c…) bold 10pt top-left.
4. Uses the Nature palette for categorical data.
5. Despines top/right.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brd4kan.viz.style import apply_style, figure_size_inches, nature_palette


def _panel_letter(ax: Any, letter: str) -> None:
    """Add a bold panel letter (a, b, c…) at the top-left of the axes."""
    ax.text(
        -0.12, 1.05, letter,
        transform=ax.transAxes,
        fontsize=10, fontweight="bold",
        va="top", ha="left",
    )


def _save_svg(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


# -----------------------------------------------------------------------
# 1. Dataset overview (counts, pIC50 histogram, scaffold UMAP)
# -----------------------------------------------------------------------

def fig_dataset_overview(
    pchembl_values: np.ndarray,
    n_compounds: int,
    n_active: int,
    out_path: Path,
    umap_coords: np.ndarray | None = None,
    umap_labels: np.ndarray | None = None,
) -> Path:
    style = apply_style()
    palette = nature_palette(style)

    n_panels = 3 if umap_coords is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figure_size_inches("two_col", 60.0, style))

    # a — pIC50 histogram
    ax = axes[0]
    _panel_letter(ax, "a")
    ax.hist(pchembl_values, bins=30, color=palette[0], edgecolor="black", linewidth=0.3)
    ax.set_xlabel("pIC50")
    ax.set_ylabel("Count")
    ax.set_title("pIC50 Distribution")

    # b — Counts bar
    ax = axes[1]
    _panel_letter(ax, "b")
    bars = ax.bar(["Total", "Active"], [n_compounds, n_active], color=[palette[3], palette[0]])
    ax.set_ylabel("Count")
    ax.set_title("Dataset Composition")

    # c — UMAP (if provided)
    if umap_coords is not None and n_panels == 3:
        ax = axes[2]
        _panel_letter(ax, "c")
        if umap_labels is not None:
            sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=umap_labels,
                           cmap="viridis", s=5, alpha=0.7)
            plt.colorbar(sc, ax=ax, label="pIC50")
        else:
            ax.scatter(umap_coords[:, 0], umap_coords[:, 1], color=palette[2], s=5, alpha=0.7)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Scaffold Chemical Space")

    fig.tight_layout()
    return _save_svg(fig, out_path)


# -----------------------------------------------------------------------
# 2. Benchmark bars: KAN vs baselines, scaffold + time split
# -----------------------------------------------------------------------

def fig_benchmark_bars(
    model_names: list[str],
    scaffold_metrics: dict[str, dict[str, float]],
    out_path: Path,
    time_metrics: dict[str, dict[str, float]] | None = None,
    metric_key: str = "rmse",
) -> Path:
    style = apply_style()
    palette = nature_palette(style)

    n_panels = 2 if time_metrics else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figure_size_inches("two_col", 70.0, style))
    if n_panels == 1:
        axes = [axes]

    for panel_idx, (ax, metrics, title) in enumerate(
        zip(axes, [scaffold_metrics, time_metrics] if time_metrics else [scaffold_metrics],
            ["Scaffold Split", "Time Split"] if time_metrics else ["Scaffold Split"])
    ):
        if metrics is None:
            continue
        _panel_letter(ax, chr(ord("a") + panel_idx))
        vals = [metrics.get(m, {}).get(f"{metric_key}_median", 0.0) for m in model_names]
        errs = [metrics.get(m, {}).get(f"{metric_key}_std", 0.0) for m in model_names]
        colors = [palette[i % len(palette)] for i in range(len(model_names))]
        bars = ax.bar(model_names, vals, yerr=errs, color=colors,
                     edgecolor="black", linewidth=0.3, capsize=3)
        ax.set_ylabel(metric_key.upper())
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    return _save_svg(fig, out_path)


# -----------------------------------------------------------------------
# 3. Parity + residual plots
# -----------------------------------------------------------------------

def fig_parity_residual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_path: Path,
) -> Path:
    style = apply_style()
    palette = nature_palette(style)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size_inches("two_col", 70.0, style))

    # a — Parity
    _panel_letter(ax1, "a")
    ax1.scatter(y_true, y_pred, color=palette[0], s=10, alpha=0.6, edgecolors="none")
    lo = min(y_true.min(), y_pred.min()) - 0.3
    hi = max(y_true.max(), y_pred.max()) + 0.3
    ax1.plot([lo, hi], [lo, hi], "--", color="black", linewidth=0.8)
    ax1.set_xlabel("Observed pIC50")
    ax1.set_ylabel("Predicted pIC50")
    ax1.set_title(f"{model_name} — Parity")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect("equal")

    # b — Residual
    _panel_letter(ax2, "b")
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, color=palette[3], s=10, alpha=0.6, edgecolors="none")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Predicted pIC50")
    ax2.set_ylabel("Residual")
    ax2.set_title(f"{model_name} — Residuals")

    fig.tight_layout()
    return _save_svg(fig, out_path)


# -----------------------------------------------------------------------
# 4. KAN architecture schematic + learned spline shapes for top descriptors
# -----------------------------------------------------------------------

def fig_kan_splines(
    descriptor_names: list[str],
    importances: list[float],
    out_path: Path,
    n_top: int = 10,
) -> Path:
    style = apply_style()
    palette = nature_palette(style)

    n_show = min(n_top, len(descriptor_names))
    fig, ax = plt.subplots(figsize=figure_size_inches("one_and_half_col", 70.0, style))
    _panel_letter(ax, "a")
    y_pos = np.arange(n_show)
    colors = [palette[i % len(palette)] for i in range(n_show)]
    ax.barh(y_pos, importances[:n_show], color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(descriptor_names[:n_show])
    ax.invert_yaxis()
    ax.set_xlabel("Edge Importance")
    ax.set_title("Top KAN Descriptors")
    fig.tight_layout()
    return _save_svg(fig, out_path)


# -----------------------------------------------------------------------
# 5. Symbolic equation panel + descriptor importance
# -----------------------------------------------------------------------

def fig_symbolic_equation(
    latex_equation: str,
    descriptor_names: list[str],
    importances: list[float],
    out_path: Path,
) -> Path:
    style = apply_style()
    palette = nature_palette(style)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size_inches("two_col", 70.0, style),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # a — Equation rendered
    _panel_letter(ax1, "a")
    ax1.axis("off")
    # Truncate long equations for display
    display_eq = latex_equation if len(latex_equation) < 300 else latex_equation[:297] + "..."
    try:
        ax1.text(0.5, 0.5, f"${display_eq}$", transform=ax1.transAxes,
                fontsize=7, ha="center", va="center", wrap=True)
    except Exception:
        ax1.text(0.5, 0.5, "Equation (see .tex file)", transform=ax1.transAxes,
                fontsize=8, ha="center", va="center")
    ax1.set_title("Symbolic pIC50 Equation")

    # b — Importance bar
    _panel_letter(ax2, "b")
    n = min(15, len(descriptor_names))
    y_pos = np.arange(n)
    colors = [palette[i % len(palette)] for i in range(n)]
    ax2.barh(y_pos, importances[:n], color=colors, edgecolor="black", linewidth=0.3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(descriptor_names[:n])
    ax2.invert_yaxis()
    ax2.set_xlabel("Importance")
    ax2.set_title("Descriptor Ranking")

    fig.tight_layout()
    return _save_svg(fig, out_path)


# -----------------------------------------------------------------------
# 6. SHAP beeswarm (KAN vs RF vs Chemprop)
# -----------------------------------------------------------------------

def fig_shap_beeswarm(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    model_name: str,
    out_path: Path,
    n_top: int = 15,
) -> Path:
    style = apply_style()
    palette = nature_palette(style)

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:n_top]

    fig, ax = plt.subplots(figsize=figure_size_inches("one_and_half_col", 80.0, style))
    _panel_letter(ax, "a")

    for rank, feat_i in enumerate(top_idx):
        sv = shap_values[:, feat_i]
        fv = X_test[:, feat_i]
        # Normalize feature values for coloring
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-12)
        y_jitter = rank + np.random.RandomState(42).uniform(-0.3, 0.3, len(sv))
        ax.scatter(sv, y_jitter, c=fv_norm, cmap="RdBu_r", s=3, alpha=0.6, edgecolors="none")

    ax.set_yticks(range(n_top))
    ax.set_yticklabels([feature_names[i] if i < len(feature_names) else f"f{i}" for i in top_idx])
    ax.invert_yaxis()
    ax.set_xlabel("SHAP Value")
    ax.set_title(f"SHAP — {model_name}")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    return _save_svg(fig, out_path)


# -----------------------------------------------------------------------
# 7. Applicability domain map (PCA + KDE)
# -----------------------------------------------------------------------

def fig_ad_map(
    pca_coords: np.ndarray,
    in_domain: np.ndarray,
    out_path: Path,
    pchembl_values: np.ndarray | None = None,
) -> Path:
    style = apply_style()
    palette = nature_palette(style)
    fig, ax = plt.subplots(figsize=figure_size_inches("one_col", 70.0, style))
    _panel_letter(ax, "a")

    in_mask = in_domain.astype(bool)
    if pchembl_values is not None:
        sc = ax.scatter(pca_coords[in_mask, 0], pca_coords[in_mask, 1],
                       c=pchembl_values[in_mask], cmap="viridis", s=8, alpha=0.7,
                       edgecolors="none", label="In-domain")
        plt.colorbar(sc, ax=ax, label="pIC50")
    else:
        ax.scatter(pca_coords[in_mask, 0], pca_coords[in_mask, 1],
                  color=palette[2], s=8, alpha=0.7, label="In-domain")
    ax.scatter(pca_coords[~in_mask, 0], pca_coords[~in_mask, 1],
              color=palette[7], s=8, alpha=0.5, marker="x", label="Out-of-domain")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Applicability Domain")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save_svg(fig, out_path)


# -----------------------------------------------------------------------
# 8. Screening funnel + top-hit UMAP
# -----------------------------------------------------------------------

def fig_screening_funnel(
    funnel_counts: dict[str, int],
    out_path: Path,
    umap_coords: np.ndarray | None = None,
    umap_pchembl: np.ndarray | None = None,
) -> Path:
    style = apply_style()
    palette = nature_palette(style)
    n_panels = 2 if umap_coords is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figure_size_inches("two_col", 70.0, style))
    if n_panels == 1:
        axes = [axes]

    # a — Funnel
    ax = axes[0]
    _panel_letter(ax, "a")
    stages = list(funnel_counts.keys())
    counts = list(funnel_counts.values())
    colors = [palette[i % len(palette)] for i in range(len(stages))]
    ax.barh(stages, counts, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Compounds")
    ax.set_title("Screening Funnel")
    ax.invert_yaxis()

    # b — UMAP
    if umap_coords is not None and n_panels == 2:
        ax = axes[1]
        _panel_letter(ax, "b")
        if umap_pchembl is not None:
            sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=umap_pchembl,
                           cmap="viridis", s=10, alpha=0.7, edgecolors="none")
            plt.colorbar(sc, ax=ax, label="Pred pIC50")
        else:
            ax.scatter(umap_coords[:, 0], umap_coords[:, 1], color=palette[0], s=10, alpha=0.7)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Top-Hit Chemical Space")

    fig.tight_layout()
    return _save_svg(fig, out_path)


# -----------------------------------------------------------------------
# 9. Top-20 hit cards (structure + bars for pIC50, AD, QED, SA)
# -----------------------------------------------------------------------

def fig_hit_cards(
    hit_data: list[dict[str, Any]],
    out_path: Path,
    n_show: int = 20,
) -> Path:
    """Bar-chart summary cards for top hits (no 2D depictions — those go in per-hit SVGs)."""
    style = apply_style()
    palette = nature_palette(style)

    hits = hit_data[:n_show]
    n = len(hits)
    fig, axes = plt.subplots(n, 1, figsize=figure_size_inches("one_col", float(n * 18), style),
                              sharex=True)
    if n == 1:
        axes = [axes]

    metric_keys = ["pIC50", "AD", "QED", "SA"]
    for idx, (ax, hit) in enumerate(zip(axes, hits)):
        vals = [hit.get(k, 0.0) for k in metric_keys]
        colors = [palette[i % len(palette)] for i in range(len(metric_keys))]
        ax.barh(metric_keys, vals, color=colors, edgecolor="black", linewidth=0.3)
        name = hit.get("name", f"Hit {idx + 1}")
        ax.set_title(name, fontsize=7, loc="left")
        if idx == 0:
            _panel_letter(ax, "a")

    if axes:
        axes[-1].set_xlabel("Score")
    fig.tight_layout()
    return _save_svg(fig, out_path)
