"""Figure generation utilities — Nature MI compliant."""

from brd4kan.viz.figures import (
    fig_ad_map,
    fig_benchmark_bars,
    fig_dataset_overview,
    fig_hit_cards,
    fig_kan_splines,
    fig_parity_residual,
    fig_screening_funnel,
    fig_shap_beeswarm,
    fig_symbolic_equation,
)
from brd4kan.viz.style import (
    FigureStyle,
    apply_style,
    figure_size_inches,
    load_figure_config,
    nature_palette,
    palette_cycler,
)

__all__ = [
    "FigureStyle",
    "apply_style",
    "fig_ad_map",
    "fig_benchmark_bars",
    "fig_dataset_overview",
    "fig_hit_cards",
    "fig_kan_splines",
    "fig_parity_residual",
    "fig_screening_funnel",
    "fig_shap_beeswarm",
    "fig_symbolic_equation",
    "figure_size_inches",
    "load_figure_config",
    "nature_palette",
    "palette_cycler",
]
