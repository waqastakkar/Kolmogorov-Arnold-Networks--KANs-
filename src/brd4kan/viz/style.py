"""Nature MI figure-style enforcement.

All rules live in ``configs/figures.yaml``. ``apply_style()`` is the single
entry point — every figure-producing module in this codebase MUST call it
before plotting. The companion test ``tests/test_style.py`` round-trips a
sample SVG and verifies font, weight, sizes, format, and palette compliance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import yaml
from cycler import cycler

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "figures.yaml"

_MM_PER_INCH = 25.4


@dataclass(frozen=True)
class FigureStyle:
    """Resolved figure-style values, loaded from ``configs/figures.yaml``."""

    format: str
    font_family: str
    font_serif: tuple[str, ...]
    font_weight: str
    svg_fonttype: str
    title_pt: float
    axis_label_pt: float
    tick_pt: float
    legend_pt: float
    annotation_pt: float
    panel_letter_pt: float
    line_width: float
    axis_spine_width: float
    tick_length: float
    tick_width: float
    nature_palette: tuple[str, ...]
    diverging_cmap: str
    sequential_cmap: str
    despine_top: bool
    despine_right: bool
    despine_left: bool
    despine_bottom: bool
    figure_widths_mm: dict[str, float]
    default_panel_height_mm: float


def load_figure_config(path: Path | None = None) -> FigureStyle:
    """Load and validate ``configs/figures.yaml`` into a :class:`FigureStyle`."""
    cfg_path = path or _DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    return FigureStyle(
        format=str(raw["format"]),
        font_family=str(raw["font"]["family"]),
        font_serif=tuple(raw["font"]["serif"]),
        font_weight=str(raw["font"]["weight"]),
        svg_fonttype=str(raw["svg"]["fonttype"]),
        title_pt=float(raw["sizes_pt"]["title"]),
        axis_label_pt=float(raw["sizes_pt"]["axis_label"]),
        tick_pt=float(raw["sizes_pt"]["tick"]),
        legend_pt=float(raw["sizes_pt"]["legend"]),
        annotation_pt=float(raw["sizes_pt"]["annotation"]),
        panel_letter_pt=float(raw["sizes_pt"]["panel_letter"]),
        line_width=float(raw["lines"]["line_width"]),
        axis_spine_width=float(raw["lines"]["axis_spine_width"]),
        tick_length=float(raw["lines"]["tick_length"]),
        tick_width=float(raw["lines"]["tick_width"]),
        nature_palette=tuple(str(c).upper() for c in raw["palette"]["nature"]),
        diverging_cmap=str(raw["palette"]["diverging_cmap"]),
        sequential_cmap=str(raw["palette"]["sequential_cmap"]),
        despine_top=bool(raw["despine"]["top"]),
        despine_right=bool(raw["despine"]["right"]),
        despine_left=bool(raw["despine"]["left"]),
        despine_bottom=bool(raw["despine"]["bottom"]),
        figure_widths_mm={k: float(v) for k, v in raw["figure_widths_mm"].items()},
        default_panel_height_mm=float(raw["default_panel_height_mm"]),
    )


def nature_palette(style: FigureStyle | None = None) -> tuple[str, ...]:
    """Return the canonical Nature/NPG categorical palette."""
    return (style or load_figure_config()).nature_palette


def palette_cycler(style: FigureStyle | None = None) -> Any:
    """Return a matplotlib color cycler bound to the Nature palette."""
    return cycler(color=list(nature_palette(style)))


def figure_size_inches(
    width: str = "one_col",
    height_mm: float | None = None,
    style: FigureStyle | None = None,
) -> tuple[float, float]:
    """Convert a Nature column width + height in mm to (w, h) inches."""
    s = style or load_figure_config()
    if width not in s.figure_widths_mm:
        raise ValueError(
            f"Unknown figure width '{width}'. Allowed: {sorted(s.figure_widths_mm)}"
        )
    w_mm = s.figure_widths_mm[width]
    h_mm = height_mm if height_mm is not None else s.default_panel_height_mm
    return (w_mm / _MM_PER_INCH, h_mm / _MM_PER_INCH)


def apply_style(style: FigureStyle | None = None) -> FigureStyle:
    """Apply every Nature MI figure rule to the global matplotlib rcParams.

    Returns the resolved :class:`FigureStyle` so callers can pull palette
    colors and column widths without re-loading the config.
    """
    s = style or load_figure_config()

    if s.format != "svg":
        raise ValueError(f"Only 'svg' is supported as figure format, got '{s.format}'.")

    mpl.rcParams.update(
        {
            # Font
            "font.family": s.font_family,
            "font.serif": list(s.font_serif),
            "font.weight": s.font_weight,
            "mathtext.default": "regular",
            # Bold everything text-like
            "axes.labelweight": s.font_weight,
            "axes.titleweight": s.font_weight,
            "figure.titleweight": s.font_weight,
            # Sizes (pt)
            "axes.titlesize": s.title_pt,
            "axes.labelsize": s.axis_label_pt,
            "xtick.labelsize": s.tick_pt,
            "ytick.labelsize": s.tick_pt,
            "legend.fontsize": s.legend_pt,
            "legend.title_fontsize": s.legend_pt,
            "figure.titlesize": s.title_pt,
            # Lines
            "lines.linewidth": s.line_width,
            "axes.linewidth": s.axis_spine_width,
            "patch.linewidth": s.axis_spine_width,
            "xtick.major.size": s.tick_length,
            "ytick.major.size": s.tick_length,
            "xtick.major.width": s.tick_width,
            "ytick.major.width": s.tick_width,
            # Despine top/right by default
            "axes.spines.top": not s.despine_top,
            "axes.spines.right": not s.despine_right,
            "axes.spines.left": not s.despine_left,
            "axes.spines.bottom": not s.despine_bottom,
            # SVG output: keep text as text (so it round-trips with the right font)
            "svg.fonttype": s.svg_fonttype,
            "svg.image_inline": True,
            "savefig.format": s.format,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "savefig.dpi": 600,
            # Palette
            "axes.prop_cycle": cycler(color=list(s.nature_palette)),
            # Misc chartjunk control
            "axes.grid": False,
            "legend.frameon": False,
        }
    )
    return s
