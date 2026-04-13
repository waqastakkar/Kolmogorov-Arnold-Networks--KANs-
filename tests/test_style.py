"""Figure-style compliance test — Stage 0 gate.

Loads ``configs/figures.yaml`` via :func:`brd4kan.viz.style.apply_style`,
saves a sample SVG, then asserts every Nature MI rule:

* SVG format
* Times New Roman, embedded as text (``svg.fonttype='none'``)
* Every rendered text element is bold
* Font sizes match the YAML
* Only colors from the Nature palette (plus pure black/white axis chrome)
  appear in the categorical fixture
* The Nature palette colors actually were emitted (positive check)

Any figure produced by this codebase MUST pass this contract — no figure
may be committed that fails it.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from brd4kan.viz.style import (  # noqa: E402
    FigureStyle,
    apply_style,
    figure_size_inches,
    load_figure_config,
    nature_palette,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_CFG_PATH = REPO_ROOT / "configs" / "figures.yaml"

# Pure neutrals are allowed for axis spines, ticks, and text.
NEUTRAL_HEX = {"#000000", "#FFFFFF"}
HEX_RE = re.compile(r"#[0-9A-Fa-f]{6}")
RGB_FUNC_RE = re.compile(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


@pytest.fixture(scope="module")
def style() -> FigureStyle:
    return apply_style(load_figure_config(FIG_CFG_PATH))


@pytest.fixture(scope="module")
def sample_svg(style: FigureStyle) -> bytes:
    palette = nature_palette(style)
    fig, ax = plt.subplots(figsize=figure_size_inches("one_col", 60.0, style))
    xs = [0.0, 1.0, 2.0, 3.0]
    for i in range(3):
        ax.plot(xs, [i + 0.5 * x for x in xs], color=palette[i], label=f"S{i}")
    ax.set_title("Sample title")
    ax.set_xlabel("X label")
    ax.set_ylabel("Y label")
    ax.legend()
    ax.annotate("a", xy=(0.0, 1.0))
    buf = BytesIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    return buf.getvalue()


def test_format_is_svg(sample_svg: bytes) -> None:
    head = sample_svg[:512]
    assert head.startswith(b"<?xml") or b"<svg" in head
    root = ET.fromstring(sample_svg)
    assert root.tag.endswith("}svg") or root.tag == "svg"


def test_font_family_times_new_roman(sample_svg: bytes) -> None:
    text = sample_svg.decode("utf-8")
    # svg.fonttype='none' should keep glyphs as <text> with the font name in style.
    assert "Times New Roman" in text, "Times New Roman not embedded in SVG output"


def test_text_is_bold(sample_svg: bytes) -> None:
    text = sample_svg.decode("utf-8").replace(" ", "")
    # Matplotlib emits style strings like "font-weight:bold" on the text groups.
    assert "font-weight:bold" in text, "Bold weight missing from rendered text"


def test_only_palette_colors(sample_svg: bytes, style: FigureStyle) -> None:
    text = sample_svg.decode("utf-8")
    palette_norm = {c.upper() for c in style.nature_palette}
    found = {h.upper() for h in HEX_RE.findall(text)}
    found |= {_rgb_to_hex(int(r), int(g), int(b)) for r, g, b in RGB_FUNC_RE.findall(text)}
    extras = found - palette_norm - NEUTRAL_HEX
    assert not extras, f"Non-palette colors found in SVG: {sorted(extras)}"


def test_palette_actually_used(sample_svg: bytes, style: FigureStyle) -> None:
    text = sample_svg.decode("utf-8").upper()
    used = sum(1 for c in style.nature_palette[:3] if c.upper() in text)
    assert used >= 3, "Expected the first three palette colors to appear in the SVG"


def test_rcparams_match_yaml(style: FigureStyle) -> None:
    assert mpl.rcParams["svg.fonttype"] == style.svg_fonttype == "none"
    assert mpl.rcParams["font.family"] == [style.font_family]
    assert mpl.rcParams["font.serif"][0] == "Times New Roman"
    assert mpl.rcParams["font.weight"] == style.font_weight == "bold"
    assert mpl.rcParams["axes.labelweight"] == "bold"
    assert mpl.rcParams["axes.titleweight"] == "bold"
    assert float(mpl.rcParams["axes.titlesize"]) == style.title_pt
    assert float(mpl.rcParams["axes.labelsize"]) == style.axis_label_pt
    assert float(mpl.rcParams["xtick.labelsize"]) == style.tick_pt
    assert float(mpl.rcParams["ytick.labelsize"]) == style.tick_pt
    assert float(mpl.rcParams["legend.fontsize"]) == style.legend_pt
    assert float(mpl.rcParams["lines.linewidth"]) == style.line_width
    assert float(mpl.rcParams["axes.linewidth"]) == style.axis_spine_width
    assert float(mpl.rcParams["xtick.major.size"]) == style.tick_length
    assert float(mpl.rcParams["ytick.major.size"]) == style.tick_length
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False


def test_figure_widths_mm(style: FigureStyle) -> None:
    assert style.figure_widths_mm["one_col"] == 89.0
    assert style.figure_widths_mm["one_and_half_col"] == 120.0
    assert style.figure_widths_mm["two_col"] == 183.0


def test_figure_size_inches_known_widths(style: FigureStyle) -> None:
    w_in, h_in = figure_size_inches("one_col", 60.0, style)
    assert abs(w_in - 89.0 / 25.4) < 1e-9
    assert abs(h_in - 60.0 / 25.4) < 1e-9


def test_figure_size_inches_unknown_width_raises(style: FigureStyle) -> None:
    with pytest.raises(ValueError):
        figure_size_inches("triple_col", 60.0, style)
