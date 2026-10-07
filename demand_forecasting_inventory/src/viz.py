"""Shared figure style: validated colour tokens, model/tier colour maps and small layout helpers.

Design rules applied to every figure (dataviz method):
* colour by *entity* with a fixed mapping (model family / demand tier), never by rank; <= 4 hues per chart, the rest in grey;
* one axis per panel (no dual axes), thin marks, hairline solid grid, text in ink colours (never in series colours);
* every chart starts from an analytical question (title) and states the computed answer (subtitle); a CSV table twin
  with the plotted numbers is written to outputs/tables for each figure family.

Palette validated with the dataviz palette validator: slots 1-4 pass adjacent-pair CVD/normal-vision gates and slots 1-3 pass
all-pairs; aqua/yellow are below 3:1 contrast on the light surface, hence direct labels + table twins are provided.
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config as C                           # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

FAMILY_COLOR = {"machine learning": BLUE, "SARIMA": ORANGE, "exponential smoothing": AQUA, "baseline": MUTED}
MODEL_FAMILY = {"Naive": "baseline", "SeasonalNaive": "baseline", "MovingAverage": "baseline", "Croston": "baseline",
                "MA7": "baseline", "MA14": "baseline", "MA28": "baseline",
                "SES": "exponential smoothing", "Holt": "exponential smoothing", "HoltWinters": "exponential smoothing",
                "SARIMA": "SARIMA", "XGBoost": "machine learning"}
TIER_COLOR = {"high": BLUE, "medium": ORANGE, "low": AQUA}

SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#1c5cab", "#6da7ec", "#f0efec", "#f09a98", "#c23030"])


def model_color(m: str) -> str:
    return FAMILY_COLOR[MODEL_FAMILY.get(m, "baseline")]


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.4, "lines.markersize": 5,
        "figure.dpi": 100, "savefig.dpi": 150, "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, AQUA, MUTED]),
    })


apply_style()


def title_block(fig, question: str, answer: str | None = None, top: float = 0.985) -> None:
    """Left-aligned analytical question (bold) and the computed answer (secondary ink, wrapped to the figure width)."""
    import textwrap
    fig.text(0.012, top, question, ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK)
    lines = 0
    if answer:
        wrapped = textwrap.fill(answer, width=max(60, int(fig.get_figwidth() * 15.5)))
        lines = wrapped.count("\n") + 1
        fig.text(0.012, top - 0.052, wrapped, ha="left", va="top", fontsize=8.8, color=INK2, linespacing=1.35)
    fig._title_lines = lines


def save(fig, name: str, rect=None) -> str:
    C.OUT_FIG.mkdir(parents=True, exist_ok=True)
    if rect is None:
        extra = max(getattr(fig, "_title_lines", 1) - 1, 0) * (0.03 * 6.0 / max(fig.get_figheight(), 4.0))
        rect = (0, 0, 1, 0.90 - extra)
    fig.tight_layout(rect=rect)
    path = C.OUT_FIG / name
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def hairline(ax, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)


def shade_periods(ax, split, label: bool = False) -> None:
    """Neutral grey background bands for the validation (light) and test (darker) periods - colour is reserved for entities."""
    d = split.dates
    ax.axvspan(d[split.train_end], d[split.val_end - 1], color="#ecebe6", lw=0, zorder=0)
    ax.axvspan(d[split.val_end], d[-1], color="#e0dfd8", lw=0, zorder=0)
    if label:
        y = ax.get_ylim()[1]
        for x0, x1, t in ((d[0], d[split.train_end - 1], "train"), (d[split.train_end], d[split.val_end - 1], "validation"), (d[split.val_end], d[-1], "test")):
            ax.text(x0 + (x1 - x0) / 2, y, t, ha="center", va="top", fontsize=7.5, color=MUTED)


def direct_label(ax, x, y, text, color=INK2, dx=4, dy=0, fontsize=8, ha="left"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=color, fontsize=fontsize, ha=ha, va="center")


def fmt_pct(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}%"
