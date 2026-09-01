"""Matplotlib figures rendered as theme-aware inline SVG.

Matplotlib bakes literal colors into its SVG output, which breaks under a
light/dark theme switch. Every chart here draws its structural elements
(text, spines, ticks, grid) in sentinel colors that `to_svg` rewrites into CSS
custom properties, so the chart inherits the page's theme. Series colors are
chosen to hold contrast on both grounds and are left alone.

Two conventions the reports rely on:

* **No log scales.** Axes are read in dollars or percent. A log axis makes a
  ninefold difference look like a small one, which is exactly the difference a
  reader is trying to judge. Where a series genuinely spans two orders of
  magnitude, split it across panels rather than compressing the axis.
* **Role, not just color.** A chart usually has one line that matters, a few
  that compete with it, and a benchmark. `ROLES` gives each a distinct weight,
  dash pattern and marker so the hierarchy survives greyscale printing and
  colorblind viewers.
"""
from __future__ import annotations

import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Sentinels swapped for CSS variables after rendering.
INK = "#101010"
MUTED = "#7a7a7a"
RULE = "#cfcfcf"

# Series palette. Mid-tone and reasonably far apart in hue and in lightness, so
# the lines stay separable on a light ground, on a dark ground, and in grey.
ROYAL = "#4169e1"       # royal blue, reserved for the featured series
SERIES = [
    ROYAL,      # royal blue
    "#cf6a1a",  # burnt orange
    "#1f9c7a",  # teal green
    "#a83c66",  # rose
    "#7a52c8",  # violet
    "#9c7d1e",  # ochre
    "#2f8f5b",  # green
]
NEUTRAL = "#8b949e"

# Benchmarks are drawn without color: a fine dotted line in the page's own ink,
# then progressively looser grey dashes if a chart carries more than one. The
# point is that they read as reference lines rather than as competing series.
BENCH = [
    dict(color=INK, linestyle=(0, (1.3, 1.9)), linewidth=1.5),
    dict(color=MUTED, linestyle=(0, (6, 2.4)), linewidth=1.3),
    dict(color=MUTED, linestyle=(0, (6, 2, 1.3, 2)), linewidth=1.3),
]

# Line treatments by role. `markevery` is a fraction of the line's own path
# length, so markers stay evenly spaced regardless of series length.
ROLES = {
    "hero":      dict(linewidth=2.6, zorder=6, marker="o", markersize=4.6,
                      markevery=0.09, markeredgecolor="none", color=ROYAL),
    "strategy":  dict(linewidth=1.6, zorder=4),
    "secondary": dict(linewidth=1.3, zorder=3, alpha=0.9),
    "benchmark": dict(zorder=5),
}

_SUBS = [(INK, "var(--ink)"), (MUTED, "var(--ink-muted)"), (RULE, "var(--rule)")]


def new_axes(width: float = 9.0, height: float = 3.4):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    style_axes(ax)
    return fig, ax


def new_stacked(width: float = 9.0, height: float = 5.0,
                ratios: tuple[int, int] = (1, 1)):
    """Two panels sharing an x-axis, for series that differ by an order of
    magnitude. Splitting them is the alternative to a log scale."""
    fig, axes = plt.subplots(2, 1, figsize=(width, height), sharex=True,
                             gridspec_kw={"height_ratios": list(ratios),
                                          "hspace": 0.16})
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.patch.set_alpha(0.0)
        style_axes(ax)
    return fig, axes


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)
    ax.grid(True, color=RULE, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(MUTED)
        label.set_fontsize(9)
    ax.title.set_color(INK)
    ax.title.set_fontsize(10.5)


def style_for(role: str, index: int = 0) -> dict:
    """Line keywords for a role.

    For "benchmark", `index` selects a dash pattern from BENCH so several
    reference lines on one chart stay distinguishable without taking a color.
    For every other role it selects the palette color; "hero" ignores it and
    always draws royal blue.
    """
    kw = dict(ROLES.get(role, ROLES["strategy"]))
    if role == "benchmark":
        kw.update(BENCH[index % len(BENCH)])
        return kw
    kw.setdefault("color", SERIES[index % len(SERIES)])
    return kw


def dollar_axis(ax, decimals: int | None = None) -> None:
    """Label the y-axis in dollars. Call after plotting, so the tick precision
    can be chosen from the range actually drawn."""
    if decimals is None:
        decimals = 2 if ax.get_ylim()[1] < 10 else 0
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"${v:,.{decimals}f}"))


def percent_axis(ax, decimals: int = 0, already_percent: bool = True) -> None:
    """Label the y-axis in percent. `already_percent` is True when the plotted
    values are 12.4 rather than 0.124."""
    scale = 1.0 if already_percent else 100.0
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v * scale:,.{decimals}f}%"))


def growth(ax, frame, roles: dict[str, str] | None = None,
           start: float = 1.0, order: list[str] | None = None):
    """Plot growth of one dollar for every column, on a linear dollar axis.

    `roles` maps a column name to "hero", "strategy", "secondary" or
    "benchmark"; anything unlisted is drawn as a strategy. Palette colors are
    assigned in `order` (default: the frame's own column order) so that a
    column keeps the same color across every chart in a report.
    """
    roles = roles or {}
    cols = order or list(frame.columns)
    # Benchmarks are counted separately: their index picks a dash pattern, not
    # a palette slot, so adding a benchmark never shifts a strategy's color.
    colored = benched = 0
    for c in cols:
        if c not in frame.columns:
            continue
        role = roles.get(c, "strategy")
        if role == "benchmark":
            kw, benched = style_for(role, benched), benched + 1
        else:
            kw, colored = style_for(role, colored), colored + 1
        s = frame[c].dropna()
        ax.plot(s.index, start * (1 + s).cumprod(), label=c, **kw)
    ax.set_ylabel("growth of $1")
    dollar_axis(ax)
    return ax


def legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8.5, **kw)
    for txt in leg.get_texts():
        txt.set_color(INK)
    return leg


def to_svg(fig) -> str:
    """Render `fig` to an inline SVG string with theme-aware colors."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True,
                metadata={"Date": None})
    plt.close(fig)
    svg = buf.getvalue()

    svg = svg[svg.index("<svg"):]                        # drop XML decl and DOCTYPE
    svg = re.sub(r'(width|height)="[\d.]+pt"', "", svg, count=2)
    svg = svg.replace("<svg ", '<svg class="chart" preserveAspectRatio="xMidYMid meet" ', 1)

    for sentinel, var in _SUBS:
        svg = svg.replace(sentinel, var).replace(sentinel.upper(), var)
    return svg
