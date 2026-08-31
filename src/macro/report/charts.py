"""Matplotlib figures rendered as theme-aware inline SVG.

Matplotlib bakes literal colors into its SVG output, which breaks under a
light/dark theme switch. Every chart here draws its structural elements
(text, spines, ticks, grid) in sentinel colors that `to_svg` rewrites into CSS
custom properties, so the chart inherits the page's theme. Series colors are
chosen to hold contrast on both grounds and are left alone.
"""
from __future__ import annotations

import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Sentinels swapped for CSS variables after rendering.
INK = "#101010"
MUTED = "#7a7a7a"
RULE = "#cfcfcf"

# Series palette: mid-tone, legible on both light and dark grounds.
SERIES = ["#2f6f9f", "#c98a2b", "#4b8f6d", "#a4576f", "#6c6f9c", "#8a7b52"]

_SUBS = [(INK, "var(--ink)"), (MUTED, "var(--ink-muted)"), (RULE, "var(--rule)")]


def new_axes(width: float = 9.0, height: float = 3.4):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    style_axes(ax)
    return fig, ax


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
