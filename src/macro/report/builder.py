"""Phase report builder.

Every phase produces one self-contained HTML page with the same structure, so
reviewing Phase 5 feels like reviewing Phase 1. Sections are appended in order
and rendered into a single file with no external assets.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .style import CSS


@dataclass
class PhaseReport:
    phase: str                      # "Phase 1"
    title: str                      # "Data Layer"
    summary: str                    # one-paragraph standfirst
    status: str = "complete"        # complete | in progress | blocked
    project: str = "Macro_26"       # shown in the masthead rail and colophon
    _blocks: list[str] = field(default_factory=list)
    _nav: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- sections

    def section(self, heading: str, note: str | None = None) -> "PhaseReport":
        self._blocks.append(f'<h2>{html.escape(heading)}</h2>')
        if note:
            self._blocks.append(f'<p class="note">{note}</p>')
        return self

    def prose(self, text: str) -> "PhaseReport":
        self._blocks.append(f"<p>{text}</p>")
        return self

    def metrics(self, items: list[tuple[str, str, str | None]]) -> "PhaseReport":
        """Headline figures: (value, label, optional state for coloring)."""
        cells = []
        for value, label, state in items:
            cls = f" metric--{state}" if state else ""
            cells.append(
                f'<div class="metric{cls}"><span class="metric__value">{html.escape(value)}</span>'
                f'<span class="metric__label">{html.escape(label)}</span></div>'
            )
        self._blocks.append(f'<div class="metrics">{"".join(cells)}</div>')
        return self

    def table(self, df: pd.DataFrame, index: bool = True,
              align_right: list[str] | None = None,
              caption: str | None = None,
              heat: list[str] | None = None,
              heat_rows: bool = False,
              row_class: dict | None = None,
              stars: list[str] | None = None,
              compact: bool = False,
              bench_row: str | None = None) -> "PhaseReport":
        """Render a frame.

        heat        columns to shade by value, green positive, red negative.
        heat_rows   scale the shading within each row instead of each column,
                    so the reader sees what a given row is relatively good at
                    rather than which row is best at a given column.
        row_class   index label -> css class, for grouping rows by kind.
        stars       p-value columns to mark: *** below .01, ** below .05,
                    * below .10.
        """
        right = set(align_right or [])
        heat = [c for c in (heat or []) if c in df.columns]
        stars = [c for c in (stars or []) if c in df.columns]
        row_class = row_class or {}
        cols = list(df.columns)

        scale, row_scale = {}, {}
        if heat_rows and heat:
            sub = df[heat].apply(pd.to_numeric, errors="coerce").abs()
            for idx, v in sub.max(axis=1).items():
                row_scale[idx] = float(v) if v and np.isfinite(v) and v > 0 else 1.0
        for c in heat:
            v = pd.to_numeric(df[c], errors="coerce").abs().max()
            scale[c] = float(v) if v and np.isfinite(v) and v > 0 else 1.0

        head = "".join(
            f'<th class="{"num" if c in right else ""}">{html.escape(str(c))}</th>'
            for c in cols
        )
        if index:
            head = f'<th class="rowhead">{html.escape(str(df.index.name or ""))}</th>' + head

        body = []
        for idx, row in df.iterrows():
            cells = ""
            for c in cols:
                cls = "num" if c in right else ""
                style = ""
                text = _cell(row[c])
                if c in heat:
                    v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
                    if pd.notna(v):
                        denom = row_scale.get(idx, scale[c]) if heat_rows else scale[c]
                        a = min(abs(float(v)) / denom, 1.0) * 0.42
                        rgb = "46,125,90" if v > 0 else "160,58,58"
                        style = f' style="background:rgba({rgb},{a:.3f})"'
                if c in stars:
                    v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
                    if pd.notna(v):
                        m = "***" if v < .01 else "**" if v < .05 else "*" if v < .10 else ""
                        if m:
                            text = f'<strong>{text}</strong><span class="star">{m}</span>'
                cells += f'<td class="{cls}"{style}>{text}</td>'
            if index:
                cells = f'<td class="rowhead">{_cell(idx)}</td>' + cells
            rc = row_class.get(idx, "")
            body.append(f'<tr class="{rc}">{cells}</tr>' if rc else f"<tr>{cells}</tr>")

        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        self._blocks.append(
            f'<figure class="tablewrap{" compact" if compact else ""}">'
            f'<table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>{cap}</figure>'
        )
        return self

    def formula(self, body: str, caption: str | None = None) -> "PhaseReport":
        """A display equation. `body` is trusted HTML, so it can carry entities
        and per-term spans that tint each component to match the table below."""
        cap = f'<span class="formula__cap">{caption}</span>' if caption else ""
        self._blocks.append(f'<div class="formula">{body}{cap}</div>')
        return self

    def figure(self, svg: str, caption: str | None = None) -> "PhaseReport":
        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        self._blocks.append(f'<figure class="chartwrap">{svg}{cap}</figure>')
        return self

    def checks(self, rows: list[tuple[bool, str, str]]) -> "PhaseReport":
        """Pass/fail list: (passed, label, detail)."""
        items = []
        for ok, label, detail in rows:
            state = "pass" if ok else "fail"
            mark = "PASS" if ok else "FAIL"
            items.append(
                f'<li class="check check--{state}"><span class="chip chip--{state}">{mark}</span>'
                f'<span class="check__label">{label}</span>'
                f'<span class="check__detail">{detail}</span></li>'
            )
        self._blocks.append(f'<ul class="checks">{"".join(items)}</ul>')
        return self

    def findings(self, rows: list[tuple[str, str, str, str]]) -> "PhaseReport":
        """Issues found: (severity, title, what happened, resolution)."""
        cards = []
        for sev, title, what, fix in rows:
            cards.append(
                f'<article class="finding finding--{sev}">'
                f'<header><span class="chip chip--{sev}">{sev.upper()}</span>'
                f'<h3>{html.escape(title)}</h3></header>'
                f'<p class="finding__what">{what}</p>'
                f'<p class="finding__fix"><span class="finding__fixlabel">Resolution</span> {fix}</p>'
                f"</article>"
            )
        self._blocks.append(f'<div class="findings">{"".join(cards)}</div>')
        return self

    def nav(self, index: tuple[str, str] | None = None,
            prev: tuple[str, str] | None = None,
            next: tuple[str, str] | None = None) -> "PhaseReport":
        """Where this page sits in the sequence. Each argument is (href, label).

        A reader who lands on one phase from a search result or a shared link
        needs a way out of it without editing the URL, so the pager is rendered
        whether or not they arrived through the index.
        """
        self._nav = {"index": index, "prev": prev, "next": next}
        return self

    def _pager(self) -> str:
        if not self._nav:
            return ""
        def link(key, kind, arrow_before=""):
            item = self._nav.get(key)
            if not item:
                return '<span class="pager__slot"></span>'
            href, label = item
            return (f'<a class="pager__slot pager__slot--{kind}" href="{html.escape(href)}">'
                    f'<span class="pager__kind">{arrow_before}</span>'
                    f'<span class="pager__label">{html.escape(label)}</span></a>')
        return ('<nav class="pager">'
                + link("prev", "prev", "&larr; Previous")
                + link("index", "index", "All phases")
                + link("next", "next", "Next &rarr;")
                + "</nav>")

    def _crumb(self) -> str:
        item = self._nav.get("index") if self._nav else None
        if not item:
            return ""
        href, _ = item
        return (f'<a class="crumb" href="{html.escape(href)}">'
                f'&larr; {html.escape(self.project)}</a>')

    def next_up(self, phase: str, items: list[str]) -> "PhaseReport":
        lis = "".join(f"<li>{i}</li>" for i in items)
        self._blocks.append(
            f'<div class="nextup"><h2>Next: {html.escape(phase)}</h2><ul>{lis}</ul></div>'
        )
        return self

    # ------------------------------------------------------------------ render

    def render(self, path: str | Path) -> Path:
        page = f"""<title>{html.escape(self.phase)} {html.escape(self.title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="page">
  <header class="masthead">
    <div class="masthead__rail">
      {self._crumb()}
    </div>
    <h1><span class="masthead__phase">{html.escape(self.phase)}</span>{html.escape(self.title)}</h1>
    <p class="standfirst">{self.summary}</p>
  </header>
  <main>
    {"".join(self._blocks)}
  </main>
  {self._pager()}
  <footer class="colophon">
    Gus Guenther · {html.escape(self.project)} · regenerate with <code>py scripts/build_site.py</code>
  </footer>
</div>
"""
        # Escape non-ASCII to numeric entities so the page renders correctly even
        # when a host serves it without declaring a charset. Safe here because the
        # stylesheet is ASCII-only - entities are not decoded inside <style>.
        page = page.encode("ascii", "xmlcharrefreplace").decode("ascii")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(page, encoding="ascii")
        return p


def _status_class(status: str) -> str:
    return {"complete": "pass", "in progress": "warn", "blocked": "fail"}.get(status, "warn")


def _cell(v) -> str:
    if isinstance(v, str):
        return v if v.startswith("<") else html.escape(v)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return '<span class="nil">—</span>'
    return html.escape(str(v))
