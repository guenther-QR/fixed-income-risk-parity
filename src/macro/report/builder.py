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

import pandas as pd

from .style import CSS


@dataclass
class PhaseReport:
    phase: str                      # "Phase 1"
    title: str                      # "Data Layer"
    summary: str                    # one-paragraph standfirst
    status: str = "complete"        # complete | in progress | blocked
    _blocks: list[str] = field(default_factory=list)

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
              caption: str | None = None) -> "PhaseReport":
        right = set(align_right or [])
        cols = list(df.columns)

        head = "".join(
            f'<th class="{"num" if c in right else ""}">{html.escape(str(c))}</th>'
            for c in cols
        )
        if index:
            head = f'<th class="rowhead">{html.escape(str(df.index.name or ""))}</th>' + head

        body = []
        for idx, row in df.iterrows():
            cells = "".join(
                f'<td class="{"num" if c in right else ""}">{_cell(row[c])}</td>' for c in cols
            )
            if index:
                cells = f'<td class="rowhead">{_cell(idx)}</td>' + cells
            body.append(f"<tr>{cells}</tr>")

        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        self._blocks.append(
            f'<figure class="tablewrap"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>{cap}</figure>'
        )
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

    def next_up(self, phase: str, items: list[str]) -> "PhaseReport":
        lis = "".join(f"<li>{i}</li>" for i in items)
        self._blocks.append(
            f'<div class="nextup"><h2>Next — {html.escape(phase)}</h2><ul>{lis}</ul></div>'
        )
        return self

    # ------------------------------------------------------------------ render

    def render(self, path: str | Path) -> Path:
        built = datetime.now().strftime("%d %B %Y, %H:%M")
        page = f"""<title>{html.escape(self.phase)} {html.escape(self.title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;600&display=swap">
<style>{CSS}</style>
<div class="page">
  <header class="masthead">
    <div class="masthead__rail">
      <span class="eyebrow">Macro_26</span>
      <span class="chip chip--{_status_class(self.status)}">{html.escape(self.status)}</span>
    </div>
    <h1><span class="masthead__phase">{html.escape(self.phase)}</span>{html.escape(self.title)}</h1>
    <p class="standfirst">{self.summary}</p>
    <p class="built">Generated {built}</p>
  </header>
  <main>
    {"".join(self._blocks)}
  </main>
  <footer class="colophon">
    Gus Guenther · Macro_26 · regenerate with <code>py scripts/report_{self.phase.lower().replace(' ', '')}.py</code>
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
