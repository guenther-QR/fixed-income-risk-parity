"""Stylesheet for phase reports.

Colors are defined as tokens on bare :root (light), then redefined for
prefers-color-scheme:dark and again for an explicit [data-theme] stamp, so the
page resolves correctly in all three viewer states.
"""

CSS = """
:root {
  /* Neutrals carry a slight blue-green bias - chosen, not inherited grey. */
  --ground:      #f7f8f7;
  --surface:     #ffffff;
  --surface-alt: #eef1f0;
  --ink:         #16201f;
  --ink-muted:   #5d6b69;
  --rule:        #d5dbd9;
  --accent:      #1d4e5f;
  --accent-soft: #e2ecef;

  --pass: #2f6b4f;  --pass-bg: #e4f0e9;
  --warn: #8a6220;  --warn-bg: #f6ecd9;
  --fail: #9a3535;  --fail-bg: #f6e3e3;
  --high: #9a3535;  --high-bg: #f6e3e3;
  --med:  #8a6220;  --med-bg:  #f6ecd9;
  --low:  #4a5b6b;  --low-bg:  #e6ecf1;

  --t1: #4169e1;  --t2: #1f9c7a;  --t3: #cf6a1a;  --t4: #a83c66;

  --measure: 68ch;
  --serif: "IBM Plex Serif", Georgia, "Times New Roman", serif;
  --sans:  "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono:  "IBM Plex Mono", ui-monospace, "SF Mono", Consolas, monospace;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:      #0f1514;
    --surface:     #161f1e;
    --surface-alt: #1d2827;
    --ink:         #e6ecea;
    --ink-muted:   #94a3a0;
    --rule:        #2c3937;
    --accent:      #7fb8c9;
    --accent-soft: #1b2e34;

    --pass: #7dc39c;  --pass-bg: #172a21;
    --warn: #d9ac63;  --warn-bg: #2c2317;
    --fail: #e08a8a;  --fail-bg: #2e1c1c;
    --high: #e08a8a;  --high-bg: #2e1c1c;
    --med:  #d9ac63;  --med-bg:  #2c2317;
    --low:  #9db2c4;  --low-bg:  #1b242c;

    --t1: #8aa4f0;  --t2: #5fc9a9;  --t3: #e8a25e;  --t4: #d98aa6;
  }
}

:root[data-theme="dark"] {
  --ground:      #0f1514;
  --surface:     #161f1e;
  --surface-alt: #1d2827;
  --ink:         #e6ecea;
  --ink-muted:   #94a3a0;
  --rule:        #2c3937;
  --accent:      #7fb8c9;
  --accent-soft: #1b2e34;

  --pass: #7dc39c;  --pass-bg: #172a21;
  --warn: #d9ac63;  --warn-bg: #2c2317;
  --fail: #e08a8a;  --fail-bg: #2e1c1c;
  --high: #e08a8a;  --high-bg: #2e1c1c;
  --med:  #d9ac63;  --med-bg:  #2c2317;
  --low:  #9db2c4;  --low-bg:  #1b242c;

  --t1: #8aa4f0;  --t2: #5fc9a9;  --t3: #e8a25e;  --t4: #d98aa6;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 16px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.page { max-width: 68rem; margin: 0 auto; padding: 3rem 1.5rem 5rem; }

/* ---------------------------------------------------------------- masthead */

.masthead { border-bottom: 2px solid var(--ink); padding-bottom: 1.75rem; margin-bottom: 2.5rem; }
.masthead__rail { display: flex; align-items: center; gap: .75rem; margin-bottom: 1.25rem; }
.eyebrow {
  font-family: var(--mono); font-size: .75rem; font-weight: 500;
  letter-spacing: .14em; text-transform: uppercase; color: var(--ink-muted);
}
.masthead h1 {
  font-family: var(--serif); font-weight: 600; font-size: clamp(2rem, 5vw, 2.9rem);
  line-height: 1.1; margin: 0 0 .85rem; text-wrap: balance; letter-spacing: -.015em;
}
.masthead__phase {
  display: block; font-family: var(--mono); font-size: .82rem; font-weight: 500;
  letter-spacing: .16em; text-transform: uppercase; color: var(--accent);
  margin-bottom: .5rem;
}
.standfirst {
  font-size: 1.1rem; color: var(--ink-muted); max-width: var(--measure);
  margin: 0 0 1rem; text-wrap: pretty;
}
.built { font-family: var(--mono); font-size: .78rem; color: var(--ink-muted); margin: 0; }

/* ----------------------------------------------------------------- headings */

h2 {
  font-family: var(--serif); font-size: 1.5rem; font-weight: 600;
  margin: 3rem 0 .5rem; padding-top: 1.25rem; border-top: 1px solid var(--rule);
  text-wrap: balance; letter-spacing: -.01em;
}
h3 { font-family: var(--sans); font-size: 1rem; font-weight: 600; margin: 0; }
p { max-width: var(--measure); text-wrap: pretty; }
p.note { color: var(--ink-muted); margin-top: 0; }
code { font-family: var(--mono); font-size: .88em; }
strong { font-weight: 600; }

/* ------------------------------------------------------------------ metrics */

.metrics {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
  gap: 1px; background: var(--rule); border: 1px solid var(--rule);
  margin: 1.75rem 0;
}
.metric { background: var(--surface); padding: 1.15rem 1.25rem; display: flex; flex-direction: column; gap: .3rem; }
.metric__value {
  font-family: var(--mono); font-size: 1.65rem; font-weight: 500;
  font-variant-numeric: tabular-nums; line-height: 1; letter-spacing: -.02em;
}
.metric__label {
  font-size: .74rem; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-muted);
}
.metric--pass .metric__value { color: var(--pass); }
.metric--warn .metric__value { color: var(--warn); }
.metric--fail .metric__value { color: var(--fail); }

/* ------------------------------------------------------------------- tables */

.tablewrap { margin: 1.5rem 0; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; min-width: 30rem; }
thead th {
  text-align: left; font-family: var(--mono); font-weight: 500; font-size: .72rem;
  letter-spacing: .08em; text-transform: uppercase; color: var(--ink-muted);
  padding: .55rem .8rem; border-bottom: 1px solid var(--ink); white-space: nowrap;
}
tbody td { padding: .55rem .8rem; border-bottom: 1px solid var(--rule); }
tbody tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; }
td.rowhead, th.rowhead { font-weight: 500; white-space: nowrap; }
.nil { color: var(--ink-muted); }
figcaption {
  font-size: .82rem; color: var(--ink-muted); margin-top: .7rem;
  max-width: var(--measure); text-wrap: pretty;
}

/* ------------------------------------------------------------------- charts */

.chartwrap { margin: 1.75rem 0; }
.chart { width: 100%; height: auto; display: block; overflow: visible; }

/* -------------------------------------------------------------------- chips */

.chip {
  display: inline-block; font-family: var(--mono); font-size: .68rem; font-weight: 500;
  letter-spacing: .07em; text-transform: uppercase; padding: .2rem .5rem;
  border-radius: 2px; white-space: nowrap;
}
.chip--pass { color: var(--pass); background: var(--pass-bg); }
.chip--warn { color: var(--warn); background: var(--warn-bg); }
.chip--fail { color: var(--fail); background: var(--fail-bg); }
.chip--high { color: var(--high); background: var(--high-bg); }
.chip--med  { color: var(--med);  background: var(--med-bg); }
.chip--low  { color: var(--low);  background: var(--low-bg); }

/* ------------------------------------------------------------------- checks */

.checks { list-style: none; padding: 0; margin: 1.5rem 0; border-top: 1px solid var(--rule); }
.check {
  display: grid; grid-template-columns: 3.6rem minmax(0, 1fr) minmax(0, 1.1fr);
  gap: .9rem; align-items: baseline;
  padding: .7rem .25rem; border-bottom: 1px solid var(--rule);
}
.check__label { font-family: var(--mono); font-size: .85rem; }
.check__detail { color: var(--ink-muted); font-size: .85rem; }
@media (max-width: 40rem) {
  .check { grid-template-columns: 3.6rem minmax(0, 1fr); }
  .check__detail { grid-column: 2; }
}

/* ----------------------------------------------------------------- findings */

.findings { display: grid; gap: 1rem; margin: 1.5rem 0; }
.finding {
  background: var(--surface); border: 1px solid var(--rule);
  border-left: 3px solid var(--rule); padding: 1.1rem 1.25rem;
}
.finding--high { border-left-color: var(--high); }
.finding--med  { border-left-color: var(--med); }
.finding--low  { border-left-color: var(--low); }
.finding header { display: flex; align-items: center; gap: .7rem; margin-bottom: .6rem; flex-wrap: wrap; }
.finding p { margin: .4rem 0; font-size: .92rem; max-width: none; }
.finding__what { color: var(--ink); }
.finding__fix { color: var(--ink-muted); }
.finding__fixlabel {
  font-family: var(--mono); font-size: .68rem; letter-spacing: .07em;
  text-transform: uppercase; color: var(--accent);
}

/* ------------------------------------------------------------------- nextup */

.nextup {
  margin-top: 3rem; padding: 1.5rem 1.75rem;
  background: var(--accent-soft); border: 1px solid var(--rule);
}
.nextup h2 { margin: 0 0 .6rem; padding: 0; border: none; font-size: 1.2rem; }
.nextup ul { margin: 0; padding-left: 1.1rem; }
.nextup li { margin: .3rem 0; font-size: .93rem; }

/* ---------------------------------------------------------- table extras */

.star { font-size: .7em; vertical-align: super; color: var(--pass); margin-left: 1px; }
tr.k-risk    td.rowhead { border-left: 3px solid var(--t1); padding-left: .55rem; }
tr.k-reg     td.rowhead { border-left: 3px solid var(--t2); padding-left: .55rem; }
tr.k-ml      td.rowhead { border-left: 3px solid var(--t3); padding-left: .55rem; }
tr.k-tech    td.rowhead { border-left: 3px solid var(--t4); padding-left: .55rem; }
tr.k-return  td.rowhead { border-left: 3px solid var(--t3); padding-left: .55rem; }
tr.k-bench   td.rowhead { border-left: 3px solid var(--ink-muted); padding-left: .55rem; }
tr.k-bench   td { color: var(--ink-muted); }
tr.k-reg     td.rowhead, tr.k-risk td.rowhead { font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 1.1rem; margin: .5rem 0 0;
          font-size: .78rem; color: var(--ink-muted); }
.legend span::before { content: ""; display: inline-block; width: 10px;
          height: 3px; margin-right: .4rem; vertical-align: middle; }
.legend .l-risk::before   { background: var(--t1); }
.legend .l-reg::before    { background: var(--t2); }
.legend .l-ml::before     { background: var(--t3); }
.legend .l-tech::before   { background: var(--t4); }
.legend .l-return::before { background: var(--t3); }
.legend .l-bench::before  { background: var(--ink-muted); }

/* -------------------------------------------------------------- formula */

.formula {
  margin: 1.5rem 0; padding: 1.25rem 1rem; text-align: center;
  border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
  font-family: var(--serif, Georgia, serif); font-size: 1.25rem;
  line-height: 1.9; color: var(--ink);
}
.formula .t { white-space: nowrap; font-weight: 600; }
.formula .t1 { color: var(--t1); }
.formula .t2 { color: var(--t2); }
.formula .t3 { color: var(--t3); }
.formula .t4 { color: var(--t4); }
.formula__cap {
  display: block; margin-top: .6rem; font-family: var(--mono);
  font-size: .72rem; letter-spacing: .1em; text-transform: uppercase;
  color: var(--ink-muted);
}

/* ------------------------------------------------------------------ nav */

.crumb {
  font-family: var(--mono); font-size: .72rem; font-weight: 500;
  letter-spacing: .14em; text-transform: uppercase; color: var(--ink-muted);
  text-decoration: none; border-bottom: 1px solid transparent;
}
.crumb:hover { color: var(--accent); border-bottom-color: var(--accent); }

.pager {
  display: grid; grid-template-columns: 1fr auto 1fr; gap: .75rem;
  align-items: stretch; margin-top: 3.5rem; padding-top: 1.5rem;
  border-top: 2px solid var(--ink);
}
.pager__slot {
  display: flex; flex-direction: column; gap: .2rem; padding: .85rem 1rem;
  border: 1px solid var(--rule); border-radius: 3px; text-decoration: none;
  color: inherit; background: var(--card, transparent);
}
a.pager__slot:hover { border-color: var(--accent); }
span.pager__slot { border-color: transparent; }  /* first and last page */
.pager__slot--next { text-align: right; }
.pager__slot--index { justify-content: center; text-align: center; }
.pager__kind {
  font-family: var(--mono); font-size: .68rem; letter-spacing: .12em;
  text-transform: uppercase; color: var(--ink-muted);
}
.pager__label { font-weight: 600; font-size: .95rem; }
.pager__slot--index .pager__label { display: none; }

@media (max-width: 34rem) {
  .pager { grid-template-columns: 1fr; }
  .pager__slot--next { text-align: left; }
  .pager__slot--index { text-align: left; justify-content: flex-start; }
}

.colophon {
  margin-top: 2rem; padding-top: 1.25rem; border-top: 1px solid var(--rule);
  font-family: var(--mono); font-size: .75rem; color: var(--ink-muted);
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""
