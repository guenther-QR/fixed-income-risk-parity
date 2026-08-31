"""Alternative regime definitions, and a way to compare them on merit.

The original scheme in `classify.py` is purely relative: growth below its own
trailing 36-month average, inflation above its own. Nothing anchors it to a
level, and an audit showed what that produced - 64 of 137 months labelled
"Stagflation" had *positive* industrial production growth and below-average
inflation. The 1990s instance averaged 4.4% CPI with IP growing 1.5%; the 2000s
instance 3.6% CPI with IP at +0.5%. Those are not stagflation by any economic
meaning, and pooling them with 1974 asks a single set of portfolio weights to
describe two unrelated environments.

That is a plausible cause of the regime failure rather than a detail. If the
label mixes states, the in-regime mean is an average across them, and it will be
neither large nor stable.

Five schemes are provided so the choice can be made on evidence:

    relative      the original, kept as the baseline to beat
    absolute      fixed thresholds on the levels themselves
    hybrid        relative direction, but only when levels are also extreme
    percentile    position within a long backward-looking distribution
    persistent    any scheme, with short episodes filtered out

`separation` scores them by how differently assets behave across their states,
which is the only property that matters for allocation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = ["Goldilocks", "Reflation", "Stagflation", "Deflation"]


def _label(growth_up: pd.Series, infl_up: pd.Series) -> pd.Series:
    out = pd.Series(index=growth_up.index, dtype=object)
    out[growth_up & ~infl_up] = "Goldilocks"
    out[growth_up & infl_up] = "Reflation"
    out[~growth_up & infl_up] = "Stagflation"
    out[~growth_up & ~infl_up] = "Deflation"
    return out


def relative(growth: pd.Series, inflation: pd.Series, window: int = 36) -> pd.Series:
    """The original scheme: each series against its own trailing mean."""
    g = growth - growth.rolling(window, min_periods=window // 3).mean()
    i = inflation - inflation.rolling(window, min_periods=window // 3).mean()
    out = _label(g > 0, i > 0)
    out[g.isna() | i.isna()] = np.nan
    return out


def absolute(growth: pd.Series, inflation: pd.Series,
             growth_threshold: float = 0.01,
             inflation_threshold: float = 0.03) -> pd.Series:
    """
    Fixed thresholds on the levels.

    Stagflation means growth genuinely weak *and* inflation genuinely high - not
    merely each moving the wrong way relative to a recent average. Defaults of 1%
    industrial production growth and 3% CPI are conventional rather than fitted;
    they are stated here so they can be argued with.
    """
    out = _label(growth > growth_threshold, inflation > inflation_threshold)
    out[growth.isna() | inflation.isna()] = np.nan
    return out


def hybrid(growth: pd.Series, inflation: pd.Series, window: int = 36,
           growth_floor: float = 0.005, inflation_floor: float = 0.025) -> pd.Series:
    """
    Direction from the trailing comparison, but only where the level agrees.

    A month qualifies as inflationary only if inflation is both rising *and*
    above the floor. This keeps the responsiveness of the relative scheme while
    refusing to call 1.8% CPI an inflation regime.
    """
    g_dev = growth - growth.rolling(window, min_periods=window // 3).mean()
    i_dev = inflation - inflation.rolling(window, min_periods=window // 3).mean()

    g_up = (g_dev > 0) | (growth > growth_floor)
    i_up = (i_dev > 0) & (inflation > inflation_floor)

    out = _label(g_up, i_up)
    out[g_dev.isna() | i_dev.isna()] = np.nan
    return out


def percentile(growth: pd.Series, inflation: pd.Series,
               window: int = 240, cutoff: float = 0.5) -> pd.Series:
    """
    Position within a long trailing distribution.

    A twenty-year window adapts to structural change far more slowly than three
    years, so it does not redefine "high inflation" every business cycle. Still
    causal: the distribution at each date uses only prior data.
    """
    def rank(s: pd.Series) -> pd.Series:
        return s.rolling(window, min_periods=window // 4).apply(
            lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else np.nan, raw=True)

    g, i = rank(growth), rank(inflation)
    out = _label(g > cutoff, i > cutoff)
    out[g.isna() | i.isna()] = np.nan
    return out


def enforce_persistence(regimes: pd.Series, min_months: int = 3) -> pd.Series:
    """
    Drop episodes shorter than `min_months`, carrying the prior label forward.

    41% of episodes under the original scheme lasted two months or fewer. With a
    one-month signal lag those are untradeable by construction, and they add
    turnover without adding information.
    """
    s = regimes.copy()
    out = s.copy()
    grp = (s != s.shift()).cumsum()

    prev = None
    for _, block in s.groupby(grp):
        if block.isna().all():
            continue
        if len(block) < min_months and prev is not None:
            out.loc[block.index] = prev
        else:
            prev = block.iloc[0]
    return out


# ------------------------------------------------------------------ scoring

def separation(returns: pd.DataFrame, regimes: pd.Series, rf: pd.Series,
               periods_per_year: int = 12) -> dict:
    """
    How differently do assets behave across a scheme's states?

    A regime scheme is only useful for allocation if the assets it is meant to
    choose between actually behave differently across its states. `spread` is
    the average across assets of (best regime mean - worst regime mean); `f_stat`
    is a one-way ANOVA on the pooled cross-section. Neither is a trading result -
    both are in-sample by construction - but a scheme that fails here cannot
    possibly work out of sample.
    """
    ex = returns.sub(rf.reindex(returns.index), axis=0).dropna()
    g = regimes.reindex(ex.index)

    spreads, fs = [], []
    for a in ex.columns:
        groups = [ex.loc[g == s, a].dropna() for s in REGIMES]
        groups = [x for x in groups if len(x) >= 12]
        if len(groups) < 2:
            continue
        means = [x.mean() * periods_per_year for x in groups]
        spreads.append(max(means) - min(means))

        grand = np.concatenate([x.to_numpy() for x in groups]).mean()
        ssb = sum(len(x) * (x.mean() - grand) ** 2 for x in groups)
        ssw = sum(((x - x.mean()) ** 2).sum() for x in groups)
        k, n = len(groups), sum(len(x) for x in groups)
        if ssw > 0 and n > k:
            fs.append((ssb / (k - 1)) / (ssw / (n - k)))

    counts = g.value_counts()
    eps = (g != g.shift()).cumsum()
    lengths = g.dropna().groupby(eps).size()

    return {
        "mean_spread": float(np.mean(spreads)) if spreads else np.nan,
        "mean_f_stat": float(np.mean(fs)) if fs else np.nan,
        "n_labelled": int(g.notna().sum()),
        "n_states_used": int(counts.gt(11).sum()),
        "balance": float(counts.min() / counts.max()) if len(counts) else np.nan,
        "median_episode": float(lengths.median()) if len(lengths) else np.nan,
        "pct_episodes_le2": float((lengths <= 2).mean()) if len(lengths) else np.nan,
    }
