"""Alternative regime definitions, and a way to compare them on merit."""
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
    """Fixed thresholds on the levels."""
    out = _label(growth > growth_threshold, inflation > inflation_threshold)
    out[growth.isna() | inflation.isna()] = np.nan
    return out


def hybrid(growth: pd.Series, inflation: pd.Series, window: int = 36,
           growth_floor: float = 0.005, inflation_floor: float = 0.025) -> pd.Series:
    """
    Direction from the trailing comparison, but only where the level agrees.
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
    """Position within a long trailing distribution."""
    def rank(s: pd.Series) -> pd.Series:
        return s.rolling(window, min_periods=window // 4).apply(
            lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else np.nan, raw=True)

    g, i = rank(growth), rank(inflation)
    out = _label(g > cutoff, i > cutoff)
    out[g.isna() | i.isna()] = np.nan
    return out


def enforce_persistence(regimes: pd.Series, min_months: int = 3) -> pd.Series:
    """
    Drop episodes shorter than `min_months`, carrying the prior label
    forward.
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
    """How differently do assets behave across a scheme's states?"""
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
