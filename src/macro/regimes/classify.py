"""Macro regime classification: growth and inflation, rising or falling.

The 2025 study reported average GDP growth and CPI for each fixed calendar block
and reasoned informally about which environment each block represented. That is
the right intuition applied at the wrong resolution: a ten-year block spanning
1971-1980 contains both the first oil shock and the recovery from it, so its
average describes neither.

Here regimes are assigned month by month from data knowable at the time, and the
blocks are dropped entirely. Two dimensions, following the framing common to
risk-parity and all-weather research:

    growth      is activity accelerating or decelerating relative to trend
    inflation   is price pressure rising or falling relative to trend

crossed into four states:

    Goldilocks   growth up,   inflation down
    Reflation    growth up,   inflation up
    Stagflation  growth down, inflation up
    Deflation    growth down, inflation down

Everything here uses *release-aware* timing. A month's CPI is not known during
that month, so signals are lagged by the publication delay before being used to
label a period a strategy could have traded. The regime study itself is
descriptive, but the same labels feed Phase 6, where the lag is load-bearing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = ["Goldilocks", "Reflation", "Stagflation", "Deflation"]

# Publication lags in months. CPI and industrial production are released the
# following month; GDP lags a quarter and is revised for years afterward.
PUBLICATION_LAG = {"cpi": 1, "indpro": 1, "gdp": 3, "unrate": 1, "payems": 1}


def yoy(series: pd.Series, periods: int = 12) -> pd.Series:
    return series / series.shift(periods) - 1.0


def trend_deviation(series: pd.Series, window: int = 36) -> pd.Series:
    """
    How far a series sits above or below its own trailing average.

    Using a trailing window rather than a full-sample mean keeps the measure
    causal: at every date it depends only on the preceding `window` months.
    """
    return series - series.rolling(window, min_periods=window // 2).mean()


def classify(growth: pd.Series, inflation: pd.Series,
             window: int = 36) -> pd.DataFrame:
    """
    Label each month by growth and inflation direction.

    `growth` and `inflation` should already be lagged for publication delay.
    Returns the two binary signals, the underlying deviations, and the regime.
    """
    g_dev = trend_deviation(growth, window)
    i_dev = trend_deviation(inflation, window)

    g_up = g_dev > 0
    i_up = i_dev > 0

    regime = pd.Series(index=growth.index, dtype=object, name="regime")
    regime[g_up & ~i_up] = "Goldilocks"
    regime[g_up & i_up] = "Reflation"
    regime[~g_up & i_up] = "Stagflation"
    regime[~g_up & ~i_up] = "Deflation"
    regime[g_dev.isna() | i_dev.isna()] = np.nan

    return pd.DataFrame({
        "growth_dev": g_dev, "inflation_dev": i_dev,
        "growth_up": g_up, "inflation_up": i_up, "regime": regime,
    })


def build_macro_signals(panel_a: pd.DataFrame, indpro: pd.Series,
                        apply_lag: bool = True) -> pd.DataFrame:
    """
    Assemble the growth and inflation series the classifier consumes.

    Growth uses industrial production rather than GDP: it is monthly rather than
    quarterly, is released with a one-month lag rather than a quarter, and is
    revised far less. GDP is the better concept but the worse instrument for a
    monthly regime label.
    """
    # FRED stamps monthly series at the *start* of the period while the panels are
    # month-end. Reindexing across that mismatch returns all NaN and silently
    # produces an empty regime series, so resample before aligning.
    indpro_me = indpro.resample("ME").last()

    infl = yoy(panel_a["cpi"])
    grow = yoy(indpro_me.reindex(panel_a.index).ffill())

    if apply_lag:
        infl = infl.shift(PUBLICATION_LAG["cpi"])
        grow = grow.shift(PUBLICATION_LAG["indpro"])

    return pd.DataFrame({"inflation_yoy": infl, "growth_yoy": grow})


def regime_stats(returns: pd.DataFrame, regime: pd.Series,
                 rf: pd.Series | None = None,
                 periods_per_year: int = 12) -> pd.DataFrame:
    """
    Annualized return, volatility and Sharpe for each asset within each regime.

    Sharpe uses the actual risk-free rate when supplied. The 2025 study's
    equivalent tables used zero.
    """
    rows = []
    for state in REGIMES:
        mask = regime == state
        if mask.sum() < 12:
            continue
        for asset in returns.columns:
            r = returns.loc[mask, asset].dropna()
            if len(r) < 12:
                continue
            vol = r.std() * np.sqrt(periods_per_year)
            if rf is not None:
                ex = (r - rf.reindex(r.index)).dropna()
                sharpe = ex.mean() * periods_per_year / vol if vol > 0 else np.nan
            else:
                sharpe = np.nan
            rows.append({
                "regime": state, "asset": asset, "months": len(r),
                "ann_return": (1 + r).prod() ** (periods_per_year / len(r)) - 1,
                "ann_vol": vol, "sharpe": sharpe,
                "hit_rate": (r > 0).mean(),
            })
    return pd.DataFrame(rows)


def transition_matrix(regime: pd.Series) -> pd.DataFrame:
    """Empirical month-to-month transition probabilities between regimes."""
    s = regime.dropna()
    pairs = pd.DataFrame({"from": s.iloc[:-1].to_numpy(), "to": s.iloc[1:].to_numpy()})
    counts = pd.crosstab(pairs["from"], pairs["to"])
    counts = counts.reindex(index=REGIMES, columns=REGIMES).fillna(0.0)
    return counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)


def persistence(regime: pd.Series) -> pd.DataFrame:
    """Average spell length in months, and number of spells, per regime."""
    s = regime.dropna()
    grp = (s != s.shift()).cumsum()
    spells = s.groupby(grp).agg(["first", "size"])
    out = spells.groupby("first")["size"].agg(["mean", "count", "max"])
    out.columns = ["mean_months", "n_spells", "longest"]
    return out.reindex(REGIMES)
