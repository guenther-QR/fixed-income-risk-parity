"""Signal transforms, all of them causal.

Every function here computes a value at time t from data at or before t. That
sounds obvious and is the single easiest thing to get wrong: a full-sample
z-score, a centred moving average, or a `fillna` that propagates backwards each
leak future information into a signal, and each produces a backtest that cannot
be reproduced live.

The rule applied throughout: if a transform needs a mean, a standard deviation,
or a rank, it comes from a trailing window, never from the whole sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(s: pd.Series, window: int = 60, min_periods: int | None = None) -> pd.Series:
    """
    Trailing z-score.

    A full-sample z-score is the classic look-ahead in a signal library: the mean
    and standard deviation it standardizes by are not knowable until the sample
    ends, so the resulting signal quietly encodes the future.
    """
    mp = min_periods or max(12, window // 3)
    mu = s.rolling(window, min_periods=mp).mean()
    sd = s.rolling(window, min_periods=mp).std()
    return ((s - mu) / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def trend_deviation(s: pd.Series, window: int = 36) -> pd.Series:
    """Level minus its own trailing mean."""
    return s - s.rolling(window, min_periods=max(6, window // 3)).mean()


def yoy(s: pd.Series, periods: int = 12) -> pd.Series:
    return s / s.shift(periods) - 1.0


def diff(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.diff(periods)


def momentum(prices_or_returns: pd.Series, window: int, is_return: bool = True
             ) -> pd.Series:
    """
    Trailing compounded return over `window` periods.

    Skips nothing: the value at t uses returns through t, which is knowable at t.
    Reversal variants that skip the most recent month are built explicitly where
    wanted rather than assumed here.
    """
    if is_return:
        return (1 + prices_or_returns).rolling(window).apply(np.prod, raw=True) - 1
    return prices_or_returns / prices_or_returns.shift(window) - 1


def realized_vol(returns: pd.Series, window: int = 12,
                 periods_per_year: int = 12) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(periods_per_year)


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 36) -> pd.Series:
    return a.rolling(window, min_periods=window // 2).corr(b)


def rank_pct(s: pd.Series, window: int = 120) -> pd.Series:
    """
    Where the current value sits within its own trailing history, in [0, 1].

    Robust to outliers and to level shifts, which matters for series like the
    unemployment rate whose mean drifts across decades.
    """
    return s.rolling(window, min_periods=window // 4).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else np.nan, raw=True)


def apply_lag(df: pd.DataFrame, lags: dict[str, int]) -> pd.DataFrame:
    """
    Shift each column by its publication lag.

    Columns absent from `lags` are left alone, so anything not explicitly
    classified stays unshifted - which is safe only for prices and yields.
    Everything derived from a statistical release must appear in the map.
    """
    out = df.copy()
    for col, k in lags.items():
        if col in out.columns and k:
            out[col] = out[col].shift(k)
    return out


def standardize_panel(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Trailing z-score every column, for models that need comparable scales."""
    return pd.DataFrame({c: zscore(df[c], window) for c in df.columns},
                        index=df.index)


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99,
              window: int = 120) -> pd.Series:
    """
    Clip to trailing quantiles.

    Trailing rather than full-sample, for the same reason as the z-score: the
    clipping bounds must be knowable at the time they are applied.
    """
    lo = s.rolling(window, min_periods=window // 4).quantile(lower)
    hi = s.rolling(window, min_periods=window // 4).quantile(upper)
    return s.clip(lower=lo, upper=hi)
