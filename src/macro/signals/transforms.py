"""Signal transforms, all of them causal."""
from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(s: pd.Series, window: int = 60, min_periods: int | None = None) -> pd.Series:
    """Trailing z-score."""
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
    """Trailing compounded return over `window` periods."""
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
    """
    return s.rolling(window, min_periods=window // 4).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else np.nan, raw=True)


def apply_lag(df: pd.DataFrame, lags: dict[str, int]) -> pd.DataFrame:
    """Shift each column by its publication lag."""
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
    """Clip to trailing quantiles."""
    lo = s.rolling(window, min_periods=window // 4).quantile(lower)
    hi = s.rolling(window, min_periods=window // 4).quantile(upper)
    return s.clip(lower=lo, upper=hi)
