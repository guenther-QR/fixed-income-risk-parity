"""Performance metrics.

The 2025 study reported annualized return, volatility, a Sharpe ratio computed
against a zero risk-free rate, and maximum drawdown. This set adds the measures
that distinguish a strategy from a lucky path: downside-only risk, the ratio of
return to worst loss, how long recovery took, and how much trading was required
to produce the result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown(returns: pd.Series) -> pd.Series:
    cum = (1 + returns.fillna(0)).cumprod()
    return cum / cum.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown(returns).min())


def time_under_water(returns: pd.Series) -> int:
    """Longest run of consecutive periods spent below a previous peak."""
    dd = drawdown(returns)
    under = (dd < -1e-12).to_numpy()
    best = run = 0
    for u in under:
        run = run + 1 if u else 0
        best = max(best, run)
    return int(best)


def sortino(returns: pd.Series, rf: pd.Series | None = None,
            periods_per_year: int = 12) -> float:
    """
    Return per unit of *downside* deviation.

    Volatility penalises upside and downside alike. For an allocation whose whole
    claim is asymmetric protection, that is the wrong denominator.
    """
    ex = returns - rf.reindex(returns.index) if rf is not None else returns
    ex = ex.dropna()
    downside = ex[ex < 0]
    if downside.empty:
        return float("inf")
    dd = np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year)
    return float(ex.mean() * periods_per_year / dd) if dd > 0 else np.nan


def performance(returns: pd.Series, rf: pd.Series | None = None,
                turnover: pd.Series | None = None,
                periods_per_year: int = 12) -> dict:
    """The full metric set for one return series."""
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {}

    vol = r.std() * np.sqrt(periods_per_year)
    cagr = (1 + r).prod() ** (periods_per_year / n) - 1
    mdd = max_drawdown(r)

    out = {
        "cagr": float(cagr),
        "ann_return": float(r.mean() * periods_per_year),
        "vol": float(vol),
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
        "sortino": sortino(r, rf, periods_per_year),
        "hit_rate": float((r > 0).mean()),
        "worst_period": float(r.min()),
        "time_under_water": time_under_water(r),
        "periods": n,
    }

    if rf is not None:
        ex = (r - rf.reindex(r.index)).dropna()
        out["sharpe"] = float(ex.mean() * periods_per_year / vol) if vol > 0 else np.nan
        out["excess_return"] = float(ex.mean() * periods_per_year)
    else:
        out["sharpe"] = float(r.mean() * periods_per_year / vol) if vol > 0 else np.nan

    if turnover is not None:
        t = turnover.reindex(r.index).fillna(0)
        out["turnover_pa"] = float(t.mean() * periods_per_year)
        out["trades_pa"] = float((t > 1e-9).mean() * periods_per_year)

    return out


def rolling_sharpe(returns: pd.Series, rf: pd.Series | None = None,
                   window: int = 36, periods_per_year: int = 12) -> pd.Series:
    ex = returns - rf.reindex(returns.index) if rf is not None else returns
    mean = ex.rolling(window).mean() * periods_per_year
    vol = returns.rolling(window).std() * np.sqrt(periods_per_year)
    return mean / vol


def comparison_table(results: dict[str, pd.Series], rf: pd.Series | None = None,
                     turnovers: dict[str, pd.Series] | None = None,
                     periods_per_year: int = 12) -> pd.DataFrame:
    rows = {}
    for name, series in results.items():
        t = (turnovers or {}).get(name)
        rows[name] = performance(series, rf, t, periods_per_year)
    return pd.DataFrame(rows).T
