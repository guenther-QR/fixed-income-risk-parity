"""Leverage-aware performance comparison.

Comparing Sharpe ratios across portfolios of very different volatility carries a
hidden assumption: that the low-volatility one can be levered up to match the
high-volatility one, borrowing at the risk-free rate. That is two-fund separation,
and it is false in three separate ways.

**Borrowing costs more than the risk-free rate.** An institution finances at
something like rf plus 25 to 50 basis points; a retail investor pays far more.
The spread applies to the borrowed portion, every year.

**The penalty is worst for exactly the portfolios that look best.** Levering to a
target volatility costs

    SR_levered = SR_unlevered - s * (L - 1) / (L * sigma)

which approaches `s / sigma` as leverage rises. The lower the volatility, the
larger the deduction. A 50 basis point spread costs a 2% volatility portfolio
about 0.19 of Sharpe and a 9.5% volatility portfolio nothing at all - so the
correction does not shift everything equally, it reorders the table.

**Leverage is often simply unavailable.** Most mandates cap it at 1x. Under that
constraint Sharpe stops being the right objective altogether: what matters is
return per unit of *achievable* risk, and a portfolio that cannot be scaled to
the risk budget is not competitive however good its ratio looks.

There is a deeper point underneath, from Frazzini and Pedersen (2014). The
low-volatility anomaly exists *because* leverage is constrained. Investors who
want more return than a low-risk asset provides, and cannot borrow to get it, bid
up high-beta assets instead - which is why low-beta assets carry higher Sharpe
ratios in the first place. A high Sharpe on a very low volatility portfolio is
therefore partly compensation for a constraint, not an edge that can be harvested
by anyone facing that constraint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Annual financing spread over the risk-free rate, in basis points.
SPREADS_BP = {"institutional": 25, "typical": 50, "retail": 150}


def required_leverage(vol: float, target_vol: float) -> float:
    """Leverage needed to scale a portfolio to `target_vol`."""
    return float(target_vol / vol) if vol > 0 else np.inf


def lever_series(returns: pd.Series, rf: pd.Series, leverage: float,
                 spread_bp: float = 50.0,
                 periods_per_year: int = 12) -> pd.Series:
    """
    Apply leverage to a return series, charging financing on the borrowed part.

    Works on the realised path rather than on moments, so compounding and
    volatility drag are captured: levering a volatile series does not simply
    scale its geometric return, and the gap grows with leverage.
    """
    rfa = rf.reindex(returns.index).fillna(0.0)
    excess = returns - rfa
    cost = (leverage - 1.0) * (spread_bp / 1e4) / periods_per_year
    return leverage * excess + rfa - max(cost, 0.0)


def levered_sharpe(sharpe: float, vol: float, leverage: float,
                   spread_bp: float = 50.0) -> float:
    """Closed-form levered Sharpe: SR - s(L-1)/(L*sigma)."""
    if leverage <= 1.0 or vol <= 0:
        return float(sharpe)
    s = spread_bp / 1e4
    return float(sharpe - s * (leverage - 1.0) / (leverage * vol))


def comparison(nets: dict[str, pd.Series], rf: pd.Series,
               target_vol: float = 0.095, spread_bp: float = 50.0,
               max_leverage: float | None = None,
               periods_per_year: int = 12) -> pd.DataFrame:
    """
    Compare strategies at a common risk target, paying for the leverage used.

    `max_leverage` caps what may be borrowed. Strategies that cannot reach the
    target within the cap are reported at the volatility they actually achieve,
    which is the honest comparison for a mandate that forbids borrowing.
    """
    from .metrics import performance

    rows = {}
    for name, series in nets.items():
        s = series.dropna()
        vol = s.std() * np.sqrt(periods_per_year)
        base = performance(s, rf, periods_per_year=periods_per_year)

        L = required_leverage(vol, target_vol)
        capped = min(L, max_leverage) if max_leverage else L
        levered = lever_series(s, rf, capped, spread_bp, periods_per_year)
        lev_perf = performance(levered, rf, periods_per_year=periods_per_year)

        rows[name] = {
            "vol": vol,
            "sharpe_unlevered": base["sharpe"],
            "leverage_needed": L,
            "leverage_used": capped,
            "reaches_target": bool(capped >= L - 1e-9),
            "vol_achieved": lev_perf["vol"],
            "cagr_levered": lev_perf["cagr"],
            "sharpe_levered": lev_perf["sharpe"],
            "sharpe_cost": base["sharpe"] - lev_perf["sharpe"],
            "max_drawdown_levered": lev_perf["max_drawdown"],
        }
    return pd.DataFrame(rows).T.sort_values("sharpe_levered", ascending=False)


def spread_sensitivity(nets: dict[str, pd.Series], rf: pd.Series,
                       target_vol: float = 0.095,
                       spreads_bp=(0, 25, 50, 100, 150),
                       periods_per_year: int = 12) -> pd.DataFrame:
    """
    Levered Sharpe across a range of financing assumptions.

    Reported as a grid rather than a single number because the financing spread
    is an assumption, not a measurement, and the ranking is sensitive to it. A
    reader with a different cost of capital can find their own row.
    """
    from .metrics import performance

    out = {}
    for name, series in nets.items():
        s = series.dropna()
        vol = s.std() * np.sqrt(periods_per_year)
        L = required_leverage(vol, target_vol)
        row = {}
        for bp in spreads_bp:
            lev = lever_series(s, rf, L, bp, periods_per_year)
            row[f"{bp}bp"] = performance(lev, rf,
                                         periods_per_year=periods_per_year)["sharpe"]
        row["leverage"] = L
        out[name] = row
    return pd.DataFrame(out).T


def unlevered_comparison(nets: dict[str, pd.Series], rf: pd.Series,
                         periods_per_year: int = 12) -> pd.DataFrame:
    """
    The no-leverage case: compare on achievable return, not on ratio.

    Under a hard 1x constraint a Sharpe ratio cannot be converted into anything.
    What matters is the return actually delivered and the risk actually taken, so
    CAGR and drawdown lead here and Sharpe is reported alongside rather than as
    the ranking variable.
    """
    from .metrics import performance

    rows = {}
    for name, series in nets.items():
        p = performance(series.dropna(), rf, periods_per_year=periods_per_year)
        rows[name] = {"cagr": p["cagr"], "vol": p["vol"], "sharpe": p["sharpe"],
                      "max_drawdown": p["max_drawdown"], "calmar": p["calmar"]}
    return pd.DataFrame(rows).T.sort_values("cagr", ascending=False)
