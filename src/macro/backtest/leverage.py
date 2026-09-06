"""Leverage-aware performance comparison."""
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
    Apply leverage to a return series, charging financing on the borrowed
    part.
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
    Compare strategies at a common risk target, paying for the leverage
    used.
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
    """Levered Sharpe across a range of financing assumptions."""
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
    """The no-leverage case: compare on achievable return, not on ratio."""
    from .metrics import performance

    rows = {}
    for name, series in nets.items():
        p = performance(series.dropna(), rf, periods_per_year=periods_per_year)
        rows[name] = {"cagr": p["cagr"], "vol": p["vol"], "sharpe": p["sharpe"],
                      "max_drawdown": p["max_drawdown"], "calmar": p["calmar"]}
    return pd.DataFrame(rows).T.sort_values("cagr", ascending=False)
