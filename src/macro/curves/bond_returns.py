"""Constant-maturity Treasury total returns, decomposed into their sources."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap import discount_at


def coupon_times(maturity: float) -> np.ndarray:
    """Semiannual cash-flow times counting backwards from `maturity`."""
    n = int(np.ceil(maturity / 0.5 - 1e-9))
    times = maturity - 0.5 * np.arange(n - 1, -1, -1)
    return times[times > 1e-9]


def par_yield_on_schedule(disc_row: pd.Series, maturity: float,
                          times: np.ndarray | None = None) -> float:
    """Par yield for a bond paying on an explicit schedule ending at `maturity`."""
    times = coupon_times(maturity) if times is None else times
    annuity = sum(discount_at(disc_row, t) for t in times)
    return 2.0 * (1.0 - discount_at(disc_row, maturity)) / annuity


def par_yield(disc_row: pd.Series, maturity: float) -> float:
    """The par yield term structure, evaluated at `maturity`."""
    lo = max(0.5, np.floor(maturity / 0.5) * 0.5)
    nodes = np.array([lo - 0.5, lo, lo + 0.5, lo + 1.0])
    nodes = nodes[nodes >= 0.5]
    vals = [par_yield_on_schedule(disc_row, float(n)) for n in nodes]
    return float(np.interp(maturity, nodes, vals))


def price_bond(disc_row: pd.Series, coupon: float, maturity: float,
               face: float = 100.0) -> float:
    """Price a semiannual coupon bond off a discount curve."""
    times = coupon_times(maturity)
    pv = sum((coupon / 2.0) * face * discount_at(disc_row, t) for t in times)
    return pv + face * discount_at(disc_row, maturity)


def duration_convexity(disc_row: pd.Series, coupon: float, maturity: float,
                       bump: float = 1e-4) -> tuple[float, float]:
    """Effective modified duration and convexity, by bumping the curve in parallel."""
    times = disc_row.index.to_numpy(dtype=float)
    zero = -np.log(disc_row.to_numpy(dtype=float)) / times

    def shifted(dy: float) -> pd.Series:
        return pd.Series(np.exp(-(zero + dy) * times), index=disc_row.index)

    p0 = price_bond(disc_row, coupon, maturity)
    up = price_bond(shifted(bump), coupon, maturity)
    dn = price_bond(shifted(-bump), coupon, maturity)

    duration = (dn - up) / (2.0 * p0 * bump)
    convexity = (up + dn - 2.0 * p0) / (p0 * bump ** 2)
    return duration, convexity


def constant_maturity_returns(discount: pd.DataFrame, maturity: float = 10.0, *,
                              max_quoted: pd.Series | None = None,
                              day_count: float = 365.0,
                              max_gap: float = 0.05) -> pd.DataFrame:
    """
    Daily total returns for a constant-maturity par bond, with attribution.
    """
    idx = discount.index
    rows, dates = [], []

    for i in range(len(idx) - 1):
        t0, t1 = idx[i], idx[i + 1]
        dt = (t1 - t0).days / day_count
        if not (0 < dt <= max_gap):
            continue

        # Refuse days where this maturity sits beyond the longest real quote:
        # the curve is flat-extrapolated there, so a "30-year" return before
        # the 30-year bond existed would be an artefact of the extrapolation.
        if max_quoted is not None:
            reach = max_quoted.reindex([t0, t1]).min()
            if not np.isfinite(reach) or reach < maturity - 1e-9:
                continue

        curve0, curve1 = discount.iloc[i], discount.iloc[i + 1]
        aged = maturity - dt
        try:
            y0 = par_yield(curve0, maturity)          # coupon; bond opens at par
            y_roll = par_yield(curve0, aged)          # same curve, shorter point
            y1 = par_yield(curve1, aged)              # new curve, shorter point

            # Exact total return: reprice the aged bond on tomorrow's curve.
            p_new = price_bond(curve1, y0, aged)
            total = (p_new - 100.0) / 100.0

            dur, conv = duration_convexity(curve0, y0, maturity)
            carry = y0 * dt
            rolldown = -dur * (y_roll - y0)
            dy = y1 - y_roll
            dur_part = -dur * dy
            cvx_part = 0.5 * conv * dy ** 2
        except (ValueError, FloatingPointError, ZeroDivisionError):
            continue

        if not np.isfinite(total):
            continue

        rows.append({
            "total": total,
            "carry": carry,
            "rolldown": rolldown,
            "duration": dur_part,
            "convexity": cvx_part,
            "residual": total - carry - rolldown - dur_part - cvx_part,
            "ytm": y0,
            "mod_dur": dur,
            "d_yield": dy,
        })
        dates.append(t1)

    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))


def annualize(returns: pd.Series, periods_per_year: float = 252.0) -> dict:
    n = len(returns)
    cum = (1 + returns).cumprod()
    return {
        "ann_return": (1 + returns).prod() ** (periods_per_year / n) - 1,
        "ann_vol": returns.std() * np.sqrt(periods_per_year),
        "max_dd": (cum / cum.cummax() - 1).min(),
    }


def constant_maturity_bill_returns(discount: pd.DataFrame, maturity: float = 0.25, *,
                                   max_quoted: pd.Series | None = None,
                                   day_count: float = 365.0,
                                   max_gap: float = 0.05) -> pd.DataFrame:
    """Daily total returns for a constant-maturity zero-coupon bill."""
    idx = discount.index
    rows, dates = [], []

    for i in range(len(idx) - 1):
        t0, t1 = idx[i], idx[i + 1]
        dt = (t1 - t0).days / day_count
        if not (0 < dt <= max_gap):
            continue

        # Refuse days where this maturity sits beyond the longest real quote:
        # the curve is flat-extrapolated there, so a "30-year" return before
        # the 30-year bond existed would be an artefact of the extrapolation.
        if max_quoted is not None:
            reach = max_quoted.reindex([t0, t1]).min()
            if not np.isfinite(reach) or reach < maturity - 1e-9:
                continue

        c0, c1 = discount.iloc[i], discount.iloc[i + 1]
        aged = maturity - dt
        try:
            p0 = discount_at(c0, maturity)
            p_roll = discount_at(c0, aged)      # unchanged curve, shorter bill
            p_new = discount_at(c1, aged)       # new curve, shorter bill
            total = p_new / p0 - 1.0
            carry_roll = p_roll / p0 - 1.0
            shift = p_new / p_roll - 1.0
            ytm = -np.log(p0) / maturity
        except (ValueError, FloatingPointError, ZeroDivisionError):
            continue

        if not np.isfinite(total):
            continue
        rows.append({"total": total, "carry": carry_roll, "rolldown": 0.0,
                     "duration": shift, "convexity": 0.0,
                     "residual": total - carry_roll - shift,
                     "ytm": ytm, "mod_dur": aged, "d_yield": np.nan})
        dates.append(t1)

    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"))
