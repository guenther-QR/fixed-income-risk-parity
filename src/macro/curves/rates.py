"""Risk-free rate construction, and the bill conventions needed to check it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap import interp_zero

DAYS_PER_YEAR = 365.0


def discount_to_price(discount_pct: pd.Series, days: int) -> pd.Series:
    """Bill price per 100 from a discount-basis quote in percent."""
    return 100.0 * (1.0 - (discount_pct / 100.0) * days / 360.0)


def discount_to_bey(discount_pct: pd.Series, days: int) -> pd.Series:
    """Discount-basis quote to bond-equivalent yield, in percent."""
    price = discount_to_price(discount_pct, days)
    return (100.0 - price) / price * (DAYS_PER_YEAR / days) * 100.0


def continuous_to_simple(cc_pct: pd.Series | float, years: float) -> pd.Series | float:
    """Continuously compounded rate to the simple rate over `years`, in percent."""
    return (np.exp(cc_pct / 100.0 * years) - 1.0) / years * 100.0


def risk_free_from_curve(zero: pd.DataFrame, maturity: float = 1 / 12) -> pd.Series:
    """
    The risk-free rate implied by the bootstrapped curve, as a simple annual
    percentage at the given maturity (one month by default).
    """
    cc = pd.Series(
        [interp_zero(row, maturity) for _, row in zero.iterrows()],
        index=zero.index, name="rf_curve",
    )
    return continuous_to_simple(cc, maturity)


def to_period_return(annual_pct: pd.Series, periods_per_year: int) -> pd.Series:
    """Annualized percentage rate to the simple return earned over one period."""
    return annual_pct / 100.0 / periods_per_year


def compare_risk_free(curve_rf: pd.Series, dtb4wk: pd.Series | None,
                      french_rf_monthly: pd.Series | None) -> pd.DataFrame:
    """
    Align the curve-implied risk-free rate with its two independent checks.
    """
    parts = {"curve_1m": curve_rf}
    if dtb4wk is not None:
        parts["dtb4wk_bey"] = discount_to_bey(dtb4wk, 28)
    out = pd.DataFrame(parts)

    if french_rf_monthly is not None:
        monthly = out.resample("ME").last()
        monthly["french_rf"] = french_rf_monthly * 12 * 100
        return monthly.dropna(how="all")
    return out.dropna(how="all")
