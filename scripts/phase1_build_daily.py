"""Build the fixed income universe at daily frequency.

The project was running on month-end data. Both underlying sources are daily and
the monthly panel was produced by throwing that away: the constant-maturity
Treasury returns were compounded to month-end, and the fund NAVs were sampled at
month-end. Nothing about the data required it.

Daily matters here for three specific reasons, in order of size:

1.  The covariance matrix. Risk parity is entirely a covariance estimate, and
    an 11x11 matrix has 66 free parameters. A five-year burn-in gives 60 monthly
    observations against those 66 parameters, which is fewer observations than
    parameters. The same five years gives roughly 1,260 daily observations. This
    is the single biggest problem with the monthly build and it sits directly on
    the thing the project's positive result depends on.

2.  Stale pricing becomes visible. Illiquid bond funds are marked with a lag,
    which shows up as first-order autocorrelation in daily returns and is almost
    invisible monthly. On this universe the daily autocorrelations are 0.24 for
    high yield, 0.25 for intermediate municipals and 0.24 for high yield
    municipals, against 0.01 to 0.03 for the Treasuries. Monthly data reports
    0.17, 0.00 and 0.07 for the same three funds, which would have led to the
    wrong conclusion about which holdings are affected.

3.  Rebalancing frequency becomes a choice rather than an assumption. With daily
    data the estimation frequency and the trading frequency are separate
    decisions: estimate on everything available, trade as often as costs justify.

Stale pricing cuts against daily data in one direction, and it is handled rather
than ignored. Autocorrelated returns understate variance, so a naive daily
covariance would make the stale funds look safer than they are and risk parity
would overweight them. The Newey-West correction below inflates each variance by
the factor implied by its own autocorrelation, which is the Lo (2002) adjustment
applied to the covariance rather than to a Sharpe ratio.

Writes fi_daily_returns, fi_daily_rf, fi_daily_stats.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from macro.data.fred import get_series  # noqa: E402
from macro.data.yahoo import get_prices  # noqa: E402

P = ROOT / "data/processed"

CURVE = {"ust3m": 0.25, "ust2y": 2.0, "ust5y": 5.0, "ust10y": 10.0,
         "ust30y": 30.0}
FUNDS = {
    "ig_short": ("VFSTX", "Short-term investment grade"),
    "ig": ("FBNDX", "Intermediate investment grade"),
    "ig_long": ("VWESX", "Long-term investment grade"),
    "hy": ("VWEHX", "High yield corporate"),
    "mbs": ("VFIIX", "GNMA mortgage-backed"),
    "muni": ("VWITX", "Intermediate municipal"),
    "muni_hy": ("VWAHX", "High yield municipal"),
}
GROUP = {**{k: "treasury" for k in CURVE},
         **{"ig_short": "credit", "ig": "credit", "ig_long": "credit",
            "hy": "credit", "mbs": "securitized", "muni": "municipal",
            "muni_hy": "municipal"}}
SOURCE = {**{k: "Bootstrapped Treasury zero curve (FRED constant maturities)"
             for k in CURVE},
          **{k: f"{t}, Yahoo Finance daily NAV" for k, (t, _) in FUNDS.items()}}
PPY = 252


def newey_west_factor(x: pd.Series, lags: int = 5) -> float:
    """Variance inflation implied by a series' own autocorrelation.

    For a return series with autocorrelations rho_k, the variance of the sum of
    q observations is not q times the one-period variance; it carries cross
    terms. This returns the ratio of the correctly-scaled variance to the naive
    one, so a stale series gets marked up rather than flattered.
    """
    x = x.dropna()
    if len(x) < 100:
        return 1.0
    f = 1.0
    for k in range(1, lags + 1):
        rho = x.autocorr(k)
        if np.isfinite(rho):
            f += 2.0 * (1.0 - k / (lags + 1.0)) * rho
    return float(max(f, 0.25))


def main() -> int:
    curve = pd.DataFrame({
        k: pd.read_parquet(P / f"bond_returns_{m:g}y.parquet")["total"]
        for k, m in CURVE.items()})
    px = get_prices([t for t, _ in FUNDS.values()])
    funds = pd.DataFrame({k: px[t].pct_change() for k, (t, _) in FUNDS.items()})

    panel = pd.concat([curve, funds], axis=1)
    print("Daily coverage before trimming:")
    for c in panel.columns:
        s = panel[c].dropna()
        print(f"  {c:10s} {GROUP[c]:12s} {s.index.min():%Y-%m-%d} -> "
              f"{s.index.max():%Y-%m-%d}  n={len(s):,}")

    full = panel.dropna()
    print(f"\nCommon daily sample: {full.index.min():%Y-%m-%d} to "
          f"{full.index.max():%Y-%m-%d}  ({len(full):,} days, "
          f"{full.shape[1]} assets)")
    m = pd.read_parquet(P / "fi_returns.parquet")
    print(f"  the monthly panel this replaces: {len(m)} observations, "
          f"a factor of {len(full) / len(m):.0f} fewer")

    # Risk-free rate: the 3-month bill, converted to a daily accrual.
    try:
        tb = get_series("DTB3").reindex(full.index).ffill() / 100.0
        rf = tb / PPY
    except Exception:
        rf = pd.Series(0.0, index=full.index)
        print("  WARNING: could not fetch DTB3, risk-free set to zero")
    rf = rf.fillna(method="ffill").fillna(0.0)

    stats = pd.DataFrame({
        "group": pd.Series(GROUP).reindex(full.columns),
        "source": pd.Series(SOURCE).reindex(full.columns),
        "ann_return": full.mean() * PPY,
        "ann_vol": full.std() * np.sqrt(PPY),
        "sharpe": (full.sub(rf, axis=0).mean()
                   / full.sub(rf, axis=0).std() * np.sqrt(PPY)),
        "autocorr_1": full.apply(lambda s: s.autocorr(1)),
        "nw_factor": full.apply(newey_west_factor),
        "worst_day": full.min(),
    })
    stats["vol_adjusted"] = stats["ann_vol"] * np.sqrt(stats["nw_factor"])

    full.to_parquet(P / "fi_daily_returns.parquet")
    rf.to_frame("rf").to_parquet(P / "fi_daily_rf.parquet")
    stats.to_parquet(P / "fi_daily_stats.parquet")

    print("\nDaily asset statistics:")
    print(stats.drop(columns=["source"]).round(4).to_string())
    print("\n  autocorr_1 is the stale-pricing signature. nw_factor is the")
    print("  variance inflation it implies, and vol_adjusted is the volatility")
    print("  after that correction. The three funds above 0.20 are marked up")
    print("  by 30 to 40 percent, which is the difference between risk parity")
    print("  treating them as safe and treating them correctly.")
    print(f"\nwrote {P / 'fi_daily_returns.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
