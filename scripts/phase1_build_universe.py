"""FI_26 Phase 1 - a fixed income only universe.

Macro_26 concluded that regime and macro signals cannot time a multi-asset book.
One result inside that conclusion pointed somewhere else: the forecasts were not
uniformly weak. Out-of-sample R-squared against the prevailing mean was +2.13% on
the two-year Treasury - rising to +2.59% once technical indicators were added -
against +0.05% on equities. Investment grade and high yield reached +1.48% and
+1.81% with regime-conditional slopes.

The multi-asset book could not use any of that, because the forecastable assets
carried almost no risk in it. Equities and duration dominated the variance, and
those were the assets nobody could forecast.

Removing equities from the universe removes that mismatch. In a fixed income only
book, the assets that carry the risk *are* the assets with measurable
predictability. Whether that is enough to beat an equal-weight benchmark is the
question this project exists to answer, and it is a genuinely open one - the
predictability is real but small, and 1/N is a famously hard benchmark to beat
(DeMiguel, Garlappi and Uppal, 2009).

Twelve assets, in three groups:

    curve       constant-maturity Treasury holdings at 3M, 2Y, 5Y, 10Y and 30Y,
                built from the bootstrapped zero curve so they share one
                methodology and differ only in duration
    credit      short, intermediate and long investment grade, plus high yield
    securitized municipals (intermediate and high yield) and GNMA mortgages

The three groups matter because they are the three distinct risk factors in fixed
income - duration, credit and prepayment/tax - and an allocator that cannot tell
them apart is not doing anything a duration target could not.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from macro.data.yahoo import get_prices  # noqa: E402

PROCESSED = ROOT / "data/processed"

# Constant-maturity Treasury holdings already built from the zero curve.
CURVE = {"ust3m": 0.25, "ust2y": 2.0, "ust5y": 5.0, "ust10y": 10.0, "ust30y": 30.0}

# Funds, total return. Chosen for length of history first: every one of these
# reaches 1982 or earlier, which keeps the panel long enough to hold out a
# decade and still leave three decades to work with.
FUNDS = {
    "ig_short": ("VFSTX", "Short-term investment grade"),
    "ig": ("FBNDX", "Intermediate investment grade"),
    "ig_long": ("VWESX", "Long-term investment grade"),
    "hy": ("VWEHX", "High yield corporate"),
    "mbs": ("VFIIX", "GNMA mortgage-backed"),
    "muni": ("VWITX", "Intermediate municipal"),
    "muni_hy": ("VWAHX", "High yield municipal"),
}

GROUP = {**{k: "curve" for k in CURVE},
         **{"ig_short": "credit", "ig": "credit", "ig_long": "credit",
            "hy": "credit", "mbs": "securitized", "muni": "securitized",
            "muni_hy": "securitized"}}


def build_curve_returns(index) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    for name, m in CURVE.items():
        f = PROCESSED / f"bond_returns_{m:g}y.parquet"
        if not f.exists():
            print(f"  missing {f.name} - skipping {name}")
            continue
        d = pd.read_parquet(f)["total"]
        out[name] = ((1 + d).resample("ME").prod() - 1).reindex(index)
    return out


def build_fund_returns(index) -> pd.DataFrame:
    tickers = [t for t, _ in FUNDS.values()]
    px = get_prices(tickers)
    m = px.resample("ME").last()
    out = pd.DataFrame(index=index)
    for name, (t, _) in FUNDS.items():
        out[name] = m[t].pct_change().reindex(index)
    return out


def main() -> int:
    base = pd.read_parquet(PROCESSED / "panel_B_monthly.parquet")
    idx = base.index
    rf = base["rf"].reindex(idx)

    curve = build_curve_returns(idx)
    funds = build_fund_returns(idx)
    panel = pd.concat([curve, funds], axis=1)

    print("Coverage before trimming:")
    for c in panel.columns:
        s = panel[c].dropna()
        print(f"  {c:10s} {GROUP.get(c,''):12s} {s.index.min():%Y-%m} -> "
              f"{s.index.max():%Y-%m}  n={len(s)}")

    # Start where every asset exists. The alternative - letting assets enter as
    # they begin - would make the equal-weight benchmark change composition
    # partway through, which is not a benchmark anyone could have held.
    full = panel.dropna()
    print(f"\nCommon sample: {full.index.min():%Y-%m} to {full.index.max():%Y-%m} "
          f"({len(full)} months, {full.shape[1]} assets)")

    ex = full.sub(rf.reindex(full.index), axis=0)
    stats = pd.DataFrame({
        "group": pd.Series(GROUP).reindex(full.columns),
        "cagr": (1 + full).prod() ** (12 / len(full)) - 1,
        "vol": full.std() * np.sqrt(12),
        "sharpe": ex.mean() / ex.std() * np.sqrt(12),
        "skew": full.skew(),
        "worst_month": full.min(),
        "corr_ust10y": full.corrwith(full["ust10y"]),
    })
    print("\nAsset statistics (common sample):")
    print(stats.to_string(float_format=lambda x: f"{x:9.4f}"))

    print("\nCorrelation matrix:")
    print(full.corr().round(2).to_string())

    # How much genuinely independent variation is there? If one factor explains
    # nearly everything, an allocator has very little to allocate between.
    Z = (ex - ex.mean()) / ex.std()
    vals = np.linalg.eigvalsh(np.cov(Z.dropna().to_numpy(), rowvar=False))[::-1]
    share = vals / vals.sum()
    print("\nPrincipal components of excess returns:")
    for i in range(min(5, len(share))):
        print(f"  PC{i+1}  {share[i]:6.1%}   cumulative {share[:i+1].sum():6.1%}")
    print("\n  A universe dominated by one factor cannot be diversified, only")
    print("  sized. The gap between PC1 and the rest is what an allocator has")
    print("  to work with.")

    full.to_parquet(PROCESSED / "fi_returns.parquet")
    rf.reindex(full.index).to_frame("rf").to_parquet(PROCESSED / "fi_rf.parquet")
    stats.to_parquet(PROCESSED / "fi_asset_stats.parquet")
    pd.Series(GROUP).to_frame("group").to_parquet(PROCESSED / "fi_groups.parquet")
    print(f"\nwrote {PROCESSED / 'fi_returns.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
