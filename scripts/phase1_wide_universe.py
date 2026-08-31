"""An expanded fixed income universe - 26 assets from 1988.

The original twelve-asset panel reached 1982 but covered the curve with only
five points and each credit sector with one fund. Two changes widen it
materially at a cost worth paying.

**More curve.** The bootstrapped zero curve supports any maturity, so one, three,
seven and twenty year constant-maturity holdings are added alongside the existing
two, five, ten and thirty. Eight points instead of four, all from the same
discount function, so they differ only in duration and nothing about the
methodology changes. The three-month bill stays out - it is a cash proxy, and the
project owner's instruction was to express de-risking through the budget rather
than by calling cash an asset.

**A later start for a wider panel.** Nine further funds become available between
1986 and 1988 - long and intermediate Treasury, an aggregate index, a second
high yield manager, a second mortgage fund, short and high yield municipals, and
convertibles. Starting in 1988 instead of 1982 costs about five years of history
and takes the universe from twelve assets to twenty-six.

That trade is worth making here specifically because the finding under test is
about *risk parity*, and risk parity is a statement about the covariance
structure. Estimating a 26x26 covariance on 460 months is harder than a 12x12 on
526, but the strategy's whole claim is that spreading risk across genuinely
distinct exposures beats equal weight - and it cannot be tested properly on a
universe with four curve points and one fund per sector.

Convertibles sit at the boundary of fixed income. They are included and flagged,
and every result is reported with and without them, because a reader may
reasonably say a convertible bond fund is a equity position in disguise.
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
MACRO = ROOT.parent / "Macro_26/data/processed"
START = "1988-06-30"

CURVE = {"ust1y": 1.0, "ust2y": 2.0, "ust3y": 3.0, "ust5y": 5.0,
         "ust7y": 7.0, "ust10y": 10.0, "ust20y": 20.0, "ust30y": 30.0}

FUNDS = {
    # Treasury and government
    "VUSTX": ("tsy_long", "govt"), "FSTGX": ("tsy_interm", "govt"),
    "FGOVX": ("govt_income", "govt"),
    # Investment grade corporate
    "VFSTX": ("ig_short", "credit"), "FBNDX": ("ig_interm", "credit"),
    "VWESX": ("ig_long", "credit"), "FTHRX": ("ig_interm2", "credit"),
    # High yield
    "VWEHX": ("hy", "credit"), "FAGIX": ("hy_capinc", "credit"),
    "PRHYX": ("hy_trp", "credit"),
    # Securitized
    "VFIIX": ("mbs", "securitized"), "FGMNX": ("mbs_fid", "securitized"),
    # Municipal
    "VWITX": ("muni_interm", "muni"), "VWAHX": ("muni_hy", "muni"),
    "VMLTX": ("muni_short", "muni"), "FRHIX": ("muni_hy2", "muni"),
    # Aggregate and hybrid
    "VBMFX": ("agg", "broad"), "VCVSX": ("convertibles", "hybrid"),
}

# Approximate modified duration, for the duration-matched benchmark.
DURATION = {
    "ust1y": 1.0, "ust2y": 1.9, "ust3y": 2.8, "ust5y": 4.6, "ust7y": 6.2,
    "ust10y": 8.4, "ust20y": 13.5, "ust30y": 18.5,
    "tsy_long": 16.0, "tsy_interm": 5.0, "govt_income": 5.5,
    "ig_short": 2.5, "ig_interm": 4.2, "ig_long": 12.0, "ig_interm2": 4.5,
    "hy": 4.0, "hy_capinc": 3.5, "hy_trp": 3.8,
    "mbs": 4.5, "mbs_fid": 4.3,
    "muni_interm": 5.0, "muni_hy": 7.5, "muni_short": 2.5, "muni_hy2": 7.0,
    "agg": 5.5, "convertibles": 3.0,
}

COST_BP = {
    "ust1y": 2, "ust2y": 3, "ust3y": 3, "ust5y": 4, "ust7y": 5,
    "ust10y": 5, "ust20y": 7, "ust30y": 8,
    "tsy_long": 8, "tsy_interm": 6, "govt_income": 10,
    "ig_short": 10, "ig_interm": 15, "ig_long": 20, "ig_interm2": 15,
    "hy": 40, "hy_capinc": 40, "hy_trp": 40,
    "mbs": 12, "mbs_fid": 12,
    "muni_interm": 25, "muni_hy": 45, "muni_short": 20, "muni_hy2": 45,
    "agg": 10, "convertibles": 30,
}


def main() -> int:
    base = pd.read_parquet(MACRO / "panel_B_monthly.parquet")
    idx = base.loc[START:].index
    out = pd.DataFrame(index=idx)
    groups = {}

    for name, m in CURVE.items():
        f = MACRO / f"bond_returns_{m:g}y.parquet"
        if not f.exists():
            print(f"  missing {f.name}")
            continue
        d = pd.read_parquet(f)["total"]
        out[name] = ((1 + d).resample("ME").prod() - 1).reindex(idx)
        groups[name] = "curve"

    px = get_prices(list(FUNDS)).resample("ME").last()
    for t, (name, grp) in FUNDS.items():
        if t in px:
            out[name] = px[t].pct_change().reindex(idx)
            groups[name] = grp

    full = out.dropna()
    rf = base["rf"].reindex(full.index)
    print(f"EXPANDED FI UNIVERSE: {full.shape[1]} assets, {len(full)} months")
    print(f"  {full.index.min():%Y-%m} to {full.index.max():%Y-%m}\n")
    gs = pd.Series(groups).reindex(full.columns)
    print(gs.value_counts().to_string())

    ex = full.sub(rf, axis=0)
    stats = pd.DataFrame({
        "group": gs, "duration": pd.Series(DURATION).reindex(full.columns),
        "cagr": (1 + full).prod() ** (12 / len(full)) - 1,
        "vol": full.std() * np.sqrt(12),
        "sharpe": ex.mean() / ex.std() * np.sqrt(12),
        "autocorr_1": [full[c].autocorr(1) for c in full.columns],
    }).sort_values("duration")
    print("\nAsset statistics:")
    print(stats.to_string(float_format=lambda x: f"{x:9.4f}"))

    C = ex.corr().to_numpy()
    vals = np.linalg.eigvalsh(C)[::-1]
    eff = float((vals.sum() ** 2) / (vals ** 2).sum())
    print(f"\nEffective independent assets: {eff:.2f} of {full.shape[1]}")
    print(f"  (the 12-asset panel scored 2.5 of 12)")
    print(f"  PC1 {vals[0]/vals.sum():.1%}   PC1-3 {vals[:3].sum()/vals.sum():.1%}"
          f"   PC1-5 {vals[:5].sum()/vals.sum():.1%}")

    print("\nMean pairwise correlation by group:")
    fam = gs.dropna().unique()
    Cd = ex.corr()
    M = pd.DataFrame(index=fam, columns=fam, dtype=float)
    for a in fam:
        for b in fam:
            ia = [c for c in full.columns if gs.get(c) == a]
            ib = [c for c in full.columns if gs.get(c) == b]
            blk = Cd.loc[ia, ib].to_numpy()
            if a == b and len(ia) > 1:
                blk = blk[~np.eye(len(ia), dtype=bool)]
            M.loc[a, b] = float(np.nanmean(blk)) if blk.size else np.nan
    print(M.round(2).to_string())

    full.to_parquet(PROCESSED / "fi_wide_returns.parquet")
    rf.to_frame("rf").to_parquet(PROCESSED / "fi_wide_rf.parquet")
    gs.to_frame("group").to_parquet(PROCESSED / "fi_wide_groups.parquet")
    stats.to_parquet(PROCESSED / "fi_wide_stats.parquet")
    pd.Series(DURATION).to_frame("duration").to_parquet(
        PROCESSED / "fi_wide_duration.parquet")
    print(f"\nwrote {PROCESSED / 'fi_wide_returns.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
