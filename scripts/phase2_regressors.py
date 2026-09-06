"""Which signals actually forecast bond returns."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from macro.predict import forecast as fc  # noqa: E402
from macro.signals import daily as ds  # noqa: E402
from macro.signals import library as lib, literature as lit  # noqa: E402

P = ROOT / "data/processed"
PPY = 252
BURN_IN = 5 * PPY
DEV_END = "2015-12-31"

# The daily and monthly signal libraries name things differently, so both
# conventions are mapped to the same families and the two scans stay
# comparable.
FAMILY = [
    # daily library
    ("d_carry", "carry and rolldown"), ("d_roll", "carry and rolldown"),
    ("d_slope", "curve shape"), ("d_curv", "curve shape"),
    ("d_level", "curve shape"), ("d_fwd", "forward rates"),
    ("d_cp", "Cochrane-Piazzesi"), ("d_baa", "credit spreads"),
    ("d_aaa", "credit spreads"), ("d_hy", "credit spreads"),
    ("d_vix", "volatility"), ("d_rvol", "volatility"),
    ("d_mom", "momentum"), ("d_ma", "moving average"),
    ("d_rev", "short-term reversal"),
    # monthly library
    ("carry_roll", "carry and rolldown"), ("mod_dur", "carry and rolldown"),
    ("curve_", "curve shape"), ("short_rate", "curve shape"),
    ("fwd_", "forward rates"), ("cp_factor", "Cochrane-Piazzesi"),
    ("baa", "credit spreads"), ("aaa", "credit spreads"),
    ("vix", "volatility"), ("rvol", "volatility"), ("vrp", "volatility"),
    ("mom", "momentum"), ("tchi", "moving average"),
    ("gw_", "valuation"), ("term_premium", "term premium"),
    ("stock_bond", "cross asset"), ("gold_vs", "cross asset"),
    ("regime_", "regime state"), ("crisis", "regime state"),
]


def family_of(name):
    for pre, lbl in FAMILY:
        if name.startswith(pre):
            return lbl
    if " x " in name:
        return "regime interaction"
    return "macro and other"


def main() -> int:
    r_all = pd.read_parquet(P / "fi_daily_returns.parquet")
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf.reindex(r.index).fillna(0.0)
    X = ds.build(r_all)
    print(f"Scoring {X.shape[1]} signals on {len(assets)} assets, "
          f"development sample only\n")

    scores, tstats, per_asset = {}, {}, {}
    for a in assets:
        y = (r[a] - rf).dropna()
        F = fc.univariate_forecasts(y, X.reindex(y.index), BURN_IN, 1)
        b = fc.prevailing_mean(y, min_obs=BURN_IN)
        dv = b.dropna().index
        dv = dv[dv <= pd.Timestamp(DEV_END)]
        for c in F.columns:
            f = F[c].reindex(dv)
            m = f.dropna().index
            if len(m) < PPY:
                continue
            r2 = fc.oos_r2(y[m], f[m], b[m])
            scores.setdefault(c, []).append(r2)
            per_asset.setdefault(a, {})[c] = r2
            # Clark-West rather than raw R squared: with 256 signals, ranking
            # on R squared alone guarantees a leaderboard whether or not
            # anything is real.
            t, _ = fc.clark_west(y[m], f[m], b[m])
            if np.isfinite(t):
                tstats.setdefault(c, []).append(t)

    rows = []
    for c, v in scores.items():
        if len(v) < len(assets) // 2:
            continue
        ts = tstats.get(c, [])
        rows.append({"signal": c, "family": family_of(c),
                     "mean_r2": float(np.mean(v)),
                     "mean_t": float(np.mean(ts)) if ts else np.nan,
                     "n_sig": int(sum(1 for x in ts if x > 1.645)),
                     "n_positive": int(sum(1 for x in v if x > 0)),
                     "n_assets": len(v)})
    T = pd.DataFrame(rows).set_index("signal").sort_values(
        "mean_r2", ascending=False)
    T.to_parquet(P / "fi_regressor_ranks.parquet")

    fam = (T.groupby("family")
             .agg(signals=("mean_r2", "size"), mean_r2=("mean_r2", "mean"),
                  best_r2=("mean_r2", "max"), mean_t=("mean_t", "mean"),
                  sig_pairs=("n_sig", "sum"), pairs=("n_assets", "sum"))
             .assign(pct_sig=lambda d: d["sig_pairs"] / d["pairs"])
             .sort_values("mean_t", ascending=False))
    fam.to_parquet(P / "fi_regressor_families.parquet")

    # How many signal-asset pairs clear the 5% one-sided threshold, against
    # how many would by chance.
    tot_pairs = int(T["n_assets"].sum())
    tot_sig = int(T["n_sig"].sum())
    expected = 0.05 * tot_pairs
    print(f"Signal-asset pairs tested: {tot_pairs:,}")
    print(f"  clearing Clark-West at 5% one-sided: {tot_sig:,} "
          f"({tot_sig / tot_pairs:.1%})")
    print(f"  expected by chance if nothing predicts: {expected:,.0f} (5.0%)")
    print(f"  ratio: {tot_sig / expected:.2f}x")
    print()
    pd.DataFrame([{"pairs": tot_pairs, "significant": tot_sig,
                   "expected_by_chance": expected,
                   "ratio": tot_sig / expected}]).to_parquet(
        P / "fi_regressor_multiplicity.parquet")

    # Top three signals for each asset.
    top = []
    for a, d in per_asset.items():
        ser = pd.Series(d).sort_values(ascending=False)
        for rank, (sig, v) in enumerate(ser.head(10).items(), start=1):
            top.append({"asset": a, "rank": rank, "signal": sig,
                        "family": family_of(sig), "r2": v})
    TOP = pd.DataFrame(top).set_index(["asset", "rank"])
    TOP.to_parquet(P / "fi_regressor_top10.parquet")
    TOP[TOP.index.get_level_values("rank") <= 3].to_parquet(
        P / "fi_regressor_top3.parquet")

    # Triangulation: which signals recur across assets.
    flat = TOP.reset_index()
    rec = (flat.groupby(["signal", "family"])
               .agg(assets=("asset", "nunique"),
                    median_r2=("r2", "median"),
                    best_r2=("r2", "max"),
                    best_on=("r2", "idxmax"))
               .reset_index())
    rec["best_on"] = [flat.loc[i, "asset"] for i in rec["best_on"]]
    rec = (rec[rec["assets"] >= 3]
           .sort_values(["assets", "median_r2"], ascending=[False, False])
           .set_index("signal"))
    rec.to_parquet(P / "fi_regressor_recurring.parquet")
    print("Signals appearing in the top ten of three or more assets:")
    print(rec.assign(median_r2=lambda d: d["median_r2"] * 100,
                     best_r2=lambda d: d["best_r2"] * 100).round(3).to_string())
    print()
    print("Top three signals per asset:")
    print(TOP[TOP.index.get_level_values("rank") <= 3]
          .assign(r2=lambda d: d["r2"] * 100).round(3).to_string())
    counts = TOP["family"].value_counts()
    print()
    print("Families appearing in the per-asset top three:")
    print(counts.to_string())
    print()

    print("Top 15 individual signals by development out-of-sample R squared:")
    print((T.head(15).assign(mean_r2=lambda d: d["mean_r2"] * 100)
             .round(4)).to_string())
    print("\nBy family:")
    print((fam.assign(mean_r2=lambda d: d["mean_r2"] * 100,
                      best_r2=lambda d: d["best_r2"] * 100)).round(4).to_string())

    # The same scan at monthly frequency so the two can be compared directly.
    rm = pd.read_parquet(P / "fi_returns.parquet")
    rfm = pd.read_parquet(P / "fi_rf.parquet").squeeze().reindex(rm.index)
    am = [c for c in rm.columns if c != "ust3m"]
    Xm = pd.concat([lib.build(rm, P), lit.build(rm, P)], axis=1)
    Xm = Xm.loc[:, ~Xm.columns.duplicated()]
    mscores = {}
    for a in am:
        y = (rm[a] - rfm).dropna()
        F = fc.univariate_forecasts(y, Xm.reindex(y.index), 60, 1)
        b = fc.prevailing_mean(y, min_obs=60)
        dv = b.dropna().index
        dv = dv[dv <= pd.Timestamp(DEV_END)]
        for c in F.columns:
            f = F[c].reindex(dv)
            m = f.dropna().index
            if len(m) < 24:
                continue
            mscores.setdefault(c, []).append(fc.oos_r2(y[m], f[m], b[m]))
    MR = pd.DataFrame([{"signal": c, "family": family_of(c),
                        "mean_r2": float(np.mean(v))}
                       for c, v in mscores.items()
                       if len(v) >= len(am) // 2]).set_index("signal")
    mfam = MR.groupby("family").agg(signals_m=("mean_r2", "size"),
                                    mean_r2=("mean_r2", "mean"),
                                    best_r2=("mean_r2", "max"))
    both = fam[["signals", "mean_r2", "best_r2"]].join(
        mfam[["mean_r2", "best_r2"]], lsuffix="_daily", rsuffix="_monthly",
        how="outer")
    both.to_parquet(P / "fi_regressor_families_both.parquet")
    print("\nBy family, daily against monthly, mean R squared in percent:")
    show = both.copy()
    for c in show.columns:
        if c != "signals":
            show[c] = show[c] * 100
    print(show.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
