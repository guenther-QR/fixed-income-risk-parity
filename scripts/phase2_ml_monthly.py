"""Machine learning on daily signals with a one-month forward target.

The daily models score well in aggregate and the gain sits entirely on the
three funds priced by matrix valuation, whose daily returns autocorrelate at
around 0.25. That points at the target rather than the features: a one-day
return carries the pricing delay, so a flexible model finds the delay instead
of finding a forecast.

Aggregating the target over a month should remove most of it, because a delay
of a day or two washes out of a twenty-one day sum. This keeps the daily
signal panel, which is timely and rich, and replaces the one-day target with
the return over the following month.

The target is the excess return over the twenty-one trading days ending at t,
and the features are lagged twenty-one days so nothing inside the window is
used to predict it. Training is purged by a further month so that overlapping
targets cannot leak across the decision point. A non-overlapping version,
sampled every twenty-first day, is reported alongside as a robustness check.

Hyperparameters are held at the values already tuned for the daily models, so
the only thing changing between the two sets of results is the target.
"""
import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from importlib import import_module  # noqa: E402

from macro.predict import forecast as fc  # noqa: E402
from macro.predict import models as ml  # noqa: E402
from macro.signals import daily as ds  # noqa: E402

DM = import_module("phase2_daily_models")

P = ROOT / "data/processed"
PPY = 252
MONTH = 21
BURN_IN = 5 * PPY
DEV_END = "2015-12-31"
STALE = ["hy", "muni", "muni_hy"]


def main() -> int:
    r_all = pd.read_parquet(P / "fi_daily_returns.parquet")
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf.reindex(r.index).fillna(0.0)
    dev_mask = r.index <= pd.Timestamp(DEV_END)

    X = pd.concat([ds.build(r_all), DM.regime_frame(r.index)], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    X, _ = DM.add_interactions(X, r, dev_mask)
    Xlag = X.shift(MONTH)
    print(f"Daily panel {X.shape[1]} columns, target is the 21-day forward "
          f"excess return\n")

    cfg = ml.MLConfig(min_train=BURN_IN, refit_every=PPY, max_features=30,
                      horizon=MONTH, embargo=MONTH)

    # Tuned for this horizon rather than inherited from the daily models. A
    # grid picked for a one day target is regularised for a problem with
    # twenty-one times more independent observations than this one has.
    print("Grid searching on the development sample, monthly horizon:")
    params = {}
    for name, grid in DM.GRIDS.items():
        best, best_score = None, -np.inf
        for prm in grid:
            sc = []
            for a in assets:
                ex = (r[a] - rf).dropna()
                y = ex.rolling(MONTH).sum().dropna()
                f = ml.walk_forward(y, Xlag.reindex(y.index),
                                    DM.FACTORY[name](**prm), cfg)
                b = fc.prevailing_mean(y, min_obs=BURN_IN)
                m = f.dropna().index.intersection(b.dropna().index)
                m = m[m <= pd.Timestamp(DEV_END)]
                if len(m) > PPY:
                    sc.append(fc.oos_r2(y[m], f[m], b[m]))
            s = float(np.mean(sc)) if sc else -np.inf
            if s > best_score:
                best, best_score = prm, s
        params[name] = best
        print(f"  {name:<18} {str(best):<52} dev R2 {best_score * 100:+.3f}%")
    pd.DataFrame({k: {"params": str(v)} for k, v in params.items()}).T.to_parquet(
        P / "fi_ml_monthly_chosen.parquet")
    print()

    rows, per_asset = [], {}
    for name, prm in params.items():
        skill = {}
        for a in assets:
            ex = (r[a] - rf).dropna()
            y = ex.rolling(MONTH).sum().dropna()
            Xa = Xlag.reindex(y.index)
            f = ml.walk_forward(y, Xa, DM.FACTORY[name](**prm), cfg)
            b = fc.prevailing_mean(y, min_obs=BURN_IN)
            m = f.dropna().index.intersection(b.dropna().index)
            dv = m[m <= pd.Timestamp(DEV_END)]
            skill[a] = fc.oos_r2(y[dv], f[dv], b[dv]) if len(dv) > PPY else np.nan

            # non-overlapping: every 21st observation
            nz = y.index[::MONTH]
            m2 = m.intersection(nz)
            m2 = m2[m2 <= pd.Timestamp(DEV_END)]
            skill[f"{a}__nonoverlap"] = (
                fc.oos_r2(y[m2], f[m2], b[m2]) if len(m2) > 24 else np.nan)
        per_asset[name] = skill
        ov = {a: skill[a] for a in assets}
        no = {a: skill[f"{a}__nonoverlap"] for a in assets}
        rows.append({
            "family": name,
            "all_11": float(np.nanmean(list(ov.values()))),
            "stale_3": float(np.nanmean([ov[a] for a in STALE])),
            "clean_8": float(np.nanmean([ov[a] for a in assets
                                         if a not in STALE])),
            "all_11_nonoverlap": float(np.nanmean(list(no.values()))),
            "stale_3_nonoverlap": float(np.nanmean([no[a] for a in STALE])),
            "clean_8_nonoverlap": float(np.nanmean([no[a] for a in assets
                                                    if a not in STALE])),
        })
        print(f"  scored {name}")

    T = pd.DataFrame(rows).set_index("family")
    T.to_parquet(P / "fi_ml_monthly_tuned.parquet")
    PA = pd.DataFrame(per_asset).T
    PA[[c for c in PA.columns if "__" not in c]].to_parquet(
        P / "fi_ml_monthly_tuned_by_asset.parquet")

    print("\nDevelopment out-of-sample R squared, percent, "
          "one-month forward target")
    print((T * 100).round(3).to_string())

    print("\nSame split for the one-day target, for comparison:")
    old = pd.read_parquet(P / "fi_dmodel_ml_skill.parquet")
    o = (old["dev_r2"].unstack() * 100)
    cmp = pd.DataFrame({
        "all_11": o.mean(axis=1),
        "stale_3": o[STALE].mean(axis=1),
        "clean_8": o[[a for a in assets if a not in STALE]].mean(axis=1),
    })
    print(cmp.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
