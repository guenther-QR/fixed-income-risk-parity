"""Overlays, unconstrained.

The overlay moves each risk parity weight by lambda times its cross-sectional
score times that weight. The cap on how far a weight may travel turned out not
to bind: at fifteen points the largest position averaged below risk parity's
own largest position, and opening the cap to sixty points moved it under two
points. Lambda, not the cap, is what controls how far the overlay goes.

So the cap is removed entirely and lambda is raised to one. The only remaining
constraints are that no weight may be negative and that the weights sum to one,
which is to say the book is long only and fully invested. At lambda of one an
asset scoring one standard deviation above the cross-section receives twice its
risk parity weight, and an asset a standard deviation below receives nothing.

The earlier settings are kept alongside so the effect of loosening is visible
rather than assumed.
"""
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

from macro.backtest import metrics  # noqa: E402
from macro.data.yahoo import get_prices  # noqa: E402
from macro.stats import inference as inf  # noqa: E402

RB = import_module("phase2_rebalance_study")
LS = import_module("phase2_long_short_factors")
BS = import_module("phase2_base_sleeve")
MC = import_module("phase2_momentum_calibration")
BM = import_module("phase2_beta_matched")
TD = import_module("phase2_technical_development")

P = ROOT / "data/processed"
PPY, MONTH, BURN_IN = 252, 21, 5 * 252
DEV_END = "2015-12-31"
LOOKBACK, SKIP = 252, 21
TILT = 0.5


def main() -> int:
    r_all = pd.read_parquet(P / "fi_daily_returns.parquet")
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf.reindex(r.index).fillna(0.0)
    rates = np.array([RB.COST_BP[a] / 1e4 for a in assets])
    dur = pd.Series(LS.CAN.DUR).reindex(assets)
    idx = r.index[BURN_IN:]
    agg = get_prices(["VBMFX"])["VBMFX"].pct_change().reindex(r.index)

    msh = (r.rolling(LOOKBACK).mean() / r.rolling(LOOKBACK).std()).shift(SKIP)
    cand = MC.momentum_candidates(r)
    Zc = {k: MC.xs_z(v) for k, v in cand.items()}
    Zr1 = BM.rolling_signal(r, cand, Zc, assets, idx, horizon=1)
    print("  rolling 60m signal done")

    SIG = {"Momentum (Sharpe)": BS.xs_z(msh).reindex(idx),
           "Rolling 60m (1d target)": Zr1}

    s_hrp, W_hrp, _, tn_hrp = RB.walk(
        r, rf, rates, lambda S_, c: RB.RP.hierarchical_rp(S_, list(c)),
        lookback=None, every=PPY)
    b = W_hrp.reindex(idx).ffill()

    out, turns, durn, maxw = {}, {}, {}, {}
    out["HRP, annual"], turns["HRP, annual"] = s_hrp, tn_hrp
    durn["HRP, annual"] = float((b * dur).sum(axis=1).mean())
    maxw["HRP, annual"] = float(b.max(axis=1).mean())

    for name, Z in SIG.items():
        for lbl, W in [
            ("lambda 0.5, cap +15pts",
             (b + 0.5 * Z * b).clip(lower=0.0).clip(upper=b + 0.15)),
            ("lambda 0.5, uncapped", (b + 0.5 * Z * b).clip(lower=0.0)),
            ("lambda 1.0, uncapped", (b + 1.0 * Z * b).clip(lower=0.0)),
            ("lambda 1.5, uncapped", (b + 1.5 * Z * b).clip(lower=0.0)),
        ]:
            W = W.div(W.sum(axis=1), axis=0)
            s, tn = BS.AS.run_long(W, r, rates, PPY)
            k = f"HRP + {name} overlay, {lbl}"
            out[k], turns[k] = s, tn
            durn[k] = float((W * dur).sum(axis=1).mean())
            maxw[k] = float(W.max(axis=1).mean())
        print(f"  built {name}")

    D = pd.DataFrame(out).dropna(how="any")
    D["Agg index"] = agg.reindex(D.index)
    D = D.dropna()
    D = D[D.index <= pd.Timestamp(DEV_END)]
    rf2 = rf.reindex(D.index)
    bench = D["Agg index"]
    bx = (bench - rf2).to_numpy()
    bm = metrics.performance(bench, rf2, periods_per_year=PPY)

    rows = []
    for c in D.columns:
        x = D[c]
        m = metrics.performance(x, rf2, periods_per_year=PPY)
        a, t = TD.nw_ols((x - rf2).to_numpy(), bx[:, None])
        rows.append({
            "strategy": c,
            "return": float((1 + x).prod() ** (PPY / len(x)) - 1),
            "vol": m["vol"], "sharpe": m["sharpe"],
            "vs_agg": np.nan if c == "Agg index" else m["sharpe"] - bm["sharpe"],
            "p": np.nan if c == "Agg index" else
                 inf.sharpe_difference(x, bench, rf=rf2, ppy=PPY)["p_one_sided"],
            "alpha_pct_yr": a[0] * PPY * 100, "t_alpha": t[0], "beta": a[1],
            "duration": durn.get(c, np.nan),
            "mean_max_weight": maxw.get(c, np.nan),
            "turnover": turns.get(c, np.nan)})
    T = pd.DataFrame(rows).set_index("strategy").sort_values(
        "sharpe", ascending=False)
    T.to_parquet(P / "fi_overlay_lambda.parquet")
    print("\n=== OVERLAY LAMBDA SWEEP, DEVELOPMENT 1987-11 to 2015-12 ===")
    print(T.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
