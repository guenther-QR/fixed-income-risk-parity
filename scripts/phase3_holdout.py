"""The holdout, opened once, on the strategies development selected."""
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
from macro.backtest import technical as TQ  # noqa: E402
from macro.stats import inference as inf  # noqa: E402

RB = import_module("phase2_rebalance_study")

P = ROOT / "data/processed"
PPY, MONTH, BURN_IN = 252, 21, 5 * 252
DEV_END = "2015-12-31"
LOOKBACK, SKIP = 252, 21

CONFIRMATORY = {
    "HRP + Rolling 60m overlay", "HRP + Momentum (Sharpe) overlay",
    "HRP, annual", "ERC, annual"}


def main() -> int:
    r_all = pd.read_parquet(P / "fi_daily_returns.parquet")
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf.reindex(r.index).fillna(0.0)
    rates = np.array([RB.COST_BP[a] / 1e4 for a in assets])
    dur = pd.Series(TQ.DUR).reindex(assets)
    idx = r.index[BURN_IN:]
    n = len(assets)
    agg = get_prices(["VBMFX"])["VBMFX"].pct_change().reindex(r.index)

    msh = (r.rolling(LOOKBACK).mean() / r.rolling(LOOKBACK).std()).shift(SKIP)
    Zm = TQ.xs_z(msh).reindex(idx)
    cand = TQ.momentum_candidates(r)
    Zr = TQ.rolling_signal(r, cand, {k: TQ.xs_z(v) for k, v in cand.items()},
                           assets, idx, BURN_IN, 5 * PPY, horizon=1)
    print("  signals built")

    out, turns, durn = {}, {}, {}

    def keep(name, s, tn, W):
        out[name], turns[name] = s, tn
        durn[name] = float((W.reindex(idx) * dur).sum(axis=1).mean())

    for lbl, fn in [("HRP, annual",
                     lambda S_, c: RB.RP.hierarchical_rp(S_, list(c))),
                    ("ERC, annual", lambda S_, c: RB.RP.erc_weights(S_))]:
        s, W, _, tn = RB.walk(r, rf, rates, fn, lookback=None, every=PPY)
        keep(lbl, s, tn, W)
        if lbl.startswith("HRP"):
            b = W.reindex(idx).ffill()

    # overlays, at the settings with the strongest development evidence
    Wo = (b + 1.0 * Zr * b).clip(lower=0.0)
    Wo = Wo.div(Wo.sum(axis=1), axis=0)
    s, tn = TQ.run_long(Wo, r, rates, PPY)
    keep("HRP + Rolling 60m overlay", s, tn, Wo)

    Wo = (b + 0.5 * Zm * b).clip(lower=0.0)
    Wo = Wo.div(Wo.sum(axis=1), axis=0)
    s, tn = TQ.run_long(Wo, r, rates, PPY)
    keep("HRP + Momentum (Sharpe) overlay", s, tn, Wo)

    # the same two signals standing on their own
    W = TQ.long_only(Zr, n)
    s, tn = TQ.run_long(W, r, rates, PPY)
    keep("Rolling 60m, long only", s, tn, W)

    W = TQ.long_only(Zm, n)
    s, tn = TQ.run_long(W, r, rates, MONTH)
    keep("Momentum (Sharpe), long only", s, tn, W)

    # Two more that appear in the development table only, kept here so the
    # whole development and holdout comparison comes out of one script.
    Zr21 = TQ.rolling_signal(r, cand, {k: TQ.xs_z(v) for k, v in cand.items()},
                             assets, idx, BURN_IN, 5 * PPY, horizon=MONTH)
    W = TQ.long_only(Zr21, n)
    s, tn = TQ.run_long(W, r, rates, PPY)
    keep("Rolling 60m (1m target), long only", s, tn, W)

    mom = r.rolling(LOOKBACK).apply(lambda x: np.prod(1 + x) - 1,
                                    raw=True).shift(MONTH)
    W = TQ.long_only(TQ.xs_z(mom).reindex(idx), n)
    s, tn = TQ.run_long(W, r, rates, MONTH)
    keep("Momentum 12-1, long only", s, tn, W)
    print("  portfolios built")

    D = pd.DataFrame(out).dropna(how="any")
    D["Agg index"] = agg.reindex(D.index)
    D = D.dropna()
    D.to_parquet(P / "fi_holdout_paths.parquet")
    dev = D.index <= pd.Timestamp(DEV_END)
    rf2 = rf.reindex(D.index)
    bench = D["Agg index"]

    rows = []
    for c in D.columns:
        row = {"strategy": c,
               "status": ("benchmark" if c == "Agg index" else
                          "confirmatory" if c in CONFIRMATORY else
                          "exploratory"),
               "duration": durn.get(c, np.nan),
               "turnover": turns.get(c, np.nan)}
        for tag, msk in [("dev", dev), ("oos", ~dev)]:
            x = D[c][msk]
            bx = (bench[msk] - rf2[msk]).to_numpy()
            m = metrics.performance(x, rf2[msk], periods_per_year=PPY)
            bmk = metrics.performance(bench[msk], rf2[msk], periods_per_year=PPY)
            a, t = TQ.nw_ols((x - rf2[msk]).to_numpy(), bx[:, None])
            row[f"{tag}_ret"] = float((1 + x).prod() ** (PPY / len(x)) - 1)
            row[f"{tag}_vol"] = m["vol"]
            row[f"{tag}_sharpe"] = m["sharpe"]
            row[f"{tag}_vs_agg"] = (np.nan if c == "Agg index"
                                    else m["sharpe"] - bmk["sharpe"])
            row[f"{tag}_p"] = (np.nan if c == "Agg index" else
                               inf.sharpe_difference(x, bench[msk],
                                                     rf=rf2[msk],
                                                     ppy=PPY)["p_one_sided"])
            row[f"{tag}_alpha"] = a[0] * PPY * 100
            row[f"{tag}_t"] = t[0]
        rows.append(row)
    T = pd.DataFrame(rows).set_index("strategy")
    T.to_parquet(P / "fi_holdout_results.parquet")

    print(f"\nHoldout {D.index[dev.sum()]:%Y-%m} to {D.index[-1]:%Y-%m}, "
          f"{(~dev).sum():,} days\n")
    print("=== DEVELOPMENT ===")
    print(T[["status", "dev_ret", "dev_vol", "dev_sharpe", "dev_vs_agg",
             "dev_p", "dev_alpha", "dev_t"]].round(4).to_string())
    print("\n=== HOLDOUT ===")
    print(T[["status", "oos_ret", "oos_vol", "oos_sharpe", "oos_vs_agg",
             "oos_p", "oos_alpha", "oos_t", "duration",
             "turnover"]].round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
