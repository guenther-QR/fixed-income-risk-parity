"""Signal-weighted books only: no bounded tilts anywhere.

A bounded tilt around equal weight is mostly equal weight. It flatters any
signal by anchoring it to a portfolio that already works, so the strategies
here are signal-weighted throughout: the signal alone sets the weights.

Each momentum signal appears in both forms a cross-sectional signal can take.
Long only clips the below-average assets to zero and keeps full market
exposure. Long-short keeps them as shorts and holds no market exposure at all,
which is how the factor literature builds these. A cross-sectional z-score is
already demeaned, so the dollar-neutral book is the z-score itself rescaled.

The one exception to signal weighting is the momentum overlay on hierarchical
risk parity, which is a bounded tilt by construction and is reported as such
rather than as a signal-weighted book.
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
AS = import_module("phase2_asness_factor")

P = ROOT / "data/processed"
PPY = 252
MONTH = 21
BURN_IN = 5 * PPY
DEV_END = "2015-12-31"
LOOKBACK = 252
SKIP = 21
TILT = 0.5
CAP = 0.15


def xs_z(F):
    return F.sub(F.mean(axis=1), axis=0).div(
        F.std(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def long_only(Z, n):
    W = Z.clip(lower=0.0)
    return W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / n)


def long_short(Z):
    """Dollar neutral: z is already demeaned, so rescale to one per side."""
    pos = Z.clip(lower=0.0).sum(axis=1).replace(0, np.nan)
    neg = (-Z.clip(upper=0.0)).sum(axis=1).replace(0, np.nan)
    return Z.div(pd.concat([pos, neg], axis=1).max(axis=1), axis=0).fillna(0.0)


def main() -> int:
    r_all = pd.read_parquet(P / "fi_daily_returns.parquet")
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf.reindex(r.index).fillna(0.0)
    rates = np.array([RB.COST_BP[a] / 1e4 for a in assets])
    dur = pd.Series(LS.CAN.DUR).reindex(assets)
    idx = r.index[BURN_IN:]
    n = len(assets)
    agg = get_prices(["VBMFX"])["VBMFX"].pct_change().reindex(r.index)

    # signals
    mom_raw = r.rolling(LOOKBACK).apply(lambda x: np.prod(1 + x) - 1,
                                        raw=True).shift(SKIP)
    msharpe = (r.rolling(LOOKBACK).mean() / r.rolling(LOOKBACK).std()).shift(SKIP)
    sig = {"Momentum 12-1": mom_raw, "Momentum (Sharpe)": msharpe}

    out, turns, notes, neutral = {}, {}, {}, {}
    for name, S in sig.items():
        Z = xs_z(S).loc[idx[0]:]
        Wl = long_only(Z, n)
        k1 = f"{name}, long only, monthly"
        out[k1], turns[k1] = AS.run_long(Wl, r, rates, MONTH)
        notes[k1], neutral[k1] = float((Wl * dur).sum(axis=1).mean()), False

        Ws = long_short(Z)
        k2 = f"{name}, long-short, monthly"
        out[k2], turns[k2] = LS.run_ls(Ws, r, rates, MONTH)
        notes[k2], neutral[k2] = float((Ws * dur).sum(axis=1).mean()), True
        print(f"  built {name}")

    # Asness, his own construction, for reference
    Wr = AS.rank_weights(mom_raw).loc[idx[0]:]
    k = "Asness factor (rank-weighted)"
    out[k], turns[k] = LS.run_ls(Wr, r, rates, MONTH)
    notes[k], neutral[k] = float((Wr * dur).sum(axis=1).mean()), True

    # references and the HRP overlay
    s_hrp, W_hrp, _, tn = RB.walk(
        r, rf, rates, lambda S_, c: RB.RP.hierarchical_rp(S_, list(c)),
        lookback=None, every=PPY)
    out["HRP, annual"], turns["HRP, annual"] = s_hrp, tn
    notes["HRP, annual"] = float((W_hrp.reindex(idx) * dur).sum(axis=1).mean())
    neutral["HRP, annual"] = False

    Za = xs_z(r.rolling(LOOKBACK).mean().shift(SKIP) * LOOKBACK)
    b = W_hrp.reindex(idx).ffill()
    Wo = (b + TILT * Za.reindex(idx) * b).clip(lower=0.0)
    Wo = Wo.clip(upper=b + CAP)
    Wo = Wo.div(Wo.sum(axis=1), axis=0)
    k = "Momentum overlay on HRP"
    out[k], turns[k] = AS.run_long(Wo, r, rates, PPY)
    notes[k], neutral[k] = float((Wo * dur).sum(axis=1).mean()), False
    print("  built references and overlay")

    D = pd.DataFrame(out).dropna(how="any")
    D["Agg index"] = agg.reindex(D.index)
    D = D.dropna()
    dev = D.index <= pd.Timestamp(DEV_END)
    rf2 = rf.reindex(D.index)
    bench = D["Agg index"]

    rows = []
    for c in D.columns:
        if c == "Agg index":
            continue
        row = {"strategy": c, "turnover": turns.get(c, np.nan),
               "duration": notes.get(c, np.nan),
               "corr_agg": float(D[c].corr(bench)),
               "market_neutral": neutral.get(c, False)}
        for tag, msk in [("dev", dev), ("oos", ~dev)]:
            x = D[c][msk]
            if neutral.get(c, False):
                row[f"{tag}_sharpe"] = float(x.mean() * PPY /
                                             (x.std() * np.sqrt(PPY)))
                row[f"{tag}_vol"] = float(x.std() * np.sqrt(PPY))
                row[f"{tag}_vs_agg"] = np.nan
                row[f"{tag}_p"] = np.nan
            else:
                m = metrics.performance(x, rf2[msk], periods_per_year=PPY)
                bm = metrics.performance(bench[msk], rf2[msk],
                                         periods_per_year=PPY)
                d = inf.sharpe_difference(x, bench[msk], rf=rf2[msk], ppy=PPY)
                row[f"{tag}_sharpe"], row[f"{tag}_vol"] = m["sharpe"], m["vol"]
                row[f"{tag}_vs_agg"] = m["sharpe"] - bm["sharpe"]
                row[f"{tag}_p"] = d["p_one_sided"]
        rows.append(row)
    T = pd.DataFrame(rows).set_index("strategy")
    T.to_parquet(P / "fi_base_sleeve.parquet")

    bd = metrics.performance(bench[dev], rf2[dev], periods_per_year=PPY)["sharpe"]
    bo = metrics.performance(bench[~dev], rf2[~dev], periods_per_year=PPY)["sharpe"]
    print(f"\nAgg Sharpe: development {bd:.4f}, holdout {bo:.4f}\n")
    print("Long-only and overlay books, against the Aggregate:")
    lo = T[~T["market_neutral"]]
    print(lo[["dev_sharpe", "dev_vol", "dev_vs_agg", "dev_p", "oos_sharpe",
              "oos_vs_agg", "duration", "turnover"]].round(4).to_string())
    print("\nMarket-neutral books (Sharpe is mean over standard deviation):")
    mn = T[T["market_neutral"]]
    print(mn[["dev_sharpe", "dev_vol", "oos_sharpe", "oos_vol", "corr_agg",
              "duration", "turnover"]].round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
