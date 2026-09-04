"""Technical signals built as classic long-short factor spreads.

The technical portfolios elsewhere in the project are long only. A long-only
book cannot express a negative view, so half of a cross-sectional signal is
discarded, and what remains is a concentrated directional position rather than
the factor itself. This builds each signal the way the factor literature does:
rank the universe, go long the top group and short the bottom group, hold a
dollar-neutral spread.

Each signal appears twice, once on raw returns and once scaled by trailing
volatility, because on a bond universe a raw signal ranks largely by duration
and the risk-scaled version is what isolates the signal from that.

The spread is assumed self-financing, which is the institutional case: short
proceeds fund the long leg, and the residual is the per-asset transaction cost
already charged elsewhere in the project. Duration is reported rather than
neutralized, so a spread that is really a duration bet is visible as one.
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
CAN = import_module("phase2_canonical")

P = ROOT / "data/processed"
PPY = 252
MONTH = 21
BURN_IN = 5 * PPY
DEV_END = "2015-12-31"
LOOKBACK = 252
SKIP = 21                 # skip the most recent month, as the literature does
LEG = 4                   # long the best four, short the worst four of eleven
VOL_TARGET = 0.02         # overlay is scaled to this annualized volatility


def legs(S, k=LEG):
    """Dollar-neutral spread: +1 spread across the top k, -1 across the bottom."""
    R = S.rank(axis=1, ascending=False, na_option="keep")
    n = S.notna().sum(axis=1)
    long_leg = (R <= k).astype(float)
    short_leg = (R > (n.values[:, None] - k)).astype(float) * (R.notna())
    W = long_leg.div(long_leg.sum(axis=1).replace(0, np.nan), axis=0) \
        - short_leg.div(short_leg.sum(axis=1).replace(0, np.nan), axis=0)
    return W.fillna(0.0)


def run_ls(W, r, rates, every):
    """Hold a spread, resetting to target every `every` days, net of costs."""
    idx = W.index
    R = r.reindex(idx).to_numpy()
    Wt = W.to_numpy()
    nets, held, traded = [], np.zeros(W.shape[1]), 0.0
    for i in range(len(idx)):
        if i % every == 0:
            t = Wt[i]
            if np.isfinite(t).all():
                cost = float(np.abs(t - held) @ rates)
                traded += float(np.abs(t - held).sum()) / 2.0
                held = t
            else:
                cost = 0.0
        else:
            cost = 0.0
        nets.append(float(R[i] @ held) - cost)
    return pd.Series(nets, index=idx), traded / (len(idx) / PPY)


def main() -> int:
    r_all = pd.read_parquet(P / "fi_daily_returns.parquet")
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf.reindex(r.index).fillna(0.0)
    rates = np.array([RB.COST_BP[a] / 1e4 for a in assets])
    dur = pd.Series(CAN.DUR).reindex(assets)
    idx = r.index[BURN_IN:]

    vol = r.rolling(LOOKBACK).std()
    carry_raw = r.rolling(LOOKBACK).mean().shift(SKIP) * LOOKBACK
    mom_raw = r.rolling(LOOKBACK).apply(lambda x: np.prod(1 + x) - 1,
                                        raw=True).shift(SKIP)
    signals = {
        "Carry, raw": carry_raw,
        "Carry, risk-scaled": (carry_raw / vol.shift(SKIP)),
        "Momentum 12-1, raw": mom_raw,
        "Momentum 12-1, risk-scaled": (r.rolling(LOOKBACK).mean()
                                       / vol).shift(SKIP),
    }

    agg = get_prices(["VBMFX"])["VBMFX"].pct_change().reindex(r.index)
    s_hrp, W_hrp, _, tn_hrp = RB.walk(
        r, rf, rates, lambda S_, c: RB.RP.hierarchical_rp(S_, list(c)),
        lookback=None, every=PPY)

    spreads, rows = {}, []
    for name, S in signals.items():
        W = legs(S.reindex(columns=assets)).loc[idx[0]:]
        for lbl, every in [("annual", PPY), ("monthly", MONTH)]:
            s, tn = run_ls(W, r, rates, every)
            key = f"{name} ({lbl})"
            spreads[key] = s
            d = s.loc[s.index <= pd.Timestamp(DEV_END)]
            o = s.loc[s.index > pd.Timestamp(DEV_END)]
            dt = (W * dur).sum(axis=1)
            rows.append({
                "factor": key,
                "dev_sharpe": float(d.mean() * PPY / (d.std() * np.sqrt(PPY))),
                "dev_vol": float(d.std() * np.sqrt(PPY)),
                "oos_sharpe": float(o.mean() * PPY / (o.std() * np.sqrt(PPY))),
                "oos_vol": float(o.std() * np.sqrt(PPY)),
                "corr_agg": float(s.corr(agg.reindex(s.index))),
                "duration_tilt": float(dt.mean()),
                "turnover": tn})
    F = pd.DataFrame(rows).set_index("factor").sort_values(
        "dev_sharpe", ascending=False)
    F.to_parquet(P / "fi_ls_factors.parquet")
    print("Long-short spreads, dollar neutral, net of per-asset costs")
    print("(Sharpe on a zero-cost spread is mean over standard deviation; "
          "there is no cash leg to subtract.)\n")
    print(F.round(4).to_string())

    # ---- overlays, so the spread can be judged against the benchmark ------
    out, turns = {"HRP, annual": s_hrp}, {"HRP, annual": tn_hrp}
    for key, s in spreads.items():
        if "(annual)" not in key:
            continue
        sv = s.rolling(PPY).std().shift(1) * np.sqrt(PPY)
        lam = (VOL_TARGET / sv).clip(upper=5.0).fillna(0.0)
        scaled = s * lam
        out[f"Agg + {key.replace(' (annual)', '')}"] = \
            agg.reindex(s.index) + scaled
        out[f"HRP + {key.replace(' (annual)', '')}"] = \
            s_hrp.reindex(s.index) + scaled

    D = pd.DataFrame(out).dropna(how="any")
    D["Agg index"] = agg.reindex(D.index)
    D = D.dropna()
    dev = D.index <= pd.Timestamp(DEV_END)
    rf2 = rf.reindex(D.index)
    b = D["Agg index"]
    orows = []
    for c in D.columns:
        if c == "Agg index":
            continue
        row = {"strategy": c}
        for tag, msk in [("dev", dev), ("oos", ~dev)]:
            m = metrics.performance(D[c][msk], rf2[msk], periods_per_year=PPY)
            bm = metrics.performance(b[msk], rf2[msk], periods_per_year=PPY)
            d = inf.sharpe_difference(D[c][msk], b[msk], rf=rf2[msk], ppy=PPY)
            row[f"{tag}_sharpe"] = m["sharpe"]
            row[f"{tag}_vol"] = m["vol"]
            row[f"{tag}_vs_agg"] = m["sharpe"] - bm["sharpe"]
            row[f"{tag}_p"] = d["p_one_sided"]
        orows.append(row)
    O = pd.DataFrame(orows).set_index("strategy").sort_values(
        "dev_sharpe", ascending=False)
    O.to_parquet(P / "fi_ls_overlays.parquet")
    for tag, lbl in [("dev", "DEVELOPMENT"), ("oos", "HOLDOUT")]:
        msk = dev if tag == "dev" else ~dev
        bm = metrics.performance(b[msk], rf2[msk], periods_per_year=PPY)["sharpe"]
        print(f"\n{lbl} overlays, spread scaled to {VOL_TARGET:.0%} vol "
              f"(Agg Sharpe {bm:.4f})")
        print(O[[f"{tag}_sharpe", f"{tag}_vol", f"{tag}_vs_agg", f"{tag}_p"]]
              .sort_values(f"{tag}_sharpe", ascending=False).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
