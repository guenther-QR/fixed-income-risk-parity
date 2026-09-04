"""Asness's factor portfolio, z-scored momentum, and the regression schemes.

Three things, all rebalanced monthly.

A.  Asness, Moskowitz and Pedersen construct their momentum factor as a
    zero-cost portfolio weighted by cross-sectional rank less the average rank,
    scaled to one dollar long and one dollar short. The signal is the raw
    twelve month return skipping the most recent month, with no volatility
    adjustment. Their tercile spread is built alongside it, since they report
    both and note the rank-weighted version does better.

B.  Momentum on the trailing information ratio rather than the trailing
    return, cross-sectionally standardised, as a bounded tilt and as a
    signal-weighted book. Monthly rebalancing, because a cross-sectional
    ranking that is only acted on once a year discards most of what it knows.

C.  Three portfolio schemes built on the rolling regression results. Within an
    asset the significant predictors almost never agree in direction, so no
    scheme averages them. Scheme one resolves the disagreement by conviction,
    scheme two separates the horizons that disagree, and scheme three is a
    plain equal-weighted long-short on the resulting directions.
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

P = ROOT / "data/processed"
PPY = 252
MONTH = 21
BURN_IN = 5 * PPY
DEV_END = "2015-12-31"
LOOKBACK = 252
SKIP = 21
TILT = 0.5
CAP = 0.15
SHORT_H = ["ret 1m"]
LONG_H = ["mom 6m", "mom 12m", "mom 24m"]


def rank_weights(S):
    """AMP equation (1): weight by rank less mean rank, scaled 1 long 1 short."""
    R = S.rank(axis=1, na_option="keep")
    dev = R.sub(R.mean(axis=1), axis=0)
    pos = dev.clip(lower=0.0).sum(axis=1).replace(0, np.nan)
    return dev.div(pos, axis=0).fillna(0.0)


def long_only(Z, n, mode):
    if mode == "tilt":
        b = 1.0 / n
        W = (b + TILT * Z * b).clip(lower=max(b - CAP, 0.0), upper=b + CAP)
    else:
        W = Z.clip(lower=0.0)
    return W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / n)


def run_long(W, r, rates, every):
    idx = W.index
    R = r.reindex(idx).to_numpy()
    Wt, n = W.to_numpy(), W.shape[1]
    nets, held, traded = [], None, 0.0
    for i in range(len(idx)):
        if i % every == 0 or held is None:
            t = Wt[i]
            t = np.ones(n) / n if not np.isfinite(t).all() or t.sum() <= 0 \
                else t / t.sum()
            pre = np.zeros(n) if held is None else held
            traded += 0.0 if pre.sum() == 0 else float(np.abs(t - pre).sum()) / 2
            cost = float(np.abs(t - pre) @ rates)
            held = t
        else:
            cost = 0.0
        nets.append(float(R[i] @ held) - cost)
        g = held * (1 + R[i])
        held = g / g.sum() if g.sum() > 0 else held
    return pd.Series(nets, index=idx), traded / (len(idx) / PPY)


def directions_to_weights(D, neutral=True):
    """Direction matrix (-1/0/+1) to weights, gross one per side."""
    W = D.astype(float)
    if neutral:
        act = (W != 0).sum(axis=1)
        W = W.sub(W.sum(axis=1) / act.replace(0, np.nan), axis=0)
        W = W.where(D != 0, 0.0)
    pos = W.clip(lower=0.0).sum(axis=1).replace(0, np.nan)
    neg = (-W.clip(upper=0.0)).sum(axis=1).replace(0, np.nan)
    scale = pd.concat([pos, neg], axis=1).max(axis=1)
    return W.div(scale, axis=0).fillna(0.0)


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

    out, turns, notes = {}, {}, {}

    # ---- A. Asness -------------------------------------------------------
    mom = r.rolling(LOOKBACK).apply(lambda x: np.prod(1 + x) - 1,
                                    raw=True).shift(SKIP)
    Wr = rank_weights(mom).loc[idx[0]:]
    out["Asness factor (rank-weighted)"], turns["Asness factor (rank-weighted)"] = \
        LS.run_ls(Wr, r, rates, MONTH)
    notes["Asness factor (rank-weighted)"] = float((Wr * dur).sum(axis=1).mean())
    Wt3 = LS.legs(mom, k=4).loc[idx[0]:]
    out["Asness tercile spread"], turns["Asness tercile spread"] = \
        LS.run_ls(Wt3, r, rates, MONTH)
    notes["Asness tercile spread"] = float((Wt3 * dur).sum(axis=1).mean())
    print("  built Asness variants")

    # ---- B. z-scored information-ratio momentum, monthly -----------------
    msharpe = (r.rolling(LOOKBACK).mean() / r.rolling(LOOKBACK).std()).shift(SKIP)
    Zm = msharpe.sub(msharpe.mean(axis=1), axis=0).div(
        msharpe.std(axis=1).replace(0, np.nan), axis=0).fillna(0.0).loc[idx[0]:]
    for mode in ("tilt", "base"):
        name = f"Momentum (Sharpe), {mode}, monthly"
        out[name], turns[name] = run_long(long_only(Zm, n, mode), r, rates, MONTH)
        notes[name] = float((long_only(Zm, n, mode) * dur).sum(axis=1).mean())
    print("  built z-scored momentum")

    # ---- C. regression schemes ------------------------------------------
    Z = pd.read_parquet(P / "fi_rolling_regression_zoo.parquet")
    Z = Z[Z["sig"]].copy()

    def build(sub, label, neutral=True):
        if sub.empty:
            return
        pick = sub.sort_values("p").groupby(["date", "asset"]).first()
        D = pick["dir"].unstack().reindex(columns=assets)
        D.index = pd.to_datetime(D.index)
        # a position fitted at month end t is held through month t+1
        D = D.shift(1).reindex(r.index, method="ffill").reindex(idx).fillna(0.0)
        W = directions_to_weights(D, neutral=neutral)
        out[label], turns[label] = LS.run_ls(W, r, rates, MONTH)
        notes[label] = float((W * dur).sum(axis=1).mean())
        print(f"  built {label}")

    build(Z, "Scheme 1: most significant wins")
    build(Z[Z["predictor"].isin(SHORT_H)], "Scheme 2a: short-horizon sleeve")
    build(Z[Z["predictor"].isin(LONG_H)], "Scheme 2b: long-horizon sleeve")
    build(Z, "Scheme 3: directional, net exposure allowed", neutral=False)

    # ---- score -----------------------------------------------------------
    s_hrp, _, _, tn_hrp = RB.walk(
        r, rf, rates, lambda S_, c: RB.RP.hierarchical_rp(S_, list(c)),
        lookback=None, every=PPY)
    out["HRP, annual"], turns["HRP, annual"] = s_hrp, tn_hrp

    D = pd.DataFrame(out).dropna(how="any")
    D["Agg index"] = agg.reindex(D.index)
    D = D.dropna()
    D.to_parquet(P / "fi_scheme_paths.parquet")
    dev = D.index <= pd.Timestamp(DEV_END)
    rf2 = rf.reindex(D.index)
    b = D["Agg index"]

    rows = []
    for c in D.columns:
        if c == "Agg index":
            continue
        neutral = c.startswith(("Asness", "Scheme"))
        row = {"strategy": c, "turnover": turns.get(c, np.nan),
               "duration_tilt": notes.get(c, np.nan),
               "corr_agg": float(D[c].corr(b))}
        for tag, msk in [("dev", dev), ("oos", ~dev)]:
            x = D[c][msk]
            if neutral:      # zero-cost book: no cash leg to subtract
                sh = float(x.mean() * PPY / (x.std() * np.sqrt(PPY)))
                vol = float(x.std() * np.sqrt(PPY))
                row[f"{tag}_sharpe"], row[f"{tag}_vol"] = sh, vol
                row[f"{tag}_vs_agg"] = np.nan
                row[f"{tag}_p"] = np.nan
            else:
                m = metrics.performance(x, rf2[msk], periods_per_year=PPY)
                bm = metrics.performance(b[msk], rf2[msk], periods_per_year=PPY)
                d = inf.sharpe_difference(x, b[msk], rf=rf2[msk], ppy=PPY)
                row[f"{tag}_sharpe"], row[f"{tag}_vol"] = m["sharpe"], m["vol"]
                row[f"{tag}_vs_agg"] = m["sharpe"] - bm["sharpe"]
                row[f"{tag}_p"] = d["p_one_sided"]
        rows.append(row)
    T = pd.DataFrame(rows).set_index("strategy")
    T.to_parquet(P / "fi_asness_schemes.parquet")

    bm_d = metrics.performance(b[dev], rf2[dev], periods_per_year=PPY)["sharpe"]
    bm_o = metrics.performance(b[~dev], rf2[~dev], periods_per_year=PPY)["sharpe"]
    print(f"\nAgg Sharpe: development {bm_d:.4f}, holdout {bm_o:.4f}")
    print("\nZero-cost books (Sharpe is mean over standard deviation):")
    zc = T[T.index.str.startswith(("Asness", "Scheme"))]
    print(zc[["dev_sharpe", "dev_vol", "oos_sharpe", "corr_agg",
              "duration_tilt", "turnover"]].round(4).to_string())
    print("\nLong-only books, against the Aggregate:")
    lo = T[~T.index.str.startswith(("Asness", "Scheme"))]
    print(lo[["dev_sharpe", "dev_vol", "dev_vs_agg", "dev_p", "oos_sharpe",
              "oos_vs_agg", "turnover"]].round(4).to_string())

    # overlays so the zero-cost books can be judged against the benchmark
    print("\nOverlays on HRP, spread scaled to 2% volatility:")
    orows = []
    for c in zc.index:
        s = D[c]
        sv = s.rolling(PPY).std().shift(1) * np.sqrt(PPY)
        lam = (0.02 / sv).clip(upper=5.0).fillna(0.0)
        comb = (s_hrp.reindex(s.index) + s * lam).dropna()
        m2 = comb.index <= pd.Timestamp(DEV_END)
        rr = {"overlay": c}
        for tag, msk in [("dev", m2), ("oos", ~m2)]:
            m = metrics.performance(comb[msk], rf2.reindex(comb.index)[msk],
                                    periods_per_year=PPY)
            bm = metrics.performance(b.reindex(comb.index)[msk],
                                     rf2.reindex(comb.index)[msk],
                                     periods_per_year=PPY)
            rr[f"{tag}_sharpe"] = m["sharpe"]
            rr[f"{tag}_vs_agg"] = m["sharpe"] - bm["sharpe"]
        orows.append(rr)
    O = pd.DataFrame(orows).set_index("overlay")
    O.to_parquet(P / "fi_scheme_overlays.parquet")
    print(O.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
