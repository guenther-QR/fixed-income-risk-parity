"""Beta-matched combinations, and a rolling selection that predicts a month.

Two changes on the previous pass.

The combinations are matched on beta rather than on duration or volatility.
Beta here is the slope of a strategy's excess return on the Aggregate's, so
matching it is matching exposure to the bond market itself rather than to one
of its components. The spread is grossed so its own beta reaches one, then held
half and half with the index, which puts the combined beta near one. Where a
spread's beta is negative the gross factor is negative too, meaning the spread
has to be held the other way round to match; that is reported rather than
hidden, because inverting a strategy to fit a target is a decision, not a
detail. Beta is estimated on trailing data only and the gross factor is capped.

The rolling selection is also run against a one month forward return instead of
a one day return. Every other portfolio here except risk parity is a monthly
construction, so scoring candidate windows on next-day skill and then trading
them for a year was inconsistent. Pairs whose forward return is not yet
realised at the decision date are excluded.
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
BS = import_module("phase2_base_sleeve")
RM = import_module("phase2_rolling_momentum")
MC = import_module("phase2_momentum_calibration")
TD = import_module("phase2_technical_development")

P = ROOT / "data/processed"
PPY, MONTH, BURN_IN = 252, 21, 5 * 252
SEL_WINDOW = 5 * PPY
DEV_END = "2015-12-31"
LOOKBACK, SKIP = 252, 21
TILT, CAP = 0.5, 0.15
MAX_GROSS = 5.0


def rolling_signal(r, cand, Zc, assets, idx, horizon=1):
    """Rolling 60-month argmax selection against an h-day forward return."""
    keys = list(cand)
    if horizon == 1:
        fwd = {a: r[a].shift(-1).to_numpy() for a in assets}
    else:
        f = r.rolling(horizon).sum().shift(-horizon)
        fwd = {a: f[a].to_numpy() for a in assets}
    sig = {k: {a: cand[k][a].to_numpy() for a in assets} for k in keys}
    rows, dates = [], []
    for i in range(len(idx)):
        if i % MONTH:
            continue
        g = BURN_IN + i
        lo = max(0, g - SEL_WINDOW)
        hi = g - horizon                 # forward return must be realised
        row = {}
        for a in assets:
            yv = fwd[a][lo:hi]
            best, best_ic = keys[0], -np.inf
            for k in keys:
                xv = sig[k][a][lo:hi]
                m = np.isfinite(xv) & np.isfinite(yv)
                if m.sum() < MC.MIN_IC_OBS:
                    continue
                ic = RM.spearman(xv[m], yv[m])
                if np.isfinite(ic) and ic > best_ic:
                    best, best_ic = k, ic
            row[a] = best
        rows.append(row)
        dates.append(idx[i])
    CH = pd.DataFrame(rows, index=dates).reindex(idx).ffill()
    return pd.DataFrame({a: [Zc[CH.loc[t, a]].at[t, a] for t in idx]
                         for a in assets}, index=idx)


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

    mom = r.rolling(LOOKBACK).apply(lambda x: np.prod(1 + x) - 1,
                                    raw=True).shift(SKIP)
    msh = (r.rolling(LOOKBACK).mean() / r.rolling(LOOKBACK).std()).shift(SKIP)
    cand = MC.momentum_candidates(r)
    Zc = {k: MC.xs_z(v) for k, v in cand.items()}
    Zr1 = rolling_signal(r, cand, Zc, assets, idx, horizon=1)
    print("  rolling 60m, 1-day target done")
    Zr21 = rolling_signal(r, cand, Zc, assets, idx, horizon=MONTH)
    print("  rolling 60m, 1-month target done")

    LONG = {"Momentum 12-1": (BS.xs_z(mom).loc[idx[0]:], MONTH),
            "Momentum (Sharpe)": (BS.xs_z(msh).loc[idx[0]:], MONTH),
            "Rolling 60m (1d target)": (Zr1, PPY),
            "Rolling 60m (1m target)": (Zr21, PPY)}

    out, turns, durn, typ, notes = {}, {}, {}, {}, {}

    def keep(name, s, tn, d, t, note=""):
        out[name], turns[name], durn[name], typ[name] = s, tn, d, t
        notes[name] = note

    for lbl, fn in [("HRP, annual",
                     lambda S_, c: RB.RP.hierarchical_rp(S_, list(c))),
                    ("ERC, annual", lambda S_, c: RB.RP.erc_weights(S_))]:
        s, W, _, tn = RB.walk(r, rf, rates, fn, lookback=None, every=PPY)
        keep(lbl, s, tn, float((W.reindex(idx) * dur).sum(axis=1).mean()),
             "long only")
        if lbl.startswith("HRP"):
            W_hrp = W

    for name, (Z, every) in LONG.items():
        W = BS.long_only(Z, n)
        s, tn = BS.AS.run_long(W, r, rates, every)
        keep(f"{name}, long only", s, tn,
             float((W.reindex(idx) * dur).sum(axis=1).mean()), "long only")

    spreads = {}
    for name, (Z, every) in LONG.items():
        Ws = BS.long_short(Z)
        sp, tn = LS.run_ls(Ws, r, rates, every)
        spreads[name] = (sp, tn,
                         float((Ws.reindex(idx) * dur).sum(axis=1).mean()))
    Wr = AS.rank_weights(mom).loc[idx[0]:]
    sp, tn = LS.run_ls(Wr, r, rates, MONTH)
    spreads["Asness factor"] = (sp, tn,
                                float((Wr.reindex(idx) * dur).sum(axis=1).mean()))
    for name, (sp, tn, d) in spreads.items():
        keep(f"{name}, long-short", sp, tn, d, "long-short")
    print("  long-only and spreads done")

    # ---- beta-matched combinations ---------------------------------------
    av = agg.rolling(PPY).var().shift(1)
    for name, (sp, tn, d) in spreads.items():
        cv = sp.rolling(PPY).cov(agg).shift(1)
        beta_sp = (cv / av).replace([np.inf, -np.inf], np.nan)
        k = (1.0 / beta_sp).clip(-MAX_GROSS, MAX_GROSS).fillna(0.0)
        comb = (0.5 * agg.reindex(sp.index) + 0.5 * k * sp).dropna()
        inv = float((k < 0).mean())
        keep(f"50-50 Agg / {name}, beta matched", comb, tn,
             0.5 * 4.16 + 0.5 * float(k.mean()) * d, "combination",
             f"gross {k.abs().mean():.2f}x, inverted {inv:.0%} of days")
    print("  beta-matched combinations done")

    b = W_hrp.reindex(idx).ffill()
    for name, (Z, _) in LONG.items():
        W = (b + TILT * Z.reindex(idx) * b).clip(lower=0.0).clip(upper=b + CAP)
        W = W.div(W.sum(axis=1), axis=0)
        s, tn = BS.AS.run_long(W, r, rates, PPY)
        keep(f"HRP + {name} overlay", s, tn,
             float((W * dur).sum(axis=1).mean()), "overlay")
    print("  overlays done")

    D = pd.DataFrame(out).dropna(how="any")
    D["Agg index"] = agg.reindex(D.index)
    D = D.dropna()
    D.to_parquet(P / "fi_technical_paths.parquet")
    pd.Series(typ).to_frame("type").to_parquet(P / "fi_technical_types.parquet")
    pd.Series(durn).to_frame("duration").to_parquet(P / "fi_technical_dur.parquet")
    pd.Series(turns).to_frame("turnover").to_parquet(P / "fi_technical_turn.parquet")
    D = D[D.index <= pd.Timestamp(DEV_END)]
    rf2 = rf.reindex(D.index)
    bench = D["Agg index"]
    bx = (bench - rf2).to_numpy()

    rows = []
    for c in D.columns:
        x = D[c]
        ls = typ.get(c) == "long-short"
        cagr = float((1 + x).prod() ** (PPY / len(x)) - 1)
        if ls:
            sh = float(x.mean() * PPY / (x.std() * np.sqrt(PPY)))
            vol = float(x.std() * np.sqrt(PPY))
            vs, p = np.nan, np.nan
        else:
            m = metrics.performance(x, rf2, periods_per_year=PPY)
            sh, vol = m["sharpe"], m["vol"]
            if c == "Agg index":
                vs, p = np.nan, np.nan
            else:
                bm = metrics.performance(bench, rf2, periods_per_year=PPY)
                vs = sh - bm["sharpe"]
                p = inf.sharpe_difference(x, bench, rf=rf2, ppy=PPY)["p_one_sided"]
        a, t = TD.nw_ols((x - rf2).to_numpy(), bx[:, None])
        rows.append({"strategy": c, "type": typ.get(c, "benchmark"),
                     "return": cagr, "vol": vol, "sharpe": sh, "vs_agg": vs,
                     "p": p, "alpha_pct_yr": a[0] * PPY * 100,
                     "t_alpha": t[0], "beta": a[1],
                     "duration": durn.get(c, 4.16),
                     "turnover": turns.get(c, np.nan),
                     "note": notes.get(c, "")})
    T = pd.DataFrame(rows).set_index("strategy").sort_values(
        "sharpe", ascending=False)
    T.to_parquet(P / "fi_beta_matched.parquet")
    print("\n=== DEVELOPMENT, BETA-MATCHED COMBINATIONS "
          "AND MONTH-FORWARD SELECTION ===")
    print(T.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
