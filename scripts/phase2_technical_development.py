"""The technical family on development: results, beta, and whether to blend.

Every strategy is signal-weighted rather than tilted toward equal weight, with
two exceptions that are labelled as overlays because that is what they are.

Market-neutral books are reported as portable alpha rather than on their own.
A dollar-neutral spread holds no market exposure, so its Sharpe is not
comparable to a benchmark that does. Holding the Aggregate for beta and adding
the spread on top makes the comparison meaningful: beta comes from the index
leg, and anything above the index has to come from the spread. The spread is
sized by its own trailing volatility so the overlay does not inherit the
spread's scale.

Whether any of these belong in the same portfolio is answered by spanning
regressions rather than by correlation. Regressing a candidate on what is
already held and testing the intercept asks whether the candidate adds
anything that cannot already be obtained; correlation only asks whether two
series move together. Eigenvalues of the correlation matrix are reported as
context on how many independent bets the family actually contains.

Development sample only.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

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
VA = import_module("phase2_momentum_vs_agg")
MC = import_module("phase2_momentum_calibration")

P = ROOT / "data/processed"
PPY, MONTH, BURN_IN = 252, 21, 5 * 252
DEV_END = "2015-12-31"
LOOKBACK, SKIP = 252, 21
TILT, CAP, VOL_TARGET = 0.5, 0.15, 0.02


def nw_ols(y, Xm, lags=21):
    """OLS with Newey-West standard errors. Returns coefs and t statistics."""
    X = np.column_stack([np.ones(len(y)), Xm])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    XtX_inv = np.linalg.pinv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        A = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (A + A.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 1e-18))
    return b, b / se


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
    SIG = {"Momentum 12-1": mom, "Momentum (Sharpe)": msh}

    out, turns, durn, kind = {}, {}, {}, {}

    def keep(name, s, tn, w, k):
        out[name], turns[name], kind[name] = s, tn, k
        durn[name] = float((w.reindex(idx) * dur).sum(axis=1).mean())

    # ---- references -------------------------------------------------------
    for lbl, fn in [("HRP, annual",
                     lambda S_, c: RB.RP.hierarchical_rp(S_, list(c))),
                    ("ERC, annual", lambda S_, c: RB.RP.erc_weights(S_))]:
        s, W, _, tn = RB.walk(r, rf, rates, fn, lookback=None, every=PPY)
        keep(lbl, s, tn, W, "risk based")
    W_hrp = None
    _, W_hrp, _, _ = RB.walk(
        r, rf, rates, lambda S_, c: RB.RP.hierarchical_rp(S_, list(c)),
        lookback=None, every=PPY)
    print("  references done")

    cand = MC.momentum_candidates(r)
    Zc = {k: MC.xs_z(v) for k, v in cand.items()}
    Zr = VA.rolling_signal(r, cand, Zc, assets, idx)
    print("  rolling 60m selection done")

    # ---- long-only signal weighted ---------------------------------------
    Wr60 = BS.long_only(Zr, n)
    s, tn = BS.AS.run_long(Wr60, r, rates, PPY)
    keep("Rolling 60m selection, long only", s, tn, Wr60, "technical")
    for name, S in SIG.items():
        Z = BS.xs_z(S).loc[idx[0]:]
        W = BS.long_only(Z, n)
        s, tn = BS.AS.run_long(W, r, rates, MONTH)
        keep(f"{name}, long only", s, tn, W, "technical")
    print("  long-only done")

    # ---- market neutral spreads, held as portable alpha on the Agg -------
    spreads = {}
    for name, S in SIG.items():
        Ws = BS.long_short(BS.xs_z(S).loc[idx[0]:])
        sp, tn = LS.run_ls(Ws, r, rates, MONTH)
        spreads[f"{name} spread"] = (sp, tn, Ws)
    Wr = AS.rank_weights(mom).loc[idx[0]:]
    sp, tn = LS.run_ls(Wr, r, rates, MONTH)
    spreads["Asness factor spread"] = (sp, tn, Wr)
    Ws60 = BS.long_short(Zr)
    sp, tn = LS.run_ls(Ws60, r, rates, PPY)
    spreads["rolling 60m spread"] = (sp, tn, Ws60)

    for name, (sp, tn, Ws) in spreads.items():
        sv = sp.rolling(PPY).std().shift(1) * np.sqrt(PPY)
        lam = (VOL_TARGET / sv).clip(upper=5.0).fillna(0.0)
        comb = (agg.reindex(sp.index) + sp * lam).dropna()
        lbl = f"Agg + {name}"
        out[lbl], turns[lbl], kind[lbl] = comb, tn, "portable alpha"
        durn[lbl] = float((Ws.reindex(idx) * dur).sum(axis=1).mean())
    print("  spreads done")

    # ---- overlays on HRP --------------------------------------------------
    b = W_hrp.reindex(idx).ffill()
    for name, S in SIG.items():
        Z = BS.xs_z(S).reindex(idx)
        W = (b + TILT * Z * b).clip(lower=0.0).clip(upper=b + CAP)
        W = W.div(W.sum(axis=1), axis=0)
        s, tn = BS.AS.run_long(W, r, rates, PPY)
        keep(f"HRP + {name} overlay", s, tn, W, "overlay")

    W = (b + TILT * Zr * b).clip(lower=0.0).clip(upper=b + CAP)
    W = W.div(W.sum(axis=1), axis=0)
    s, tn = BS.AS.run_long(W, r, rates, PPY)
    keep("HRP + rolling 60m selection overlay", s, tn, W, "overlay")
    print("  overlays done")

    # ---- score, development only -----------------------------------------
    D = pd.DataFrame(out).dropna(how="any")
    D["Agg index"] = agg.reindex(D.index)
    D = D.dropna()
    D = D[D.index <= pd.Timestamp(DEV_END)]
    rf2 = rf.reindex(D.index)
    bench = D["Agg index"]
    bx = (bench - rf2).to_numpy()

    rows = []
    for c in D.columns:
        x = D[c]
        m = metrics.performance(x, rf2, periods_per_year=PPY)
        bm = metrics.performance(bench, rf2, periods_per_year=PPY)
        cagr = float((1 + x).prod() ** (PPY / len(x)) - 1)
        beta = float(np.cov(x - rf2, bx)[0, 1] / np.var(bx))
        if c == "Agg index":
            vs, p = np.nan, np.nan
        else:
            d = inf.sharpe_difference(x, bench, rf=rf2, ppy=PPY)
            vs, p = m["sharpe"] - bm["sharpe"], d["p_one_sided"]
        rows.append({"strategy": c, "kind": kind.get(c, "benchmark"),
                     "return": cagr, "vol": m["vol"], "sharpe": m["sharpe"],
                     "vs_agg": vs, "p": p, "beta_agg": beta,
                     "duration": durn.get(c, np.nan),
                     "turnover": turns.get(c, np.nan)})
    T = pd.DataFrame(rows).set_index("strategy").sort_values(
        "sharpe", ascending=False)
    T.to_parquet(P / "fi_technical_development.parquet")
    print("\n=== DEVELOPMENT RESULTS, 1987-11 to 2015-12 ===")
    print(T.round(4).to_string())

    # ---- is any of this distinctive? -------------------------------------
    print("\n=== Correlation of strategy returns (development) ===")
    C = D.drop(columns=["Agg index"]).corr()
    print(C.round(2).to_string())
    ev = np.linalg.eigvalsh(C.to_numpy())[::-1]
    ev = ev / ev.sum()
    print(f"\nFirst component explains {ev[0]:.1%}; "
          f"first two {ev[:2].sum():.1%}; "
          f"effective independent bets {1 / (ev ** 2).sum():.2f} "
          f"of {len(ev)}")

    print("\n=== Spanning regressions: does it add anything? ===")
    print("alpha in percent a year, with Newey-West t statistics\n")
    sp_rows = []
    for c in D.columns:
        if c == "Agg index":
            continue
        yv = (D[c] - rf2).to_numpy()
        b1, t1 = nw_ols(yv, bx[:, None])
        sp_rows.append({"strategy": c, "alpha_pct_yr": b1[0] * PPY * 100,
                        "t_stat": t1[0], "beta_agg": b1[1],
                        "significant_5pct": abs(t1[0]) > 1.96})
    S = pd.DataFrame(sp_rows).set_index("strategy").sort_values(
        "alpha_pct_yr", ascending=False)
    S.to_parquet(P / "fi_technical_spanning.parquet")
    print(S.round(3).to_string())
    print("\n(|t| above 1.96 is significant at 5%.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
