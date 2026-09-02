"""Return forecasting on daily data: regression, machine learning, technicals.

The risk-based side of this project moved to daily data and improved. This does
the same for the forecasting side, so the comparison between the two is made on
one dataset with one estimation window rather than across two builds.

What changes at daily frequency:

    more observations   256 signals against roughly 1,260 training days after
                        burn-in, growing to nearly 11,000. The monthly build had
                        60 observations at the same point.

    publication lags    macro signals carry their release delay in calendar
                        days rather than being assumed known at month end,
                        which is a real tightening: CPI for March is not
                        knowable on April 10th.

    stale pricing       shows up directly. Three funds autocorrelate above 0.23
                        daily, so a flexible model can score well by predicting
                        a move that already happened. That is reported rather
                        than corrected, and it is the reason the credit columns
                        need reading separately.

Portfolios are rebalanced annually, matching the best calendar frequency found
for the risk-based methods, so the two sides differ in what they estimate rather
than in how often they trade.

Writes fi_dmodel_*.parquet.
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
from macro.predict import forecast as fc, models as ml  # noqa: E402
from macro.signals import daily as ds  # noqa: E402
from macro.stats import inference  # noqa: E402

RB = import_module("phase2_rebalance_study")

P = ROOT / "data/processed"
MACRO = ROOT.parent / "Macro_26/data/processed"

PPY = 252
BURN_IN = 5 * PPY
DEV_END = "2015-12-31"
OOS_START = "2016-01-01"
REBAL = 252                      # annual, matching the risk-based side
BENCH = "Agg index"

TILT_STRENGTH = 0.5
TILT_CAP = 0.15
REGIME_LAG = 45                  # calendar days; IP and CPI publish with a lag

COST_BP = RB.COST_BP


def regime_frame(index):
    """Daily regime dummies from the monthly labels, lagged for publication."""
    src = P / "regimes_monthly.parquet"
    if not src.exists() and (MACRO / "regimes_monthly.parquet").exists():
        pd.read_parquet(MACRO / "regimes_monthly.parquet").to_parquet(src)
    if not src.exists():
        return pd.DataFrame(index=index)
    g = pd.read_parquet(src)["regime"]
    g.index = g.index + pd.Timedelta(days=REGIME_LAG)
    g = g.reindex(index, method="ffill")
    out = {}
    for st in ["Goldilocks", "Reflation", "Stagflation", "Deflation"]:
        out[f"regime_{st.lower()}"] = (g == st).astype(float)
    return pd.DataFrame(out, index=index)


def add_interactions(X, R, dev_mask, n_key=6):
    """Regime interactions on the signals that matter most in development.

    A regime dummy alone moves the intercept. Interacting it with a signal lets
    the slope change by state, which is the claim regime conditioning makes.
    """
    reg = [c for c in X.columns if c.startswith("regime_")]
    if not reg:
        return X, []
    y = R.mean(axis=1)
    cand = [c for c in X.columns if not c.startswith("regime_")]
    Xd = X.loc[dev_mask, cand]
    yd = y.loc[dev_mask]
    corr = Xd.apply(lambda s: abs(np.corrcoef(
        s.fillna(0.0), yd.reindex(s.index).fillna(0.0))[0, 1])
        if s.notna().sum() > 100 else np.nan)
    key = list(corr.dropna().sort_values(ascending=False).head(n_key).index)
    inter = {f"{k} x {d.replace('regime_', '')}": X[k] * X[d]
             for k in key for d in reg}
    return pd.concat([X, pd.DataFrame(inter, index=X.index)], axis=1), key


def zrows(F):
    mu, sd = F.mean(axis=1), F.std(axis=1).replace(0, np.nan)
    return F.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)


def portfolio(Z, r, rates, mode):
    """Turn a standardised forecast into weights, then run it annually."""
    n = Z.shape[1]
    if mode == "tilt":
        base = 1.0 / n
        W = (base + TILT_STRENGTH * Z * base).clip(
            lower=max(base - TILT_CAP, 0.0), upper=base + TILT_CAP)
    else:
        W = Z.clip(lower=0.0)
    W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / n)

    idx = W.index
    R = r.reindex(idx).to_numpy()
    Wt = W.to_numpy()
    nets, held, traded = [], None, 0.0
    for i in range(len(idx)):
        if i % REBAL == 0 or held is None:
            t = Wt[i]
            t = np.ones(n) / n if not np.isfinite(t).all() or t.sum() <= 0 else t / t.sum()
            pre = np.zeros(n) if held is None else held
            cost = float(np.abs(t - pre) @ rates)
            traded += 0.0 if pre.sum() == 0 else float(np.abs(t - pre).sum()) / 2
            held = t
        else:
            cost = 0.0
        nets.append(float(R[i] @ held) - cost)
        g = held * (1 + R[i])
        held = g / g.sum() if g.sum() > 0 else held
    return pd.Series(nets, index=idx), traded / (len(idx) / PPY)


GRIDS = {
    "elastic_net": [{"alpha": a, "l1_ratio": l}
                    for a in (0.1, 1.0) for l in (0.5, 0.9)],
    "ridge": [{"alpha": a} for a in (10.0, 100.0, 1000.0)],
    "random_forest": [{"max_depth": d, "n_estimators": 150} for d in (2, 3)],
    "gradient_boosting": [{"max_depth": 1, "learning_rate": lr,
                           "n_estimators": 100} for lr in (0.01, 0.05)],
}
FACTORY = {"elastic_net": ml.elastic_net, "ridge": ml.ridge,
           "random_forest": ml.random_forest,
           "gradient_boosting": ml.gradient_boosting}


def main() -> int:
    r_all = pd.read_parquet(P / "fi_daily_returns.parquet")
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf.reindex(r.index).fillna(0.0)
    rates = np.array([COST_BP[a] / 1e4 for a in assets])
    dev_mask = r.index <= pd.Timestamp(DEV_END)
    agg = get_prices(["VBMFX"])["VBMFX"].pct_change().reindex(r.index)

    X = pd.concat([ds.build(r_all), regime_frame(r.index)], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    X, key = add_interactions(X, r, dev_mask)
    print(f"Daily signal panel: {X.shape[1]} columns over {len(X):,} days")
    print(f"  regime dummies {len([c for c in X.columns if c.startswith('regime_')])}"
          f", interactions {len([c for c in X.columns if ' x ' in c])}")
    print(f"  interacted on: {key}\n")

    # ---------------------------------------------------- univariate combination
    print("Univariate combination across every signal:")
    uni, skill = {}, {}
    for a in assets:
        y = (r[a] - rf).dropna()
        F = fc.univariate_forecasts(y, X.reindex(y.index), BURN_IN, 1)
        f = fc.combine(F, "mean")
        b = fc.prevailing_mean(y, min_obs=BURN_IN)
        m = f.dropna().index.intersection(b.dropna().index)
        dv, oo = m[m <= pd.Timestamp(DEV_END)], m[m >= pd.Timestamp(OOS_START)]
        uni[a] = f
        skill[a] = {"dev_r2": fc.oos_r2(y[dv], f[dv], b[dv]),
                    "oos_r2": fc.oos_r2(y[oo], f[oo], b[oo])}
    uni = pd.DataFrame(uni)
    U = pd.DataFrame(skill).T
    U.to_parquet(P / "fi_dmodel_regression_skill.parquet")
    print((U * 100).round(3).to_string())
    print(f"  mean development {U['dev_r2'].mean() * 100:+.3f}%, "
          f"holdout {U['oos_r2'].mean() * 100:+.3f}%\n")

    # ---------------------------------------------------- machine learning
    print("Machine learning, tuned on development:")
    cfg = ml.MLConfig(min_train=BURN_IN, refit_every=PPY, max_features=30)
    chosen, fcst, mskill = {}, {}, {}
    for name, grid in GRIDS.items():
        best, best_score, best_fc = None, -np.inf, None
        for params in grid:
            preds, scores = {}, []
            for a in assets:
                y = (r[a] - rf).dropna()
                f = ml.walk_forward(y, X.reindex(y.index),
                                    FACTORY[name](**params), cfg)
                b = fc.prevailing_mean(y, min_obs=BURN_IN)
                m = f.dropna().index.intersection(b.dropna().index)
                m = m[m <= pd.Timestamp(DEV_END)]
                if len(m) > PPY:
                    scores.append(fc.oos_r2(y[m], f[m], b[m]))
                preds[a] = f
            sc = float(np.mean(scores)) if scores else -np.inf
            if sc > best_score:
                best, best_score, best_fc = params, sc, pd.DataFrame(preds)
        chosen[name] = {"params": str(best), "dev_r2": best_score}
        fcst[name] = best_fc
        per = {}
        for a in assets:
            y = (r[a] - rf).dropna()
            b = fc.prevailing_mean(y, min_obs=BURN_IN)
            f = best_fc[a]
            m = f.dropna().index.intersection(b.dropna().index)
            dv, oo = m[m <= pd.Timestamp(DEV_END)], m[m >= pd.Timestamp(OOS_START)]
            per[a] = {"dev_r2": fc.oos_r2(y[dv], f[dv], b[dv]) if len(dv) > PPY else np.nan,
                      "oos_r2": fc.oos_r2(y[oo], f[oo], b[oo]) if len(oo) > PPY else np.nan}
        mskill[name] = pd.DataFrame(per).T
        print(f"  {name:<18} {str(best):<52} dev R2 {best_score * 100:+.3f}%")
    pd.DataFrame(chosen).T.to_parquet(P / "fi_dmodel_ml_chosen.parquet")
    pd.concat(mskill, axis=0).to_parquet(P / "fi_dmodel_ml_skill.parquet")

    # ---------------------------------------------------- technical signals
    carry = pd.DataFrame(
        {a: X[f"d_carry_{a}"] if f"d_carry_{a}" in X else np.nan for a in assets},
        index=X.index)
    if carry.isna().all().all():
        carry = r.rolling(PPY).mean() * PPY          # realised yield proxy
    mom = r.rolling(PPY).apply(lambda x: np.prod(1 + x) - 1, raw=True).shift(21)
    # Momentum on risk-adjusted return, not raw return: the same trailing window
    # divided by its own volatility. On a bond book the raw version simply ranks
    # by duration, which is a risk bet wearing a signal's clothes.
    msharpe = (r.rolling(PPY).mean() / r.rolling(PPY).std()).shift(21)

    signals = {"Carry": carry, "Momentum": mom, "Momentum (Sharpe)": msharpe,
               "Regression": uni}
    for k, v in fcst.items():
        signals[f"ML {k.replace('_', ' ')}"] = v

    first = r.index[BURN_IN]
    strat, turns = {}, {}
    for name, F in signals.items():
        Z = zrows(F.reindex(columns=assets)).loc[first:]
        for mode in ("tilt", "base"):
            s, tn = portfolio(Z, r, rates, mode)
            strat[f"{name}, {mode}"] = s
            turns[f"{name}, {mode}"] = tn

    # Risk-based reference, annual rebalancing, on the same dates.
    for lbl, fn in [("HRP", lambda S, c: RB.RP.hierarchical_rp(S, list(c))),
                    ("ERC", lambda S, c: RB.RP.erc_weights(S))]:
        s, W, tr, tn = RB.walk(r, rf, rates, fn, lookback=None, every=REBAL)
        strat[f"{lbl}, annual"] = s
        turns[f"{lbl}, annual"] = tn
    strat[BENCH] = agg

    S = pd.DataFrame(strat).dropna(how="any")
    S.to_parquet(P / "fi_dmodel_strategies.parquet")

    dev = S.index <= pd.Timestamp(DEV_END)
    oos = S.index >= pd.Timestamp(OOS_START)
    rf2 = rf.reindex(S.index)
    rows = []
    for c in S.columns:
        if c == BENCH:
            continue
        row = {"strategy": c,
               "class": ("risk based" if c.endswith("annual")
                         else "machine learning" if c.startswith("ML ")
                         else "regression" if c.startswith("Regression")
                         else "technical"),
               "turnover": turns.get(c, np.nan)}
        for tag, msk in [("dev", dev), ("oos", oos)]:
            m = metrics.performance(S[c][msk], rf2[msk], periods_per_year=PPY)
            bm = metrics.performance(S[BENCH][msk], rf2[msk], periods_per_year=PPY)
            d = inference.sharpe_difference(S[c][msk], S[BENCH][msk],
                                            rf=rf2[msk], ppy=PPY)
            row[f"{tag}_sharpe"] = m["sharpe"]
            row[f"{tag}_vol"] = m["vol"]
            row[f"{tag}_vs_agg"] = m["sharpe"] - bm["sharpe"]
            row[f"{tag}_p"] = d["p_one_sided"]
        rows.append(row)
    T = pd.DataFrame(rows).set_index("strategy").sort_values(
        "dev_sharpe", ascending=False)
    T.to_parquet(P / "fi_dmodel_summary.parquet")

    bd = metrics.performance(S[BENCH][dev], rf2[dev], periods_per_year=PPY)["sharpe"]
    bo = metrics.performance(S[BENCH][oos], rf2[oos], periods_per_year=PPY)["sharpe"]
    print(f"\nAgg: development {bd:.4f}, holdout {bo:.4f}")
    print(f"Window {S.index.min():%Y-%m-%d} to {S.index.max():%Y-%m-%d}, "
          f"{len(S):,} days\n")
    print(T.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
