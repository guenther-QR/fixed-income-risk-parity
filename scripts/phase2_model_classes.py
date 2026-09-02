"""Every model class on one sample, one burn-in, and one benchmark.

The project had accumulated four different estimation windows across its
scripts, which meant the regression classes were being judged on a five-year
shorter sample than the risk-based ones. Everything here uses a single 60-month
burn-in from the first month of data (1982-11), so every model produces its
first weight in 1987-11 and every comparison below is like for like.

Structure:

    signal          carry and momentum. Purely technical, no estimation.
    regression      univariate combination across the signal panel, which now
                    includes regime dummies and regime interactions.
    machine learning elastic net, ridge, random forest, gradient boosting, each
                    tuned on the development sample before it is scored.
    regime          the covariance or the mean estimated within regime.
    risk based      ERC and HRP on Ledoit-Wolf and on DCC-GARCH.

Each return-forecasting signal is turned into a portfolio two ways:

    tilt            bounded deviation from equal weight, w = clip(w_eq + L*z)
    base            the signal alone decides the weights, long only, normalised

The tilt is what a desk would run. The base version is what the signal actually
wants to hold, and it is shown because a tilt that barely moves can hide a
signal that is badly wrong.

Writes fi_uni_*.parquet.
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
from macro.portfolio import covariance as cov  # noqa: E402
from macro.predict import forecast as fc, models as ml  # noqa: E402
from macro.signals import library as lib, literature as lit  # noqa: E402
from macro.stats import inference  # noqa: E402

RP = import_module("phase2_risk_parity")

P = ROOT / "data/processed"
MACRO = ROOT.parent / "Macro_26/data/processed"

# One burn-in for the whole project. Data begins 1982-11, so the first weight
# is formed for 1987-11 and every model below shares that start.
BURN_IN = 60
DEV_END = "2015-12-31"
OOS_START = "2016-01-01"
BENCH = "Agg index (VBMFX)"

TILT_STRENGTH = 0.5
TILT_CAP = 0.15

COST_BP = {"ust2y": 2, "ust5y": 3, "ust10y": 3, "ust30y": 4, "ig_short": 15,
           "ig": 20, "ig_long": 25, "hy": 40, "mbs": 12, "muni": 25,
           "muni_hy": 45}

DUR = {"ust2y": 1.9, "ust5y": 4.6, "ust10y": 8.4, "ust30y": 18.5,
       "ig_short": 2.5, "ig": 4.2, "ig_long": 12.0, "hy": 4.0, "mbs": 4.5,
       "muni": 5.0, "muni_hy": 7.5}


# ------------------------------------------------------------------ helpers

def erc(S):
    return RP.erc_weights(S)


def hrp(S, cols):
    return RP.hierarchical_rp(S, list(cols))


def zscore_rows(F):
    """Cross-sectional z-score, one row at a time."""
    mu = F.mean(axis=1)
    sd = F.std(axis=1).replace(0, np.nan)
    return F.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)


def tilt_weights(Z, n):
    """Bounded deviation from equal weight."""
    base = 1.0 / n
    W = base + TILT_STRENGTH * Z * base
    W = W.clip(lower=max(base - TILT_CAP, 0.0), upper=base + TILT_CAP)
    return W.div(W.sum(axis=1), axis=0)


def base_weights(Z):
    """The signal alone decides. Long only, so negatives are floored at zero."""
    W = Z.clip(lower=0.0)
    tot = W.sum(axis=1).replace(0, np.nan)
    W = W.div(tot, axis=0)
    return W.fillna(1.0 / Z.shape[1])


def run_weights(W, r, rates):
    """Net returns from a weight path, charging turnover, with weight drift."""
    idx = W.dropna(how="all").index
    nets, held = [], None
    for t in idx:
        target = W.loc[t].to_numpy(dtype=float)
        if not np.isfinite(target).all() or target.sum() <= 0:
            nets.append(np.nan)
            continue
        target = target / target.sum()
        pre = np.zeros(len(target)) if held is None else held
        cost = float(np.abs(target - pre) @ rates)
        held = target
        nets.append(float(r.loc[t].to_numpy() @ held) - cost)
        grown = held * (1 + r.loc[t].to_numpy())
        held = grown / grown.sum() if grown.sum() > 0 else held
    return pd.Series(nets, index=idx).dropna()


def walk_cov(r, rf, weight_fn, rates, labels=None, min_regime=24):
    """Walk forward on the covariance matrix only. Returns nets and weights."""
    assets = list(r.columns)
    nets, held, rows, dates = [], None, [], []
    for i in range(len(r)):
        if i < BURN_IN:
            continue
        ex = r.iloc[:i].sub(rf.iloc[:i], axis=0).dropna()
        if labels is not None:
            now = labels.iloc[i]
            same = labels.iloc[:i] == now
            sub = ex[same.reindex(ex.index).fillna(False).to_numpy()]
            if len(sub) >= min_regime:
                ex = sub
        target = np.nan_to_num(weight_fn(ex), nan=0.0)
        if target.sum() <= 0:
            target = np.ones(len(assets)) / len(assets)
        target = target / target.sum()
        pre = np.zeros(len(assets)) if held is None else held
        cost = float(np.abs(target - pre) @ rates)
        held = target
        nets.append(float(r.iloc[i].to_numpy() @ held) - cost)
        rows.append(held.copy())
        dates.append(r.index[i])
        grown = held * (1 + r.iloc[i].to_numpy())
        held = grown / grown.sum() if grown.sum() > 0 else held
    return (pd.Series(nets, index=dates),
            pd.DataFrame(rows, index=dates, columns=assets))


def regime_labels(index):
    src = P / "regimes_monthly.parquet"
    if not src.exists() and (MACRO / "regimes_monthly.parquet").exists():
        pd.read_parquet(MACRO / "regimes_monthly.parquet").to_parquet(src)
    if not src.exists():
        return None
    g = pd.read_parquet(src)["regime"].reindex(index, method="ffill")
    return g if g.notna().sum() > len(index) * 0.5 else None


# ------------------------------------------------------------------ signals

def build_panel(r, dev_mask):
    """Signal panel, plus regime dummies and regime interactions.

    The interaction terms are the point of this function. A regime dummy on its
    own only shifts the intercept; it says the average return differs by state
    but not that a signal *works* differently by state. Interacting the regime
    with a signal lets the slope change, which is the claim regime conditioning
    actually makes.

    Which signals get interacted is decided on the development sample only, by
    the magnitude of their standardised univariate slope against the equally
    weighted portfolio.
    """
    X = pd.concat([lib.build(r, P), lit.build(r, P)], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    reg = [c for c in X.columns if c.startswith("regime_")]
    if not reg:
        print("  WARNING: no regime dummies in the panel")
        return X, []

    y = r.mean(axis=1)
    cand = [c for c in X.columns if not c.startswith("regime_")]
    Xd, yd = X.loc[dev_mask, cand], y.loc[dev_mask]
    Z = (Xd - Xd.mean()) / Xd.std().replace(0, np.nan)
    slope = Z.apply(lambda s: np.abs(np.corrcoef(
        s.fillna(0), yd.reindex(s.index).fillna(0))[0, 1]))
    key = list(slope.dropna().sort_values(ascending=False).head(8).index)

    inter = {}
    for k in key:
        for d in reg:
            inter[f"{k} x {d.replace('regime_', '')}"] = X[k] * X[d]
    X = pd.concat([X, pd.DataFrame(inter, index=X.index)], axis=1)
    return X, key


# ------------------------------------------------------------------ tuning

GRIDS = {
    "elastic_net": [{"alpha": a, "l1_ratio": l}
                    for a in (0.01, 0.1, 1.0) for l in (0.15, 0.5, 0.85)],
    "ridge": [{"alpha": a} for a in (1.0, 10.0, 100.0)],
    "random_forest": [{"max_depth": d, "n_estimators": 200}
                      for d in (2, 3, 4)],
    "gradient_boosting": [{"max_depth": d, "learning_rate": lr,
                           "n_estimators": 150}
                          for d in (1, 2) for lr in (0.01, 0.05)],
}
FACTORY = {"elastic_net": ml.elastic_net, "ridge": ml.ridge,
           "random_forest": ml.random_forest,
           "gradient_boosting": ml.gradient_boosting}


def tune_and_forecast(r, rf, X, assets, dev_mask):
    """Grid search on development only, then one forecast at the chosen setting.

    Selection never sees the holdout. The chosen parameters are printed so the
    search is visible rather than implied.
    """
    cfg = ml.MLConfig(min_train=BURN_IN, refit_every=24, max_features=20)
    chosen, out, skill = {}, {}, {}
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
                m = m[m <= pd.Timestamp(DEV_END)]          # development only
                if len(m) > 24:
                    scores.append(fc.oos_r2(y[m], f[m], b[m]))
                preds[a] = f
            score = float(np.mean(scores)) if scores else -np.inf
            if score > best_score:
                best, best_score, best_fc = params, score, pd.DataFrame(preds)
        chosen[name] = {"params": best, "dev_r2": best_score}
        out[name] = best_fc
        print(f"  {name:<18} best {best}  development R2 {best_score:+.4%}")
        # Score the chosen model on both windows.
        per = {}
        for a in assets:
            y = (r[a] - rf).dropna()
            b = fc.prevailing_mean(y, min_obs=BURN_IN)
            f = best_fc[a]
            m = f.dropna().index.intersection(b.dropna().index)
            dv = m[m <= pd.Timestamp(DEV_END)]
            oo = m[m >= pd.Timestamp(OOS_START)]
            per[a] = {"dev_r2": fc.oos_r2(y[dv], f[dv], b[dv]) if len(dv) > 24 else np.nan,
                      "oos_r2": fc.oos_r2(y[oo], f[oo], b[oo]) if len(oo) > 24 else np.nan}
        skill[name] = pd.DataFrame(per).T
    return chosen, out, skill


def main() -> int:
    r_all = pd.read_parquet(P / "fi_returns.parquet")
    rf_all = pd.read_parquet(P / "fi_rf.parquet").squeeze().reindex(r_all.index)
    bench = pd.read_parquet(P / "fi_bench_final.parquet")[BENCH]
    assets = [c for c in r_all.columns if c != "ust3m"]
    r = r_all[assets]
    rf = rf_all
    rates = np.array([COST_BP.get(a, 20) / 1e4 for a in assets])
    dev_mask = r.index <= pd.Timestamp(DEV_END)
    print(f"Data {r.index.min():%Y-%m} to {r.index.max():%Y-%m}, "
          f"{len(r)} months, {len(assets)} assets")
    print(f"Burn-in {BURN_IN} months, so every model starts "
          f"{r.index[BURN_IN]:%Y-%m}\n")

    labels = regime_labels(r.index)
    X, key = build_panel(r, dev_mask)
    n_reg = len([c for c in X.columns if c.startswith("regime_")])
    n_int = len([c for c in X.columns if " x " in c])
    print(f"Signal panel {X.shape[1]} columns: {n_reg} regime dummies, "
          f"{n_int} regime interactions")
    print(f"  interacted signals: {key}\n")

    # ---------------------------------------------------- return forecasts
    print("Machine learning, tuned on development:")
    chosen, ml_fc, ml_skill = tune_and_forecast(r, rf, X, assets, dev_mask)
    pd.DataFrame({k: {"params": str(v["params"]), "dev_r2": v["dev_r2"]}
                  for k, v in chosen.items()}).T.to_parquet(
        P / "fi_uni_ml_chosen.parquet")
    SK = pd.concat({k: v for k, v in ml_skill.items()}, axis=0)
    SK.to_parquet(P / "fi_uni_ml_skill.parquet")

    print("\nUnivariate combination, with regimes and interactions:")
    uni, uni_skill = {}, {}
    for a in assets:
        y = (r[a] - rf).dropna()
        F = fc.univariate_forecasts(y, X.reindex(y.index), BURN_IN, 1)
        f = fc.combine(F, "mean")
        b = fc.prevailing_mean(y, min_obs=BURN_IN)
        m = f.dropna().index.intersection(b.dropna().index)
        dv, oo = m[m <= pd.Timestamp(DEV_END)], m[m >= pd.Timestamp(OOS_START)]
        uni[a] = f
        uni_skill[a] = {"dev_r2": fc.oos_r2(y[dv], f[dv], b[dv]),
                        "oos_r2": fc.oos_r2(y[oo], f[oo], b[oo])}
    uni = pd.DataFrame(uni)
    U = pd.DataFrame(uni_skill).T
    U.to_parquet(P / "fi_uni_regression_skill.parquet")
    print(U.round(4).to_string())
    print(f"  mean development R2 {U['dev_r2'].mean():+.4%}, "
          f"holdout {U['oos_r2'].mean():+.4%}\n")

    # ---------------------------------------------------- technical signals
    F03 = import_module("phase2_forecast_tilts")
    carry = F03.carry_edges(r).reindex(r.index)
    mom = F03.momentum_edges(r).reindex(r.index)

    # ---------------------------------------------------- portfolios
    strat, weights = {}, {}
    signals = {"Carry": carry, "Momentum": mom, "Regression": uni}
    for name, f in ml_fc.items():
        signals[f"ML {name.replace('_', ' ')}"] = f.reindex(r.index)

    first = r.index[BURN_IN]
    for name, F in signals.items():
        Z = zscore_rows(F.reindex(columns=assets)).loc[first:]
        strat[f"{name}, tilt"] = run_weights(tilt_weights(Z, len(assets)),
                                             r, rates)
        strat[f"{name}, base"] = run_weights(base_weights(Z), r, rates)

    # ---------------------------------------------------- risk based
    lw = lambda ex: cov.ledoit_wolf(ex)

    def dcc(ex):
        try:
            return cov.dcc_garch(ex)
        except Exception:
            return cov.ledoit_wolf(ex)

    strat["ERC"], _ = walk_cov(r, rf, lambda ex: erc(lw(ex)), rates)
    strat["HRP"], hrp_w = walk_cov(
        r, rf, lambda ex: hrp(lw(ex), ex.columns), rates)
    strat["ERC, DCC-GARCH"], _ = walk_cov(r, rf, lambda ex: erc(dcc(ex)), rates)
    strat["HRP, DCC-GARCH"], _ = walk_cov(
        r, rf, lambda ex: hrp(dcc(ex), ex.columns), rates)
    if labels is not None:
        strat["ERC, regime covariance"], _ = walk_cov(
            r, rf, lambda ex: erc(lw(ex)), rates, labels)
        strat["HRP, regime covariance"], _ = walk_cov(
            r, rf, lambda ex: hrp(lw(ex), ex.columns), rates, labels)
    strat["Equal weight"] = run_weights(
        pd.DataFrame(1.0 / len(assets), index=r.index[BURN_IN:],
                     columns=assets), r, rates)
    hrp_w.to_parquet(P / "fi_uni_hrp_weights.parquet")

    S = pd.DataFrame(strat)
    S[BENCH] = bench
    S = S.dropna(how="any")
    S.to_parquet(P / "fi_uni_strategies.parquet")
    print(f"All strategies on a common window: {S.index.min():%Y-%m} to "
          f"{S.index.max():%Y-%m}, {len(S)} months\n")

    # ---------------------------------------------------- evaluation
    CLASS = {"Carry": "technical", "Momentum": "technical",
             "Regression": "regression"}
    def cls(n):
        if n in (BENCH, "Equal weight"):
            return "benchmark"
        if n.startswith("ML "):
            return "machine learning"
        if "regime" in n:
            return "regime conditional"
        if n.startswith(("ERC", "HRP")):
            return "risk based"
        return CLASS.get(n.split(",")[0], "other")

    dev = S.index <= pd.Timestamp(DEV_END)
    oos = S.index >= pd.Timestamp(OOS_START)
    rf2 = rf.reindex(S.index)
    bd = metrics.performance(S[BENCH][dev], rf2[dev])["sharpe"]
    bo = metrics.performance(S[BENCH][oos], rf2[oos])["sharpe"]
    rows = []
    for c in S.columns:
        if c == BENCH:
            continue
        d = metrics.performance(S[c][dev], rf2[dev])
        o = metrics.performance(S[c][oos], rf2[oos])
        pdv = inference.sharpe_difference(S[c][dev], S[BENCH][dev], rf=rf2[dev])
        poo = inference.sharpe_difference(S[c][oos], S[BENCH][oos], rf=rf2[oos])
        rows.append({"strategy": c, "class": cls(c),
                     "dev_sharpe": d["sharpe"], "dev_vol": d["vol"],
                     "dev_vs_agg": d["sharpe"] - bd, "dev_p": pdv["p_one_sided"],
                     "oos_sharpe": o["sharpe"], "oos_vs_agg": o["sharpe"] - bo,
                     "oos_p": poo["p_one_sided"]})
    T = pd.DataFrame(rows).set_index("strategy").sort_values(
        "dev_sharpe", ascending=False)
    T.to_parquet(P / "fi_uni_summary.parquet")
    print(f"Agg benchmark: development {bd:.4f}, holdout {bo:.4f}\n")
    print(T.round(4).to_string())

    W = hrp_w.copy()
    W.index.name = "date"
    print("\nHRP average weights:")
    print((W.mean() * 100).round(2).sort_values(ascending=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
