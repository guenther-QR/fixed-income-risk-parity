"""Does shorting unlock anything in fixed income?

The expanded universe answered a question it was not asked. Going from twelve
assets to twenty-six *reduced* effective breadth, from 2.5 to 2.22, because the
added instruments were redundant - four interpolated curve points, a second
mortgage fund correlated 0.98 with the first, duplicate managers in the same
sector. The first principal component explains 64.3% of variance.

Fixed income is one factor. That is not a data problem to be solved by finding
more bonds; it is what the asset class is. Every long-only portfolio in this
universe is therefore a duration bet plus a small residual, and the long-only
constraint means the residual can never be isolated.

Shorting is the only way to trade the residual, which makes it the natural next
test rather than an exotic one:

    unconstrained ERC     risk parity without the long-only constraint. Lets the
                          optimiser short the assets whose risk contribution is
                          negative-valued, which it cannot express otherwise.
    PC1-neutral           hedge the first principal component to zero. What is
                          left is, by construction, everything that is not the
                          level of rates.
    duration-neutral      long and short legs matched on modified duration, so
                          the book has no directional rate exposure at all.
    relative value        explicit curve, credit and municipal trades - the views
                          a fixed income desk actually expresses.
    cross-sectional L/S   rank on carry, momentum and value; long the top,
                          short the bottom, dollar-neutral.

Two costs are charged that long-only books avoid. Short positions pay a borrow
fee, taken here at the same per-asset transaction rate, and a market-neutral book
earns no risk premium, so it has to be funded - its returns are reported as an
overlay on cash rather than as a standalone portfolio that can be compared to
1/N on a Sharpe ratio without qualification.
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

from macro.backtest import leverage, metrics, speclog  # noqa: E402
from macro.portfolio import covariance as cov, cross_section as xs  # noqa: E402
from macro.stats import inference as inf  # noqa: E402

P7 = import_module("07_wide_fi_universe")
PROCESSED = ROOT / "data/processed"
DEV_END = "2015-12"
OOS_START = "2016-01"
MIN_TRAIN = 60
MAX_GROSS = 2.0
BORROW_BP = 30          # annual borrow fee on short positions
FINANCE_BP = 50         # spread over risk-free on borrowed capital


def load():
    r = pd.read_parquet(PROCESSED / "fi_wide_returns.parquet")
    rf = pd.read_parquet(PROCESSED / "fi_wide_rf.parquet")["rf"].reindex(r.index)
    g = pd.read_parquet(PROCESSED / "fi_wide_groups.parquet")["group"]
    dur = pd.read_parquet(PROCESSED / "fi_wide_duration.parquet")["duration"]
    return r, rf, g.reindex(r.columns), dur.reindex(r.columns)


def erc(S, long_only=True, iters=400):
    n = S.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        rc = w * (S @ w)
        w = np.maximum(w * ((rc.sum() / n) /
                            np.where(rc > 1e-16, rc, 1e-16)) ** 0.5, 1e-9)
        w = w / w.sum()
    if long_only:
        return w
    # Unconstrained risk parity: solve Sigma^-1 * (1/vol) which permits shorts
    # where an asset's marginal contribution is best expressed negatively.
    try:
        inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(S)
    v = 1.0 / np.sqrt(np.diag(S))
    u = inv @ v
    return u / np.abs(u).sum()


def pc1_neutral(S, base_w):
    """Remove the first principal component from a weight vector."""
    vals, vecs = np.linalg.eigh(S)
    p1 = vecs[:, -1]
    p1 = p1 / np.linalg.norm(p1)
    return base_w - (base_w @ p1) * p1


def duration_neutral(w, dur):
    """Shift weights so portfolio duration is zero, keeping relative bets."""
    d = dur.to_numpy()
    if np.abs(d).sum() < 1e-9:
        return w
    return w - (w @ d) / (d @ d) * d


def scale_gross(w, max_gross=MAX_GROSS):
    g = np.abs(w).sum()
    return w * (max_gross / g) if g > max_gross and g > 0 else w


def backtest(r, rf, W, rates, allow_short=True):
    """
    Hold `W`, charging transaction costs, borrow fees and financing.

    A book with gross exposure above one is borrowing and pays the financing
    spread; a book with short positions pays a borrow fee on them. Both are
    charged here because a long-short strategy that ignores them is not a
    strategy, it is an accounting choice.
    """
    A = r.to_numpy()
    Wt = W.to_numpy()
    nets, held, gross_l, net_l = [], None, [], []
    for i in range(len(r)):
        w = Wt[i]
        if not np.isfinite(w).all():
            nets.append(np.nan)
            continue
        pre = np.zeros(len(w)) if held is None else held
        cost = float(np.abs(w - pre) @ rates)
        shorts = float(np.abs(np.minimum(w, 0.0)).sum())
        gross = float(np.abs(w).sum())
        cost += shorts * (BORROW_BP / 1e4) / 12.0
        cost += max(gross - 1.0, 0.0) * (FINANCE_BP / 1e4) / 12.0
        held = w
        port = float(A[i] @ held)
        nets.append(port - cost)
        gross_l.append(gross)
        net_l.append(float(w.sum()))
        # Weight drift. The divisor is the portfolio's gross return, not the sum
        # of the grown weights: for a long-only book with weights summing to one
        # those are identical, but for a dollar-neutral book the sum is near zero
        # and dividing by it explodes the weights. That bug produced 150%
        # annualised volatility in an earlier version of this script.
        held = held * (1 + A[i]) / (1.0 + port) if abs(1.0 + port) > 1e-9 else held
    return (pd.Series(nets, index=r.index).dropna(),
            float(np.mean(gross_l)), float(np.mean(net_l)))


def walk(r, rf, fn, rates):
    cols = list(r.columns)
    W = []
    for i in range(len(r)):
        if i < MIN_TRAIN:
            W.append([np.nan] * len(cols))
            continue
        ex = r.iloc[:i].sub(rf.iloc[:i], axis=0).dropna()
        S = cov.ledoit_wolf(ex)
        w = fn(S, cols, ex)
        w = np.nan_to_num(w, nan=0.0)
        W.append(scale_gross(w))
    return pd.DataFrame(W, index=r.index, columns=cols)


def main() -> int:
    r, rf, groups, dur = load()
    cols = list(r.columns)
    rates = np.array([P7.COST_BP.get(c, 20) / 1e4 for c in cols])
    dev, oos = slice(None, DEV_END), slice(OOS_START, None)
    print(f"Universe {len(cols)} assets, {len(r)} months "
          f"({r.index.min():%Y-%m} to {r.index.max():%Y-%m})\n")

    strat, meta = {}, {}

    def add(name, Wd, **kw):
        s, g, n = backtest(r, rf, Wd, rates, **kw)
        strat[name] = s
        meta[name] = {"gross": g, "net": n}

    add("1/N (long only)", pd.DataFrame(1.0 / len(cols), index=r.index, columns=cols))
    add("Risk parity (long only)",
        walk(r, rf, lambda S, c, e: erc(S, True), rates))
    add("Risk parity (unconstrained)",
        walk(r, rf, lambda S, c, e: erc(S, False), rates))
    add("PC1-neutral risk parity",
        walk(r, rf, lambda S, c, e: pc1_neutral(S, erc(S, True)), rates))
    add("Duration-neutral risk parity",
        walk(r, rf, lambda S, c, e: duration_neutral(erc(S, True), dur), rates))

    # cross-sectional long-short on fixed income signals
    carry = r.rolling(12, min_periods=6).mean().shift(1)
    mom = ((1 + r).rolling(12, min_periods=12).apply(np.prod, raw=True) - 1).shift(1)
    value = -(r.rolling(60, min_periods=36).mean().shift(1))
    for nm, sig in [("carry", carry), ("momentum", mom), ("5y reversal", value)]:
        W = xs.long_short(sig, 5, 5, gross=1.0)
        add(f"XS long-short: {nm}", W.reindex(r.index).fillna(0.0))

    comp = xs.composite({"carry": carry, "momentum": mom})
    add("XS long-short: carry+momentum",
        xs.long_short(comp, 5, 5, gross=1.0).reindex(r.index).fillna(0.0))

    # explicit relative value trades a desk would actually put on
    def rv(long_leg, short_leg, scale=1.0):
        W = pd.DataFrame(0.0, index=r.index, columns=cols)
        for c in long_leg:
            W[c] = scale / len(long_leg)
        for c in short_leg:
            W[c] = -scale / len(short_leg)
        return W
    if {"ust2y", "ust10y"} <= set(cols):
        add("RV: curve steepener (2s10s)", rv(["ust2y"], ["ust10y"]))
    if {"ig_interm", "ust5y"} <= set(cols):
        add("RV: credit spread (IG vs UST)", rv(["ig_interm"], ["ust5y"]))
    if {"hy", "ig_interm"} <= set(cols):
        add("RV: HY vs IG", rv(["hy"], ["ig_interm"]))
    if {"muni_interm", "ust5y"} <= set(cols):
        add("RV: muni vs Treasury", rv(["muni_interm"], ["ust5y"]))

    # overlays: the long-only book plus a market-neutral sleeve
    base = strat["Risk parity (long only)"]
    for nm in ["XS long-short: carry+momentum", "PC1-neutral risk parity"]:
        common = base.index.intersection(strat[nm].index)
        strat[f"RP + 30% {nm.split(':')[-1].strip()}"] = (
            base.loc[common] + 0.30 * strat[nm].loc[common])

    t = metrics.comparison_table({k: v.loc[dev] for k, v in strat.items()},
                                 rf.loc[dev])
    t_o = metrics.comparison_table({k: v.loc[oos] for k, v in strat.items()},
                                   rf.loc[oos])
    sh_b = t.loc["1/N (long only)", "sharpe"]
    sh_bo = t_o.loc["1/N (long only)", "sharpe"]
    t["vs_1N"] = t["sharpe"] - sh_b
    t["oos_sharpe"] = t_o["sharpe"]
    t["oos_edge"] = t_o["sharpe"] - sh_bo
    t["gross"] = pd.Series({k: v["gross"] for k, v in meta.items()})
    t["net"] = pd.Series({k: v["net"] for k, v in meta.items()})
    T = t.sort_values("vs_1N", ascending=False)

    print("SHORTING-ENABLED STRATEGIES (development, all costs charged)")
    print(T[["cagr", "vol", "sharpe", "vs_1N", "oos_sharpe", "oos_edge",
             "max_drawdown", "gross", "net"]]
          .to_string(float_format=lambda x: f"{x:9.4f}"))

    print(f"\n   positive on development: {(T.vs_1N > 0).sum()} of {len(T)}")
    print(f"   positive on holdout:     {(T.oos_edge > 0).sum()} of {len(T)}")
    print(f"   dev/holdout rank corr:   "
          f"{T['vs_1N'].corr(T['oos_edge'], method='spearman'):+.3f}")

    print("\nBLOCK BOOTSTRAP vs 1/N (development) - top 6")
    for k in T.head(7).index:
        if k == "1/N (long only)":
            continue
        d = inf.sharpe_difference(strat[k].loc[dev],
                                  strat["1/N (long only)"].loc[dev],
                                  rf.loc[dev], n_boot=3000, mean_block=12.0)
        star = "  *" if d["ci_lo"] > 0 else ""
        print(f"   {k:34s} {d['difference']:+.4f}  "
              f"CI [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  p={d['p_one_sided']:.4f}{star}")

    print("\nMARKET-NEUTRAL SLEEVES ON THEIR OWN (development)")
    neutral = [k for k in strat if k.startswith(("XS long-short", "RV:",
                                                 "PC1-neutral", "Duration-neutral"))]
    tn = metrics.comparison_table({k: strat[k].loc[dev] for k in neutral},
                                  rf.loc[dev])
    tn["oos_sharpe"] = metrics.comparison_table(
        {k: strat[k].loc[oos] for k in neutral}, rf.loc[oos])["sharpe"]
    print(tn[["cagr", "vol", "sharpe", "oos_sharpe", "max_drawdown"]]
          .sort_values("sharpe", ascending=False)
          .to_string(float_format=lambda x: f"{x:9.4f}"))
    print("\n   These earn no risk premium by construction, so a Sharpe ratio")
    print("   here is an information ratio on a funded overlay, not a portfolio")
    print("   that can be held on its own.")

    for name in T.index:
        speclog.record(speclog.Spec(
            phase="FI-8", family="shorting", name=name,
            config={"n_assets": len(cols), "max_gross": MAX_GROSS,
                    "borrow_bp": BORROW_BP},
            metrics={"sharpe": float(T.loc[name, "sharpe"]),
                     "vs_1N": float(T.loc[name, "vs_1N"]),
                     "oos_edge": float(T.loc[name, "oos_edge"])},
            n_periods=int(len(strat[name].loc[dev]))))

    pd.DataFrame(strat).to_parquet(PROCESSED / "fi_short_strategies.parquet")
    T.to_parquet(PROCESSED / "fi_short_summary.parquet")
    print("\n" + speclog.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
