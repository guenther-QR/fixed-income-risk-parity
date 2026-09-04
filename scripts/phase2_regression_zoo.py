"""Rolling regressions on trailing-return horizons: do any signals exist?

Six predictors per asset, each a trailing return over a different horizon.
Short horizons are where the literature finds reversal and long horizons where
it finds momentum, so the sign is left to the regression rather than imposed:
a negative slope on a one-month trailing return is reversal, a positive slope
on a twelve-month trailing return is momentum.

At each month end, for each asset and each predictor, fit an ordinary least
squares regression of next month's excess return on the predictor using the
trailing sixty months. Record the slope and its p-value. A predictor is
tradeable next month only if it clears five percent.

Monthly, non-overlapping observations are used deliberately. Sampling a
one-month forward return daily produces sixty overlapping observations for
every genuinely independent one, which inflates t-statistics several fold and
would manufacture significance that is not there.

This script only measures. With six predictors on eleven assets, sixty-six
tests are run every month, so roughly three will clear five percent by chance
alone. Nothing here is worth building a portfolio on unless the observed count
is well above that floor, so the counting is the point.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

P = ROOT / "data/processed"
WINDOW = 60                  # months in the rolling estimation window
DEV_END = "2015-12-31"
ALPHA = 0.05

# Trailing-return horizons in months. The first three are the short horizons
# where reversal is normally found, the last three are momentum horizons and
# skip the most recent month in the standard way.
HORIZONS = {"ret 1m": (1, 0), "ret 2m": (2, 0), "ret 3m": (3, 0),
            "mom 6m": (6, 1), "mom 12m": (12, 1), "mom 24m": (24, 1)}


def ols_t(x, y):
    """Slope, t statistic and two-sided p for a univariate regression."""
    n = len(x)
    if n < 20:
        return np.nan, np.nan, np.nan
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    if sxx <= 0:
        return np.nan, np.nan, np.nan
    beta = float((xc * (y - y.mean())).sum() / sxx)
    alpha = float(y.mean() - beta * x.mean())
    resid = y - (alpha + beta * x)
    s2 = float((resid ** 2).sum() / (n - 2))
    se = np.sqrt(s2 / sxx)
    if not np.isfinite(se) or se <= 0:
        return beta, np.nan, np.nan
    t = beta / se
    return beta, float(t), float(2 * st.t.sf(abs(t), n - 2))


def main() -> int:
    rd = pd.read_parquet(P / "fi_daily_returns.parquet")
    rfd = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    assets = [c for c in rd.columns if c != "ust3m"]
    rm = (1 + rd[assets]).resample("ME").prod() - 1
    rfm = (1 + rfd).resample("ME").prod() - 1
    rfm = rfm.reindex(rm.index).fillna(0.0)
    ex = rm.sub(rfm, axis=0)
    print(f"{len(rm)} months, {len(assets)} assets, "
          f"{rm.index[0]:%Y-%m} to {rm.index[-1]:%Y-%m}")
    print(f"{len(HORIZONS)} predictors, {WINDOW}-month rolling window\n")

    # predictors, each known at the end of month t
    X = {}
    for name, (h, skip) in HORIZONS.items():
        cum = (1 + rm).rolling(h).apply(lambda v: np.prod(v) - 1, raw=True)
        X[name] = cum.shift(skip)

    rows = []
    months = rm.index
    start = max(WINDOW + max(h + s for h, s in HORIZONS.values()) + 1, WINDOW + 2)
    for i in range(start, len(months) - 1):
        t = months[i]
        lo = i - WINDOW
        for a in assets:
            yv = ex[a].iloc[lo + 1:i + 1]          # returns at s+1
            for name in HORIZONS:
                xv = X[name][a].iloc[lo:i]         # predictor at s
                d = pd.DataFrame({"x": xv.values, "y": yv.values}).dropna()
                if len(d) < 40:
                    continue
                b, tt, p = ols_t(d["x"].values, d["y"].values)
                if not np.isfinite(p):
                    continue
                rows.append({"date": t, "asset": a, "predictor": name,
                             "beta": b, "t": tt, "p": p,
                             "sig": bool(p < ALPHA),
                             "dir": int(np.sign(b)) if p < ALPHA else 0})
    R = pd.DataFrame(rows)
    R.to_parquet(P / "fi_rolling_regression_zoo.parquet")
    dev = R[R["date"] <= pd.Timestamp(DEV_END)]
    print(f"{len(R):,} regressions, {len(dev):,} on development\n")

    n_tests = dev.groupby("date").size()
    n_sig = dev.groupby("date")["sig"].sum()
    print("=== Is anything there at all? (development) ===")
    print(f"  tests per month           {n_tests.mean():.1f}")
    print(f"  significant per month     {n_sig.mean():.2f}")
    print(f"  expected by chance at 5%  {n_tests.mean() * ALPHA:.2f}")
    print(f"  ratio                     {n_sig.mean() / (n_tests.mean() * ALPHA):.2f}x")
    print(f"  months with zero signals  {(n_sig == 0).mean():.1%}")
    print(f"  overall hit rate          {dev['sig'].mean():.1%}")

    print("\n=== By predictor (development) ===")
    bp = dev.groupby("predictor").agg(
        tests=("sig", "size"), sig_rate=("sig", "mean"),
        mean_beta=("beta", "mean"))
    bp["ratio_vs_chance"] = bp["sig_rate"] / ALPHA
    s = dev[dev["sig"]]
    bp["pct_positive_when_sig"] = s.groupby("predictor")["dir"].apply(
        lambda v: (v > 0).mean())
    print(bp.round(4).to_string())

    print("\n=== By asset (development) ===")
    ba = dev.groupby("asset").agg(sig_rate=("sig", "mean"))
    ba["ratio_vs_chance"] = ba["sig_rate"] / ALPHA
    ba["pct_positive_when_sig"] = s.groupby("asset")["dir"].apply(
        lambda v: (v > 0).mean())
    print(ba.sort_values("sig_rate", ascending=False).round(4).to_string())

    print("\n=== Persistence: if significant this month, then next? ===")
    piv = dev.pivot_table(index="date", columns=["asset", "predictor"],
                          values="sig", aggfunc="first")
    cur, nxt = piv.iloc[:-1].to_numpy(), piv.iloc[1:].to_numpy()
    m = ~pd.isna(cur) & ~pd.isna(nxt)
    cur, nxt = cur[m].astype(bool), nxt[m].astype(bool)
    print(f"  P(sig next | sig now)     {nxt[cur].mean():.1%}")
    print(f"  P(sig next | not sig now) {nxt[~cur].mean():.1%}")
    print(f"  unconditional             {nxt.mean():.1%}")

    print("\n=== How many assets are tradeable in a given month? ===")
    per = dev[dev["sig"]].groupby("date")["asset"].nunique().reindex(
        n_tests.index).fillna(0)
    print(per.describe().round(2).to_string())
    print("\n  distribution of tradeable assets per month:")
    print(per.value_counts().sort_index().to_string())

    print("\n=== Sign agreement when an asset has several signals ===")
    g = dev[dev["sig"]].groupby(["date", "asset"])["dir"]
    agree = g.apply(lambda v: abs(v.sum()) == len(v) if len(v) > 1 else np.nan)
    agree = agree.dropna()
    print(f"  asset-months with 2+ signals   {len(agree):,}")
    print(f"  all signals agree in direction {agree.mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
