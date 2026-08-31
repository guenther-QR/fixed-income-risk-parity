"""FI_26 Phase 2 - a dynamic allocation engine against 1/N.

1/N is not a soft benchmark. DeMiguel, Garlappi and Uppal (2009) showed that
across fourteen datasets no optimising model they tested beat equal weighting out
of sample, because the estimation error in expected returns swamps the
optimisation gain. In a twelve-asset fixed income universe where the first
principal component explains 65% of variance, that argument is stronger still:
most of what an optimiser can do is choose a duration, and equal weight already
picks a reasonable one.

So the engine has to earn its complexity. Four families are run against 1/N and
against each other, in increasing order of what they claim to know:

    naive          1/N, and 1/N rebalanced annually rather than monthly, to
                   separate the allocation from the rebalancing premium
    risk-based     inverse volatility, risk parity, minimum variance - these use
                   the covariance matrix only, never expected returns, and are
                   the fair test of "can structure alone beat equal weight"
    carry-based    yield and roll-down tilts, which are fixed income's native
                   signal and require no statistical forecast
    forecast-based tilts driven by the walk-forward return forecasts, which is
                   the approach Macro_26 found worked best on exactly these
                   assets

Every comparison charges transaction costs and reports leverage honestly. The
short end of the curve has almost no volatility, so any objective that minimises
risk will pile into it - min variance took 100% of the three-month bill in
Macro_26 - and a Sharpe ratio earned at 0.8% volatility is not comparable to one
earned at 8% unless the levering is priced.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from macro.backtest import leverage, metrics, speclog  # noqa: E402
from macro.portfolio import covariance as cov  # noqa: E402

PROCESSED = ROOT / "data/processed"
MIN_TRAIN = 60
DEV_END = "2015-12"

# Per-asset round-trip cost in annualised terms, applied to turnover. Treasuries
# are cheap, high yield and municipals are not; using one number for all of them
# would flatter exactly the strategies that trade the illiquid legs hardest.
COST_BP = {"ust3m": 2, "ust2y": 3, "ust5y": 4, "ust10y": 5, "ust30y": 8,
           "ig_short": 10, "ig": 15, "ig_long": 20, "hy": 40,
           "mbs": 12, "muni": 25, "muni_hy": 45}


def load():
    r = pd.read_parquet(PROCESSED / "fi_returns.parquet")
    rf = pd.read_parquet(PROCESSED / "fi_rf.parquet")["rf"].reindex(r.index)
    return r, rf


# --------------------------------------------------------------- allocators

def equal_weight(train, rf, **kw):
    n = train.shape[1]
    return np.ones(n) / n


def inverse_vol(train, rf, **kw):
    v = train.std().to_numpy()
    w = 1.0 / np.where(v > 1e-9, v, 1e-9)
    return w / w.sum()


def risk_parity(train, rf, **kw):
    """Equal risk contribution, solved by the standard fixed-point iteration."""
    S = cov.ledoit_wolf(train)
    n = S.shape[0]
    w = np.ones(n) / n
    for _ in range(500):
        mrc = S @ w
        rc = w * mrc
        target = rc.sum() / n
        w = np.maximum(w * (target / np.where(rc > 1e-14, rc, 1e-14)) ** 0.5, 1e-9)
        w = w / w.sum()
    return w


def min_variance(train, rf, **kw):
    S = cov.ledoit_wolf(train)
    try:
        inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(S)
    ones = np.ones(S.shape[0])
    w = inv @ ones
    w = np.maximum(w / w.sum(), 0.0)
    return w / w.sum() if w.sum() > 0 else ones / len(ones)


def max_diversification(train, rf, **kw):
    """
    Choquet's diversification ratio: weighted average vol over portfolio vol.

    Included because it is the risk-based rule that most explicitly rewards low
    correlation, which is where this universe's only real structure lives - high
    yield correlates 0.18 with the ten-year and nothing else does.
    """
    S = cov.ledoit_wolf(train)
    v = np.sqrt(np.diag(S))
    try:
        inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(S)
    w = inv @ v
    w = np.maximum(w, 0.0)
    return w / w.sum() if w.sum() > 0 else np.ones(len(v)) / len(v)


def make_tilt(edge_fn, strength=0.5, cap=0.10):
    """
    Equal weight plus a bounded tilt proportional to a standardised edge.

    Bounded rather than optimised, for the reason DeMiguel et al. give: an
    optimiser converts expected-return error into extreme weights, and the edges
    available here are small. The cap is on deviation from 1/N, not on the weight
    itself, so the benchmark is always recoverable by setting strength to zero.
    """
    def alloc(train, rf, edge=None, **kw):
        n = train.shape[1]
        base = np.ones(n) / n
        if edge is None or not np.isfinite(edge).any():
            return base
        e = np.nan_to_num(edge, nan=0.0)
        sd = e.std()
        if sd < 1e-12:
            return base
        z = (e - e.mean()) / sd
        w = base + np.clip(z * strength / n, -cap, cap)
        w = np.maximum(w, 0.0)
        return w / w.sum() if w.sum() > 0 else base
    return alloc


# --------------------------------------------------------------- backtest

def backtest(r, rf, alloc, edges=None, rebalance=1, label=""):
    """
    Walk-forward: fit on everything through t-1, hold through t, pay for trades.

    `rebalance` of 1 is monthly; larger values hold the weight and let it drift,
    which is how the annual-rebalancing variant is produced.
    """
    assets = list(r.columns)
    rates = np.array([COST_BP.get(a, 20) / 1e4 for a in assets])
    nets, held, weights = [], None, []

    for i in range(len(r)):
        if i < MIN_TRAIN:
            nets.append(np.nan)
            weights.append(np.full(len(assets), np.nan))
            continue

        due = (held is None) or (i % rebalance == 0)
        if due:
            train = r.iloc[:i]
            edge = None
            if edges is not None:
                row = edges.reindex([r.index[i]])
                if not row.isna().all(axis=1).iloc[0]:
                    edge = row.iloc[0].reindex(assets).to_numpy()
            target = alloc(train, rf.iloc[:i], edge=edge)
            pre = np.zeros(len(assets)) if held is None else held
            cost = float(np.abs(target - pre) @ rates)
            held = target
        else:
            cost = 0.0

        gross = float(r.iloc[i].to_numpy() @ held)
        nets.append(gross - cost)
        weights.append(held.copy())
        # Weights drift with returns between rebalances.
        grown = held * (1 + r.iloc[i].to_numpy())
        held = grown / grown.sum() if grown.sum() > 0 else held

    return (pd.Series(nets, index=r.index, name=label).dropna(),
            pd.DataFrame(weights, index=r.index, columns=assets).dropna())


def main() -> int:
    r, rf = load()
    dev = r.loc[:DEV_END]
    rf_dev = rf.loc[:DEV_END]
    print(f"Development sample: {dev.index.min():%Y-%m} to {dev.index.max():%Y-%m} "
          f"({len(dev)} months, {dev.shape[1]} assets)")
    print(f"Holdout sealed from {DEV_END} onward, opened in Phase 4.\n")

    strategies = {
        "1/N monthly": (equal_weight, None, 1),
        "1/N annual": (equal_weight, None, 12),
        "Inverse volatility": (inverse_vol, None, 1),
        "Risk parity": (risk_parity, None, 1),
        "Minimum variance": (min_variance, None, 1),
        "Max diversification": (max_diversification, None, 1),
    }

    nets, weights = {}, {}
    for name, (fn, ed, rb) in strategies.items():
        nets[name], weights[name] = backtest(dev, rf_dev, fn, ed, rb, name)
        print(f"  ran {name}")

    tab = metrics.comparison_table(nets, rf_dev)
    tab["vs_1N"] = tab["sharpe"] - tab.loc["1/N monthly", "sharpe"]
    print("\nRISK-BASED ALLOCATORS vs 1/N (development, net of costs)")
    print(tab[["cagr", "vol", "sharpe", "vs_1N", "max_drawdown"]]
          .to_string(float_format=lambda x: f"{x:9.4f}"))

    print("\nThe volatility column is the story. A rule that concentrates in the")
    print("short end posts a high Sharpe on a book nobody would hold.")
    print("\nAt a common 4% risk target with financing charged at 50bp:")
    print(leverage.comparison(nets, rf_dev, target_vol=0.04)[
        ["sharpe_unlevered", "leverage_needed", "sharpe_levered", "cagr_levered"]]
        .to_string(float_format=lambda x: f"{x:9.4f}"))

    print("\nAverage weights:")
    avg = pd.DataFrame({k: v.mean() for k, v in weights.items()})
    print(avg.to_string(float_format=lambda x: f"{x:7.3f}"))

    for name in tab.index:
        speclog.record(speclog.Spec(
            phase="FI-2", family="risk_based", name=name,
            config={"layer": "development", "n_assets": int(dev.shape[1])},
            metrics={k: float(tab.loc[name, k]) for k in
                     ["sharpe", "cagr", "vol", "max_drawdown"]},
            n_periods=int(len(nets[name]))))

    pd.DataFrame(nets).to_parquet(PROCESSED / "fi_riskbased_strategies.parquet")
    tab.to_parquet(PROCESSED / "fi_riskbased_summary.parquet")
    avg.to_parquet(PROCESSED / "fi_riskbased_weights.parquet")
    print("\n" + speclog.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
