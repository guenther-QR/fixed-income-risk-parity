"""The tables that go in the writeup, built from the same return paths.

Covariance is estimated on daily data and performance is measured on monthly.
The two want different things from the sampling frequency. Merton (1980) is the
reason: the precision of a variance estimate improves as the same span is
sampled more finely, while the precision of a mean depends on the length of the
span and not on how often it is observed. So daily observations make the
covariance matrix better and do nothing for expected returns.

Measurement runs the other way. Three of the eleven holdings are priced by
matrix valuation and their daily returns autocorrelate around 0.25, which
distorts a daily Sharpe ratio and the inference built on it. A month of
compounding removes most of that. Portfolios are therefore constructed, traded
and costed daily, and the resulting return series is compounded to monthly
before anything is scored.

The development table and the holdout table have to describe the same
portfolios, so both are scored from one set of daily return series rather than
assembled from separate runs. The two overlays appear at the settings that were
carried forward, not at the settings an earlier sweep happened to report.

Development covers everything tested. The holdout covers only what development
selected, plus the two signals in their standalone form, which are marked
exploratory because neither cleared the significance bar that admitted the
others.
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
from macro.stats import inference as inf  # noqa: E402

TD = import_module("phase2_technical_development")

P = ROOT / "data/processed"
PPY = 12                 # scoring frequency: monthly
BLOCK = 12               # one year of monthly blocks in the bootstrap
DEV_END = "2015-12-31"
BENCH = "Agg index"


def to_monthly(D: pd.DataFrame) -> pd.DataFrame:
    """Compound daily strategy returns into monthly ones."""
    return (1.0 + D).resample("ME").prod() - 1.0

HOLDOUT_SET = {
    "HRP + Rolling 60m overlay": "confirmatory",
    "HRP, annual": "confirmatory",
    "HRP + Momentum (Sharpe) overlay": "confirmatory",
    "ERC, annual": "comparison",
    "Rolling 60m, long only": "exploratory",
    "Momentum (Sharpe), long only": "exploratory",
}
FROM_TECH = ["Rolling 60m (1m target), long only", "Momentum 12-1, long only"]
FROM_ZOO = ["Regression, base", "ML ridge, base", "ML elastic net, base",
            "ML gradient boosting, base", "ML random forest, base"]


def score(D, cols, rf, mask, durn, turns):
    rf2 = rf.reindex(D.index)[mask]
    bench = D[BENCH][mask]
    bx = (bench - rf2).to_numpy()
    bm = metrics.performance(bench, rf2, periods_per_year=PPY)
    rows = []
    for c in cols:
        x = D[c][mask]
        m = metrics.performance(x, rf2, periods_per_year=PPY)
        a, t = TD.nw_ols((x - rf2).to_numpy(), bx[:, None])
        rows.append({
            "strategy": c,
            "return": float((1 + x).prod() ** (PPY / len(x)) - 1),
            "vol": m["vol"], "sharpe": m["sharpe"],
            "vs_agg": np.nan if c == BENCH else m["sharpe"] - bm["sharpe"],
            "p": (np.nan if c == BENCH else
                  inf.sharpe_difference(x, bench, rf=rf2, ppy=PPY,
                                        mean_block=BLOCK)["p_one_sided"]),
            "alpha_pct_yr": a[0] * PPY * 100, "t_alpha": t[0], "beta": a[1],
            "duration": durn.get(c, np.nan), "turnover": turns.get(c, np.nan)})
    return pd.DataFrame(rows).set_index("strategy")


def main() -> int:
    rf = pd.read_parquet(P / "fi_daily_rf.parquet").squeeze()
    rf = (1.0 + rf).resample("ME").prod() - 1.0
    H = pd.read_parquet(P / "fi_holdout_paths.parquet")
    T = pd.read_parquet(P / "fi_technical_paths.parquet")
    HR = pd.read_parquet(P / "fi_holdout_results.parquet")

    durn = dict(HR["duration"].dropna())
    turns = dict(HR["turnover"].dropna())
    td = pd.read_parquet(P / "fi_technical_dur.parquet")["duration"]
    tt = pd.read_parquet(P / "fi_technical_turn.parquet")["turnover"]
    durn.update({k: v for k, v in td.items() if k in FROM_TECH})
    turns.update({k: v for k, v in tt.items() if k in FROM_TECH})

    frames = [H]
    frames.append(T[[c for c in FROM_TECH if c in T.columns]])
    zoo = P / "fi_dmodel_strategies.parquet"
    if zoo.exists():
        Z = pd.read_parquet(zoo)
        frames.append(Z[[c for c in FROM_ZOO if c in Z.columns]])
        S = pd.read_parquet(P / "fi_dmodel_summary.parquet")
        turns.update({c: float(S["turnover"].get(c, np.nan)) for c in FROM_ZOO})
    zd = P / "fi_dmodel_duration.parquet"
    if zd.exists():
        durn.update(pd.read_parquet(zd).mean().to_dict())

    D = pd.concat(frames, axis=1)
    D = D.loc[:, ~D.columns.duplicated()].dropna(how="any")
    print(f"daily paths: {len(D):,} days, {D.shape[1]} series")
    D = to_monthly(D)
    print(f"scored on {len(D)} months, {D.index[0]:%Y-%m} to {D.index[-1]:%Y-%m}")
    dev = D.index <= pd.Timestamp(DEV_END)

    dev_cols = [c for c in D.columns]
    DEV = score(D, dev_cols, rf, dev, durn, turns).sort_values(
        "sharpe", ascending=False)
    DEV.to_parquet(P / "fi_table_development.parquet")
    print(f"=== DEVELOPMENT, {D.index[0]:%Y-%m} to {D.index[dev.sum()-1]:%Y-%m} ===")
    print(DEV.round(4).to_string())

    oos_cols = [c for c in D.columns if c in HOLDOUT_SET or c == BENCH]
    OOS = score(D, oos_cols, rf, ~dev, durn, turns)
    OOS.insert(0, "status", [HOLDOUT_SET.get(c, "benchmark") for c in OOS.index])
    OOS = OOS.sort_values("sharpe", ascending=False)
    OOS.to_parquet(P / "fi_table_holdout.parquet")

    # Full sample, same portfolios, nothing refitted. The settings were fixed
    # on development and the holdout was run forward on them, so concatenating
    # the two periods measures the strategies as they would actually have been
    # held rather than as two separate studies.
    ALL = score(D, oos_cols, rf, pd.Series(True, index=D.index), durn, turns)
    ALL.insert(0, "status", [HOLDOUT_SET.get(c, "benchmark") for c in ALL.index])
    ALL = ALL.sort_values("sharpe", ascending=False)
    ALL.to_parquet(P / "fi_table_fullsample.parquet")
    print(f"\n=== HOLDOUT, {D.index[dev.sum()]:%Y-%m} to {D.index[-1]:%Y-%m}, "
          f"{(~dev).sum()} months ===")
    print(OOS.round(4).to_string())
    print(f"\n=== FULL SAMPLE, {D.index[0]:%Y-%m} to {D.index[-1]:%Y-%m}, "
          f"settings frozen on development ===")
    print(ALL.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
