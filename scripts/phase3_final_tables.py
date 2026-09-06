"""The tables that go in the writeup, built from the same return paths."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from macro.backtest import metrics  # noqa: E402
from macro.backtest import technical as TQ  # noqa: E402
from macro.stats import inference as inf  # noqa: E402

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
        a, t = TQ.nw_ols((x - rf2).to_numpy(), bx[:, None])
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
    HR = pd.read_parquet(P / "fi_holdout_results.parquet")

    durn = dict(HR["duration"].dropna())
    turns = dict(HR["turnover"].dropna())
    frames = [H]
    zoo = P / "fi_dmodel_strategies.parquet"
    if zoo.exists():
        Z = pd.read_parquet(zoo)
        frames.append(Z[[c for c in FROM_ZOO if c in Z.columns]])
        S = pd.read_parquet(P / "fi_dmodel_summary.parquet")
        turns.update({c: float(S["turnover"].get(c, np.nan)) for c in FROM_ZOO})
    # No duration for the regression and machine learning books.

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

    # Full sample, same portfolios, nothing refitted.
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
