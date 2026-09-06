"""Forecastability against risk, one observation per holding."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from macro.backtest import technical as TQ  # noqa: E402

P = ROOT / "data/processed"
DEV_END = "2015-12-31"


def main() -> int:
    skill = pd.read_parquet(P / "fi_uni_regression_skill.parquet")
    rd = pd.read_parquet(P / "fi_daily_returns.parquet")
    assets = [c for c in rd.columns if c != "ust3m"]
    rm = (1 + rd[assets]).resample("ME").prod() - 1
    rm = rm[rm.index <= pd.Timestamp(DEV_END)]

    vol = rm.std() * np.sqrt(12)
    var = vol ** 2
    out = pd.DataFrame({
        "oos_r2": skill["dev_r2"].reindex(assets),
        "vol": vol,
        "duration": pd.Series(TQ.DUR).reindex(assets),
        "risk_share": var / var.sum(),
    }).sort_values("oos_r2", ascending=False)
    out.to_parquet(P / "fi_predictability_risk.parquet")

    rho = out["oos_r2"].corr(out["duration"], method="spearman")
    print(f"{len(out)} holdings, development sample, monthly\n")
    print((out * [100, 100, 1, 100]).round(2).to_string())
    print(f"\nrank correlation of forecastability with duration: {rho:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
