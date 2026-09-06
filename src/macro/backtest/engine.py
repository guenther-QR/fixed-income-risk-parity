"""Walk-forward backtest engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .costs import CostModel
from .rebalance import RebalanceSchedule

WeightFn = Callable[[pd.DataFrame, pd.Series | None], np.ndarray]


@dataclass
class BacktestConfig:
    min_obs: int = 60                    # shortest usable estimation window
    window: int | None = None            # None = expanding, int = rolling
    lag: int = 1                         # periods between signal and execution
    periods_per_year: int = 12
    allow_cash: bool = False

    def train_slice(self, returns: pd.DataFrame, i: int) -> pd.DataFrame:
        """The estimation window for a decision made at position `i`."""
        end = i - self.lag + 1
        start = 0 if self.window is None else max(0, end - self.window)
        return returns.iloc[start:end]


@dataclass
class BacktestResult:
    weights: pd.DataFrame                # target weights at each rebalance
    held: pd.DataFrame                   # weights actually held each period
    gross: pd.Series
    net: pd.Series
    costs: pd.Series
    turnover: pd.Series
    cash: pd.Series
    name: str = ""
    meta: dict = field(default_factory=dict)

    def summary(self, rf: pd.Series | None = None) -> dict:
        from .metrics import performance
        return {"name": self.name,
                **performance(self.net, rf, turnover=self.turnover)}


def _drift(weights: np.ndarray, period_returns: np.ndarray) -> np.ndarray:
    """Weights after one period of returns, before any rebalancing."""
    cash = 1.0 - weights.sum()
    grown = weights * (1.0 + period_returns)
    total = grown.sum() + cash
    return grown / total if total > 0 else weights


def run(returns: pd.DataFrame, weight_fn: WeightFn,
        schedule: RebalanceSchedule, config: BacktestConfig,
        costs: CostModel | None = None, rf: pd.Series | None = None,
        name: str = "") -> BacktestResult:
    """Run one strategy through the walk-forward loop."""
    costs = costs or CostModel.free()
    assets = list(returns.columns)
    n = len(assets)
    idx = returns.index

    held = np.full(n, np.nan)
    rows_w, rows_h, dates_w = [], [], []
    gross, net, cost_series, turnover, cash_series = [], [], [], [], []

    for i in range(config.min_obs + config.lag, len(idx)):
        date = idx[i]
        train = config.train_slice(returns, i)
        first = bool(np.isnan(held).any())

        # `may_trade` is side-effect free, so periods that cannot trade skip
        # the optimizer entirely.
        if first or schedule.may_trade(date, i):
            if len(train) < config.min_obs:
                continue
            rf_train = rf.reindex(train.index) if rf is not None else None
            proposed = np.asarray(weight_fn(train, rf_train), dtype=float)
            rebalance_now = first or schedule.is_rebalance(date, i, held, proposed)
            target = proposed if rebalance_now else held
        else:
            rebalance_now = False
            target = held

        # Cost must be priced against the weights held *before* trading. Updating
        # `held` first makes `target - held` identically zero, which silently
        # zeroes every transaction cost while turnover still looks correct.
        pre_trade = np.zeros(n) if first else held
        if first or rebalance_now:
            traded = float(np.abs(target - pre_trade).sum())
            held = target
        else:
            traded = 0.0

        period = returns.iloc[i].to_numpy(dtype=float)
        period_cost = costs.charge(pre_trade, target, traded, assets) if traded > 0 else 0.0

        g = float(held @ np.nan_to_num(period))
        gross.append(g)
        cost_series.append(period_cost)
        net.append(g - period_cost)
        turnover.append(traded)
        cash_series.append(1.0 - held.sum())

        rows_h.append(held.copy())
        if rebalance_now:
            rows_w.append(target.copy())
            dates_w.append(date)

        held = _drift(held, np.nan_to_num(period))

    dates = idx[config.min_obs + config.lag: config.min_obs + config.lag + len(gross)]
    return BacktestResult(
        weights=pd.DataFrame(rows_w, index=pd.DatetimeIndex(dates_w, name="date"),
                             columns=assets),
        held=pd.DataFrame(rows_h, index=dates, columns=assets),
        gross=pd.Series(gross, index=dates, name="gross"),
        net=pd.Series(net, index=dates, name="net"),
        costs=pd.Series(cost_series, index=dates, name="cost"),
        turnover=pd.Series(turnover, index=dates, name="turnover"),
        cash=pd.Series(cash_series, index=dates, name="cash"),
        name=name,
        meta={"min_obs": config.min_obs, "window": config.window,
              "lag": config.lag, "schedule": schedule.describe()},
    )


def break_even_cost(result_gross: pd.Series, benchmark: pd.Series,
                    turnover: pd.Series, periods_per_year: int = 12) -> float:
    """
    The per-unit-turnover cost at which a strategy's gross edge disappears.
    """
    edge = (result_gross - benchmark.reindex(result_gross.index)).mean() * periods_per_year
    traded = turnover.mean() * periods_per_year
    return float(edge / traded) if traded > 0 else np.inf
