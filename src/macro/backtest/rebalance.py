"""Rebalance schedules.

Calendar rebalancing is the obvious default and usually not the best one. A
portfolio that trades every month pays for a great deal of drift that would have
reversed on its own; one that trades only when it has drifted materially pays
less and often tracks its target just as closely. Comparing the two is the point
of having several schedules behind one interface.

The interface is deliberately split in two, because schedules fall into two
kinds and conflating them corrupts their state:

    may_trade(date, i)                   cheap, side-effect free. Answers "is
                                         this period eligible at all?" so the
                                         engine can skip the optimizer entirely
                                         on periods that cannot trade.

    is_rebalance(date, i, held, target)  the decision, called at most once per
                                         period and only after `may_trade`
                                         allowed it. May update internal state.

An earlier version had a single method that the engine called twice - once
before the proposed weights existed and once after. Stateful schedules recorded
the first call and then declined the second, so monthly rebalancing fired once
in 239 periods.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class RebalanceSchedule:
    """Decides whether a given period is a rebalance."""

    def may_trade(self, date: pd.Timestamp, i: int) -> bool:
        """Cheap eligibility check. Must not mutate state."""
        return True

    def is_rebalance(self, date: pd.Timestamp, i: int,
                     held: np.ndarray, target: np.ndarray) -> bool:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


@dataclass
class Periodic(RebalanceSchedule):
    """Rebalance every `every` periods, counted from the start of the backtest."""
    every: int = 1
    label: str = ""

    def may_trade(self, date, i) -> bool:
        return i % self.every == 0

    def is_rebalance(self, date, i, held, target) -> bool:
        return i % self.every == 0

    def describe(self) -> str:
        return self.label or f"every {self.every} periods"


@dataclass
class CalendarMonthly(RebalanceSchedule):
    """Rebalance on the first observation of each calendar month or quarter."""
    months: int = 1
    _last: tuple | None = field(default=None, repr=False)

    def _key(self, date: pd.Timestamp) -> tuple:
        return (date.year, (date.month - 1) // self.months)

    def may_trade(self, date, i) -> bool:
        return self._key(date) != self._last

    def is_rebalance(self, date, i, held, target) -> bool:
        key = self._key(date)
        if key != self._last:
            self._last = key
            return True
        return False

    def describe(self) -> str:
        return "monthly" if self.months == 1 else f"every {self.months} months"


@dataclass
class Threshold(RebalanceSchedule):
    """
    No-trade band: rebalance only when the portfolio has drifted past `band`.

    Drift is total absolute deviation between the weights held and the weights
    the objective now wants. Below the band the portfolio is left alone, which is
    what makes this cheaper than calendar rebalancing rather than merely less
    frequent: it skips the trades that would have reversed anyway.
    """
    band: float = 0.05
    min_gap: int = 1
    _last_trade: int = field(default=-10**9, repr=False)

    def may_trade(self, date, i) -> bool:
        return i - self._last_trade >= self.min_gap

    def is_rebalance(self, date, i, held, target) -> bool:
        if i - self._last_trade < self.min_gap:
            return False
        if float(np.abs(target - held).sum()) >= self.band:
            self._last_trade = i
            return True
        return False

    def describe(self) -> str:
        return f"no-trade band {self.band:.0%}"


def standard_schedules(periods_per_year: int = 12) -> dict[str, RebalanceSchedule]:
    """The set compared in Phase 4. Fresh instances - these carry state."""
    if periods_per_year == 12:
        return {
            "monthly": CalendarMonthly(1),
            "quarterly": CalendarMonthly(3),
            "annual": CalendarMonthly(12),
            "band_5pct": Threshold(0.05),
            "band_10pct": Threshold(0.10),
            "band_20pct": Threshold(0.20),
        }
    return {
        "weekly": Periodic(5, "weekly"),
        "monthly": CalendarMonthly(1),
        "quarterly": CalendarMonthly(3),
        "band_5pct": Threshold(0.05, min_gap=5),
        "band_10pct": Threshold(0.10, min_gap=5),
        "band_20pct": Threshold(0.20, min_gap=5),
    }
