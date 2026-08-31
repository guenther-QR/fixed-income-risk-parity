"""Transaction costs.

Costs are charged per unit of turnover, per asset, because they differ by an
order of magnitude across this universe: Treasury futures and large-cap equity
trade for a basis point or two, high yield credit for tens of basis points. A
single blended rate would flatter any strategy that trades the cheap assets and
penalise one that trades the expensive ones, which is exactly the distinction a
cost model exists to make.

The defaults are round-trip half-spreads plus market impact for institutional
size, deliberately on the conservative side. Every result should be reported
gross and net, with the break-even cost stated, so a reader can substitute their
own assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# One-way cost in basis points per unit traded.
DEFAULT_BPS = {
    "sp500": 2.0,      # liquid equity index
    "gold": 5.0,       # futures roll plus spread
    "ust3m": 0.5,      # bills
    "ust2y": 1.0,
    "ust5y": 1.5,
    "ust10y": 2.0,
    "ust30y": 3.0,
    "ig": 8.0,         # investment grade credit
    "hy": 25.0,        # high yield, the expensive leg
    "nikkei": 5.0,
    "ftse": 5.0,
    "wti": 8.0,
    "copper": 10.0,
}
FALLBACK_BPS = 10.0


@dataclass
class CostModel:
    bps: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BPS))
    fallback: float = FALLBACK_BPS
    multiplier: float = 1.0      # scale every cost, for sensitivity analysis

    @classmethod
    def free(cls) -> "CostModel":
        """Zero costs - for isolating the gross effect of a change."""
        return cls(bps={}, fallback=0.0)

    @classmethod
    def flat(cls, bps: float) -> "CostModel":
        return cls(bps={}, fallback=bps)

    def rate(self, asset: str) -> float:
        return self.bps.get(asset, self.fallback) * self.multiplier / 10_000.0

    def charge(self, held: np.ndarray, target: np.ndarray,
               total_turnover: float, assets: list[str]) -> float:
        """Cost of moving from `held` to `target`, as a fraction of portfolio value."""
        traded = np.abs(np.asarray(target) - np.asarray(held))
        rates = np.array([self.rate(a) for a in assets])
        return float(traded @ rates)

    def average_rate(self, assets: list[str]) -> float:
        return float(np.mean([self.rate(a) for a in assets]))
