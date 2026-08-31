"""Episodic regime learning: build a regime's portfolio only once you have seen it.

Phase 6b's regime allocation failed, and the placebo test showed why - shuffled
labels beat real ones. But the implementation was doing something subtly wrong.
It fitted each regime's weights on *all* prior months of that regime the moment a
24-month threshold was crossed, which meant most of its early decisions rested on
a partial, in-progress episode.

This is the cleaner rule:

    A regime's portfolio may only be built from episodes that have **finished**.

    First time a regime appears - no completed episodes, so nothing has been
    learned about it. Hold the benchmark: 60/40.

    Second time - one completed episode. Fit the weights on it.

    Third time - two completed episodes. Refit on both. And so on.

The distinction matters because a regime in progress is a regime whose outcome is
unknown. Fitting on it is a soft form of look-ahead: you are estimating the
weights for a period partly from that period. Requiring completion removes it,
at the cost of learning more slowly - which is the correct trade, since the
alternative is learning things that are not true.

The cold-start default is 60/40 rather than an unconditional optimizer. That is
the honest counterfactual: it is what the portfolio would hold if it had never
heard of regimes, and it is the benchmark the whole project is measured against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import covariance as cov, optimize as opt

REGIMES = ["Goldilocks", "Reflation", "Stagflation", "Deflation"]


def episode_table(regimes: pd.Series) -> pd.DataFrame:
    """
    Every contiguous run of a regime, with its start and end.

    An episode is 'completed' at date t if it ended strictly before t. The run
    currently in progress is deliberately excluded from every fit.
    """
    s = regimes.dropna()
    if s.empty:
        return pd.DataFrame(columns=["regime", "start", "end", "months"])

    grp = (s != s.shift()).cumsum()
    rows = []
    for _, block in s.groupby(grp):
        rows.append({"regime": block.iloc[0], "start": block.index[0],
                     "end": block.index[-1], "months": len(block)})
    return pd.DataFrame(rows)


def completed_months(episodes: pd.DataFrame, regime: str,
                     asof: pd.Timestamp) -> list[pd.Timestamp]:
    """Index of every month belonging to a *finished* episode of `regime`."""
    done = episodes[(episodes["regime"] == regime) & (episodes["end"] < asof)]
    out = []
    for _, ep in done.iterrows():
        out.append(pd.date_range(ep["start"], ep["end"], freq="ME"))
    return list(pd.DatetimeIndex(np.concatenate(out))) if out else []


def n_completed_episodes(episodes: pd.DataFrame, regime: str,
                         asof: pd.Timestamp) -> int:
    return int(((episodes["regime"] == regime) & (episodes["end"] < asof)).sum())


@dataclass
class EpisodicRegime:
    """
    Hold the benchmark until a regime has completed once; then use what it taught.

    Each regime unlocks **independently**. Goldilocks finishing its first episode
    does not entitle the strategy to a Stagflation portfolio - that one stays on
    the benchmark until Stagflation itself has completed. The strategy therefore
    starts entirely on 60/40 and earns its way off it one regime at a time.

    `shade_rate` handles the other problem, which is that regimes are short. The
    median episode here runs four months and 41% run two or fewer, so with a
    one-month signal lag, committing fully on detection means committing to a
    regime that has often already ended. Instead the weight on the regime
    portfolio approaches 1 geometrically with each month it persists:

        alpha_k = 1 - (1 - shade_rate)^k     k = consecutive months in the regime

    At the default 0.5 that is 50% in the first month, 75% in the second, 88% in
    the third. A regime that ends immediately costs half a month of
    mispositioning rather than a full commitment; one that persists is fully
    expressed within a quarter. `shade_rate = 1.0` recovers hard switching.
    """
    regimes: pd.Series
    assets: list[str]
    benchmark: dict[str, float]
    objective: str = "risk_parity"
    min_episodes: int = 1
    min_months: int = 12
    shade_rate: float = 0.5
    log: list = field(default_factory=list)

    def _benchmark_vector(self) -> np.ndarray:
        return np.array([self.benchmark.get(a, 0.0) for a in self.assets])

    def __call__(self, train: pd.DataFrame, rf: pd.Series | None = None) -> np.ndarray:
        reg = self.regimes.reindex(train.index).dropna()
        if reg.empty:
            return self._benchmark_vector()

        asof = train.index[-1]
        state = reg.iloc[-1]
        episodes = episode_table(reg)

        n_done = n_completed_episodes(episodes, state, asof)
        if n_done < self.min_episodes:
            self.log.append({"date": asof, "regime": state, "episodes": n_done,
                             "action": "benchmark", "alpha": 0.0})
            return self._benchmark_vector()

        months = [m for m in completed_months(episodes, state, asof)
                  if m in train.index]
        if len(months) < self.min_months:
            self.log.append({"date": asof, "regime": state, "episodes": n_done,
                             "action": "benchmark_thin", "alpha": 0.0})
            return self._benchmark_vector()

        sub = train.loc[months]
        subrf = rf.reindex(sub.index) if rf is not None else None
        try:
            m = opt.estimate_moments(sub, subrf)
            m.sigma = cov.ledoit_wolf(
                sub.sub(subrf, axis=0).dropna() if subrf is not None else sub)
            w = (opt.risk_parity(m) if self.objective == "risk_parity"
                 else opt.max_sharpe(m) if self.objective == "max_sharpe"
                 else opt.min_variance(m))
        except Exception:
            self.log.append({"date": asof, "regime": state, "episodes": n_done,
                             "action": "benchmark_failed", "alpha": 0.0})
            return self._benchmark_vector()

        # Shade in over consecutive months rather than committing on detection.
        k = _run_length(reg, state)
        alpha = 1.0 - (1.0 - self.shade_rate) ** k if self.shade_rate < 1.0 else 1.0
        blended = alpha * w + (1.0 - alpha) * self._benchmark_vector()

        self.log.append({"date": asof, "regime": state, "episodes": n_done,
                         "action": "regime", "months_used": len(months),
                         "run_length": k, "alpha": alpha})
        return blended

    def activity(self) -> pd.DataFrame:
        """When the strategy used a learned portfolio and when it fell back."""
        return pd.DataFrame(self.log)


def _run_length(regimes: pd.Series, state: str) -> int:
    """Consecutive months the current regime has run, counting back from the end."""
    k = 0
    for v in reversed(regimes.dropna().to_numpy()):
        if v == state:
            k += 1
        else:
            break
    return k


def learning_curve(regimes: pd.Series, start: pd.Timestamp) -> pd.DataFrame:
    """
    How long before each regime has enough completed episodes to be usable.

    Reported because it bounds what the strategy can possibly do: a regime that
    completes its first episode ten years into the sample contributes nothing
    before then, whatever its portfolio would have been.
    """
    eps = episode_table(regimes)
    rows = []
    for r in REGIMES:
        sub = eps[eps["regime"] == r].sort_values("end")
        rows.append({
            "regime": r,
            "episodes_total": len(sub),
            "first_completed": sub["end"].iloc[0] if len(sub) else pd.NaT,
            "second_completed": sub["end"].iloc[1] if len(sub) > 1 else pd.NaT,
            "median_months": sub["months"].median() if len(sub) else np.nan,
        })
    return pd.DataFrame(rows).set_index("regime")
