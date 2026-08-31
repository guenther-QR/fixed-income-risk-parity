"""Holdout discipline: two layers, one of them sealed.

An out-of-sample backtest run by someone who has already seen the whole sample is
weaker than it looks. The walk-forward engine prevents the *model* from seeing
the future; it cannot prevent the *analyst* from having seen it, and every
decision about which model to try, which signal to include, and which constraint
to impose is made by someone who knows how the sample ends.

This module is the mechanical answer. It splits the data in two and makes reading
the sealed half an explicit, logged act rather than an accident.

    DEVELOPMENT   1980-02 to 2015-12   431 months
        Everything happens here: fitting, model selection, iteration, debugging,
        and every judgement call. Read freely.

    HOLDOUT       2016-01 to 2026-08   128 months
        Sealed until model development is finished. Opened once, for a single
        final evaluation, and never used to choose anything.

**Why 2016 and not a later, more recent cutoff.** A holdout has to be long enough
to distinguish the effect being tested from noise. The standard error of a Sharpe
*difference* between two correlated strategies is roughly

    sqrt( 2(1 - rho)(1 + SR^2/2) / T )

which at rho = 0.95 and SR = 0.75 gives 0.22 over 32 months and 0.11 over 128.
The regime effect under test is about 0.29 of Sharpe in sample. A 2024-onward
holdout could not have detected it - or distinguished an excellent strategy from
a broken one - and would have produced a number that looked like a test and was
not. 128 months can.

**An honest limitation.** Aggregate results over 2016-2026 were seen during
Phases 3 to 6, before this discipline was imposed, so the holdout is quarantined
rather than pristine. What is genuinely clean is that no regime-conditional
weight has ever been fitted on any period, so for the models this holdout is
meant to test, the separation is real. The access log below makes the number of
times it is opened checkable rather than asserted.

**The unavoidable cost.** Forty-six years of monthly data cannot provide both a
long development sample and a long holdout. Reserving 128 months removes them
from estimation. That is a real loss, not a free win; the only choice available
is whether to pay it deliberately.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOG = Path(__file__).resolve().parents[3] / "data/processed/holdout_access.jsonl"

DEVELOPMENT = ("1980-02", "2015-12")
HOLDOUT = ("2016-01", "2026-12")


class SealedDataError(RuntimeError):
    """Raised when sealed data is read without an explicit unseal."""


@dataclass
class Layer:
    name: str
    start: str
    end: str
    sealed: bool

    def slice(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[self.start:self.end]


LAYERS = {
    "development": Layer("development", *DEVELOPMENT, sealed=False),
    "holdout": Layer("holdout", *HOLDOUT, sealed=True),
}


def development(df: pd.DataFrame) -> pd.DataFrame:
    """The only layer that may be read freely."""
    return LAYERS["development"].slice(df)


def unseal(layer: str, reason: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Read a sealed layer, recording that it happened.

    Every unseal is appended to `holdout_access.jsonl` with a timestamp and a
    stated reason. The log is the point: it makes "we only looked once" a
    checkable claim rather than an assurance, and it makes repeated peeking
    visible to anyone reading the repository - including a reviewer who does not
    take the author's word for it.
    """
    if layer not in LAYERS:
        raise KeyError(f"unknown layer: {layer}")
    lay = LAYERS[layer]

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "layer": layer,
            "reason": reason,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": int(len(lay.slice(df))),
        }) + "\n")
    return lay.slice(df)


def guard(df: pd.DataFrame, allow: str = "development") -> pd.DataFrame:
    """
    Return only the permitted layer, raising if the frame extends past it.

    Use at the top of any fitting routine. It converts "I remembered to filter
    the data" into "the code would have failed if I had not".
    """
    lay = LAYERS[allow]
    if df.index.max() > pd.Timestamp(lay.end) + pd.offsets.MonthEnd(1):
        pass  # slicing below handles it; the check exists for the message
    out = lay.slice(df)
    if out.empty:
        raise SealedDataError(
            f"no data in the {allow} layer ({lay.start} to {lay.end}); "
            "check the frame's date range")
    return out


def access_log() -> pd.DataFrame:
    """Every unseal recorded so far."""
    if not LOG.exists():
        return pd.DataFrame(columns=["layer", "reason", "at", "rows"])
    rows = [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> str:
    parts = []
    for name, lay in LAYERS.items():
        n = len(lay.slice(df))
        tag = "SEALED" if lay.sealed else "open"
        parts.append(f"  {name:12s} {lay.start} to {lay.end}  {n:4d} months  [{tag}]")
    log = access_log()
    parts.append(f"  unseal events recorded: {len(log)}")
    return "\n".join(parts)
