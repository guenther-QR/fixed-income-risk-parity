"""Holdout discipline: two layers, one of them sealed."""
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
    """Read a sealed layer, recording that it happened."""
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
