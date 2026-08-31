"""Specification log: a record of every configuration actually tried.

A multiple-testing correction needs to know how many things were tested. White's
Reality Check and Hansen's SPA both compare the best observed result against the
distribution of the best result under the null - and that distribution depends
entirely on how wide the search was. Correcting for the specifications you
remember trying, rather than the ones you actually tried, produces a p-value that
looks rigorous and is not.

So every run appends here automatically, whether or not it looked promising, and
Phase 7 reads the file rather than a list reconstructed from memory. The
discipline only works if nothing is exempt: a specification abandoned because it
performed badly is exactly the kind the correction exists to account for.

Entries are append-only JSON lines, one per run, keyed by a hash of the
configuration so repeated identical runs collapse to one entry.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data/processed/specification_log.jsonl"


@dataclass
class Spec:
    """One tested configuration and what it produced."""
    phase: str
    family: str                      # e.g. "fixed_weight", "mean_variance", "ml_tilt"
    name: str
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    oos_start: str | None = None
    oos_end: str | None = None
    n_periods: int | None = None
    note: str = ""

    def fingerprint(self) -> str:
        payload = json.dumps(
            {"phase": self.phase, "family": self.family, "name": self.name,
             "config": self.config},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def record(spec: Spec, path: Path | None = None) -> str:
    """Append one specification. Returns its fingerprint."""
    path = path or DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    fp = spec.fingerprint()
    if fp in _fingerprints(path):
        return fp

    entry = asdict(spec) | {
        "fingerprint": fp,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return fp


def _fingerprints(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.add(json.loads(line)["fingerprint"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def load(path: Path | None = None) -> pd.DataFrame:
    """Every recorded specification, flattened for analysis."""
    path = path or DEFAULT_PATH
    if not path.exists():
        return pd.DataFrame()

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        flat = {k: v for k, v in e.items() if k not in ("config", "metrics")}
        flat |= {f"cfg_{k}": v for k, v in (e.get("config") or {}).items()}
        flat |= {k: v for k, v in (e.get("metrics") or {}).items()}
        rows.append(flat)
    return pd.DataFrame(rows)


def count_by_phase(path: Path | None = None) -> pd.Series:
    df = load(path)
    return df.groupby("phase").size() if len(df) else pd.Series(dtype=int)


def summary(path: Path | None = None) -> str:
    df = load(path)
    if df.empty:
        return "specification log is empty"
    lines = [f"{len(df)} specifications recorded"]
    for phase, n in df.groupby("phase").size().items():
        fams = df[df.phase == phase]["family"].nunique()
        lines.append(f"  {phase}: {n} specs across {fams} families")
    return "\n".join(lines)
