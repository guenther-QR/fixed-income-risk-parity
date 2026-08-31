"""Gurkaynak-Sack-Wright (FEDS 2006-28) Treasury yield curve.

The Federal Reserve staff publish daily Nelson-Siegel-Svensson parameters and the
fitted zero-coupon, par and forward curves back to 1961. This project does not
use GSW as its curve - Phase 2 bootstraps its own from published constant-maturity
par yields - but GSW is the reference the bootstrap is validated against, and its
NSS parameters are the reference for our own NSS fit.

Source: https://www.federalreserve.gov/data/nominal-yield-curve.htm
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
CACHE = Path(__file__).resolve().parents[3] / "data/raw/fed/feds200628.csv"

NSS_PARAMS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]


def _raw(refresh: bool = False) -> pd.DataFrame:
    if refresh or not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            CACHE.write_bytes(resp.read())

    text = CACHE.read_text(encoding="utf-8", errors="ignore")
    header = next(i for i, line in enumerate(text.splitlines())
                  if line.startswith("Date,BETA0"))
    df = pd.read_csv(io.StringIO(text), skiprows=header, na_values=["NA"],
                     low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def zero_yields(refresh: bool = False) -> pd.DataFrame:
    """Fitted continuously-compounded zero yields, columns = maturity in years."""
    df = _raw(refresh)
    cols = {c: int(c[5:]) for c in df.columns if c.startswith("SVENY")}
    out = df[list(cols)].rename(columns=cols)
    out.index.name = "date"
    return out.sort_index(axis=1)


def par_yields(refresh: bool = False) -> pd.DataFrame:
    """Fitted coupon-equivalent par yields, columns = maturity in years."""
    df = _raw(refresh)
    cols = {c: int(c[6:]) for c in df.columns if c.startswith("SVENPY")}
    out = df[list(cols)].rename(columns=cols)
    out.index.name = "date"
    return out.sort_index(axis=1)


def nss_parameters(refresh: bool = False) -> pd.DataFrame:
    """The six published Svensson parameters, for validating our own fit."""
    out = _raw(refresh)[NSS_PARAMS].dropna(how="all")
    out.index.name = "date"
    return out
