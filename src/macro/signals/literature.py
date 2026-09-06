"""Predictors from the equity-premium prediction literature."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.fred import get_series
from ..data.yahoo import get_prices

SHILLER = Path(__file__).resolve().parents[3] / "data/external/shiller_ie_data.xls"
SHILLER_URL = ("https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-"
               "4763ac982e53/downloads/ie_data.xls")

# Neely-Rapach-Tu-Zhou moving-average and momentum windows.
MA_PAIRS = [(1, 9), (1, 12), (2, 9), (2, 12), (3, 9), (3, 12)]
MOM_WINDOWS = [9, 12]


def shiller_data() -> pd.DataFrame:
    """
    Shiller's monthly S&P composite series: price, dividend, earnings, CAPE.
    """
    if not SHILLER.exists():
        raise FileNotFoundError(
            f"{SHILLER} missing. Download from {SHILLER_URL}")

    raw = pd.read_excel(SHILLER, sheet_name="Data", header=None)
    body = raw.iloc[8:].copy()
    cols = {0: "date", 1: "P", 2: "D", 3: "E", 4: "CPI", 6: "GS10", 12: "CAPE"}
    body = body[list(cols)].rename(columns=cols)

    d = body["date"].astype(str).str.strip()
    ok = d.str.match(r"^\d{4}\.\d{1,2}$")
    body, d = body[ok], d[ok]

    year = d.str.split(".").str[0].astype(int)
    month = d.str.split(".").str[1].str.ljust(2, "0").astype(int)
    idx = pd.to_datetime(dict(year=year, month=month, day=1)) + pd.offsets.MonthEnd(0)

    out = body.drop(columns="date").apply(pd.to_numeric, errors="coerce")
    out.index = idx
    return out.dropna(subset=["P"])


def goyal_welch(index: pd.DatetimeIndex) -> pd.DataFrame:
    """The Goyal-Welch valuation and issuance block."""
    s = shiller_data()
    out = pd.DataFrame(index=s.index)

    out["gw_dp"] = np.log(s["D"]) - np.log(s["P"])
    out["gw_dy"] = np.log(s["D"]) - np.log(s["P"].shift(1))
    out["gw_ep"] = np.log(s["E"]) - np.log(s["P"])
    out["gw_de"] = np.log(s["D"]) - np.log(s["E"])
    out["gw_cape"] = s["CAPE"]
    out["gw_cape_inv"] = 1.0 / s["CAPE"].replace(0, np.nan)

    # Maio (2013): the "Fed model" gap between the equity earnings yield and
    # the ten-year Treasury yield.
    out["gw_ygap"] = (s["E"] / s["P"]) - s["GS10"] / 100.0

    # Each ratio's deviation from its own trailing decade, which is how a
    # practitioner would read it - the level is dominated by a slow trend that
    # no allocator could have traded on.
    for c in ["gw_dp", "gw_ep", "gw_cape"]:
        roll = out[c].rolling(120, min_periods=60)
        out[f"{c}_dev"] = out[c] - roll.mean()
        out[f"{c}_z"] = (out[c] - roll.mean()) / roll.std()

    # Published one month after the fact, as with any accounting quantity.
    return out.shift(1).reindex(index)


def technical_indicators(index: pd.DatetimeIndex) -> pd.DataFrame:
    """The fourteen Neely-Rapach-Tu-Zhou indicators."""
    px = get_prices(["^GSPC"])["^GSPC"]
    m = px.resample("ME").last()
    out = pd.DataFrame(index=m.index)

    for s, l in MA_PAIRS:
        short = m.rolling(s, min_periods=s).mean()
        long = m.rolling(l, min_periods=l).mean()
        sig = (short >= long).astype(float)
        sig[short.isna() | long.isna()] = np.nan
        out[f"tchi_ma{s}_{l}"] = sig

    for k in MOM_WINDOWS:
        sig = (m >= m.shift(k)).astype(float)
        sig[m.shift(k).isna()] = np.nan
        out[f"tchi_mom{k}"] = sig

    vol = _index_volume(m.index)
    if vol is not None:
        ret_sign = np.sign(m.pct_change()).fillna(0.0)
        obv = (ret_sign * vol).cumsum()
        for s, l in MA_PAIRS:
            short = obv.rolling(s, min_periods=s).mean()
            long = obv.rolling(l, min_periods=l).mean()
            sig = (short >= long).astype(float)
            sig[short.isna() | long.isna()] = np.nan
            out[f"tchi_vol{s}_{l}"] = sig

    # The paper's own aggregate: the average of the individual states, which
    # is the fraction of rules currently long.
    cols = [c for c in out.columns if c.startswith("tchi_")]
    out["tchi_mean"] = out[cols].mean(axis=1, skipna=True)

    return out.shift(1).reindex(index)


def _index_volume(index: pd.DatetimeIndex) -> pd.Series | None:
    """Monthly S&P share volume, or None if Yahoo does not supply it."""
    try:
        import yfinance as yf
        raw = yf.download("^GSPC", start="1950-01-01", progress=False,
                          auto_adjust=True)["Volume"]
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
        v = raw.resample("ME").sum().reindex(index)
        return v.where(v > 0)
    except Exception:
        return None


def output_gap(index: pd.DatetimeIndex, min_obs: int = 120) -> pd.DataFrame:
    """
    Cooper and Priestley's output gap: log industrial production against a
    recursively fitted quadratic trend.
    """
    ip = np.log(get_series("INDPRO").resample("ME").last())
    gap = pd.Series(index=ip.index, dtype=float)

    for i in range(min_obs, len(ip)):
        y = ip.iloc[: i + 1].dropna()
        if len(y) < min_obs:
            continue
        t = np.arange(len(y), dtype=float)
        X = np.column_stack([np.ones(len(y)), t, t ** 2])
        beta, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
        gap.iloc[i] = float(y.iloc[-1] - X[-1] @ beta)

    out = pd.DataFrame({"ogap": gap})
    out["ogap_chg"] = gap.diff()
    return out.shift(1).reindex(index)


def investment_capital(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cochrane's investment-to-capital ratio, approximated."""
    try:
        inv = get_series("PNFI").resample("ME").last().ffill()
        gdp = get_series("GDPC1").resample("ME").last().ffill()
    except Exception:
        return pd.DataFrame(index=index)
    ik = (inv / gdp).reindex(index)
    out = pd.DataFrame({"ik_proxy": ik})
    out["ik_proxy_dev"] = ik - ik.rolling(120, min_periods=60).mean()
    return out.shift(1)


def stock_variance(returns: pd.DataFrame, processed: Path) -> pd.DataFrame:
    """
    Goyal-Welch stock variance: the sum of squared daily returns in the
    month.
    """
    out = pd.DataFrame(index=returns.index)
    daily = processed / "panel_C_daily.parquet"
    if not daily.exists():
        return out
    d = pd.read_parquet(daily)
    col = next((c for c in d.columns if "sp500" in c.lower()), None)
    if col is None:
        return out
    sv = (d[col] ** 2).resample("ME").sum()
    out["gw_svar"] = sv.reindex(returns.index)
    out["gw_svar_log"] = np.log(out["gw_svar"].replace(0, np.nan))
    return out.shift(1)


def return_spreads(returns: pd.DataFrame) -> pd.DataFrame:
    """Goyal-Welch long-term return and default return spread."""
    out = pd.DataFrame(index=returns.index)
    if "ust30y" in returns:
        out["gw_ltr"] = returns["ust30y"]
    elif "ust10y" in returns:
        out["gw_ltr"] = returns["ust10y"]
    if "ig" in returns and "gw_ltr" in out:
        out["gw_dfr"] = returns["ig"] - out["gw_ltr"]
    return out.shift(1)


def build(returns: pd.DataFrame, processed: Path) -> pd.DataFrame:
    """Every literature predictor this project can construct, in one frame."""
    idx = returns.index
    blocks = [
        goyal_welch(idx),
        technical_indicators(idx),
        output_gap(idx),
        investment_capital(idx),
        stock_variance(returns, processed),
        return_spreads(returns),
    ]
    out = pd.concat([b for b in blocks if not b.empty], axis=1)
    return out.loc[:, ~out.columns.duplicated()].reindex(idx)


UNAVAILABLE = {
    "ntis": "net equity expansion (Goyal-Welch) - needs CRSP NYSE market cap",
    "eqis": "percent equity issuing (Baker-Wurgler) - needs CRSP/Compustat",
    "bm": "book-to-market (Goyal-Welch) - needs Value Line / Compustat book values",
    "shtint": "short interest (Rapach-Ringgenberg-Zhou 2016) - needs Compustat",
    "csp": "cross-sectional beta premium (Polk-Thompson-Vuolteenaho) - ends 2002",
    "cay": "consumption-wealth ratio (Lettau-Ludvigson) - author-maintained series",
    "accrual": "aggregate accruals (Hirshleifer-Hou-Teoh) - needs Compustat",
    "sntm": "aligned investor sentiment (Huang et al) - author-maintained series",
}
