"""Fetch daily price history from free, keyless sources and cache it as CSV.

Sources:
  - CoinMetrics Community API: crypto price + MVRV (no key)
  - Yahoo Finance chart endpoint: gold / silver futures (no key, unofficial)
  - FRED fredgraph CSV: macro series (no key)
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0"}  # Yahoo needs a browser UA; FRED rejects one
CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"


def _get(url: str, params: dict | None = None, tries: int = 4, timeout: int = 90,
         headers: dict | None = None) -> requests.Response:
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429:
                time.sleep(5 * (i + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed: {url}")


def coinmetrics(asset: str, metrics: list[str]) -> pd.DataFrame:
    params = {
        "assets": asset,
        "metrics": ",".join(metrics),
        "frequency": "1d",
        "page_size": 10000,
        "paging_from": "start",
    }
    rows, url = [], CM_URL
    while url:
        data = _get(url, params).json()
        rows.extend(data["data"])
        url, params = data.get("next_page_url"), None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("date")[metrics].astype(float)
    return df.rename(columns={"PriceUSD": "price", "CapMVRVCur": "mvrv"})


def yahoo(symbol: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    # range=max silently downsamples to monthly; explicit bounds keep daily bars
    params = {"period1": 0, "period2": int(time.time()), "interval": "1d"}
    res = _get(url, params, headers=UA).json()["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
    close = res["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"price": close}, index=idx).dropna()
    df.index.name = "date"
    return df[~df.index.duplicated(keep="last")]


def fred(series: str) -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    df = pd.read_csv(io.StringIO(_get(url, {"id": series}).text))
    df.columns = ["date", series]
    df["date"] = pd.to_datetime(df["date"])
    df[series] = pd.to_numeric(df[series], errors="coerce")
    return df.set_index("date").dropna()


SOURCES = {
    "btc": lambda: coinmetrics("btc", ["PriceUSD", "CapMVRVCur"]),
    "eth": lambda: coinmetrics("eth", ["PriceUSD", "CapMVRVCur"]),
    "ada": lambda: coinmetrics("ada", ["PriceUSD"]),
    "gold": lambda: yahoo("GC=F"),
    "silver": lambda: yahoo("SI=F"),
    "real_yield": lambda: fred("DFII10"),
    "m2": lambda: fred("M2SL"),
    "tbill": lambda: fred("DTB3"),
}


def load(name: str, refresh: bool = True) -> pd.DataFrame:
    """Return cached data, refreshing from the source when possible.

    If the source fails, fall back to the cache so one flaky API
    doesn't take down the whole dashboard.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.csv"
    if refresh:
        try:
            df = SOURCES[name]()
            df.to_csv(path)
            return df
        except Exception as e:  # noqa: BLE001
            if not path.exists():
                raise
            print(f"[warn] {name}: refresh failed ({e}); using cache")
    return pd.read_csv(path, index_col=0, parse_dates=True)


if __name__ == "__main__":
    for n in SOURCES:
        d = load(n)
        print(f"{n:11s} {d.index.min().date()} -> {d.index.max().date()}  rows={len(d)}")
