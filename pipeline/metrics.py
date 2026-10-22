"""Risk metrics. Every metric is computed point-in-time (no lookahead) and
normalized to 0.0 (historically cheap) .. 1.0 (historically expensive)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---------- normalization ----------

HALFLIFE_DAYS = 4 * 365  # ~one BTC cycle; older history fades so diminishing returns are respected


def ewm_score(x: pd.Series, halflife: int = HALFLIFE_DAYS, min_periods: int = 730) -> pd.Series:
    """Z-score vs exponentially weighted history (strictly past values), mapped to 0..1 via normal CDF."""
    past = x.shift(1)
    mean = past.ewm(halflife=halflife, min_periods=min_periods).mean()
    std = past.ewm(halflife=halflife, min_periods=min_periods).std()
    z = (x - mean) / std
    return z.apply(lambda v: 0.5 * (1 + math.erf(v / math.sqrt(2))) if pd.notna(v) else np.nan)


# ---------- raw metric building blocks ----------


def expanding_loglog_residual(price: pd.Series, genesis: str, min_obs: int = 730) -> pd.Series:
    """Residual of log10(price) vs an OLS fit on log10(days since genesis),
    refit each day using only data up to that day (closed-form cumulative sums)."""
    days = (price.index - pd.Timestamp(genesis)).days.values.astype(float)
    x = np.log10(days)
    y = np.log10(price.values)
    n = np.arange(1, len(x) + 1)
    sx, sy = np.cumsum(x), np.cumsum(y)
    sxx, sxy = np.cumsum(x * x), np.cumsum(x * y)
    denom = n * sxx - sx**2
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
    resid = y - (a + b * x)
    resid[n < min_obs] = np.nan
    return pd.Series(resid, index=price.index)


def rsi(price: pd.Series, period: int = 14) -> pd.Series:
    d = price.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def weekly_rsi_daily(price: pd.Series, period: int = 14) -> pd.Series:
    """Weekly RSI, forward-filled onto the daily index (uses only completed weeks)."""
    wk = price.resample("W-SUN").last()
    return rsi(wk, period).reindex(price.index, method="ffill").shift(1)


def sma(price: pd.Series, days: int) -> pd.Series:
    return price.rolling(days, min_periods=days).mean()


# ---------- asset models ----------
# Each component: (key, label, group, weight, raw series, score series)
# "cycle" components carry most of the weight per the user's preference.


def btc_components(df: pd.DataFrame) -> list[dict]:
    p = df["price"]
    comps = []

    def add(key, label, group, weight, raw, score, fmt):
        comps.append(dict(key=key, label=label, group=group, weight=weight, raw=raw, score=score, fmt=fmt))

    resid = expanding_loglog_residual(p, genesis="2009-01-03")
    add("logreg", "Log-regression residual", "cycle", 0.30, resid, ewm_score(resid), "resid")

    log_mvrv = np.log(df["mvrv"])
    add("mvrv", "MVRV (market / realized cap)", "cycle", 0.25, df["mvrv"], ewm_score(log_mvrv), "x")

    r200w = p / sma(p, 1400)
    add("sma200w", "Price / 200-week SMA", "cycle", 0.20, r200w, ewm_score(np.log(r200w)), "x")

    pi = sma(p, 111) / (2 * sma(p, 350))
    add("pi", "Pi Cycle (111D / 2x350D)", "cycle", 0.10, pi, ewm_score(np.log(pi)), "x")

    mayer = p / sma(p, 200)
    add("mayer", "Mayer Multiple (price / 200D)", "trend", 0.10, mayer, ewm_score(np.log(mayer)), "x")

    wrsi = weekly_rsi_daily(p)
    add("rsi_w", "Weekly RSI(14)", "momentum", 0.05, wrsi, ewm_score(wrsi), "rsi")
    return comps


def composite(comps: list[dict]) -> pd.Series:
    """Weighted mean of available component scores (weights renormalized when some are warming up)."""
    scores = pd.concat({c["key"]: c["score"] for c in comps}, axis=1)
    w = pd.Series({c["key"]: c["weight"] for c in comps})
    avail = scores.notna().astype(float).mul(w, axis=1)
    total = avail.sum(axis=1)
    out = scores.fillna(0).mul(w, axis=1).sum(axis=1) / total
    # require at least half the weight to be live
    return out.where(total >= 0.5)


# ---------- zones ----------

ZONES = [  # (upper bound, key, label)
    (0.20, "strong_buy", "Strong Buy"),
    (0.40, "buy", "Buy"),
    (0.60, "hold", "Hold"),
    (0.80, "trim", "Trim"),
    (1.01, "sell", "Sell"),
]


def zone_of(r: float) -> str:
    for ub, key, _ in ZONES:
        if r < ub:
            return key
    return "sell"


def weekly_signal(daily_risk: pd.Series, confirm: int = 2) -> pd.DataFrame:
    """Weekly-close risk with hysteresis: the active zone only changes after
    `confirm` consecutive weekly closes inside a new zone."""
    wk = daily_risk.resample("W-SUN").last().dropna()
    raw = wk.apply(zone_of)
    active, cur, streak, cand = [], None, 0, None
    for z in raw:
        if cur is None:
            cur = z
        elif z != cur:
            streak = streak + 1 if z == cand else 1
            cand = z
            if streak >= confirm:
                cur, streak, cand = z, 0, None
        else:
            streak, cand = 0, None
        active.append(cur)
    return pd.DataFrame({"risk": wk, "raw_zone": raw, "zone": active})
