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


def expanding_ols_residual(y: pd.Series, x: pd.Series, min_obs: int = 730) -> pd.Series:
    """Residual of y vs an OLS fit on x, refit each day using only data up to
    that day (closed-form cumulative sums)."""
    m = y.notna() & x.notna()
    xv, yv = x[m].values.astype(float), y[m].values.astype(float)
    n = np.arange(1, len(xv) + 1)
    sx, sy = np.cumsum(xv), np.cumsum(yv)
    sxx, sxy = np.cumsum(xv * xv), np.cumsum(xv * yv)
    denom = n * sxx - sx**2
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
    resid = yv - (a + b * xv)
    resid[n < min_obs] = np.nan
    return pd.Series(resid, index=y[m].index).reindex(y.index)


def loglog_residual(price: pd.Series, genesis: str) -> pd.Series:
    """Power-law trend: log10(price) vs log10(days since genesis). Suits crypto adoption curves."""
    days = pd.Series((price.index - pd.Timestamp(genesis)).days, index=price.index).astype(float)
    return expanding_ols_residual(np.log10(price), np.log10(days.where(days > 0)))


def loglinear_residual(price: pd.Series) -> pd.Series:
    """Exponential trend: log10(price) vs time. Suits metals' long-run compounding."""
    days = pd.Series((price.index - price.index[0]).days, index=price.index).astype(float)
    return expanding_ols_residual(np.log10(price), days)


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


def to_daily(s: pd.Series, index: pd.DatetimeIndex, lag_days: int = 0) -> pd.Series:
    """Align a (possibly lower-frequency) series to a daily index. `lag_days`
    delays it to when it was actually published (e.g. M2 is released ~6 weeks late)."""
    s = s.copy()
    s.index = s.index + pd.Timedelta(days=lag_days)
    return s.reindex(s.index.union(index)).ffill().reindex(index)


# ---------- asset models ----------
# Each component: key, label, group, weight, raw series, score series, display format.
# "cycle" components carry most of the weight per the user's preference.


class Model:
    def __init__(self, halflife: int = HALFLIFE_DAYS):
        self.halflife = halflife
        self.comps: list[dict] = []

    def add(self, key, label, group, weight, raw, scored=None, fmt="x", invert=False, normalized=False):
        """`scored` is the transformed series to normalize (defaults to raw);
        `invert` flips it so that high raw values mean low risk;
        `normalized` means raw is already a 0..1 risk score."""
        s = raw if normalized else ewm_score(raw if scored is None else scored, self.halflife)
        self.comps.append(dict(key=key, label=label, group=group, weight=weight, raw=raw,
                               score=1 - s if invert else s, fmt=fmt))
        return self

    def common(self, p: pd.Series, w200w: float, wmayer: float, wrsi: float):
        """Price / 200-week SMA, Mayer Multiple, weekly RSI: shared by every asset."""
        r200w = p / sma(p, 1400)
        mayer = p / sma(p, 200)
        self.add("sma200w", "Price / 200-week SMA", "cycle", w200w, r200w, np.log(r200w))
        self.add("mayer", "Mayer Multiple (price / 200D)", "trend", wmayer, mayer, np.log(mayer))
        self.add("rsi_w", "Weekly RSI(14)", "momentum", wrsi, weekly_rsi_daily(p), fmt="rsi")
        return self.comps


def btc_components(df: pd.DataFrame, ctx: dict) -> list[dict]:
    p = df["price"]
    m = Model()
    m.add("logreg", "Log-regression residual", "cycle", 0.30, loglog_residual(p, "2009-01-03"), fmt="resid")
    m.add("mvrv", "MVRV (market / realized cap)", "cycle", 0.25, df["mvrv"], np.log(df["mvrv"]))
    pi = sma(p, 111) / (2 * sma(p, 350))
    m.add("pi", "Pi Cycle (111D / 2x350D)", "cycle", 0.10, pi, np.log(pi))
    return m.common(p, w200w=0.20, wmayer=0.10, wrsi=0.05)


def eth_components(df: pd.DataFrame, ctx: dict) -> list[dict]:
    p = df["price"]
    m = Model()
    m.add("logreg", "Log-regression residual", "cycle", 0.25, loglog_residual(p, "2015-07-30"), fmt="resid")
    m.add("mvrv", "MVRV (market / realized cap)", "cycle", 0.30, df["mvrv"], np.log(df["mvrv"]))
    return m.common(p, w200w=0.20, wmayer=0.15, wrsi=0.10)


def ada_components(df: pd.DataFrame, ctx: dict) -> list[dict]:
    """Only ~1.5 cycles of history, so lean on BTC's cycle and ADA's strength vs BTC."""
    p = df["price"]
    m = Model()
    btc_risk = to_daily(ctx["btc_risk"], p.index)
    m.add("btc_cycle", "BTC cycle risk (market regime)", "cycle", 0.20, btc_risk, fmt="score", normalized=True)
    ada_btc = p / to_daily(ctx["btc_price"], p.index)
    m.add("ada_btc", "ADA/BTC vs its 1-year average", "cycle", 0.25, ada_btc, np.log(ada_btc / sma(ada_btc, 365)))
    m.add("logreg", "Log-regression residual", "cycle", 0.20, loglog_residual(p, "2017-09-29"), fmt="resid")
    return m.common(p, w200w=0.20, wmayer=0.10, wrsi=0.05)


METALS_HALFLIFE = 8 * 365  # metal cycles run roughly a decade


def gold_components(df: pd.DataFrame, ctx: dict) -> list[dict]:
    p = df["price"]
    m = Model(METALS_HALFLIFE)
    m.add("trend", "Long-run trend residual (log-linear)", "cycle", 0.25, loglinear_residual(p), fmt="resid")
    m2 = to_daily(ctx["m2"], p.index, lag_days=45)
    m.add("m2", "Gold / M2 money supply", "cycle", 0.15, p / m2, np.log(p / m2), fmt="ratio")
    ry = to_daily(ctx["real_yield"], p.index, lag_days=1)
    ry_resid = expanding_ols_residual(np.log(p), ry)
    m.add("real_yield", "Gold vs 10Y real yield (residual)", "macro", 0.15, ry_resid, fmt="resid")
    return m.common(p, w200w=0.25, wmayer=0.10, wrsi=0.10)


def silver_components(df: pd.DataFrame, ctx: dict) -> list[dict]:
    p = df["price"]
    m = Model(METALS_HALFLIFE)
    gsr = to_daily(ctx["gold_price"], p.index) / p
    # high gold/silver ratio = silver cheap relative to gold = lower risk
    m.add("gsr", "Gold / silver ratio (inverted)", "cycle", 0.25, gsr, np.log(gsr), invert=True)
    m.add("trend", "Long-run trend residual (log-linear)", "cycle", 0.20, loglinear_residual(p), fmt="resid")
    m2 = to_daily(ctx["m2"], p.index, lag_days=45)
    m.add("m2", "Silver / M2 money supply", "cycle", 0.10, p / m2, np.log(p / m2), fmt="ratio")
    return m.common(p, w200w=0.20, wmayer=0.15, wrsi=0.10)


def composite(comps: list[dict]) -> pd.Series:
    """Weighted mean of available component scores (weights renormalized when some are warming up)."""
    scores = pd.concat({c["key"]: c["score"] for c in comps}, axis=1)
    w = pd.Series({c["key"]: c["weight"] for c in comps})
    avail = scores.notna().astype(float).mul(w, axis=1)
    total = avail.sum(axis=1)
    out = scores.fillna(0).mul(w, axis=1).sum(axis=1) / total
    # require at least half the weight to be live
    return out.where(total >= 0.5 - 1e-9)


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
