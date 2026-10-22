"""Fetch data, compute risk, backtest, and write JSON for the dashboard (docs/data)."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import backtest
from fetch import load
import metrics as M
from metrics import ZONES, composite, weekly_signal

OUT = Path(__file__).resolve().parent.parent / "docs" / "data"

ASSETS = {
    "btc": dict(name="Bitcoin", symbol="BTC", components=M.btc_components, chart_start="2013-01-01",
                backtest_starts=["2015-01-01", "2017-01-01", "2019-01-01", "2021-01-01", "2023-01-01"],
                confidence="high", note="5 full cycles of history; on-chain MVRV available."),
    "eth": dict(name="Ethereum", symbol="ETH", components=M.eth_components, chart_start="2017-01-01",
                backtest_starts=["2018-06-01", "2019-01-01", "2021-01-01", "2023-01-01"],
                confidence="medium", note="~2.5 cycles of history; on-chain MVRV available."),
    "gold": dict(name="Gold", symbol="XAU", components=M.gold_components, chart_start="2004-01-01",
                 backtest_starts=["2005-01-01", "2008-01-01", "2011-01-01", "2015-01-01", "2020-01-01"],
                 confidence="medium",
                 note="COMEX front-month futures, ~8-year normalization window. Backtest TRAILS plain DCA from "
                      "every start: in secular bulls (2005–08, 2024–26) gold stays 'expensive' for years. "
                      "Treat this as a valuation gauge, not a timing rule."),
    "silver": dict(name="Silver", symbol="XAG", components=M.silver_components, chart_start="2004-01-01",
                   backtest_starts=["2005-01-01", "2008-01-01", "2011-01-01", "2015-01-01", "2020-01-01"],
                   confidence="medium", note="COMEX front-month futures. Gold/silver ratio is the anchor metric."),
    "ada": dict(name="Cardano", symbol="ADA", components=M.ada_components, chart_start="2020-01-01",
                backtest_starts=["2021-01-01", "2022-01-01", "2023-01-01"],
                confidence="low",
                note="Only ~1.5 cycles of history (Dec 2017+), so it leans on BTC's cycle risk and ADA/BTC strength."),
}

DAILY_FILL = {"gold", "silver"}  # trading-day series -> calendar days so day-based windows match crypto


def _num(v, nd=4):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(float(v), nd)


def _series(s: pd.Series, nd=4) -> list:
    return [_num(v, nd) for v in s]


def _dates(idx) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in idx]


def build_asset(key: str, cfg: dict, df: pd.DataFrame, ctx: dict, tbill: pd.Series) -> tuple[pd.Series, dict]:
    comps = cfg["components"](df, ctx)
    risk = composite(comps)
    sig = weekly_signal(risk)
    last_day = df.index[-1]
    # the final weekly row is the week in progress (labelled with its upcoming Sunday)
    in_progress = sig.index[-1] > last_day

    price_w = df["price"].resample("W-SUN").last().reindex(sig.index)
    bt_runs = backtest.by_start(price_w, sig, tbill, cfg["backtest_starts"])
    bt_main = backtest.run(price_w[price_w.index >= cfg["backtest_starts"][0]],
                           sig[sig.index >= cfg["backtest_starts"][0]], tbill)["log"]

    confirmed = sig.iloc[-2] if in_progress else sig.iloc[-1]
    zone = sig.zone.iloc[-1]  # hysteresis-confirmed zone including this week so far
    r_today = float(risk.dropna().iloc[-1])

    def ago(days):
        prior = risk.loc[: last_day - pd.Timedelta(days=days)].dropna()
        return _num(r_today - prior.iloc[-1]) if len(prior) else None

    chart_start = max(pd.Timestamp(cfg["chart_start"]), risk.first_valid_index())
    chart = df.loc[chart_start:]
    cs = pd.concat({c["key"]: c["score"] for c in comps}, axis=1).loc[chart_start:]
    cs_w = cs.resample("W-SUN").last()

    return risk, {
        "key": key, "name": cfg["name"], "symbol": cfg["symbol"],
        "confidence": cfg["confidence"], "note": cfg["note"],
        "latest": {
            "date": last_day.strftime("%Y-%m-%d"),
            "price": _num(df["price"].iloc[-1], 6),
            "risk": _num(r_today),
            "risk_week_close": _num(confirmed.risk),
            "week_close_date": sig.index[-2 if in_progress else -1].strftime("%Y-%m-%d"),
            "raw_zone": sig.raw_zone.iloc[-1],
            "zone": zone,
            "change_1w": ago(7), "change_4w": ago(28), "change_13w": ago(91),
            "plan": backtest.plan_for(zone, r_today),
        },
        "components": [
            {"key": c["key"], "label": c["label"], "group": c["group"], "weight": c["weight"],
             "fmt": c["fmt"], "raw": _num(c["raw"].dropna().iloc[-1]), "score": _num(c["score"].dropna().iloc[-1])}
            for c in comps
        ],
        "daily": {"date": _dates(chart.index), "price": _series(chart["price"], 6),
                  "risk": _series(risk.reindex(chart.index))},
        "weekly": {"date": _dates(sig.index), "risk": _series(sig.risk), "zone": list(sig.zone)},
        "component_weekly": {"date": _dates(cs_w.index), **{k: _series(cs_w[k], 3) for k in cs_w.columns}},
        "backtest": {
            "contribution": backtest.CONTRIBUTION,
            "runs": bt_runs,
            "curve": {"date": _dates(bt_main.index), "value": _series(bt_main.value, 2),
                      "dca_value": _series(bt_main.dca_value, 2), "contributed": _series(bt_main.contributed, 2),
                      "in_market": _series((bt_main.units * bt_main.price / bt_main.value).fillna(0), 3)},
        },
    }


def main(refresh: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {}
    for name in ["tbill", "m2", "real_yield", *ASSETS]:
        df = load(name, refresh=refresh)
        raw[name] = df.asfreq("D").ffill() if name in DAILY_FILL else df
    tbill = raw["tbill"]["DTB3"]
    # cross-asset inputs; btc_risk is filled in once BTC is built (ASSETS lists BTC first)
    ctx = {"m2": raw["m2"]["M2SL"], "real_yield": raw["real_yield"]["DFII10"],
           "btc_price": raw["btc"]["price"], "gold_price": raw["gold"]["price"]}

    index = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
             "zones": [{"max": ub, "key": k, "label": lbl} for ub, k, lbl in ZONES], "assets": []}
    for key, cfg in ASSETS.items():
        risk, data = build_asset(key, cfg, raw[key], ctx, tbill)
        ctx[f"{key}_risk"] = risk
        (OUT / f"{key}.json").write_text(json.dumps(data, separators=(",", ":")))
        index["assets"].append({"key": key, "name": data["name"], "symbol": data["symbol"],
                                "confidence": data["confidence"], **data["latest"]})
        lt = data["latest"]
        print(f"{data['symbol']:5s} {lt['date']} price={lt['price']:,} risk={lt['risk']:.3f} zone={lt['zone']}")
    (OUT / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use cached CSVs only")
    main(refresh=not ap.parse_args().offline)
