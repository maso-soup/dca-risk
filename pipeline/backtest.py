"""Weekly backtest: risk-based DCA in/out vs plain weekly DCA.

Both strategies receive the same weekly contribution. The plain DCA buys
every week. The risk strategy parks contributions in cash (earning T-bill
yield) and deploys or sells according to the hysteresis-confirmed zone.
"""
from __future__ import annotations

import pandas as pd

CONTRIBUTION = 100.0  # $ per week into both strategies

# fraction of the cash pile deployed per week while in a buy zone
BUY_PACE = {"strong_buy": 1 / 8, "buy": 1 / 20}  # ~2 months / ~5 months


def sell_pace(zone: str, risk: float) -> float:
    """Fraction of holdings sold per week. Trim is gentle (0-1%/wk); above 0.8
    the pace rises convexly to 15%/wk at 1.0 so a blow-off top gets sold hard
    while an early "sell" reading in a long bull run doesn't exit too soon."""
    if zone not in ("trim", "sell"):
        return 0.0
    r = min(max(risk, 0.6), 1.0)
    if r < 0.8:
        return 0.01 * (r - 0.6) / 0.2
    return 0.01 + 0.14 * ((r - 0.8) / 0.2) ** 2


def plan_for(zone: str, risk: float) -> dict:
    """Human-readable action for the dashboard."""
    if zone in BUY_PACE:
        return {"action": "buy", "pace": BUY_PACE[zone],
                "text": f"Deploy {BUY_PACE[zone]:.1%} of cash reserve per week (+ new contributions)"}
    if zone == "hold":
        return {"action": "hold", "pace": 0.0, "text": "Accumulate cash; no buys or sells"}
    p = sell_pace(zone, risk)
    return {"action": "sell", "pace": p, "text": f"Sell {p:.1%} of holdings per week"}


def run(price_w: pd.Series, sig: pd.DataFrame, tbill_pct: pd.Series) -> dict:
    """price_w, sig (risk, zone) indexed by week; tbill_pct = annual % yield (daily FRED)."""
    c = CONTRIBUTION
    rate = tbill_pct.reindex(price_w.index, method="ffill").fillna(0) / 100 / 52

    units = cash = 0.0
    dca_units = 0.0
    contributed = 0.0
    rows = []
    for (d, px), z, rk, r in zip(price_w.items(), sig.zone, sig.risk, rate):
        contributed += c
        dca_units += c / px

        cash = cash * (1 + r) + c
        buy_frac, sell_frac = BUY_PACE.get(z, 0.0), sell_pace(z, rk)
        trade = 0.0
        if buy_frac:
            spend = cash * buy_frac
            units += spend / px
            cash -= spend
            trade = spend
        if sell_frac and units:
            sold = units * sell_frac
            units -= sold
            cash += sold * px
            trade = -sold * px
        rows.append(dict(date=d, price=px, zone=z, trade=trade, units=units, cash=cash,
                         value=units * px + cash, dca_value=dca_units * px, contributed=contributed))
    log = pd.DataFrame(rows).set_index("date")
    return dict(log=log, summary=summarize(log))


def _max_dd(v: pd.Series) -> float:
    return float((v / v.cummax() - 1).min())


def summarize(log: pd.DataFrame) -> dict:
    last = log.iloc[-1]
    return dict(
        start=str(log.index[0].date()),
        end=str(log.index[-1].date()),
        contributed=round(float(last.contributed), 2),
        risk_value=round(float(last.value), 2),
        dca_value=round(float(last.dca_value), 2),
        risk_multiple=round(float(last.value / last.contributed), 2),
        dca_multiple=round(float(last.dca_value / last.contributed), 2),
        risk_max_drawdown=round(_max_dd(log.value), 3),
        dca_max_drawdown=round(_max_dd(log.dca_value), 3),
        pct_in_market_now=round(float(last.units * last.price / last.value), 3) if last.value else 0.0,
    )


def by_start(price_w: pd.Series, sig: pd.DataFrame, tbill: pd.Series, starts: list[str]) -> list[dict]:
    """Rerun from several start dates so one lucky entry can't flatter the result."""
    out = []
    for s in starts:
        m = price_w.index >= s
        if m.sum() > 52:
            out.append(run(price_w[m], sig[m], tbill)["summary"])
    return out
