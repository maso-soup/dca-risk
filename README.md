# DCA Risk Dashboard

A long-horizon risk metric (0.0 = historically cheap, 1.0 = historically expensive) for
multi-year DCA in/out decisions, with a static dashboard on GitHub Pages. No paid APIs.

**Not financial advice.** It's a model fit to a handful of market cycles.

## How it works

| Step | File | Notes |
|---|---|---|
| Fetch | `pipeline/fetch.py` | CoinMetrics Community (crypto price, MVRV), Yahoo chart API (gold/silver futures), FRED CSV (T-bill, real yields, M2). No keys. Cached to `data/raw/`, falls back to cache if a source fails. |
| Metrics | `pipeline/metrics.py` | Each component is z-scored against an exponentially weighted history of *past* values (≈4-year half-life) and mapped to 0–1. No lookahead. Weighted composite; cycle metrics carry ~85%. |
| Signal | `metrics.weekly_signal` | Weekly close; zone changes only after 2 consecutive closes in a new zone. |
| Backtest | `pipeline/backtest.py` | Same weekly contribution into risk strategy vs plain DCA, from several start dates. Idle cash earns the 3-month T-bill. |
| Output | `pipeline/build.py` → `docs/data/*.json` | Read by `docs/index.html`. |

### Zones and actions

| Risk | Zone | Action per week |
|---|---|---|
| < 0.2 | Strong Buy | Deploy 1/8 of cash reserve (~2 months) |
| 0.2–0.4 | Buy | Deploy 1/20 of cash reserve (~5 months) |
| 0.4–0.6 | Hold | Accumulate cash |
| 0.6–0.8 | Trim | Sell 0–1% of holdings |
| > 0.8 | Sell | Sell 1% → 15% of holdings (convex in risk), can reach 100% cash |

### Components by asset

Weights in parentheses. "Cycle" metrics carry most of the weight.

| Asset | Components | Backtest vs plain DCA |
|---|---|---|
| **BTC** | Log-regression residual (30), MVRV (25), Price/200W SMA (20), Pi Cycle (10), Mayer Multiple (10), Weekly RSI (5) | Wins 4 of 5 start years, ~half the drawdown |
| **ETH** | MVRV (30), Log-regression residual (25), Price/200W SMA (20), Mayer (15), Weekly RSI (10) | Wins every start year |
| **Gold** | Log-linear trend residual (25), Price/200W SMA (25), Gold/M2 (15), Gold vs 10Y real yield residual (15), Mayer (10), Weekly RSI (10) | **Trails** every start year: secular bulls stay "expensive" for years. Use as a valuation gauge. |
| **Silver** | Gold/silver ratio, inverted (25), Log-linear trend residual (20), Price/200W SMA (20), Mayer (15), Silver/M2 (10), Weekly RSI (10) | Wins every start year |
| **ADA** | ADA/BTC vs 1-year average (25), BTC cycle risk (20), Log-regression residual (20), Price/200W SMA (20), Mayer (10), Weekly RSI (5) | Wins every start year, but low confidence (~1.5 cycles of data) |

Crypto uses a ~4-year normalization half-life; metals use ~8 years. M2 is lagged 45 days to match its publication delay.

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python pipeline/build.py          # fetch + build
.venv/bin/python pipeline/build.py --offline # rebuild from cache
python3 -m http.server 8765 --directory docs # open http://localhost:8765
```

## Deploy

1. Push this repo to GitHub (Pages from a free account requires a public repo).
2. Settings → Pages → Build and deployment → *Deploy from a branch* → `main` / `/docs`.
3. Settings → Actions → General → Workflow permissions → *Read and write*.
4. The `Update risk data` workflow runs daily at 06:30 UTC (or trigger it manually from the Actions tab).
