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

### BTC components

| Metric | Group | Weight |
|---|---|---|
| Log-regression residual (log price vs log days since genesis, refit daily on past data) | cycle | 30% |
| MVRV (market cap / realized cap) | cycle | 25% |
| Price / 200-week SMA | cycle | 20% |
| Pi Cycle (111D SMA / 2×350D SMA) | cycle | 10% |
| Mayer Multiple (price / 200D SMA) | trend | 10% |
| Weekly RSI(14) | momentum | 5% |

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
