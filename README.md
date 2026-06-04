# US Swing Trading System v2

[中文说明](README.zh-CN.md)

US swing trading workflow for `stocks + options`, with a `macro and market first` bias. The system prioritizes:

- macro direction and event risk
- market regime and breadth
- Futu/moomoo community sentiment
- capital flow and company events
- lightweight technical execution confirmation

Technical structure still matters, but it no longer dominates the decision stack.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Optional environment variables
export FRED_API_KEY=your_key
export FINNHUB_API_KEY=your_key

# 3. Optional Futu/moomoo snapshot inputs
# data/inputs/community/futu_comments.json
# data/inputs/community/moomoo_comments.json
# data/inputs/flows/futu_capital.json
# data/inputs/flows/moomoo_capital.json

# 4. Run after market close
python -m pipeline.run_daily

# 5. Read the report
cat "/Users/xiao1/Desktop/2026 Day Scan output/daily_scan_latest.md"
```

## Architecture

```text
L1 Event Risk  ->  L2 Macro  ->  L3 Regime/Breadth  ->
L4 Community   ->  L5 Flow/Event  ->  L6 Structure/Trigger  ->
Composite Score -> watchlist
```

| Layer | Module | Veto | Source |
|------|------|------|------|
| L1 Event Risk | `event_risk.py` | Yes | `events_calendar.csv` |
| L2 Macro | `macro.py` | Yes | FRED + event window |
| L3 Regime | `regime.py` | Yes | `yfinance` + optional FRED |
| L4 Breadth | `breadth.py` | No | Internal universe breadth |
| L5 Community | `community.py` | No | Futu/moomoo comment snapshots |
| L6 Flow / Event | `flow.py` + `stock_events.py` | No | Futu/moomoo flow snapshots + Finnhub company news |
| L7 Structure / Trigger | `structure.py` + `triggers.py` | No | `yfinance` |

## Core Weights

- Macro: 25%
- Market: 20%
- Community: 20%
- Flow: 15%
- Stock Events: 15%
- Technical: 5%

This weighting is designed for swing trading where `macro + market` decide whether we should even lean risk-on, and technicals mostly decide timing quality.

## Project Layout

```text
config/
  thresholds.yaml
  universe.yaml
  events_calendar.csv

src/adapters/
  price_client.py
  fred_client.py
  news_client.py
  community_client.py
  flow_client.py

src/features/
  event_risk.py
  macro.py
  regime.py
  breadth.py
  community.py
  flow.py
  stock_events.py
  structure.py
  triggers.py
  scoring.py

pipeline/
  run_daily.py

research/
  factor_tests.py

tests/
  regression tests for P0 logic and upgraded factor layers
```

## Futu / moomoo Integration

The current version supports snapshot-driven integration first.

1. Export or generate comment sentiment snapshots from Futu/moomoo.
2. Export or generate capital-flow snapshots from Futu/moomoo.
3. Save them into `data/inputs/...`.
4. Run the pipeline and the scores will automatically include them.

Community snapshot example:

```json
[
  {"symbol": "MARKET", "sentiment": "bullish", "mentions": 42},
  {"symbol": "NVDA", "sentiment": "bearish", "mentions": 8}
]
```

Flow snapshot example:

```json
[
  {"symbol": "NVDA", "net_inflow": 2400000, "anomaly": true},
  {"symbol": "AAPL", "net_inflow": -500000, "anomaly": false}
]
```

The adapters were split out so we can later upgrade from static snapshots to real OpenD / SDK ingestion.

## P0 Fixes Included

- `--refresh-prices` now really refreshes bulk price downloads instead of being a dead CLI flag.
- `rs_vs_sector_min` now uses decimal return units correctly as `-0.02`.
- Breadth now actually respects configured `adv_ratio`, `nh_nl_ratio`, and `% above 200EMA` thresholds.
- Composite scoring now matches the documented 100-point architecture instead of an incomplete partial-weight implementation.

## Risk Rules

- Option position loss over 50%: exit.
- Single ticker exposure over 40% of capital: no adding.
- Macro/Event veto: avoid new long risk.
- Time stop: if no real follow-through after 7 days, reduce or exit.

## Validation

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall pipeline src research tests
```

## Next Up

- Replace snapshot loading with direct Futu/moomoo OpenD ingestion
- Upgrade community sentiment with FinBERT or custom finance classifiers
- Expand stock event tagging for earnings, guidance, litigation, dilution, M&A
- Add rolling validation, slippage, transaction costs, and regime-segmented research
