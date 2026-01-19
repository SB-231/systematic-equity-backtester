# Systematic Equity Backtester (Performance-Focused)

A vectorized backtesting engine for cross-sectional equity strategies with
explicit execution timing, transaction costs, and scalable universe selection.

![CI](https://github.com/SB-231/systematic-equity-backtester/actions/workflows/ci.yml/badge.svg)

---

## Overview

- Loads daily equity close prices from a local data source (Stooq dump)
- Aligns prices into a dense `[T, N]` matrix with coverage rules
- Generates portfolio weights via a strategy interface
- Applies no-lookahead execution (signals at t, traded at t+1)
- Computes gross and net performance with turnover-based costs
- Reports returns, volatility, Sharpe, drawdowns, turnover, and runtime metrics

---

## Why this project

This project focuses on infrastructure correctness and performance, not on optimizing a trading strategy. The goal is to demonstrate how a quant developer would design a reproducible, scalable backtesting system: enforcing execution timing, avoiding lookahead bias, modeling trading frictions, and benchmarking runtime and memory usage.

---

## Design principles

- No lookahead bias: weights generated at time t, executed at t+1
- Vectorized engine: NumPy-based, no pandas in hot path
- Config-driven: universe, strategy, costs, and dates defined in YAML
- Reproducible: cached universe selection and price matrices
- Tested: unit tests validate timing, turnover math, and cost accounting

---

## Repository structure

```text
.
├── backtester
│   ├── core
│   │   └── engine.py            # Vectorized engine: weights→PnL + turnover + costs
│   ├── data
│   │   └── loader.py            # Local Stooq loader + alignment + caching
│   ├── strategies
│   │   ├── base.py              # Strategy interface
│   │   └── momentum.py          # Cross-sectional momentum strategy
│   └── universe.py              # Universe discovery + coverage filter + caching
│
├── benchmarks
│   ├── __init__.py
│   └── benchmark.py             # Cold/warm/engine-only benchmarks
│
├── configs
│   └── base_config.yml          # Universe, dates, strategy, cost params
│
├── tests
│   └── test_engine.py           # Timing + turnover + cost accounting tests
│
├── .github
│   └── workflows
│       └── ci.yml               # GitHub Actions CI (pytest)
│
├── run_backtest.py              # End-to-end run + metrics + memory/timing report
├── requirements.txt
├── .gitignore
└── README.md

```
---

## Data and assumptions

- Data source: local Stooq US daily TXT dump
- Prices are unadjusted (no split/dividend adjustment)
- Universe selection is based on data coverage, not liquidity
- Not survivorship-bias free
- Intended for infrastructure development, not live trading

---

## Execution model

- Signals computed using close prices up to day t
- Target weights generated at t
- Executed weights applied to returns at t+1
- Returns computed close-to-close
- Long-only portfolio (no shorting or borrow costs)

---

## Transaction costs

- Turnover = 0.5 × sum(|w_t − w_{t−1}|)
- Linear costs in basis points:
    -	generic cost
    -	commission
    -	slippage
- Optional quadratic market impact term
- Costs are deducted in return space after portfolio aggregation.

---

## How to run

```bash
pip install -r requirements.txt

pytest -q
python run_backtest.py
python -m benchmarks.benchmark
```

## Example output and interpretation

The backtest reports both **gross** (before costs) and **net** (after costs) performance.
This separation makes the impact of turnover and transaction costs explicit.

In large universes, high turnover leads to significant cost drag, which is expected
for cross-sectional strategies without liquidity constraints. Poor absolute returns
in some runs reflect universe quality and data limitations, not backtesting errors.

The primary focus is correctness, execution realism, and performance benchmarking.

---

## Testing

Unit tests validate core correctness assumptions:

- No-lookahead execution (weights are shifted by one period)
- Turnover math on known weight transitions
- Transaction costs reduce performance when turnover is positive
- Strategy weights sum to 1 after warmup (and 0 during warmup)

Tests are run automatically via GitHub Actions CI.

---

## Performance and scalability

Typical run (500 equities × ~200 trading days):

- Cold run (universe selection + load + run): ~30–40 ms
- Warm run (cached universe + prices): ~15–20 ms
- Engine-only (100 runs): ~3 ms per run
- Memory footprint (price matrix): < 1 MB

The engine is fully vectorized and avoids pandas in the hot path,
making it suitable for larger universes and longer histories.

---

## Limitations and future work

- Use adjusted prices (splits/dividends)
- Liquidity-based universe construction
- Survivorship-bias-safe datasets
- Market benchmark comparison (e.g., equal-weight proxy)
- Additional execution models (open, VWAP)
- Packaging via `pyproject.toml` and CLI entry points

---

## Disclaimer

This project is for educational and demonstration purposes only.
It is not intended as trading advice or a production trading system.
