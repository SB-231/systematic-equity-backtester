from __future__ import annotations

import time
import yaml
import numpy as np
import matplotlib.pyplot as plt

from backtester.universe import select_universe
from backtester.data.loader import load_close_matrix
from backtester.core.engine import BacktestEngine
from backtester.strategies.momentum import CrossSectionalMomentum


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None or not isinstance(cfg, dict):
        raise ValueError(f"Config '{path}' is empty/invalid YAML.")
    return cfg


def _mb(nbytes: int) -> float:
    return float(nbytes) / (1024.0 * 1024.0)


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def _metrics(daily: np.ndarray, equity: np.ndarray):
    total_return = float(equity[-1] - 1.0)
    vol = float(np.std(daily) * np.sqrt(252))
    sharpe = float((np.mean(daily) / (np.std(daily) + 1e-12)) * np.sqrt(252))
    mdd = float(max_drawdown(equity))
    return total_return, vol, sharpe, mdd


def main():
    t_start_total = time.perf_counter()
    cfg = _load_config("configs/base_config.yml")

    # -------------------------------------------------
    # Universe selection (shared + cached)
    # -------------------------------------------------
    tickers, timings = select_universe(cfg)
    if not tickers:
        raise RuntimeError("No tickers selected (universe empty)")

    start = cfg["dates"]["start"]
    end = cfg["dates"]["end"]
    nasdaq_root = cfg["universe"]["root"]
    requested = int(cfg["universe"]["max_tickers"])

    # -------------------------------------------------
    # Data load
    # -------------------------------------------------
    t_data0 = time.perf_counter()
    data = load_close_matrix(
        tickers=tickers,
        start=start,
        end=end,
        nasdaq_stocks_root=nasdaq_root,
    )
    t_data1 = time.perf_counter()

    T, N = data.close.shape
    print(f"Universe selected: requested={requested} used(N)={N}  Days(T)={T}")
    print(f"Data matrix: T={T} days, N={N} tickers")

    # -------------------------------------------------
    # Strategy params
    # -------------------------------------------------
    strat_cfg = cfg.get("strategy", {})
    lookback = int(strat_cfg.get("lookback", 20))
    top_k_cfg = int(strat_cfg.get("top_k", 3))
    top_k = min(top_k_cfg, N)

    # -------------------------------------------------
    # Costs params
    # -------------------------------------------------
    costs_cfg = cfg.get("costs", {})
    cost_bps = float(costs_cfg.get("cost_bps", 0.0))
    commission_bps = float(costs_cfg.get("commission_bps", 0.0))
    slippage_bps = float(costs_cfg.get("slippage_bps", 0.0))
    impact_k = float(costs_cfg.get("impact_k", 0.0))

    strategy = CrossSectionalMomentum(lookback=lookback, top_k=top_k)
    engine = BacktestEngine()

    # -------------------------------------------------
    # Engine run
    # -------------------------------------------------
    t_eng0 = time.perf_counter()
    res = engine.run(
        close=data.close,
        strategy=strategy,
        cost_bps=cost_bps,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        impact_k=impact_k,
    )
    t_eng1 = time.perf_counter()

    # -------------------------------------------------
    # Memory footprint
    # -------------------------------------------------
    print("\nMemory footprint")
    print(f"close matrix: {_mb(data.close.nbytes):.2f} MB ({data.close.dtype}, shape={data.close.shape})")
    print(f"weights:      {_mb(res.weights.nbytes):.2f} MB (shape={res.weights.shape})")
    print(f"turnover:     {_mb(res.turnover.nbytes):.4f} MB (shape={res.turnover.shape})")
    print(f"costs:        {_mb(res.costs.nbytes):.4f} MB (shape={res.costs.shape})")
    print(f"daily_gross:  {_mb(res.daily_gross.nbytes):.4f} MB (shape={res.daily_gross.shape})")
    print(f"daily_net:    {_mb(res.daily_net.nbytes):.4f} MB (shape={res.daily_net.shape})")

    # -------------------------------------------------
    # Timings
    # -------------------------------------------------
    t_total = time.perf_counter() - t_start_total

    discover_ms = float(timings.get("discover_ms", 0.0))
    filter_ms = float(timings.get("filter_ms", 0.0))
    found = int(timings.get("found", 0))
    cache_hit = bool(timings.get("universe_cache_hit", 0.0) == 1.0)

    print("\nTiming breakdown")
    print(f"discover tickers: {discover_ms:.1f} ms (found {found})")
    if cache_hit:
        print("coverage filter:  0.0 ms (universe cache hit)")
    else:
        print(f"coverage filter:  {filter_ms:.1f} ms")
    print(f"data load:        {(t_data1 - t_data0) * 1000:.1f} ms")
    print(f"engine run:       {(t_eng1 - t_eng0) * 1000:.1f} ms")
    print(f"TOTAL script:     {t_total * 1000:.1f} ms")

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    tr_g, vol_g, sh_g, mdd_g = _metrics(res.daily_gross, res.equity_gross)
    tr_n, vol_n, sh_n, mdd_n = _metrics(res.daily_net, res.equity_net)

    print("\nConfig")
    print(f"lookback: {lookback}")
    print(f"top_k: {top_k}")

    print("\nCosts model")
    print(f"cost_bps (generic): {cost_bps:.2f}")
    print(f"commission_bps:    {commission_bps:.2f}")
    print(f"slippage_bps:      {slippage_bps:.2f}")
    print(f"impact_k:          {impact_k:.6f}")

    print("\nGROSS")
    print(f"Total return: {tr_g:.2%}")
    print(f"Annualized vol: {vol_g:.2%}")
    print(f"Sharpe: {sh_g:.2f}")
    print(f"Max Drawdown: {mdd_g:.2%}")

    print("\nNET")
    print(f"Total return: {tr_n:.2%}")
    print(f"Annualized vol: {vol_n:.2%}")
    print(f"Sharpe: {sh_n:.2f}")
    print(f"Max Drawdown: {mdd_n:.2%}")

    avg_turnover = float(np.mean(res.turnover))
    avg_cost_bps = float(np.mean(res.costs) * 1e4)

    print("\nTrading frictions")
    print(f"Avg daily turnover: {avg_turnover:.3f}")
    print(f"Avg daily cost: {avg_cost_bps:.2f} bps")

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    plt.plot(res.equity_gross, label="Gross")
    plt.plot(res.equity_net, label="Net")
    plt.legend()
    plt.title("Equity Curve")
    plt.show()


if __name__ == "__main__":
    main()