import yaml
import numpy as np
import matplotlib.pyplot as plt
import time

from backtester.data.loader import load_close_matrix
from backtester.core.engine import BacktestEngine
from backtester.strategies.momentum import CrossSectionalMomentum


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None or not isinstance(cfg, dict):
        raise ValueError(f"Config '{path}' is empty/invalid YAML.")
    return cfg


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def _metrics(daily: np.ndarray, equity: np.ndarray) -> tuple:
    total_return = float(equity[-1] - 1.0)
    vol = float(np.std(daily) * np.sqrt(252))
    sharpe = float((np.mean(daily) / (np.std(daily) + 1e-12)) * np.sqrt(252))
    mdd = float(max_drawdown(equity))
    return total_return, vol, sharpe, mdd


def main():
    cfg = _load_config("configs/base_config.yml")

    # Required config
    tickers = cfg["universe"]["tickers"]
    start = cfg["dates"]["start"]
    end = cfg["dates"]["end"]

    # Optional strategy params from config
    strat_cfg = cfg.get("strategy", {})
    lookback = int(strat_cfg.get("lookback", 20))
    top_k_cfg = int(strat_cfg.get("top_k", 3))

    # Optional cost params from config
    cost_bps = float(cfg.get("costs", {}).get("cost_bps", 0.0))

    # Load market data (cached if available)
    data = load_close_matrix(tickers, start, end)

    # Clamp top_k to universe size
    top_k = min(top_k_cfg, len(data.tickers))

    # Strategy: cross-sectional momentum
    strategy = CrossSectionalMomentum(lookback=lookback, top_k=top_k)
    engine = BacktestEngine()

    # Run backtest
    t0 = time.perf_counter()
    res = engine.run(close=data.close, strategy=strategy, cost_bps=cost_bps)
    t1 = time.perf_counter()

    print(f"Backtest runtime: {(t1 - t0) * 1000:.1f} ms")
    print("Backtest complete")
    print("Tickers:", data.tickers)

    # Gross vs Net metrics
    tr_g, vol_g, sh_g, mdd_g = _metrics(res.daily_gross, res.equity_gross)
    tr_n, vol_n, sh_n, mdd_n = _metrics(res.daily_net, res.equity_net)

    avg_turnover = float(res.turnover.mean())
    avg_cost_bps = float(res.costs.mean() * 1e4)

    print("\nConfig")
    print(f"lookback: {lookback}")
    print(f"top_k: {top_k}")
    print(f"cost_bps: {cost_bps:.2f}")

    print("\nGROSS (no costs)")
    print(f"Total return: {tr_g:.2%}")
    print(f"Annualized vol: {vol_g:.2%}")
    print(f"Sharpe (naive): {sh_g:.2f}")
    print(f"Max drawdown: {mdd_g:.2%}")

    print("\nNET (with costs)")
    print(f"Cost model: cost_bps={cost_bps:.2f} applied to turnover")
    print(f"Total return: {tr_n:.2%}")
    print(f"Annualized vol: {vol_n:.2%}")
    print(f"Sharpe (naive): {sh_n:.2f}")
    print(f"Max drawdown: {mdd_n:.2%}")

    print("\nTrading frictions")
    print(f"Avg daily turnover: {avg_turnover:.3f}")
    print(f"Avg daily cost: {avg_cost_bps:.2f} bps")

    # Plot equity curves
    plt.figure()
    plt.plot(res.equity_gross, label="Gross")
    plt.plot(res.equity_net, label="Net")
    plt.title("Equity Curve (Cross-Sectional Momentum)")
    plt.xlabel("Days")
    plt.ylabel("Equity")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()