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


def main():
    cfg = _load_config("configs/base_config.yml")

    tickers = cfg["universe"]["tickers"]
    start = cfg["dates"]["start"]
    end = cfg["dates"]["end"]

    data = load_close_matrix(tickers, start, end)

    strategy = CrossSectionalMomentum(lookback=20, top_k=min(3, len(data.tickers)))
    engine = BacktestEngine()

    t0 = time.perf_counter()
    res = engine.run(data.close, strategy)
    t1 = time.perf_counter()

    print(f"Backtest runtime: {(t1 - t0) * 1000:.1f} ms")

    total_return = res.equity[-1] - 1.0
    vol = np.std(res.daily_ret) * np.sqrt(252)
    sharpe = (np.mean(res.daily_ret) / (np.std(res.daily_ret) + 1e-12)) * np.sqrt(252)
    mdd = max_drawdown(res.equity)

    print("Backtest complete")
    print("Tickers:", data.tickers)
    print(f"Total return: {total_return:.2%}")
    print(f"Annualized vol: {vol:.2%}")
    print(f"Sharpe (naive): {sharpe:.2f}")
    print(f"Max drawdown: {mdd:.2%}")

    plt.figure()
    plt.plot(res.equity)
    plt.title("Equity Curve (Momentum, MVP)")
    plt.xlabel("Days")
    plt.ylabel("Equity")
    plt.show()


if __name__ == "__main__":
    main()