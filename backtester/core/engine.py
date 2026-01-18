from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from backtester.strategies.base import Strategy


@dataclass(frozen=True)
class BacktestResult:
    equity: np.ndarray      # [T]
    daily_ret: np.ndarray   # [T]
    weights: np.ndarray     # [T, N] executed weights (shifted)


class BacktestEngine:
    def run(self, close: np.ndarray, strategy: Strategy) -> BacktestResult:
        T, N = close.shape

        # Close-to-close asset returns
        asset_ret = np.zeros((T, N), dtype=np.float64)
        asset_ret[1:] = close[1:] / close[:-1] - 1.0

        # Strategy outputs target weights at time t
        w_target = strategy.generate_weights(close)

        # Execute those weights on next day (prevents lookahead)
        w_exec = np.zeros_like(w_target)
        w_exec[1:] = w_target[:-1]

        # Portfolio daily return
        daily = np.sum(w_exec * asset_ret, axis=1)

        # Equity curve (start at 1.0)
        equity = np.cumprod(1.0 + daily)

        return BacktestResult(equity=equity, daily_ret=daily, weights=w_exec)
    