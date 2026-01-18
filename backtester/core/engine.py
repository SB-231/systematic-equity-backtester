from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from backtester.strategies.base import Strategy


@dataclass(frozen=True)
class BacktestResult:
    equity_gross: np.ndarray   # [T]
    equity_net: np.ndarray     # [T]
    daily_gross: np.ndarray    # [T]
    daily_net: np.ndarray      # [T]
    weights: np.ndarray        # [T, N] executed weights (shifted)
    turnover: np.ndarray       # [T] sum(|w_t - w_{t-1}|)/2
    costs: np.ndarray          # [T] cost in return units deducted from gross


class BacktestEngine:
    def run(
        self,
        close: np.ndarray,
        strategy: Strategy,
        cost_bps: float = 0.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        impact_k: float = 0.0,
    ) -> BacktestResult:
        """
        Vectorized backtest engine.

        Assumptions:
        - Strategy produces target weights at day t using close[0..t]
        - Execution uses weights at t-1 applied to returns at t (no lookahead)
        - Turnover = 0.5 * sum_i |w_t - w_{t-1}|
          (0.5 so "round-trip" not double-counted)

        Costs:
        - Linear bps cost applied to turnover:
            total_linear_bps = cost_bps + commission_bps + slippage_bps
            linear_cost_t = turnover_t * total_linear_bps * 1e-4
        - Quadratic market impact (optional):
            impact_cost_t = impact_k * turnover_t^2
          where impact_k is in "return units" (e.g. 1e-3 means 10 bps when turnover=1.0)
        """
        if close.ndim != 2:
            raise ValueError("close must be 2D array [T, N]")
        T, N = close.shape
        if T < 2:
            raise ValueError("Need at least 2 rows of prices")

        # Close-to-close returns
        asset_ret = np.zeros((T, N), dtype=np.float64)
        asset_ret[1:] = close[1:] / close[:-1] - 1.0

        # Target weights from strategy at time t
        w_target = strategy.generate_weights(close).astype(np.float64, copy=False)
        if w_target.shape != (T, N):
            raise ValueError(f"Strategy returned weights {w_target.shape}, expected {(T, N)}")

        # Execute weights next day (no-lookahead)
        w_exec = np.zeros_like(w_target, dtype=np.float64)
        w_exec[1:] = w_target[:-1]

        # Portfolio gross daily return
        daily_gross = np.sum(w_exec * asset_ret, axis=1)

        # Turnover (executed weights change day-to-day)
        dw = np.zeros_like(w_exec)
        dw[1:] = w_exec[1:] - w_exec[:-1]
        turnover = 0.5 * np.sum(np.abs(dw), axis=1)  # [T]

        # Costs in return units
        total_linear_bps = float(cost_bps) + float(commission_bps) + float(slippage_bps)
        linear_cost = turnover * (total_linear_bps * 1e-4)
        impact_cost = float(impact_k) * (turnover ** 2)
        costs = linear_cost + impact_cost

        daily_net = daily_gross - costs

        equity_gross = np.cumprod(1.0 + daily_gross)
        equity_net = np.cumprod(1.0 + daily_net)

        return BacktestResult(
            equity_gross=equity_gross,
            equity_net=equity_net,
            daily_gross=daily_gross,
            daily_net=daily_net,
            weights=w_exec,
            turnover=turnover,
            costs=costs,
        )