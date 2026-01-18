import numpy as np
from backtester.core.engine import BacktestEngine
from backtester.strategies.base import Strategy


class ConstantWeightStrategy(Strategy):
    def __init__(self, w):
        self.w = np.asarray(w, dtype=np.float64)

    def generate_weights(self, close: np.ndarray) -> np.ndarray:
        T, N = close.shape
        w = np.zeros((T, N), dtype=np.float64)
        w[:] = self.w
        return w


class FlippingStrategy(Strategy):
    """
    Alternates 100% allocation between asset 0 and asset 1 each day.
    This intentionally creates turnover.
    """
    def generate_weights(self, close: np.ndarray) -> np.ndarray:
        T, N = close.shape
        w = np.zeros((T, N), dtype=np.float64)
        for t in range(T):
            w[t, t % 2] = 1.0
        return w


def _toy_close(T=6, N=2) -> np.ndarray:
    # Simple deterministic prices, strictly positive
    close = np.ones((T, N), dtype=np.float64)
    close[:, 0] = np.linspace(100, 105, T)
    close[:, 1] = np.linspace(200, 195, T)
    return close


def test_shapes():
    close = _toy_close(T=8, N=2)
    strat = ConstantWeightStrategy([0.5, 0.5])
    eng = BacktestEngine()
    res = eng.run(close, strat, cost_bps=0.0)

    assert res.weights.shape == close.shape
    assert res.daily_gross.shape == (close.shape[0],)
    assert res.daily_net.shape == (close.shape[0],)
    assert res.equity_gross.shape == (close.shape[0],)
    assert res.equity_net.shape == (close.shape[0],)
    assert res.turnover.shape == (close.shape[0],)
    assert res.costs.shape == (close.shape[0],)


def test_no_lookahead_weight_shift():
    close = _toy_close(T=5, N=2)

    strat = ConstantWeightStrategy([1.0, 0.0])
    eng = BacktestEngine()
    res = eng.run(close, strat, cost_bps=0.0)

    # Day 0 executed weights should be zeros due to shift
    assert np.allclose(res.weights[0], [0.0, 0.0])

    # Days 1.. should be [1,0]
    assert np.allclose(res.weights[1:], np.array([[1.0, 0.0]] * (close.shape[0] - 1)))


def test_cost_zero_net_equals_gross():
    close = _toy_close(T=10, N=2)
    strat = FlippingStrategy()
    eng = BacktestEngine()
    res = eng.run(close, strat, cost_bps=0.0)

    assert np.allclose(res.daily_net, res.daily_gross)
    assert np.allclose(res.equity_net, res.equity_gross)
    assert np.allclose(res.costs, 0.0)


def test_costs_reduce_performance_when_turnover_positive():
    close = _toy_close(T=20, N=2)
    strat = FlippingStrategy()
    eng = BacktestEngine()
    res = eng.run(close, strat, cost_bps=10.0)

    assert res.turnover.mean() > 0.0
    assert res.costs.mean() > 0.0

    # Net equity should not exceed gross equity (cost drag)
    assert res.equity_net[-1] <= res.equity_gross[-1] + 1e-12