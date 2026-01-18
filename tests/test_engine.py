import numpy as np
from backtester.core.engine import BacktestEngine
from backtester.strategies.base import Strategy
from backtester.strategies.momentum import CrossSectionalMomentum


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


class KnownTurnoverStrategy(Strategy):
    """
    Deterministic weights for testing turnover math.

    Target weights (w_target):
      t=0: [0, 0]
      t=1: [1, 0]
      t=2: [0, 1]
      t>=3: [0, 1]

    Engine executes with 1-day delay (w_exec[t] = w_target[t-1]):

      w_exec[0] = [0,0]
      w_exec[1] = [0,0]
      w_exec[2] = [1,0]
      w_exec[3] = [0,1]
      ...

    Turnover[t] = 0.5 * sum(|w_exec[t] - w_exec[t-1]|)

    So:
      turnover[0] = 0
      turnover[1] = 0
      turnover[2] = 0.5 * (|1-0| + |0-0|) = 0.5
      turnover[3] = 0.5 * (|0-1| + |1-0|) = 1.0   (full switch)
    """
    def generate_weights(self, close: np.ndarray) -> np.ndarray:
        T, N = close.shape
        if N < 2:
            raise ValueError("Need at least 2 assets for this test")

        w = np.zeros((T, N), dtype=np.float64)
        if T > 1:
            w[1, 0] = 1.0
        if T > 2:
            w[2, 1] = 1.0
        if T > 3:
            w[3:, 1] = 1.0
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


# -------------------------------
# NEW TEST A: Turnover math sanity
# -------------------------------
def test_turnover_known_case():
    close = _toy_close(T=6, N=2)
    strat = KnownTurnoverStrategy()
    eng = BacktestEngine()
    res = eng.run(close, strat, cost_bps=0.0)

    # Explicit expected turnover series for first few days
    # Based on the docstring above.
    expected = np.zeros(close.shape[0], dtype=np.float64)
    expected[0] = 0.0
    expected[1] = 0.0
    expected[2] = 0.5
    expected[3] = 1.0

    assert np.allclose(res.turnover[:4], expected[:4], atol=1e-12)


# -----------------------------------------
# NEW TEST B: Momentum weights sum to 1/0
# -----------------------------------------
def test_momentum_weights_sum_to_one_after_warmup():
    # Make a simple price matrix with enough rows for lookback
    T, N = 40, 5
    close = np.ones((T, N), dtype=np.float64)
    # add small trends so scores are well-defined
    for j in range(N):
        close[:, j] = 100 + (j + 1) * np.linspace(0, 1, T)

    lookback = 10
    top_k = 3
    strat = CrossSectionalMomentum(lookback=lookback, top_k=top_k)
    w = strat.generate_weights(close)

    row_sums = w.sum(axis=1)

    # Warmup rows should be exactly 0
    assert np.allclose(row_sums[:lookback], 0.0, atol=1e-12)

    # After warmup, weights should sum to 1 (long-only fully invested)
    assert np.allclose(row_sums[lookback:], 1.0, atol=1e-12)

    # Also sanity: no negatives
    assert np.all(w >= -1e-15)