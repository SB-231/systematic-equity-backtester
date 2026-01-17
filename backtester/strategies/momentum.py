from __future__ import annotations
import numpy as np
from .base import Strategy


class CrossSectionalMomentum(Strategy):
    def __init__(self, lookback: int = 20, top_k: int = 3):
        self.lookback = lookback
        self.top_k = top_k

    def generate_weights(self, close: np.ndarray) -> np.ndarray:
        T, N = close.shape
        w = np.zeros((T, N), dtype=np.float64)

        lb = self.lookback
        k = min(self.top_k, N)

        # momentum score matrix for t>=lb
        scores = np.full((T, N), -np.inf, dtype=np.float64)
        scores[lb:] = close[lb:] / close[:-lb] - 1.0

        # pick top-k each day (vectorized)
        topk_idx = np.argpartition(scores, -k, axis=1)[:, -k:]  # [T, k]

        # assign equal weights to top-k for each row
        rows = np.arange(T)[:, None]
        w[rows, topk_idx] = 1.0 / k

        # optional: zero-out warmup rows explicitly
        w[:lb] = 0.0
        return w