from __future__ import annotations
import numpy as np
from .base import Strategy


class CrossSectionalMomentum(Strategy):
    def __init__(self, lookback: int = 20, top_k: int = 3):
        self.lookback = int(lookback)
        self.top_k = int(top_k)

    def generate_weights(self, close: np.ndarray) -> np.ndarray:
        if close.ndim != 2:
            raise ValueError("close must be [T, N]")

        T, N = close.shape
        w = np.zeros((T, N), dtype=np.float64)

        lb = self.lookback
        if lb <= 0:
            raise ValueError("lookback must be > 0")
        if T <= lb:
            # Not enough history to form momentum
            return w

        k = min(self.top_k, N)
        if k <= 0:
            return w

        # Raw momentum: close[t]/close[t-lb] - 1
        scores = np.full((T, N), -np.inf, dtype=np.float64)
        raw = close[lb:] / close[:-lb] - 1.0

        # Mask invalid values to -inf so they never get selected
        raw = np.where(np.isfinite(raw), raw, -np.inf)
        scores[lb:] = raw

        # For each day, select top-k by score.
        # If a day has fewer than k finite scores, it will still pick some -inf names,
        # so we explicitly handle that by selecting only valid names.
        rows = np.arange(T)

        # Candidate selection via argpartition (fast)
        topk_idx = np.argpartition(scores, -k, axis=1)[:, -k:]  # [T, k]

        # Build weights with a per-row validity check
        for t in range(lb, T):
            idx = topk_idx[t]
            valid = np.isfinite(scores[t, idx]) & (scores[t, idx] > -np.inf / 2)
            idx = idx[valid]
            if idx.size == 0:
                continue
            w[t, idx] = 1.0 / float(idx.size)

        # Warmup explicitly zero
        w[:lb] = 0.0
        return w