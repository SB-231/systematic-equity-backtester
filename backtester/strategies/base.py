from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Strategy(ABC):
    @abstractmethod
    def generate_weights(self, close: np.ndarray) -> np.ndarray:
        """
        Input:
            close: np.ndarray [T, N] close prices

        Output:
            weights: np.ndarray [T, N] target portfolio weights.

        Notes:
            - Long-only strategies typically have weights >= 0 and sum(weights[t]) ~= 1.
            - Long/short strategies may have negative weights and non-1 gross exposure.
            - The engine will apply a 1-day execution lag to prevent lookahead.
        """
        raise NotImplementedError