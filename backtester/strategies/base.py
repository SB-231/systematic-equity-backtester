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
            weights: np.ndarray [T, N] target weights (sum to 1 for long-only)
        """
        raise NotImplementedError