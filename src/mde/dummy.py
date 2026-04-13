import numpy as np

from mde.base import BaseMDE


class DummyMDE(BaseMDE):
    def __init__(self, max_depth: float = 10.0):
        self._max_depth = max_depth

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        return np.random.uniform(0.1, self._max_depth, (h, w)).astype(np.float32)

    def get_max_depth(self) -> float:
        return self._max_depth
