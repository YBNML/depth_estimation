from abc import ABC, abstractmethod

import numpy as np


class BaseMDE(ABC):
    @abstractmethod
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def get_max_depth(self) -> float:
        ...
