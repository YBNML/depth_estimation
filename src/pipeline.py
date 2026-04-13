from typing import Any, Dict
import numpy as np
from config import Config
from mde.base import BaseMDE
from mde.dummy import DummyMDE

def _create_mde(cfg: Config) -> BaseMDE:
    name = cfg.model_name
    max_depth = cfg.max_depth
    if name == "dummy":
        return DummyMDE(max_depth=max_depth)
    raise ValueError(f"Unknown model: {name}")

class DepthEstimationPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mde = _create_mde(cfg)

    def run(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> Dict[str, Any]:
        left_depth = self.mde.predict(left_rgb)
        right_depth = self.mde.predict(right_rgb)
        result = {"left_depth": left_depth, "right_depth": right_depth}
        if self.cfg.refinement_method != "none":
            result = self._refine(result, left_rgb, right_rgb)
        return result

    def _refine(self, result: Dict[str, Any], left_rgb: np.ndarray, right_rgb: np.ndarray) -> Dict[str, Any]:
        return result
