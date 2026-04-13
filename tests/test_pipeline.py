import numpy as np
import pytest
from pipeline import DepthEstimationPipeline
from config import Config

def test_pipeline_creation():
    cfg = Config(overrides={"model_name": "dummy"})
    pipe = DepthEstimationPipeline(cfg)
    assert pipe is not None

def test_pipeline_run_mono():
    cfg = Config(overrides={"model_name": "dummy", "refinement_method": "none"})
    pipe = DepthEstimationPipeline(cfg)
    left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = pipe.run(left, right)
    assert "left_depth" in result
    assert "right_depth" in result
    assert result["left_depth"].shape == (480, 640)
    assert result["right_depth"].shape == (480, 640)

def test_pipeline_run_returns_float32():
    cfg = Config(overrides={"model_name": "dummy", "refinement_method": "none"})
    pipe = DepthEstimationPipeline(cfg)
    left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = pipe.run(left, right)
    assert result["left_depth"].dtype == np.float32
    assert result["right_depth"].dtype == np.float32
