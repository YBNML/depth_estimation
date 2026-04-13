import numpy as np
import pytest
from evaluation import compute_metrics

def test_perfect_prediction():
    gt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pred = gt.copy()
    metrics = compute_metrics(pred, gt)
    assert metrics["delta1"] == pytest.approx(1.0)
    assert metrics["delta2"] == pytest.approx(1.0)
    assert metrics["delta3"] == pytest.approx(1.0)
    assert metrics["absrel"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)

def test_skip_zero_gt():
    gt = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    pred = np.array([[999.0, 2.0], [3.0, 999.0]], dtype=np.float32)
    metrics = compute_metrics(pred, gt)
    assert metrics["delta1"] == pytest.approx(1.0)
    assert metrics["absrel"] == pytest.approx(0.0)

def test_scaled_prediction():
    gt = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    pred = gt * 1.2
    metrics = compute_metrics(pred, gt)
    assert metrics["delta1"] == pytest.approx(1.0)
    assert metrics["absrel"] == pytest.approx(0.2)

def test_bad_prediction():
    gt = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    pred = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    metrics = compute_metrics(pred, gt)
    # max(2/1, 1/2) = 2.0 > 1.25, so delta1 = 0
    assert metrics["delta1"] == pytest.approx(0.0)
    # 2.0 > 1.25^2 = 1.5625, so delta2 = 0
    assert metrics["delta2"] == pytest.approx(0.0)
    # 2.0 < 1.25^3 = 1.953125, so delta3 = 0 (2.0 > 1.953)
    assert metrics["delta3"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(1.0)
