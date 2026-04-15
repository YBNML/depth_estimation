"""Depth estimation 정확도 메트릭.

TIE 논문 eq. (7) 및 Table II 기준 메트릭 계산. 이미지 하나 단위로 호출되며,
여러 이미지에 대한 평균은 호출부에서 집계한다.

메트릭 정의:
    delta_i (threshold accuracy):
        d_p, d_p* 각각 estimated/ground-truth depth 라 할 때,
        max(d_p / d_p*, d_p* / d_p) < 1.25^i 를 만족하는 픽셀 비율.
        delta1 = threshold 1.25
        delta2 = threshold 1.25^2 ≈ 1.5625
        delta3 = threshold 1.25^3 ≈ 1.9531

    absrel (absolute relative error):
        mean( |d_p - d_p*| / d_p* )

    rmse (root mean squared error):
        sqrt( mean( (d_p - d_p*)^2 ) )

gt = 0 인 픽셀 (LiDAR 미관측 등)은 평가에서 제외.
"""

from typing import Dict

import numpy as np


def compute_metrics(
    pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.001
) -> Dict[str, float]:
    """픽셀 단위 depth 정확도 메트릭 계산.

    Args:
        pred: (H, W) 추정 depth (meters).
        gt:   (H, W) ground-truth depth (meters). 0 또는 min_depth 이하는 invalid.
        min_depth: 이 값 이하인 gt 픽셀은 평가에서 제외.

    Returns:
        {"delta1", "delta2", "delta3", "absrel", "rmse"} 의 float dict.
        유효 픽셀이 0이면 모두 0.0 반환.
    """
    # 유효 픽셀 마스크 (gt > min_depth)
    valid = gt > min_depth
    pred_v = pred[valid]
    gt_v = gt[valid]

    if len(gt_v) == 0:
        # 유효 픽셀이 없으면 평가 불가 — 0.0으로 대체
        return {"delta1": 0.0, "delta2": 0.0, "delta3": 0.0, "absrel": 0.0, "rmse": 0.0}

    # threshold 계산: max(pred/gt, gt/pred) - 양쪽 비율 중 큰 쪽
    thresh = np.maximum(pred_v / gt_v, gt_v / pred_v)

    # 픽셀 비율 (0~1)
    delta1 = float(np.mean(thresh < 1.25))
    delta2 = float(np.mean(thresh < 1.25 ** 2))
    delta3 = float(np.mean(thresh < 1.25 ** 3))

    absrel = float(np.mean(np.abs(pred_v - gt_v) / gt_v))
    rmse = float(np.sqrt(np.mean((pred_v - gt_v) ** 2)))

    return {"delta1": delta1, "delta2": delta2, "delta3": delta3, "absrel": absrel, "rmse": rmse}
