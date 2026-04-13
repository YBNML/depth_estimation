from typing import Dict
import numpy as np

def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.001) -> Dict[str, float]:
    valid = gt > min_depth
    pred_v = pred[valid]
    gt_v = gt[valid]

    if len(gt_v) == 0:
        return {"delta1": 0.0, "delta2": 0.0, "delta3": 0.0, "absrel": 0.0, "rmse": 0.0}

    thresh = np.maximum(pred_v / gt_v, gt_v / pred_v)
    delta1 = float(np.mean(thresh < 1.25))
    delta2 = float(np.mean(thresh < 1.25 * 2))
    delta3 = float(np.mean(thresh < 1.25 * 3))
    absrel = float(np.mean(np.abs(pred_v - gt_v) / gt_v))
    rmse = float(np.sqrt(np.mean((pred_v - gt_v) ** 2)))

    return {"delta1": delta1, "delta2": delta2, "delta3": delta3, "absrel": absrel, "rmse": rmse}
