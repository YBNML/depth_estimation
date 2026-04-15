"""학습된 MDE 모델 평가.

KITTI test split 이미지들에 모델을 돌려 delta/absrel/rmse 평균을 출력한다.
TIE 논문 Table II와 비교 가능한 메트릭 산출.
"""

from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation import compute_metrics
from mde.convnext_mde import ConvNeXtMDE
from mde.dataset.kitti import KITTIDepthDataset
from mde.train import get_device


def evaluate_kitti(cfg: Dict[str, Any], weights_path: str) -> Dict[str, float]:
    """학습된 ConvNeXtMDE checkpoint를 KITTI test set으로 평가.

    Args:
        cfg: 학습 config (kitti.yaml). raw_dir/depth_dir/test_file 필수.
        weights_path: .pth checkpoint 경로.

    Returns:
        이미지별 metric의 평균 dict:
            {delta1, delta2, delta3, absrel, rmse}
    """
    device = get_device()

    # pretrained=False: timm weight 다운로드 방지. weights_path에서 state_dict 로드.
    model = ConvNeXtMDE(max_depth=cfg["max_depth"], pretrained=False).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_ds = KITTIDepthDataset(
        split_file=cfg["test_file"],
        raw_dir=cfg["raw_dir"],
        depth_dir=cfg["depth_dir"],
        crop_height=cfg["augmentation"]["crop_height"],
        crop_width=cfg["augmentation"]["crop_width"],
        training=False,  # center crop only
    )
    # batch_size=1: 이미지별 metric을 따로 계산해 평균내기 위함
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    # 이미지별 metric 수집. 나중에 평균.
    all_metrics = {"delta1": [], "delta2": [], "delta3": [], "absrel": [], "rmse": []}

    with torch.no_grad():
        for rgb, gt in tqdm(loader, desc="Evaluating"):
            rgb = rgb.to(device)
            gt = gt.to(device)
            pred = model(rgb)

            # 이미지 단위로 numpy 변환 후 compute_metrics 호출
            pred_np = pred.squeeze().cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()

            metrics = compute_metrics(pred_np, gt_np, min_depth=cfg["min_depth"])
            for k, v in metrics.items():
                all_metrics[k].append(v)

    # 이미지별 metric을 평균해 최종 스칼라로 집계
    avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    return avg
