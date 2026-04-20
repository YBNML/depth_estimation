"""학습된 ConvNeXtMDE를 NYU Depth v2 val set (654 images) 으로 평가.

NYU 표준 벤치마크: FastDepth 전처리판의 val/ 폴더 (654 h5 files).
"""

from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation import compute_metrics
from mde.convnext_mde import ConvNeXtMDE
from mde.dataset.nyu_h5 import NYUH5Dataset
from mde.train import get_device


def evaluate_nyu(cfg: Dict[str, Any], weights_path: str) -> Dict[str, float]:
    """학습된 ConvNeXtMDE checkpoint를 NYU val set으로 평가.

    Args:
        cfg: 학습 config (nyu.yaml). max_depth, min_depth, val_dir 필수.
        weights_path: .pth checkpoint 경로.

    Returns:
        이미지별 metric의 평균 dict.
    """
    device = get_device()

    model = ConvNeXtMDE(max_depth=cfg["max_depth"], pretrained=False).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    val_ds = NYUH5Dataset(
        h5_dir=cfg["val_dir"],
        crop_height=cfg["augmentation"]["crop_height"],
        crop_width=cfg["augmentation"]["crop_width"],
        training=False,
    )
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    all_metrics = {"delta1": [], "delta2": [], "delta3": [], "absrel": [], "rmse": []}

    with torch.no_grad():
        for rgb, gt in tqdm(loader, desc="Evaluating NYU"):
            rgb = rgb.to(device)
            gt = gt.to(device)
            pred = model(rgb)

            pred_np = pred.squeeze().cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()

            metrics = compute_metrics(pred_np, gt_np, min_depth=cfg["min_depth"])
            for k, v in metrics.items():
                all_metrics[k].append(v)

    avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    return avg
