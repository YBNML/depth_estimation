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
    device = get_device()

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
        training=False,
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    all_metrics = {"delta1": [], "delta2": [], "delta3": [], "absrel": [], "rmse": []}

    with torch.no_grad():
        for rgb, gt in tqdm(loader, desc="Evaluating"):
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
