"""NYU Depth v2 학습 루프 (FastDepth 전처리판 사용).

KITTI 학습과 동일한 구조이지만 dataset만 NYUH5Dataset으로 교체.
"""

import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from mde.convnext_mde import ConvNeXtMDE
from mde.dataset.nyu_h5 import NYUH5Dataset
from mde.loss import ScaleInvariantLoss
from mde.train import get_device


def train_nyu(cfg: Dict[str, Any]) -> None:
    """NYU depth estimation 학습.

    Args:
        cfg: nyu.yaml 로부터 로드된 config dict. 필수 키:
            - train_dir, val_dir: h5 파일 디렉토리
            - augmentation.crop_height, .crop_width
            - batch_size, num_workers, epochs, learning_rate, weight_decay
            - si_alpha, si_lambda, min_depth, max_depth
            - weights_dir (optional)
    """
    device = get_device()
    print(f"Device: {device}")

    train_ds = NYUH5Dataset(
        h5_dir=cfg["train_dir"],
        crop_height=cfg["augmentation"]["crop_height"],
        crop_width=cfg["augmentation"]["crop_width"],
        training=True,
    )
    val_ds = NYUH5Dataset(
        h5_dir=cfg["val_dir"],
        crop_height=cfg["augmentation"]["crop_height"],
        crop_width=cfg["augmentation"]["crop_width"],
        training=False,
    )
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    num_workers = cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # max_depth를 NYU 용 10m 으로 설정. Scaling block이 자동 대응.
    model = ConvNeXtMDE(max_depth=cfg["max_depth"], pretrained=True).to(device)
    loss_fn = ScaleInvariantLoss(alpha=cfg["si_alpha"], lambd=cfg["si_lambda"])
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    weights_dir = Path(cfg.get("weights_dir", "weights"))
    weights_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        t0 = time.time()

        for rgb, gt_depth in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            rgb = rgb.to(device)
            gt_depth = gt_depth.to(device)

            pred = model(rgb)
            mask = (gt_depth > cfg["min_depth"]) & (gt_depth < cfg["max_depth"])
            if mask.sum() == 0:
                continue

            loss = loss_fn(pred, gt_depth, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = train_loss_sum / max(n_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for rgb, gt_depth in val_loader:
                rgb = rgb.to(device)
                gt_depth = gt_depth.to(device)
                pred = model(rgb)
                mask = (gt_depth > cfg["min_depth"]) & (gt_depth < cfg["max_depth"])
                if mask.sum() == 0:
                    continue
                val_loss_sum += loss_fn(pred, gt_depth, mask).item()
                n_val += 1
        val_loss = val_loss_sum / max(n_val, 1)

        dt = time.time() - t0
        print(f"[Epoch {epoch+1}] train={train_loss:.4f} val={val_loss:.4f} "
              f"time={dt:.1f}s lr={scheduler.get_last_lr()[0]:.6f}")

        ckpt_path = weights_dir / f"convnext_mde_nyu_epoch{epoch+1:02d}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  saved -> {ckpt_path}")
