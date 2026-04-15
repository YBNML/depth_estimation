"""MDE 학습 루프.

ConvNeXtMDE를 KITTI depth dataset으로 학습.

TIE 논문 학습 설정 (Section V, Table II 기준):
    - Optimizer: AdamW
    - LR scheduler: Cosine annealing (논문에서는 1e-4 start → 0)
    - Loss: Scale-Invariant (alpha=10, lambd=0.85)
    - Image crop: 352x704 (KITTI Eigen convention)
    - Epochs: 25 (논문 값)
    - Batch size: GPU 사양에 맞게 (RTX 3050 Mobile에서 ~8)

각 epoch 종료 시 checkpoint(.pth) 저장.
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
from mde.dataset.kitti import KITTIDepthDataset
from mde.loss import ScaleInvariantLoss


def get_device() -> torch.device:
    """사용 가능한 가장 빠른 device 선택.

    우선순위: CUDA (Ubuntu/Nvidia) > MPS (Mac Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: Dict[str, Any]) -> None:
    """KITTI depth estimation 학습 실행.

    Args:
        cfg: 학습 hyperparameter dict. 필수 키:
            - train_file, val_file: split 파일 경로
            - raw_dir, depth_dir: KITTI 데이터 경로
            - augmentation.crop_height, .crop_width
            - batch_size, num_workers
            - epochs, learning_rate, weight_decay
            - si_alpha, si_lambda: loss 하이퍼파라미터
            - min_depth, max_depth: 유효 depth 범위
            - weights_dir (optional): checkpoint 저장 폴더 (default "weights")
    """
    device = get_device()
    print(f"Device: {device}")

    # --- Dataset & DataLoader -------------------------------------------------
    train_ds = KITTIDepthDataset(
        split_file=cfg["train_file"],
        raw_dir=cfg["raw_dir"],
        depth_dir=cfg["depth_dir"],
        crop_height=cfg["augmentation"]["crop_height"],
        crop_width=cfg["augmentation"]["crop_width"],
        training=True,
    )
    val_ds = KITTIDepthDataset(
        split_file=cfg["val_file"],
        raw_dir=cfg["raw_dir"],
        depth_dir=cfg["depth_dir"],
        crop_height=cfg["augmentation"]["crop_height"],
        crop_width=cfg["augmentation"]["crop_width"],
        training=False,
    )

    num_workers = cfg.get("num_workers", 4)
    # pin_memory 는 CUDA일 때만 유효 (host->GPU 전송 속도 향상)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # --- Model, loss, optimizer ------------------------------------------------
    model = ConvNeXtMDE(max_depth=cfg["max_depth"], pretrained=True).to(device)
    loss_fn = ScaleInvariantLoss(alpha=cfg["si_alpha"], lambd=cfg["si_lambda"])
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    # Cosine annealing: lr이 epochs 진행에 따라 코사인으로 감소
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    weights_dir = Path(cfg.get("weights_dir", "weights"))
    weights_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---------------------------------------------------------
    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        t0 = time.time()

        for rgb, gt_depth in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            rgb = rgb.to(device)
            gt_depth = gt_depth.to(device)

            pred = model(rgb)
            # 유효 픽셀만 loss에 포함 (depth=0은 LiDAR 미관측 영역)
            mask = (gt_depth > cfg["min_depth"]) & (gt_depth < cfg["max_depth"])
            if mask.sum() == 0:
                # 이번 배치에 유효 depth가 전혀 없으면 스킵
                continue

            loss = loss_fn(pred, gt_depth, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = train_loss_sum / max(n_batches, 1)

        # --- Validation --------------------------------------------------------
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

        # 매 epoch 체크포인트 저장 (디버깅/재개용)
        ckpt_path = weights_dir / f"convnext_mde_epoch{epoch+1:02d}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  saved -> {ckpt_path}")
