#!/usr/bin/env python3
"""ConvNeXtMDE forward pass + backward pass 스모크 테스트.

실제 데이터 없이 random tensor로 한 iteration이 문제없이 돌아가는지 확인.
Mac Mini MPS 에서 메모리/속도 체크용.

주의: PyTorch 2.11 MPS 는 grouped-conv + BatchNorm backward 조합과
adaptive avg pool (non-divisible) 에 버그가 있어, MPS 실패 시 CPU로
fallback 한다. KITTI 해상도 (352x704) 대신 MPS 호환 해상도 (384x768)
를 사용한다 (PPM adaptive pool 1,2,3,6 divisor 충족).
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mde.convnext_mde import ConvNeXtMDE
from mde.loss import ScaleInvariantLoss
from mde.train import get_device


def run(device: torch.device, batch_size: int, h: int, w: int) -> float:
    model = ConvNeXtMDE(max_depth=80.0, pretrained=False).to(device)
    loss_fn = ScaleInvariantLoss(alpha=10.0, lambd=0.85)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Batch size: {batch_size}, Image: {h}x{w}")
    rgb = torch.randn(batch_size, 3, h, w, device=device)
    gt = torch.rand(batch_size, 1, h, w, device=device) * 80.0
    mask = torch.ones_like(gt, dtype=torch.bool)

    # Warmup
    for _ in range(2):
        pred = model(rgb)
        loss = loss_fn(pred, gt, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()

    t0 = time.time()
    n_iter = 5
    for _ in range(n_iter):
        pred = model(rgb)
        loss = loss_fn(pred, gt, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()
    dt = (time.time() - t0) / n_iter

    print(f"Avg iteration time: {dt*1000:.1f}ms")
    print(f"Final loss: {loss.item():.4f}")
    return dt


def main():
    device = get_device()
    print(f"Device: {device}")

    # MPS 호환 해상도 (adaptive pool divisibility)
    h, w = 384, 768
    batch_size = 4

    try:
        run(device, batch_size, h, w)
    except RuntimeError as e:
        if device.type == "mps":
            print(f"MPS run failed: {e}")
            print("Falling back to CPU.")
            run(torch.device("cpu"), batch_size=2, h=h, w=w)
        else:
            raise

    print("Smoke test PASSED.")


if __name__ == "__main__":
    main()
