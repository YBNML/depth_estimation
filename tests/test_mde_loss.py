import torch

from mde.loss import ScaleInvariantLoss


def test_si_loss_zero_when_perfect():
    loss_fn = ScaleInvariantLoss(alpha=10.0, lambd=0.85)
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    gt = pred.clone()
    mask = torch.ones_like(pred, dtype=torch.bool)
    loss = loss_fn(pred, gt, mask)
    assert loss.item() < 1e-3


def test_si_loss_small_when_scaled():
    loss_fn = ScaleInvariantLoss(alpha=10.0, lambd=0.85)
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    gt = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
    mask = torch.ones_like(pred, dtype=torch.bool)
    loss = loss_fn(pred, gt, mask)
    # pred = gt * 0.5 같은 비율이라 scale-invariant loss는 매우 작음
    assert loss.item() < 0.01


def test_si_loss_masks_invalid():
    loss_fn = ScaleInvariantLoss(alpha=10.0, lambd=0.85)
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    gt = torch.tensor([[[[0.0, 2.0], [3.0, 0.0]]]])
    mask = gt > 0
    # mask된 픽셀만 사용 -> 완벽 일치
    loss = loss_fn(pred, gt, mask)
    assert loss.item() < 1e-3


def test_si_loss_nonzero_when_different():
    loss_fn = ScaleInvariantLoss(alpha=10.0, lambd=0.85)
    pred = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    gt = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]])
    mask = torch.ones_like(pred, dtype=torch.bool)
    loss = loss_fn(pred, gt, mask)
    # 비율이 다르므로 loss가 양수
    assert loss.item() > 0.1
