"""Depth estimation 학습용 loss function.

Scale-Invariant Loss (Eigen et al., NIPS 2014)를 TIE 논문 (eq. 1) 과
동일한 형태로 구현.
"""

import torch
import torch.nn as nn


class ScaleInvariantLoss(nn.Module):
    """Scale-Invariant loss (Eigen 2014, TIE 논문 eq. (1)).

    단일 이미지 MDE 학습용. depth의 로그 차이를 이용해 스케일 불확실성을
    부분적으로 보정한다.

    수식:
        g_i   = log(pred_i) - log(gt_i)
        L = alpha * sqrt( (1/T) * Σ g_i^2  -  (lambd / T^2) * (Σ g_i)^2 )

        여기서 T = 유효 픽셀 수, alpha=10, lambd=0.85 (TIE 논문).

    lambd 해석:
        lambd=1.0  -> 완전한 scale invariance (log-depth 분산만 측정)
        lambd=0.0  -> 일반 log MSE
        lambd=0.85 -> 중간. Scale error를 약하게 penalize (논문 선택).

    Args:
        alpha: loss scaling factor. 10.0이 관례.
        lambd: scale invariance 정도 (0~1). 0.85가 TIE 논문 기본값.
        eps: 수치 안정성을 위한 작은 값. sqrt(0)과 log(0) 방지.
    """

    def __init__(self, alpha: float = 10.0, lambd: float = 0.85, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.lambd = lambd
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) 추정 depth.
            gt:   (B, 1, H, W) ground-truth depth. 0 또는 invalid 값 포함 가능.
            mask: (B, 1, H, W) bool. True 인 픽셀만 loss 계산에 포함.
                보통 (gt > min_depth) & (gt < max_depth) 로 만든다.

        Returns:
            scalar tensor. 학습 시 backward 대상.
        """
        # 마스크로 유효 픽셀만 flatten. eps 로 clamp 해서 log(0) 방지.
        pred_v = pred[mask].clamp(min=self.eps)
        gt_v = gt[mask].clamp(min=self.eps)

        # 로그 depth 차이
        g = torch.log(pred_v) - torch.log(gt_v)

        # 유효 픽셀이 없으면 0 loss (gradient 살려서 그래프 유지)
        if g.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # 원논문 eq: (1/T) Σ g_i^2 - (λ/T^2) (Σ g_i)^2
        term1 = (g ** 2).mean()
        term2 = self.lambd * (g.mean() ** 2)

        # sqrt 내부가 음수가 되는 수치 불안정 방지
        loss = torch.sqrt((term1 - term2).clamp(min=self.eps))
        return self.alpha * loss
