import torch
import torch.nn as nn


class ScaleInvariantLoss(nn.Module):
    """Scale-Invariant loss (Eigen et al. 2014, TIE paper eq. 1).

    L = alpha * sqrt( (1/T) * sum(g_i^2) - (lambd/T^2) * (sum g_i)^2 )
    where g_i = log(pred_i) - log(gt_i).

    TIE paper uses alpha=10.0, lambd=0.85. With lambd<1, a pure log-scale
    offset does not cancel out completely — this is intentional so the model
    is lightly penalized for scale errors.
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
        pred_v = pred[mask].clamp(min=self.eps)
        gt_v = gt[mask].clamp(min=self.eps)

        g = torch.log(pred_v) - torch.log(gt_v)
        if g.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        term1 = (g ** 2).mean()
        term2 = self.lambd * (g.mean() ** 2)
        loss = torch.sqrt((term1 - term2).clamp(min=self.eps))
        return self.alpha * loss
