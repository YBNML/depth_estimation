import torch
import torch.nn as nn


class ScaleInvariantLoss(nn.Module):
    """Scale-Invariant loss (Eigen et al. 2014).

    L = alpha * sqrt(var(g))
    where g_i = log(pred_i) - log(gt_i).

    This is the fully scale-invariant variant (lambd=1 in the generalized
    Eigen formulation: mean(g^2) - lambd*mean(g)^2). The ``lambd`` attribute
    is retained on the module for configuration compatibility (TIE paper
    uses lambd=0.85 as a soft weight), but the variance form is used so that
    a pure log-scale offset produces zero loss.
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

        mean_g = g.mean()
        var_g = ((g - mean_g) ** 2).mean()
        loss = torch.sqrt(var_g + self.eps) - (self.eps ** 0.5)
        return self.alpha * loss
