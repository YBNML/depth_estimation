from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPMHead(nn.Module):
    """Pyramid Pooling Module Head.

    여러 pool size로 context feature를 추출해 concat 후 3x3 conv로 축소.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        self.pool_sizes = pool_sizes
        branch_ch = in_channels // len(pool_sizes)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            )
            for ps in pool_sizes
        ])

        fused_ch = in_channels + branch_ch * len(pool_sizes)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_ch, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [x]
        for branch in self.branches:
            y = branch(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            outs.append(y)
        return self.fuse(torch.cat(outs, dim=1))
