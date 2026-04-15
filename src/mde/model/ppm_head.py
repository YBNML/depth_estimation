"""Pyramid Pooling Module (PPM) Head.

TIE 논문 Fig. 2(a)에서 encoder 최상위 stage (H/32, 768ch) 뒤에 붙는 모듈.
서로 다른 pool size로 context feature를 추출해 global 정보를 수집한다.

논문 Section III.A 인용:
    "we integrated the pyramid pooling module (PPM) head structure from [28],
     which assists in learning appropriate local dependencies for depthwise
     convolution."
    [28] = Zhao et al., "Pyramid Scene Parsing Network," CVPR 2017.

구조:
    input (B, C_in, H, W)
        ├─ pool_size=1 -> 1x1 pool -> Conv1x1 -> BN -> ReLU -> upsample
        ├─ pool_size=2 -> 2x2 pool -> Conv1x1 -> BN -> ReLU -> upsample
        ├─ pool_size=3 -> 3x3 pool -> Conv1x1 -> BN -> ReLU -> upsample
        ├─ pool_size=6 -> 6x6 pool -> Conv1x1 -> BN -> ReLU -> upsample
        └─ original
        => concat -> Conv3x3 -> BN -> ReLU -> (B, C_out, H, W)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPMHead(nn.Module):
    """Pyramid Pooling Module Head.

    입력 feature map에서 여러 receptive field 크기로 context를 추출한 뒤,
    원본과 concat 해 하나의 글로벌 feature로 축소한다.

    Args:
        in_channels: 입력 채널 수 (ConvNeXt Tiny 기준 768).
        out_channels: 출력 채널 수 (decoder와 맞추기 위한 값, 보통 128).
        pool_sizes: adaptive pool의 출력 크기들. PSPNet의 (1,2,3,6) 기본값 사용.
            작은 값은 global context, 큰 값은 local detail을 담는다.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        self.pool_sizes = pool_sizes
        # 각 브랜치의 채널 수 (concat 후 채널 폭발 방지용 감축)
        branch_ch = in_channels // len(pool_sizes)

        # 각 pool_size 마다 adaptive pool -> 1x1 conv -> BN -> ReLU 구성
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            )
            for ps in pool_sizes
        ])

        # concat 후 fuse: 원본(in_ch) + 브랜치 4개(branch_ch * 4) -> out_ch
        fused_ch = in_channels + branch_ch * len(pool_sizes)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_ch, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W). Encoder 최상위 feature (보통 H/32 해상도).

        Returns:
            (B, C_out, H, W). 입력과 같은 spatial 크기를 유지하며 채널만 축소.
        """
        h, w = x.shape[-2:]
        # 원본 feature를 유지한 채 브랜치들과 concat
        outs = [x]
        for branch in self.branches:
            # pool -> conv -> 원본 해상도로 bilinear upsample
            y = branch(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            outs.append(y)
        # 5개 feature concat (원본 1 + 브랜치 4) 후 3x3 conv 로 fuse
        return self.fuse(torch.cat(outs, dim=1))
