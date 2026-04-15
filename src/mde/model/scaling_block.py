import torch
import torch.nn as nn

from mde.model.lwa_decoder import DepthwiseSeparableConv


class HardSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + 3.0) / 6.0, 0.0, 1.0)


class ScalingBlock(nn.Module):
    """Adaptive depth scaling block.

    LWA decoder 최종 출력에서 max_depth에 맞는 depth map을 생성한다.
    7x7 depthwise -> GELU -> 3x3 separable conv -> hard sigmoid -> * max_depth
    """

    def __init__(self, in_channels: int, max_depth: float):
        super().__init__()
        self.max_depth = max_depth

        self.dw7 = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, padding=3,
            groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.gelu = nn.GELU()

        self.sep = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3)
        self.out = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.hard_sigmoid = HardSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gelu(self.bn1(self.dw7(x)))
        y = self.sep(y)
        y = self.out(y)
        y = self.hard_sigmoid(y)
        return y * self.max_depth
