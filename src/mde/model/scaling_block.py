"""Adaptive Depth Scaling Block.

TIE 논문 Fig. 2(a) 우하단의 "Scaling Block". Decoder 최종 출력에서
max_depth에 맞는 depth map을 만든다.

TIE 논문 Section III.A 인용:
    "The major limitation of the state-of-the-art algorithms including NewCRFs
     and global-local path network (GLP) algorithm lies in its fixed estimation
     range, which is constrained to a maximum distance of 80 m in the KITTI
     dataset. ... To address this issue, we have developed a novel scaling
     block to overcome this limitation. Due to this structure, our depth model
     can learn from diverse datasets with varying maximum depths."

파이프라인:
    input feature -> 7x7 depthwise conv -> BN -> GELU
                  -> 3x3 depthwise separable conv
                  -> 1x1 conv (1ch)
                  -> hard sigmoid ([0, 1])
                  -> * max_depth (meters)
"""

import torch
import torch.nn as nn

from mde.model.lwa_decoder import DepthwiseSeparableConv


class HardSigmoid(nn.Module):
    """Hard sigmoid: clamp((x + 3) / 6, 0, 1).

    일반 sigmoid보다 빠르고 extreme 영역에서 gradient가 살아있음.
    MobileNetV3, EfficientNet 등에서 채택된 activation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + 3.0) / 6.0, 0.0, 1.0)


class ScalingBlock(nn.Module):
    """Depth map을 [0, max_depth] 범위로 생성하는 최종 head.

    다양한 max_depth (KITTI 80m, NYU 10m 등)를 지원하기 위해
    hard sigmoid 출력에 max_depth를 곱해 실제 미터 단위 depth를 만든다.

    Args:
        in_channels: 입력 feature 채널 (보통 decoder_ch=128).
        max_depth: 이 depth 이하 범위로 출력을 제한 (meters).
    """

    def __init__(self, in_channels: int, max_depth: float):
        super().__init__()
        self.max_depth = max_depth

        # 큰 receptive field를 위한 7x7 depthwise conv
        self.dw7 = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, padding=3,
            groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        # GELU: ConvNeXt와 일관성, smooth activation
        self.gelu = nn.GELU()

        self.sep = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3)
        # 1x1 conv로 채널을 1로 축소 (depth는 스칼라 맵)
        self.out = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.hard_sigmoid = HardSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W). Decoder 최종 LWA block 출력.

        Returns:
            (B, 1, H, W) float. 0 ~ max_depth 범위의 meter 단위 depth map.
        """
        y = self.gelu(self.bn1(self.dw7(x)))
        y = self.sep(y)
        y = self.out(y)
        y = self.hard_sigmoid(y)  # [0, 1]
        return y * self.max_depth  # [0, max_depth] meters
