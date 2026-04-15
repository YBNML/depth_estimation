"""ConvNeXt v2 백본 인코더.

TIE 논문 (Cho et al., 2025) Fig. 2(a)의 "Overall architecture" 왼쪽 부분에 해당.
timm 라이브러리에서 ImageNet pretrained ConvNeXt v2 Tiny를 로드하여
4개 stage의 feature map을 추출한다.

TIE 논문 Section III.A 인용:
    "we utilized the encoder block employing the ConvNeXt v2 model.
     The stem block in the encoder downsamples the input images to a proper
     feature map size. The stem block comprises a 4x4 convolutional layer
     with a stride of 4, which results in a 4x downsampling of the input image."

각 stage에서 입력 해상도 대비:
    stage 0: H/4,  96 ch
    stage 1: H/8,  192 ch
    stage 2: H/16, 384 ch
    stage 3: H/32, 768 ch
"""

from typing import List

import timm
import torch
import torch.nn as nn


class ConvNeXtV2Encoder(nn.Module):
    """ConvNeXt v2 backbone을 multi-stage feature extractor로 래핑.

    timm의 `features_only=True` 옵션으로 4 stage 출력을 모두 받는다.
    ConvNeXt v2 Tiny 기준 약 15M 파라미터.

    Args:
        variant: timm 모델 이름 (예: "convnextv2_tiny", "convnextv2_base").
        pretrained: ImageNet pretrained weight 로드 여부. 학습 시 True,
            테스트 시 False로 두면 빠르게 로딩 가능.

    Attributes:
        backbone: timm feature extractor.
        channels: 각 stage 출력 채널 수 리스트. e.g. [96, 192, 384, 768].
    """

    def __init__(self, variant: str = "convnextv2_tiny", pretrained: bool = True):
        super().__init__()
        # timm feature extractor: 각 stage 출력을 list로 반환
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
        )
        # feature_info 는 stage별 {num_chs, reduction} 정보
        self.channels: List[int] = [f["num_chs"] for f in self.backbone.feature_info]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) RGB 이미지. 0-1 정규화 가정.
                H, W는 32의 배수여야 함 (stage 3에서 H/32 해상도).

        Returns:
            4개 텐서 리스트. shape 예시 (input 352x704 기준):
                [0] (B,  96,  88, 176)   # H/4
                [1] (B, 192,  44,  88)   # H/8
                [2] (B, 384,  22,  44)   # H/16
                [3] (B, 768,  11,  22)   # H/32
        """
        return self.backbone(x)
