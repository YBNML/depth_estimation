from typing import List

import timm
import torch
import torch.nn as nn


class ConvNeXtV2Encoder(nn.Module):
    """ConvNeXt v2 encoder - timm 기반.

    입력: (B, 3, H, W) RGB
    출력: 4개 stage feature list (H/4, H/8, H/16, H/32)
    """

    def __init__(self, variant: str = "convnextv2_tiny", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
        )
        self.channels: List[int] = [f["num_chs"] for f in self.backbone.feature_info]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)
