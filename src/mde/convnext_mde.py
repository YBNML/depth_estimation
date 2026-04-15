from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mde.base import BaseMDE
from mde.model.encoder import ConvNeXtV2Encoder
from mde.model.lwa_decoder import LWABlock
from mde.model.ppm_head import PPMHead
from mde.model.scaling_block import ScalingBlock


class ConvNeXtMDE(nn.Module, BaseMDE):
    """TIE 논문의 fast MDE 모델.

    ConvNeXt v2 encoder + PPM head + 3 LWA decoder blocks + Scaling block.
    """

    def __init__(
        self,
        max_depth: float = 80.0,
        variant: str = "convnextv2_tiny",
        pretrained: bool = True,
        decoder_ch: int = 128,
    ):
        super().__init__()
        self._max_depth = max_depth

        self.encoder = ConvNeXtV2Encoder(variant=variant, pretrained=pretrained)
        e_ch = self.encoder.channels  # [96, 192, 384, 768]

        self.ppm = PPMHead(in_channels=e_ch[3], out_channels=decoder_ch)

        self.lwa1 = LWABlock(local_ch=e_ch[2], global_ch=decoder_ch, out_ch=decoder_ch, kernel_size=7)
        self.lwa2 = LWABlock(local_ch=e_ch[1], global_ch=decoder_ch, out_ch=decoder_ch, kernel_size=3)
        self.lwa3 = LWABlock(local_ch=e_ch[0], global_ch=decoder_ch, out_ch=decoder_ch, kernel_size=3)

        self.scaling = ScalingBlock(in_channels=decoder_ch, max_depth=max_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = self.encoder(x)

        g = self.ppm(feats[3])
        d1 = self.lwa1(feats[2], g)
        d2 = self.lwa2(feats[1], d1)
        d3 = self.lwa3(feats[0], d2)

        depth = self.scaling(d3)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        return depth

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            depth = self.forward(x)
        return depth.squeeze().cpu().numpy().astype(np.float32)

    def get_max_depth(self) -> float:
        return self._max_depth
