"""ConvNeXtMDE: TIE 논문의 Fast Monocular Depth Estimation 전체 모델.

TIE 논문 Fig. 2(a) 전체 구조를 하나로 조립.

아키텍처:
    Input RGB (B, 3, H, W)
         |
         v
    [ConvNeXt v2 Encoder]  → 4개 stage feature (H/4, H/8, H/16, H/32)
         |
         v
    [PPM Head] (H/32 feature → decoder_ch 글로벌 feature)
         |
         v
    [LWA Block 1] (7x7, local=H/16 + global=PPM)   → H/8 feature
    [LWA Block 2] (3x3, local=H/8  + global=lwa1)  → H/4 feature
    [LWA Block 3] (3x3, local=H/4  + global=lwa2)  → H/2 feature
         |
         v
    [Scaling Block] → depth (B, 1, H/2, W/2), [0, max_depth]
         |
         v
    [bilinear interpolation] → (B, 1, H, W) 원 해상도 depth

TIE 논문 Table I 참고한 컴퓨팅 시간 (laptop RTX 3050):
    Monocular depth: 42.9 ms
    전체 파이프라인 (stereo 포함): 98.2 ms
"""

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
    """TIE 논문 (Cho et al., IEEE TIE 2025) 의 fast MDE 모델.

    ConvNeXt v2 encoder + PPM head + 3 LWA decoder blocks + Scaling block.
    ImageNet pretrained backbone + 경량 decoder로 RTX 3050 모바일에서
    ~20-45ms/image 추론.

    Args:
        max_depth: 출력 depth 범위 상한 (meters). KITTI=80, NYU=10.
        variant: timm ConvNeXt v2 변종. "convnextv2_tiny" (15M) 권장.
        pretrained: ImageNet pretrained weight 사용 여부.
        decoder_ch: decoder 전체의 통일 채널 수. 128이 논문 default.
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

        # 1) Encoder: multi-stage feature 추출
        self.encoder = ConvNeXtV2Encoder(variant=variant, pretrained=pretrained)
        # Tiny 기준 [96, 192, 384, 768]
        e_ch = self.encoder.channels

        # 2) PPM Head: 최상위 stage (H/32, 768) -> global context (decoder_ch)
        self.ppm = PPMHead(in_channels=e_ch[3], out_channels=decoder_ch)

        # 3) LWA Decoder blocks (bottom-up)
        # 첫 블록만 kernel_size=7로 넓은 receptive field 확보 (논문 권고)
        self.lwa1 = LWABlock(local_ch=e_ch[2], global_ch=decoder_ch, out_ch=decoder_ch, kernel_size=7)
        self.lwa2 = LWABlock(local_ch=e_ch[1], global_ch=decoder_ch, out_ch=decoder_ch, kernel_size=3)
        self.lwa3 = LWABlock(local_ch=e_ch[0], global_ch=decoder_ch, out_ch=decoder_ch, kernel_size=3)

        # 4) Scaling Block: feature -> depth map [0, max_depth]
        self.scaling = ScalingBlock(in_channels=decoder_ch, max_depth=max_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training/inference 공통 forward.

        Args:
            x: (B, 3, H, W) RGB 텐서. 값 범위 [0, 1] 가정.
                H, W는 32의 배수 권장 (encoder stride 제약).

        Returns:
            (B, 1, H, W) depth map. 값 범위 [0, max_depth] meters.
        """
        h, w = x.shape[-2:]

        # Encoder 4 stage feature
        feats = self.encoder(x)
        # feats[0]: H/4, feats[1]: H/8, feats[2]: H/16, feats[3]: H/32

        # PPM: 최상위 stage에서 글로벌 context 추출
        g = self.ppm(feats[3])  # (B, decoder_ch, H/32, W/32)

        # Decoder: 해상도를 단계적으로 2배씩 늘려감
        d1 = self.lwa1(feats[2], g)   # (B, decoder_ch, H/8,  W/8)
        d2 = self.lwa2(feats[1], d1)  # (B, decoder_ch, H/4,  W/4)
        d3 = self.lwa3(feats[0], d2)  # (B, decoder_ch, H/2,  W/2)

        # Scaling: feature -> depth (H/2 해상도)
        depth = self.scaling(d3)
        # 원 해상도로 upsample
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        return depth

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """BaseMDE 인터페이스: numpy RGB → numpy depth (inference 편의용).

        Args:
            rgb: (H, W, 3) uint8 RGB 이미지.

        Returns:
            (H, W) float32 depth map (meters). 자동으로 현재 device에서 실행,
            결과는 CPU numpy로 반환.
        """
        device = next(self.parameters()).device
        # HWC uint8 -> (1, 3, H, W) float32 [0, 1]
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            depth = self.forward(x)
        # (1, 1, H, W) -> (H, W) numpy
        return depth.squeeze().cpu().numpy().astype(np.float32)

    def get_max_depth(self) -> float:
        """BaseMDE 인터페이스: 모델이 출력 가능한 최대 depth (meters)."""
        return self._max_depth
