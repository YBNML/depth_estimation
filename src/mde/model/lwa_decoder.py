"""Lightweight Attention (LWA) Decoder Block.

TIE 논문 Fig. 2(a)의 "LWA Block" 구현. Decoder 단계마다 하나씩 사용.
기존 GLPDepth (Fig. 2b의 3x3 conv)를 depthwise separable conv로 경량화.

TIE 논문 Section III.A 인용:
    "In the LWA layer, we employed depthwise convolution, which was originally
     introduced in [27]. Depthwise convolution can be processed faster than
     conventional convolutions, but performance may decrease accordingly.
     In our article, we also adopted a relatively large depthwise convolution
     layer with dimensions of 7x7 in the first decoder to better capture
     local dependencies, employing a single multiplication."

연산량 비교 (논문 eq. 앞뒤):
    기존 conv:  C * H * W * 1452
    LWA layer:  C * H * W * 274  (약 5배 빠름)

Block 3개를 stacking하며 각 단계에서:
    1) encoder의 local feature를 받아 1x1 conv로 채널 축소
    2) 이전 decoder의 global feature와 concat
    3) depthwise separable conv 2단 (첫 block은 7x7, 이후는 3x3)
    4) attention gate (sigmoid 게이팅)
    5) 2x upsample

attention은 중요한 영역의 feature를 강조하기 위한 self-gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block.

    (Depthwise conv) -> (Pointwise 1x1 conv) -> BN -> LeakyReLU

    일반 conv 대비 연산량과 파라미터가 크게 줄어듦:
        일반:    K^2 * C_in * C_out * H * W
        DSC :   (K^2 * C_in + C_in * C_out) * H * W

    Args:
        in_ch: 입력 채널.
        out_ch: 출력 채널.
        kernel_size: depthwise conv 커널 크기.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        # Depthwise: 각 채널에 독립적으로 K x K conv 적용 (groups=in_ch)
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=in_ch, bias=False,
        )
        # Pointwise: 1x1 conv 로 채널 섞음
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_ch, H, W)
        Returns:
            (B, out_ch, H, W) — spatial 크기 보존.
        """
        return self.act(self.bn(self.pw(self.dw(x))))


class LWABlock(nn.Module):
    """Lightweight Attention decoder block.

    Encoder의 동일 해상도 feature (local)와 이전 decoder stage의 feature (global)를
    결합해, attention으로 가중치를 주고 2x upsample 해서 다음 stage로 전달한다.

    논문 decoder 구성에서 3개 block이 순차적으로 사용됨:
        lwa1 (kernel=7): local=encoder f2 (H/16), global=PPM (H/32)
        lwa2 (kernel=3): local=encoder f1 (H/8),  global=lwa1 output
        lwa3 (kernel=3): local=encoder f0 (H/4),  global=lwa2 output

    Args:
        local_ch: 입력 local feature 채널 (encoder stage 채널).
        global_ch: 입력 global feature 채널 (이전 decoder stage).
        out_ch: 출력 채널 (decoder 통일 채널 수, 보통 128).
        kernel_size: 첫 번째 depthwise conv 커널. 첫 block만 7, 나머지 3.
    """

    def __init__(
        self,
        local_ch: int,
        global_ch: int,
        out_ch: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        # Local feature 채널 수를 out_ch 로 맞추기 위한 1x1 reduce
        self.local_reduce = nn.Conv2d(local_ch, out_ch, kernel_size=1, bias=False)

        # concat(local_reduced + global_up) 을 받아 out_ch 로 fuse
        self.conv_a = DepthwiseSeparableConv(out_ch + global_ch, out_ch, kernel_size=kernel_size)
        # 추가 depthwise conv 로 표현력 보강
        self.conv_b = DepthwiseSeparableConv(out_ch, out_ch, kernel_size=3)

        # Attention gate: 어떤 위치의 feature를 강조할지 sigmoid 맵 생성
        self.attn = nn.Sequential(
            DepthwiseSeparableConv(out_ch, out_ch, kernel_size=3),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_feat: (B, local_ch, Hl, Wl). Encoder 동일 해상도 feature.
            global_feat: (B, global_ch, Hg, Wg). 이전 decoder stage 또는 PPM.
                Hg, Wg는 local 보다 작거나 같으며 forward 안에서 local 해상도로 upsample.

        Returns:
            (B, out_ch, 2*Hl, 2*Wl). Local 해상도의 2배로 upsample된 feature.
        """
        # 1) Local feature 채널 축소
        local_reduced = self.local_reduce(local_feat)
        h, w = local_reduced.shape[-2:]
        # 2) Global feature 를 local 해상도에 맞춤
        global_up = F.interpolate(global_feat, size=(h, w), mode="bilinear", align_corners=False)

        # 3) concat 후 두 번의 depthwise separable conv 로 fuse
        fused = torch.cat([local_reduced, global_up], dim=1)
        fused = self.conv_a(fused)
        fused = self.conv_b(fused)

        # 4) Self-attention gating
        attn_map = self.attn(fused)
        out = fused * attn_map

        # 5) 2x upsample (다음 decoder stage 또는 최종 depth head 입력용)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        return out
