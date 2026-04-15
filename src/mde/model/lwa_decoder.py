import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=in_ch, bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class LWABlock(nn.Module):
    """Lightweight Attention decoder block.

    Local feature (encoder stage)와 Global feature (이전 decoder stage)를
    결합해 2x upsample된 feature를 출력한다.
    """

    def __init__(
        self,
        local_ch: int,
        global_ch: int,
        out_ch: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.local_reduce = nn.Conv2d(local_ch, out_ch, kernel_size=1, bias=False)

        self.conv_a = DepthwiseSeparableConv(out_ch + global_ch, out_ch, kernel_size=kernel_size)
        self.conv_b = DepthwiseSeparableConv(out_ch, out_ch, kernel_size=3)

        self.attn = nn.Sequential(
            DepthwiseSeparableConv(out_ch, out_ch, kernel_size=3),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        local_reduced = self.local_reduce(local_feat)
        h, w = local_reduced.shape[-2:]
        global_up = F.interpolate(global_feat, size=(h, w), mode="bilinear", align_corners=False)

        fused = torch.cat([local_reduced, global_up], dim=1)
        fused = self.conv_a(fused)
        fused = self.conv_b(fused)

        attn_map = self.attn(fused)
        out = fused * attn_map

        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        return out
