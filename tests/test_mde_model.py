import pytest
import torch

from mde.model.encoder import ConvNeXtV2Encoder


def test_encoder_output_shapes():
    enc = ConvNeXtV2Encoder(variant="convnextv2_tiny", pretrained=False)
    x = torch.randn(1, 3, 352, 704)
    features = enc(x)
    assert len(features) == 4
    assert features[0].shape == (1, 96, 88, 176)   # H/4
    assert features[1].shape == (1, 192, 44, 88)   # H/8
    assert features[2].shape == (1, 384, 22, 44)   # H/16
    assert features[3].shape == (1, 768, 11, 22)   # H/32


def test_encoder_channels():
    enc = ConvNeXtV2Encoder(variant="convnextv2_tiny", pretrained=False)
    assert enc.channels == [96, 192, 384, 768]


from mde.model.ppm_head import PPMHead


def test_ppm_head_output_shape():
    ppm = PPMHead(in_channels=768, out_channels=128, pool_sizes=(1, 2, 3, 6))
    ppm.eval()
    x = torch.randn(1, 768, 11, 22)
    out = ppm(x)
    assert out.shape == (1, 128, 11, 22)


from mde.model.lwa_decoder import LWABlock


def test_lwa_block_first_stage():
    # 첫 단계: 7x7 depthwise
    block = LWABlock(local_ch=384, global_ch=128, out_ch=128, kernel_size=7)
    block.eval()
    local_feat = torch.randn(1, 384, 22, 44)
    global_feat = torch.randn(1, 128, 11, 22)
    out = block(local_feat, global_feat)
    # 2x upscale from local
    assert out.shape == (1, 128, 44, 88)


def test_lwa_block_later_stage():
    # 이후 단계: 3x3 depthwise
    block = LWABlock(local_ch=192, global_ch=128, out_ch=128, kernel_size=3)
    block.eval()
    local_feat = torch.randn(1, 192, 44, 88)
    global_feat = torch.randn(1, 128, 22, 44)
    out = block(local_feat, global_feat)
    assert out.shape == (1, 128, 88, 176)
