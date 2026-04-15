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
