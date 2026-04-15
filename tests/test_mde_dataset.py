import numpy as np
import pytest

from mde.dataset.transforms import DepthAugmentation


def test_augmentation_shapes_training():
    aug = DepthAugmentation(crop_height=352, crop_width=704, training=True)
    rgb = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
    depth = np.random.uniform(0, 80, (375, 1242)).astype(np.float32)
    rgb_t, depth_t = aug(rgb, depth)
    assert rgb_t.shape == (3, 352, 704)
    assert depth_t.shape == (1, 352, 704)


def test_augmentation_eval_mode():
    aug = DepthAugmentation(crop_height=352, crop_width=704, training=False)
    rgb = np.random.randint(0, 255, (352, 704, 3), dtype=np.uint8)
    depth = np.random.uniform(0, 80, (352, 704)).astype(np.float32)
    rgb_t, depth_t = aug(rgb, depth)
    assert rgb_t.shape == (3, 352, 704)
    assert depth_t.shape == (1, 352, 704)


def test_augmentation_rgb_normalized_to_0_1():
    aug = DepthAugmentation(crop_height=100, crop_width=100, training=False)
    rgb = np.full((100, 100, 3), 255, dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32)
    rgb_t, _ = aug(rgb, depth)
    import torch
    assert torch.all(rgb_t <= 1.0)
    assert torch.all(rgb_t >= 0.0)
