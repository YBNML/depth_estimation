import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF


class DepthAugmentation:
    """RGB + depth 동시 변환.

    training=True: random crop, horizontal flip, color jitter
    training=False: center crop
    출력: rgb (3,H,W) float32 [0,1], depth (1,H,W) float32
    """

    def __init__(
        self,
        crop_height: int = 352,
        crop_width: int = 704,
        training: bool = True,
    ):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.training = training

    def __call__(
        self, rgb: np.ndarray, depth: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()

        h, w = rgb_t.shape[-2:]

        if self.training:
            if h > self.crop_height and w > self.crop_width:
                top = random.randint(0, h - self.crop_height)
                left = random.randint(0, w - self.crop_width)
                rgb_t = rgb_t[:, top:top + self.crop_height, left:left + self.crop_width]
                depth_t = depth_t[:, top:top + self.crop_height, left:left + self.crop_width]

            if random.random() < 0.5:
                rgb_t = torch.flip(rgb_t, dims=[-1])
                depth_t = torch.flip(depth_t, dims=[-1])

            if random.random() < 0.5:
                rgb_t = TF.adjust_brightness(rgb_t, random.uniform(0.8, 1.2))
                rgb_t = TF.adjust_contrast(rgb_t, random.uniform(0.8, 1.2))
                rgb_t = TF.adjust_saturation(rgb_t, random.uniform(0.8, 1.2))
                rgb_t = torch.clamp(rgb_t, 0.0, 1.0)
        else:
            if h >= self.crop_height and w >= self.crop_width:
                top = (h - self.crop_height) // 2
                left = (w - self.crop_width) // 2
                rgb_t = rgb_t[:, top:top + self.crop_height, left:left + self.crop_width]
                depth_t = depth_t[:, top:top + self.crop_height, left:left + self.crop_width]

        return rgb_t, depth_t
