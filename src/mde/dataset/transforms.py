"""Depth 학습용 data augmentation.

RGB 이미지와 depth ground-truth를 동시에 변환한다.
flip, crop 같은 기하 변환은 두 입력 모두에 적용하고, color jitter 는 RGB에만
적용한다 (depth는 시각 속성에 영향받지 않음).
"""

import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF


class DepthAugmentation:
    """RGB + depth 동시 변환 transform.

    Training 모드:
        1) Random crop (crop_height x crop_width)
        2) Horizontal flip (50%)
        3) Color jitter: brightness/contrast/saturation (50%)

    Eval 모드:
        1) Center crop (입력이 크면)

    출력:
        rgb:   (3, H, W) float32, 값 [0, 1]
        depth: (1, H, W) float32, 값 meters (변환 없음)

    Args:
        crop_height: 학습/평가 공통 crop 높이.
        crop_width: 학습/평가 공통 crop 너비.
        training: True 이면 random crop + flip + color jitter,
            False 이면 center crop 만.
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
        """
        Args:
            rgb: (H, W, 3) uint8 RGB numpy array.
            depth: (H, W) float32 depth numpy array (meters).

        Returns:
            rgb_t: (3, H', W') float32 tensor, [0, 1].
            depth_t: (1, H', W') float32 tensor.
        """
        # HWC uint8 [0, 255] -> CHW float [0, 1]
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        # depth에 채널 축 추가: (H, W) -> (1, H, W)
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()

        h, w = rgb_t.shape[-2:]

        if self.training:
            # 1) Random crop — 원본이 crop 크기보다 크면 랜덤 위치에서 잘라냄
            if h > self.crop_height and w > self.crop_width:
                top = random.randint(0, h - self.crop_height)
                left = random.randint(0, w - self.crop_width)
                rgb_t = rgb_t[:, top:top + self.crop_height, left:left + self.crop_width]
                depth_t = depth_t[:, top:top + self.crop_height, left:left + self.crop_width]

            # 2) Horizontal flip (50%) — RGB와 depth 모두 뒤집음
            if random.random() < 0.5:
                rgb_t = torch.flip(rgb_t, dims=[-1])
                depth_t = torch.flip(depth_t, dims=[-1])

            # 3) Color jitter (50%) — depth는 손대지 않음
            if random.random() < 0.5:
                rgb_t = TF.adjust_brightness(rgb_t, random.uniform(0.8, 1.2))
                rgb_t = TF.adjust_contrast(rgb_t, random.uniform(0.8, 1.2))
                rgb_t = TF.adjust_saturation(rgb_t, random.uniform(0.8, 1.2))
                # jitter 후 [0, 1] 범위 벗어날 수 있으므로 clip
                rgb_t = torch.clamp(rgb_t, 0.0, 1.0)
        else:
            # Eval: center crop only. 모델은 항상 고정 크기 입력 기대.
            if h >= self.crop_height and w >= self.crop_width:
                top = (h - self.crop_height) // 2
                left = (w - self.crop_width) // 2
                rgb_t = rgb_t[:, top:top + self.crop_height, left:left + self.crop_width]
                depth_t = depth_t[:, top:top + self.crop_height, left:left + self.crop_width]

        return rgb_t, depth_t
