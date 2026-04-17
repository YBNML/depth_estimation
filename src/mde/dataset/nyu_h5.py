"""FastDepth 전처리판 NYU Depth v2 데이터셋 (.h5 기반).

FastDepth (MIT CSAIL) 저장소에서 제공하는 전처리된 NYU v2 데이터셋.
각 .h5 파일에 다음 키가 있음:
    rgb   : (3, H, W) uint8 (CHW 순서 주의)
    depth : (H, W)   float32 meters

디렉토리 구조:
    data/nyu/
    └── nyudepthv2/
        ├── train/  (47584 h5 files)
        └── val/    (654 h5 files, 평가용 표준 벤치마크)
"""

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from torch.utils.data import Dataset

from mde.dataset.transforms import DepthAugmentation


class NYUH5Dataset(Dataset):
    """FastDepth 형식 NYU Depth v2 .h5 dataset.

    Args:
        h5_dir: .h5 파일들이 있는 디렉토리 (예: data/nyu/nyudepthv2/train).
        crop_height, crop_width: augmentation crop 크기. NYU 표준 416x544.
        training: True = random crop + flip + color jitter, False = center crop.
    """

    def __init__(
        self,
        h5_dir: str,
        crop_height: int = 416,
        crop_width: int = 544,
        training: bool = True,
    ):
        self.h5_dir = Path(h5_dir)
        self.transform = DepthAugmentation(crop_height, crop_width, training=training)
        # 디렉토리 내 모든 .h5 파일 리스팅 (재귀)
        self.files = sorted(self.h5_dir.rglob("*.h5"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .h5 files in {h5_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns:
            (rgb_t, depth_t):
                rgb_t:   (3, H, W) float32 [0, 1]
                depth_t: (1, H, W) float32 meters
        """
        with h5py.File(self.files[idx], "r") as f:
            # FastDepth 는 CHW 순서 uint8. HWC로 변환해서 transform에 전달.
            rgb = np.array(f["rgb"]).transpose(1, 2, 0)  # (H, W, 3) uint8
            depth = np.array(f["depth"]).astype(np.float32)  # (H, W) meters

        return self.transform(rgb, depth)
