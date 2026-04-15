"""NYU Depth v2 데이터셋 로더.

실내 환경 depth estimation 학습/평가용. KITTI와 달리 RGB-D 센서 (Kinect)로
촬영되어 dense depth GT를 가진다. max_depth ≈ 10m.

Split 파일 각 줄 형식:
    "<rgb_path> <depth_path>"
    경로는 root_dir 기준 상대 경로.

Depth 파일 포맷 (확장자로 구분):
    .png : uint16, mm 단위. /1000.0 하면 meters.
    .npy : float32, 이미 meters.
"""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from mde.dataset.transforms import DepthAugmentation


class NYUDepthDataset(Dataset):
    """NYU Depth v2 labeled dataset.

    Args:
        split_file: 공백 구분 split 파일. 각 줄: "rgb_path depth_path".
        root_dir: NYU 루트. split 파일의 경로는 이 디렉토리 기준.
        crop_height: NYU 기본 사용 crop 높이 (보통 416).
        crop_width:  NYU 기본 사용 crop 너비 (보통 544).
        training: True = augmentation, False = center crop only.
    """

    def __init__(
        self,
        split_file: str,
        root_dir: str,
        crop_height: int = 416,
        crop_width: int = 544,
        training: bool = True,
    ):
        self.root = Path(root_dir)
        self.transform = DepthAugmentation(crop_height, crop_width, training=training)

        with open(split_file, "r") as f:
            self.samples = [line.strip().split() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns:
            (rgb_t, depth_t):
                rgb_t:   (3, H, W) float32 [0, 1]
                depth_t: (1, H, W) float32 meters
        """
        # Import 지연: PIL 은 dataloader worker 에서 import 되도록
        from PIL import Image

        rgb_path, depth_path = self.samples[idx][:2]
        rgb = np.array(Image.open(self.root / rgb_path).convert("RGB"))

        # 확장자로 depth 포맷 판별
        dp = self.root / depth_path
        suffix = dp.suffix.lower()
        if suffix == ".png":
            # uint16 mm -> float32 m
            depth = np.array(Image.open(dp), dtype=np.uint16).astype(np.float32) / 1000.0
        elif suffix == ".npy":
            # 이미 meters 단위 float
            depth = np.load(dp).astype(np.float32)
        else:
            raise ValueError(f"Unsupported depth format: {depth_path}")

        return self.transform(rgb, depth)
