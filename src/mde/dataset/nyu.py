from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from mde.dataset.transforms import DepthAugmentation


class NYUDepthDataset(Dataset):
    """NYU Depth v2 labeled dataset.

    split_file: 각 줄이 "rgb_path depth_path" 형식.
    root_dir: NYU 데이터셋 루트.
    depth 파일은 .png (uint16, mm 단위) 또는 .npy (float32, m 단위) 지원.
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
        from PIL import Image

        rgb_path, depth_path = self.samples[idx][:2]
        rgb = np.array(Image.open(self.root / rgb_path).convert("RGB"))

        dp = self.root / depth_path
        suffix = dp.suffix.lower()
        if suffix == ".png":
            depth = np.array(Image.open(dp), dtype=np.uint16).astype(np.float32) / 1000.0
        elif suffix == ".npy":
            depth = np.load(dp).astype(np.float32)
        else:
            raise ValueError(f"Unsupported depth format: {depth_path}")

        return self.transform(rgb, depth)
