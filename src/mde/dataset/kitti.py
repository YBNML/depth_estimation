from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from mde.dataset.transforms import DepthAugmentation


def _read_depth_png(path: str) -> np.ndarray:
    """KITTI depth PNG (uint16) -> meters (float32)."""
    d = np.array(Image.open(path), dtype=np.uint16)
    return d.astype(np.float32) / 256.0


class KITTIDepthDataset(Dataset):
    """KITTI Eigen split depth dataset.

    split_file 각 줄: "<seq_path> <image_idx> <side>"
      e.g. "2011_09_26/2011_09_26_drive_0001_sync 0000000000 l"

    raw_dir: KITTI raw data 루트 (예: data/kitti/raw)
    depth_dir: data_depth_annotated 압축 해제 루트 (train/, val/ 디렉토리 포함)
    """

    def __init__(
        self,
        split_file: str,
        raw_dir: str,
        depth_dir: str,
        crop_height: int = 352,
        crop_width: int = 704,
        training: bool = True,
    ):
        self.raw_dir = Path(raw_dir)
        self.depth_dir = Path(depth_dir)
        self.transform = DepthAugmentation(crop_height, crop_width, training=training)

        with open(split_file, "r") as f:
            self.samples = [line.strip().split() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def _image_paths(self, seq: str, idx: str, side: str) -> Tuple[str, str]:
        cam = "image_02" if side == "l" else "image_03"
        rgb_path = self.raw_dir / seq / cam / "data" / f"{idx}.png"
        seq_name = Path(seq).name
        depth_path = (
            self.depth_dir / "train" / seq_name /
            "proj_depth" / "groundtruth" / cam / f"{idx}.png"
        )
        if not depth_path.exists():
            depth_path = (
                self.depth_dir / "val" / seq_name /
                "proj_depth" / "groundtruth" / cam / f"{idx}.png"
            )
        return str(rgb_path), str(depth_path)

    def __getitem__(self, idx: int):
        if len(self.samples[idx]) >= 3:
            seq, img_idx, side = self.samples[idx][:3]
        else:
            seq, img_idx = self.samples[idx][:2]
            side = "l"

        rgb_path, depth_path = self._image_paths(seq, img_idx, side)

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        depth = _read_depth_png(depth_path)

        rgb_t, depth_t = self.transform(rgb, depth)
        return rgb_t, depth_t
