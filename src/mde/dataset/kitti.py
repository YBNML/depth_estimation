"""KITTI depth estimation 데이터셋 로더.

TIE 논문의 학습 기본 데이터셋 (Section V.A). Eigen split 기준 ~43k 학습 이미지.

데이터 구조 (기본 가정):
    data/kitti/
    ├── raw/                               # KITTI raw data (수동 다운로드)
    │   └── 2011_09_26/
    │       └── 2011_09_26_drive_0001_sync/
    │           ├── image_02/data/*.png   # 좌측 RGB
    │           └── image_03/data/*.png   # 우측 RGB
    ├── train/                             # data_depth_annotated.zip 압축 해제
    │   └── 2011_09_26_drive_0001_sync/
    │       └── proj_depth/groundtruth/image_02/*.png  # 좌측 depth
    ├── val/                               # 동일 구조
    ├── eigen_train_files.txt              # split (scripts/download_kitti.py 생성)
    ├── eigen_val_files.txt
    └── eigen_test_files.txt

Split 파일 각 줄 형식:
    "<seq_path> <image_idx> <side>"
    예: "2011_09_26/2011_09_26_drive_0001_sync 0000000000 l"

Depth PNG 포맷:
    uint16 값을 256.0으로 나누면 meter 단위 depth.
    값 0은 invalid (LiDAR 미관측 영역).
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from mde.dataset.transforms import DepthAugmentation


def _read_depth_png(path: str) -> np.ndarray:
    """KITTI depth PNG (uint16) -> meters (float32).

    KITTI depth map은 값에 256을 곱해 uint16으로 저장한다.
    (https://github.com/cleinc/bts/blob/master/README.md 참고)
    """
    d = np.array(Image.open(path), dtype=np.uint16)
    return d.astype(np.float32) / 256.0


class KITTIDepthDataset(Dataset):
    """KITTI Eigen split depth dataset.

    Args:
        split_file: Eigen split 파일 경로 (train/val/test).
        raw_dir: KITTI raw data 루트 (예: data/kitti/raw).
        depth_dir: data_depth_annotated 압축 해제 루트 (train/, val/ 포함).
        crop_height: augmentation crop 높이. TIE 논문 352.
        crop_width:  augmentation crop 너비. TIE 논문 704.
        training: True = 학습 augmentation, False = center crop만.
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

        # split 파일 로드: 빈 줄 무시, 공백으로 필드 분리
        with open(split_file, "r") as f:
            self.samples = [line.strip().split() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def _image_paths(self, seq: str, idx: str, side: str) -> Tuple[str, str]:
        """Split 필드 -> 실제 파일 경로 변환.

        Args:
            seq: 예) "2011_09_26/2011_09_26_drive_0001_sync"
            idx: 예) "0000000000"
            side: "l" 또는 "r"

        Returns:
            (rgb_path, depth_path) 문자열 페어.
        """
        # 좌측 카메라: image_02, 우측: image_03
        cam = "image_02" if side == "l" else "image_03"
        rgb_path = self.raw_dir / seq / cam / "data" / f"{idx}.png"

        # depth annotation 경로는 sequence 이름만 사용 (날짜 prefix 빠짐)
        seq_name = Path(seq).name
        depth_path = (
            self.depth_dir / "train" / seq_name /
            "proj_depth" / "groundtruth" / cam / f"{idx}.png"
        )
        # train/ 에 없으면 val/ 에서 찾음 (Eigen val split 때문)
        if not depth_path.exists():
            depth_path = (
                self.depth_dir / "val" / seq_name /
                "proj_depth" / "groundtruth" / cam / f"{idx}.png"
            )
        return str(rgb_path), str(depth_path)

    def __getitem__(self, idx: int):
        """
        Returns:
            (rgb_t, depth_t) — DepthAugmentation 출력 tensor.
                rgb_t:   (3, H, W) float32, [0, 1]
                depth_t: (1, H, W) float32 meters.
        """
        # split line 형식이 2필드 (seq idx) 또는 3필드 (seq idx side) 모두 지원
        if len(self.samples[idx]) >= 3:
            seq, img_idx, side = self.samples[idx][:3]
        else:
            seq, img_idx = self.samples[idx][:2]
            side = "l"  # 기본: 좌측 카메라

        rgb_path, depth_path = self._image_paths(seq, img_idx, side)

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        depth = _read_depth_png(depth_path)

        rgb_t, depth_t = self.transform(rgb, depth)
        return rgb_t, depth_t
