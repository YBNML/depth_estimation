#!/usr/bin/env python3
"""HuggingFace에서 sayakpaul/nyu_depth_v2 다운로드 후 tar 해제하여
NYUH5Dataset이 읽을 수 있는 디렉토리 구조로 만든다.

FastDepth 전처리판 NYU Depth v2 (~35GB, 47584 train + 654 val).

최종 구조:
    data/nyu/nyudepthv2/train/*.h5
    data/nyu/nyudepthv2/val/*.h5

Usage:
    python scripts/download_nyu_hf.py
"""
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download


OUT = Path("data/nyu")
OUT.mkdir(parents=True, exist_ok=True)

print("Downloading HuggingFace snapshot (sayakpaul/nyu_depth_v2, ~35GB)...")
local = snapshot_download(
    repo_id="sayakpaul/nyu_depth_v2",
    repo_type="dataset",
    local_dir=str(OUT / "_hf"),
    max_workers=4,
    allow_patterns=["data/*.tar"],
)
print(f"Downloaded to {local}")

# tar 파일 압축 해제
for split in ["train", "val"]:
    target = OUT / "nyudepthv2" / split
    target.mkdir(parents=True, exist_ok=True)
    tars = sorted(Path(local, "data").glob(f"{split}-*.tar"))
    print(f"Extracting {len(tars)} {split} tars -> {target}")
    for t in tars:
        with tarfile.open(t, "r") as tf:
            tf.extractall(target)
        print(f"  extracted {t.name}")

# 카운트 확인
for split in ["train", "val"]:
    n = len(list((OUT / "nyudepthv2" / split).rglob("*.h5")))
    print(f"{split}: {n} h5 files")
