#!/usr/bin/env python3
"""KITTI 데이터셋으로 ConvNeXtMDE 학습.

Usage:
    python scripts/train_kitti.py
    python scripts/train_kitti.py --epochs 5 --batch-size 8   # Mac Mini 스모크 테스트
    python scripts/train_kitti.py --epochs 25 --batch-size 8  # RTX 5070 본격 학습
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mde.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/training/kitti.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--raw-dir", type=str, default="data/kitti/raw")
    parser.add_argument("--depth-dir", type=str, default="data/kitti")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    cfg["raw_dir"] = args.raw_dir
    cfg["depth_dir"] = args.depth_dir
    cfg["num_workers"] = args.num_workers

    train(cfg)


if __name__ == "__main__":
    main()
