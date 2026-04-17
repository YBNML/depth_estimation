#!/usr/bin/env python3
"""NYU Depth v2 데이터셋으로 ConvNeXtMDE 학습.

Usage:
    python scripts/train_nyu.py
    python scripts/train_nyu.py --epochs 7 --batch-size 8
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mde.train_nyu import train_nyu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/training/nyu.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.train_dir is not None:
        cfg["train_dir"] = args.train_dir
    if args.val_dir is not None:
        cfg["val_dir"] = args.val_dir
    cfg["num_workers"] = args.num_workers

    train_nyu(cfg)


if __name__ == "__main__":
    main()
