#!/usr/bin/env python3
"""NYU Depth v2 val set (654 images) 으로 학습된 모델 평가.

Usage:
    python scripts/evaluate_nyu.py --weights weights/convnext_mde_nyu_epoch06.pth
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mde.evaluate_nyu import evaluate_nyu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/training/nyu.yaml")
    parser.add_argument("--val-dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.val_dir is not None:
        cfg["val_dir"] = args.val_dir

    metrics = evaluate_nyu(cfg, args.weights)
    print("=== NYU Depth v2 Val Set Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
