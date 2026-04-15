#!/usr/bin/env python3
"""학습된 ConvNeXtMDE를 KITTI test set으로 평가.

Usage:
    python scripts/evaluate_mde.py --weights weights/convnext_mde_epoch25.pth
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mde.evaluate import evaluate_kitti


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/training/kitti.yaml")
    parser.add_argument("--raw-dir", type=str, default="data/kitti/raw")
    parser.add_argument("--depth-dir", type=str, default="data/kitti")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["raw_dir"] = args.raw_dir
    cfg["depth_dir"] = args.depth_dir

    metrics = evaluate_kitti(cfg, args.weights)
    print("=== KITTI Test Set Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
