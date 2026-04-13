#!/usr/bin/env python3
"""오프라인 이미지 파일로 파이프라인을 테스트한다.

Usage:
    python scripts/test_offline.py --left path/to/left.png --right path/to/right.png
    python scripts/test_offline.py --left path/to/left.png --right path/to/right.png --gt path/to/gt.npy
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import Config
from evaluation import compute_metrics
from pipeline import DepthEstimationPipeline


def main():
    parser = argparse.ArgumentParser(description="Offline depth estimation test")
    parser.add_argument("--left", type=str, required=True, help="Left RGB image path")
    parser.add_argument("--right", type=str, required=True, help="Right RGB image path")
    parser.add_argument("--gt", type=str, default=None, help="Ground-truth depth (.npy)")
    parser.add_argument("--model", type=str, default="dummy", help="MDE model name")
    parser.add_argument("--camera", type=str, default="wide_stereo_160", help="Camera config name")
    args = parser.parse_args()

    cfg = Config(overrides={"model_name": args.model, "camera": args.camera})
    pipe = DepthEstimationPipeline(cfg)

    left_rgb = cv2.imread(args.left)
    right_rgb = cv2.imread(args.right)

    if left_rgb is None:
        print(f"Error: cannot read {args.left}")
        sys.exit(1)
    if right_rgb is None:
        print(f"Error: cannot read {args.right}")
        sys.exit(1)

    left_rgb = cv2.cvtColor(left_rgb, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_rgb, cv2.COLOR_BGR2RGB)

    result = pipe.run(left_rgb, right_rgb)

    left_depth = result["left_depth"]
    right_depth = result["right_depth"]

    if args.gt is not None:
        gt = np.load(args.gt).astype(np.float32)
        metrics = compute_metrics(left_depth, gt)
        print("=== Evaluation Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(left_rgb)
    axes[0, 0].set_title("Left RGB")
    axes[0, 1].imshow(right_rgb)
    axes[0, 1].set_title("Right RGB")
    axes[1, 0].imshow(left_depth, cmap="plasma")
    axes[1, 0].set_title("Left Depth")
    axes[1, 1].imshow(right_depth, cmap="plasma")
    axes[1, 1].set_title("Right Depth")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("output_depth.png", dpi=150)
    print("Saved: output_depth.png")
    plt.show()


if __name__ == "__main__":
    main()
