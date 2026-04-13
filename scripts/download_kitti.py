#!/usr/bin/env python3
"""KITTI depth estimation 데이터셋을 다운로드한다.

Usage:
    python scripts/download_kitti.py --output data/kitti
    python scripts/download_kitti.py --output data/kitti --split-only
"""
import argparse
import os
import subprocess
import sys


KITTI_DEPTH_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"

EIGEN_TRAIN_URL = (
    "https://raw.githubusercontent.com/nianticlabs/monodepth2/master/splits/eigen_full/train_files.txt"
)
EIGEN_VAL_URL = (
    "https://raw.githubusercontent.com/nianticlabs/monodepth2/master/splits/eigen_full/val_files.txt"
)
EIGEN_TEST_URL = (
    "https://raw.githubusercontent.com/nianticlabs/monodepth2/master/splits/eigen/test_files.txt"
)


def download_file(url: str, output_path: str) -> None:
    print(f"Downloading: {url}")
    print(f"       -> {output_path}")
    subprocess.run(["curl", "-L", "-o", output_path, url], check=True)


def download_splits(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    download_file(EIGEN_TRAIN_URL, os.path.join(output_dir, "eigen_train_files.txt"))
    download_file(EIGEN_VAL_URL, os.path.join(output_dir, "eigen_val_files.txt"))
    download_file(EIGEN_TEST_URL, os.path.join(output_dir, "eigen_test_files.txt"))
    print("Eigen split files downloaded.")


def download_depth_annotations(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "data_depth_annotated.zip")
    download_file(KITTI_DEPTH_URL, zip_path)
    print("Extracting depth annotations...")
    subprocess.run(["unzip", "-o", zip_path, "-d", output_dir], check=True)
    os.remove(zip_path)
    print("Depth annotations extracted.")


def main():
    parser = argparse.ArgumentParser(description="Download KITTI dataset")
    parser.add_argument("--output", type=str, default="data/kitti", help="Output directory")
    parser.add_argument("--split-only", action="store_true", help="Download only split files")
    parser.add_argument("--depth-only", action="store_true", help="Download only depth annotations")
    args = parser.parse_args()

    if args.split_only:
        download_splits(args.output)
        return

    if args.depth_only:
        download_depth_annotations(args.output)
        return

    download_splits(args.output)
    download_depth_annotations(args.output)

    print("\n=== KITTI raw data ===")
    print("KITTI raw data는 용량이 크므로 수동 다운로드를 권장합니다.")
    print("필요한 시퀀스만 선택적으로 다운로드하세요.")


if __name__ == "__main__":
    main()
