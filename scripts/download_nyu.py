#!/usr/bin/env python3
"""NYU Depth v2 dataset 다운로드 가이드.

원본 .mat 파일은 크기가 크고 전처리가 필요하므로,
BTS/Adabins 저장소가 제공하는 이미지 단위 전처리 버전 사용을 권장한다.

Usage:
    python scripts/download_nyu.py --output data/nyu
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="NYU Depth v2 download guide")
    parser.add_argument("--output", type=str, default="data/nyu", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print("NYU Depth v2 다운로드는 수동으로 진행을 권장합니다.")
    print()
    print("추천 소스:")
    print("  1. BTS 저장소: https://github.com/cleinc/bts (README의 데이터 링크 참조)")
    print("  2. Adabins 저장소: https://github.com/shariqfarooq123/AdaBins")
    print("  3. 공식 NYU v2: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html")
    print()
    print(f"다운로드 후 {args.output}/ 에 압축 해제하세요.")
    print()
    print("예상 디렉토리 구조:")
    print(f"  {args.output}/sync/ (RGB+depth 페어)")
    print(f"  {args.output}/official_splits/train/")
    print(f"  {args.output}/official_splits/test/")
    print()
    print("Split 파일 형식 (공백 구분):")
    print("  <rgb_path> <depth_path>")
    print()
    print("depth 파일은 .png (uint16 mm) 또는 .npy (float32 m) 지원.")


if __name__ == "__main__":
    main()
