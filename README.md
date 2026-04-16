# Depth Estimation and Obstacle Avoidance for UAV

Fast monocular depth estimation with stereo refinement for wide-FOV stereo cameras,
combined with a behavior-arbitration-based obstacle avoidance algorithm for quadrotor UAVs.

## Overview

This project provides:
- A lightweight CNN for monocular depth estimation based on a **ConvNeXt v2** encoder
  and a custom **Lightweight Attention (LWA) decoder**.
- A **depth refinement** module that fuses sparse stereo-matching depth with dense
  monocular depth via superpixel-based linear regression.
- A **3D obstacle avoidance** algorithm that generates steering, thrust, and velocity
  commands directly from the refined depth image.
- A modular design where the core algorithms are decoupled from ROS, enabling offline
  development and testing as well as ROS 2 + Gazebo Harmonic integration for UAV
  simulation.

## Architecture

```
Stereo RGB ──► Monocular Depth Estimation (ConvNeXt v2 + LWA decoder)
           │
           ├──► Rectification ──► Stereo Matching (SGBM)
           │                         │
           └──► Superpixel-based Depth Refinement
                         │
                         ▼
                 Refined Depth Image
                         │
                         ▼
        Behavior Arbitration Obstacle Avoidance
                         │
                         ▼
           (steering, altitude, velocity) → UAV
```

## Repository Structure

```
depth_estimation/
├── config/                 # YAML configurations
│   ├── camera/             # Camera parameter sets (wide stereo variants)
│   ├── training/           # Training hyperparameters
│   └── default.yaml
├── src/
│   ├── mde/                # Monocular Depth Estimation
│   │   ├── model/          # Encoder, PPM head, LWA decoder, scaling block
│   │   ├── dataset/        # KITTI, NYU Depth v2 loaders + augmentation
│   │   ├── convnext_mde.py # Full MDE model
│   │   ├── loss.py         # Scale-invariant loss
│   │   ├── train.py        # Training loop
│   │   └── evaluate.py     # KITTI test-set evaluation
│   ├── stereo/             # Rectification + stereo matching (Phase 3)
│   ├── refinement/         # Depth refinement (Phase 3)
│   ├── navigation/         # Obstacle avoidance (Phase 4)
│   ├── pipeline.py         # End-to-end pipeline orchestration
│   ├── evaluation.py       # Depth accuracy metrics (delta, absrel, rmse)
│   └── config.py           # YAML config loader
├── scripts/
│   ├── download_kitti.py
│   ├── download_nyu.py
│   ├── train_kitti.py
│   ├── evaluate_mde.py
│   ├── smoke_test_mde.py
│   └── test_offline.py
└── tests/                  # pytest unit tests
```

## Installation

### 1. Create a conda environment

```bash
conda create -n depth_estimation python=3.10 -y
conda activate depth_estimation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies: PyTorch 2.x, timm, OpenCV, NumPy, PyYAML, h5py.

### 3. Verify

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

All 35 tests should pass.

## Usage

### Smoke test (forward/backward on random tensors)

```bash
python scripts/smoke_test_mde.py
```

### Offline pipeline test

```bash
python scripts/test_offline.py \
    --left path/to/left.png \
    --right path/to/right.png \
    --model convnext_mde
```

### Train on KITTI

Download the KITTI depth dataset:

```bash
python scripts/download_kitti.py --output data/kitti
```

Then launch training:

```bash
python scripts/train_kitti.py \
    --raw-dir data/kitti/raw \
    --depth-dir data/kitti \
    --epochs 25 --batch-size 8
```

### Evaluate on KITTI

```bash
python scripts/evaluate_mde.py --weights weights/convnext_mde_epoch25.pth
```

Outputs `delta1/2/3`, `absrel`, and `rmse` on the KITTI test set.

## Monocular Depth Estimation Network

| Component      | Details |
| -------------- | ------- |
| Encoder        | ConvNeXt v2 Tiny (timm), ImageNet pretrained, ~15M params |
| Global feature | Pyramid Pooling Module (PPM) on the H/32 stage |
| Decoder        | 3 × LWA blocks (first with 7×7 depthwise, others 3×3) |
| Output head    | Adaptive Scaling Block with hard sigmoid × `max_depth` |
| Loss           | Scale-invariant loss (α = 10, λ = 0.85) |

Input resolution during training: 352 × 704 (KITTI Eigen convention).
Supports arbitrary `max_depth` at inference time via the scaling block.

## Simulation Environment

Planned for Phase 5:
- **ROS 2** node wrappers for the MDE, refinement, and navigation modules.
- **Gazebo Harmonic** world with a wide-FOV stereo camera model.
- **PX4 SITL** integration for quadrotor flight.

## Testing

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Unit tests cover:
- Config loader + camera YAML parsing
- BaseMDE interface + DummyMDE
- Encoder / PPM head / LWA decoder / scaling block shapes
- Full MDE model forward pass
- Scale-invariant loss
- Data augmentation transforms
- Depth accuracy metrics
- Pipeline orchestration

## License

MIT
