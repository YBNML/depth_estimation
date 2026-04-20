# Depth Estimation and Obstacle Avoidance for UAV

Wide-FOV 스테레오 카메라용 빠른 단안 깊이 추정(Monocular Depth Estimation)과 스테레오 refinement,
그리고 쿼드로터 UAV를 위한 behavior-arbitration 기반 장애물 회피 알고리즘을 제공하는 프로젝트.

## 개요 (Overview)

본 프로젝트는 다음을 제공한다:

- **ConvNeXt v2** 인코더와 직접 설계한 **Lightweight Attention (LWA) 디코더** 기반 경량 CNN 단안 깊이 추정 네트워크
- 슈퍼픽셀 기반 선형 회귀(linear regression)로 sparse stereo depth와 dense monocular depth를 융합하는 **Depth Refinement** 모듈
- Refined depth image 로부터 steering / thrust / velocity 명령을 직접 생성하는 **3D 장애물 회피** 알고리즘
- 코어 알고리즘이 ROS와 분리되어 있어 오프라인 개발/테스트와 ROS 2 + Gazebo Harmonic 연동이 모두 가능한 **모듈형 설계**

## 아키텍처 (Architecture)

```
Stereo RGB ──► Monocular Depth Estimation (ConvNeXt v2 + LWA decoder)
           │
           ├──► Rectification ──► Stereo Matching (SGBM)
           │                         │
           └──► 슈퍼픽셀 기반 Depth Refinement
                         │
                         ▼
                  Refined Depth Image
                         │
                         ▼
              Behavior Arbitration 회피 알고리즘
                         │
                         ▼
           (heading, altitude, velocity) → UAV
```

## 저장소 구조 (Repository Structure)

```
depth_estimation/
├── config/                 # YAML 설정
│   ├── camera/             # 카메라 파라미터 (wide stereo 변형)
│   ├── training/           # 학습 하이퍼파라미터
│   └── default.yaml
├── src/
│   ├── mde/                # 단안 깊이 추정
│   │   ├── model/          # Encoder, PPM head, LWA decoder, scaling block
│   │   ├── dataset/        # KITTI, NYU Depth v2 로더 + augmentation
│   │   ├── convnext_mde.py # 전체 MDE 모델
│   │   ├── loss.py         # Scale-invariant loss
│   │   ├── train.py        # KITTI 학습 루프
│   │   ├── train_nyu.py    # NYU 학습 루프
│   │   ├── evaluate.py     # KITTI test 평가
│   │   └── evaluate_nyu.py # NYU val 평가
│   ├── stereo/             # Rectification + stereo matching (Phase 3)
│   ├── refinement/         # Depth refinement (Phase 3)
│   ├── navigation/         # 장애물 회피 (Phase 4)
│   ├── pipeline.py         # End-to-end 파이프라인 오케스트레이션
│   ├── evaluation.py       # 깊이 정확도 메트릭 (delta, absrel, rmse)
│   └── config.py           # YAML config 로더
├── scripts/
│   ├── download_kitti.py
│   ├── download_kitti_full.sh
│   ├── download_nyu_hf.py
│   ├── train_kitti.py
│   ├── train_nyu.py
│   ├── evaluate_mde.py
│   ├── evaluate_nyu.py
│   ├── smoke_test_mde.py
│   └── test_offline.py
├── docs/
│   ├── algorithms/         # 알고리즘 해설 (refinement, avoidance)
│   ├── mde_survey/         # MDE 네트워크 서베이 PPT 및 빌드 스크립트
│   └── superpowers/        # 프로젝트 설계 스펙 / 구현 계획
└── tests/                  # pytest 단위 테스트
```

## 설치 (Installation)

### 1. Conda 가상환경 생성

```bash
conda create -n depth_estimation python=3.10 -y
conda activate depth_estimation
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

주요 의존성: PyTorch 2.x, timm, OpenCV, NumPy, PyYAML, h5py.

### 3. 동작 검증

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

총 35개 테스트 모두 통과해야 한다.

## 사용법 (Usage)

### 스모크 테스트 (random tensor 로 forward/backward 검증)

```bash
python scripts/smoke_test_mde.py
```

### 오프라인 파이프라인 테스트

```bash
python scripts/test_offline.py \
    --left path/to/left.png \
    --right path/to/right.png \
    --model convnext_mde
```

### KITTI 학습

KITTI depth 데이터셋 다운로드:

```bash
python scripts/download_kitti.py --output data/kitti
# 또는 전체 다운로드 (raw data 포함, ~175GB)
bash scripts/download_kitti_full.sh data/kitti
```

학습 실행:

```bash
python scripts/train_kitti.py \
    --raw-dir data/kitti/raw \
    --depth-dir data/kitti \
    --epochs 25 --batch-size 8
```

평가:

```bash
python scripts/evaluate_mde.py --weights weights/convnext_mde_epoch25.pth
```

### NYU Depth v2 학습

```bash
# HuggingFace 미러에서 FastDepth 전처리판 다운로드 (~35GB)
python scripts/download_nyu_hf.py

# 학습
python scripts/train_nyu.py --epochs 7 --batch-size 8 \
    --train-dir data/nyu/nyudepthv2/train \
    --val-dir data/nyu/nyudepthv2/val

# 평가 (표준 654 val 이미지)
python scripts/evaluate_nyu.py --weights weights/convnext_mde_nyu_epoch06.pth
```

## 단안 깊이 추정 네트워크 (Monocular Depth Estimation Network)

| 구성 | 내용 |
|------|------|
| Encoder | ConvNeXt v2 Tiny (timm), ImageNet pretrained, 약 15M params |
| Global feature | H/32 stage 위의 Pyramid Pooling Module (PPM) |
| Decoder | 3 × LWA block (첫 블록 7×7 depthwise, 이후 3×3) |
| Output head | Hard sigmoid × `max_depth` 를 곱하는 Adaptive Scaling Block |
| Loss | Scale-invariant loss (α = 10, λ = 0.85) |

학습 시 해상도: KITTI 352×704 (Eigen 관례), NYU 416×544.
추론 시 Scaling Block 덕분에 임의의 `max_depth` 에 대응 가능.

## 시뮬레이션 환경 (Simulation Environment)

Phase 5 에서 진행 예정:
- MDE / refinement / navigation 각 모듈의 **ROS 2** 노드 wrapper
- Wide-FOV 스테레오 카메라 모델이 포함된 **Gazebo Harmonic** 월드
- **PX4 SITL** 을 통한 쿼드로터 비행 통합

## 테스트 (Testing)

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

단위 테스트 범위:
- Config 로더 + camera YAML 파싱
- BaseMDE 인터페이스 + DummyMDE
- Encoder / PPM head / LWA decoder / scaling block shape 검증
- ConvNeXtMDE 전체 forward pass
- Scale-invariant loss
- Data augmentation
- 깊이 정확도 메트릭
- Pipeline 오케스트레이션

## 라이선스 (License)

MIT
