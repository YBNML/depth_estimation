# Depth Estimation & Obstacle Avoidance 재구현 설계 문서

## 개요 (Overview)

두 논문의 알고리즘을 ROS2 기반으로 새롭게 구현하는 프로젝트.

- **졸업논문 (2022):** "Refined Depth Estimation and Safety Navigation with a Binocular Camera"
- **TIE 논문 (2025):** "Obstacle Avoidance of a UAV Using Fast Monocular Depth Estimation for a Wide Stereo Camera"

TIE 논문의 개선된 알고리즘에 초점을 맞추되, 졸업논문의 알고리즘도 옵션으로 포함.

## 아키텍처 (Architecture)

**방식: 코어 라이브러리 + ROS2 래퍼 분리**

- 알고리즘 코어: 순수 Python + C++ (pybind11) 패키지 (ROS 의존 없음)
- ROS2 노드: 코어를 import해서 래핑
- Mac Mini에서 알고리즘 개발/테스트, Ubuntu에서 ROS2 + PX4 + Gz Harmonic 연동

**언어 분담:**
- Python: MDE 네트워크 학습/추론 (PyTorch), 데이터 로딩, 설정 관리, 시각화, ROS2 래퍼
- C++ (pybind11): 슈퍼픽셀 depth 통계, depth refinement 연산, navigation command 계산
- OpenCV C++ backend (Python 호출): SGBM 스테레오 매칭, rectification, 슈퍼픽셀 분할

## 프로젝트 구조 (Project Structure)

```
depth_estimation/
├── config/
│   ├── camera/                  # 카메라 파라미터 YAML (FOV, baseline, intrinsic)
│   └── training/                # 학습 하이퍼파라미터 YAML
├── src/
│   ├── mde/                     # Monocular Depth Estimation (Python/PyTorch)
│   │   ├── model/               # ConvNeXt v2 encoder + LWA decoder + scaling block
│   │   ├── train.py
│   │   ├── infer.py
│   │   ├── loss.py              # Scale-invariant loss
│   │   └── dataset.py           # KITTI, NYU dataloader
│   ├── stereo/                  # 스테레오 매칭 (Python - OpenCV backend)
│   │   ├── rectification.py
│   │   └── matching.py          # SGBM / Libelas 래퍼
│   ├── refinement/              # Depth Refinement (C++ core + pybind11)
│   │   ├── cpp/
│   │   │   ├── superpixel_stats.cpp
│   │   │   ├── depth_refine.cpp
│   │   │   └── bindings.cpp
│   │   └── refine.py            # Python 인터페이스
│   ├── navigation/              # Obstacle Avoidance (C++ core + pybind11)
│   │   ├── cpp/
│   │   │   ├── avoidance.cpp    # Algorithm 1
│   │   │   └── bindings.cpp
│   │   └── navigator.py         # Python 인터페이스
│   ├── pipeline.py              # 전체 파이프라인 오케스트레이션
│   └── evaluation.py            # 정확도 평가 (delta, Absrel, RMSE)
├── ros2_ws/
│   └── src/
│       └── depth_avoidance/     # ROS2 패키지
│           ├── depth_node.py
│           ├── nav_node.py
│           └── launch/
├── scripts/
│   ├── download_dataset.py
│   └── test_offline.py
├── CMakeLists.txt               # C++ pybind11 빌드
├── setup.py
└── requirements.txt
```

## 모듈 설계 (Module Design)

### 1. MDE 네트워크 (TIE 논문 기준)

**교체 가능한 인터페이스:**
```python
class BaseMDE:
    def predict(self, rgb: np.ndarray) -> np.ndarray: ...
    def get_max_depth(self) -> float: ...

class ConvNeXtMDE(BaseMDE): ...   # TIE 논문
class AdabinsMDE(BaseMDE): ...    # 졸업논문
class CustomMDE(BaseMDE): ...     # 향후 새 네트워크
```

**ConvNeXt v2 + LWA Decoder 구조:**
- Encoder: ConvNeXt v2 (Tiny, 15M params), pretrained backbone
  - Stem block: 4x4 conv, stride 4 (4배 다운샘플링)
  - 4 stage 출력: H/4, H/8, H/16, H/32
- Decoder: 3개 LWA block
  - 7x7 depthwise separable conv (1단계만)
  - 3x3 depthwise separable conv
  - Local + Global feature 결합 (skip connection)
  - PPM (Pyramid Pooling Module) Head
- Scaling Block: 가변 max depth 대응
  - 7x7 depthwise conv -> GELU -> separable conv 3x3 -> hard sigmoid
- Loss: Scale-Invariant loss (alpha=10, lambda=0.85)

**연산량 비교 (논문):** 기존 대비 약 5배 빠름 (fconv_t = C*H*W*1452 vs four_t = C*H*W*274)

### 2. 깊이 보정 (Depth Refinement)

**TIE 논문 방식 (linear regression, 기본):**
1. Rectification: 두 이미지 정렬 (cv2.stereoRectify + cv2.remap)
2. 스테레오 매칭: SGBM으로 겹치는 영역에서 sparse depth 획득
3. 슈퍼픽셀 분할: SEEDS 알고리즘으로 RGB 이미지 그룹핑
4. Outlier 제거:
   - superpixel 내 depth 변화가 큰 그룹 제외
   - MDE와 stereo depth 50% 이상 차이나는 데이터 제외
5. Linear regression: d = sd * d_hat + td (계수 추정)
6. Temporal smoothing: sd, td에 1차 필터 적용

**졸업논문 방식 (비율 스케일링, 옵션):**
- d_ref,i = r^g * (sum(d_stereo,c) / sum(d_mono,c)) * d_mono,i
- r^g = 1 if d_max/d_min < 1.5, else 0

### 3. 장애물 회피 (Obstacle Avoidance) — TIE 논문 Algorithm 1

**Steering/Thrust (eq.4-6):**
- Exponential depth: D_i = exp(dc - d_i) * N*Sn / (H*W)
- Horizontal weight: W_h = exp(-(u_i / HFOV)^2)
- Vertical weight: W_v = exp(-(v_i / VFOV)^2)
- Steering: delta_h = phi_h * exp(-(phi_h / Ga)^2)
- Altitude: delta_v = phi_v * exp(-(phi_v / Ga)^2)

**Target Point Navigation (Algorithm 1):**
- get-steering-and-collision-prob: delta_h, delta_v, p_coll 계산
- command-for-desired-location: 목표점 + 회피 명령 합성
- get-weight: 회피 각도 크기에 따라 가중치 조절
- Collision probability 기반 속도 조절: vd = Vmax * (1 - p_coll)

### 4. 카메라 구성

파라미터로 설정 가능:
- 졸업논문: 각 82도 FOV, 총 112도, 15도 외향 배치
- TIE 논문: 각 110도 FOV, 총 160도
- config YAML로 FOV, baseline, intrinsic, extrinsic 관리

## 데이터 파이프라인 및 학습 (Data Pipeline & Training)

**데이터셋:**
- KITTI: outdoor, max 80m, Eigen split (~43k images)
- NYU Depth v2: indoor, max 10m (~50k images)

**학습:**
- Augmentation: horizontal flip, color jitter, random crop
- Optimizer: AdamW
- Scheduler: Cosine annealing
- Scaling block으로 다른 max depth 데이터셋 혼합 학습 가능

**평가 메트릭:**
- delta_1 (1.25), delta_2 (1.25^2), delta_3 (1.25^3)
- Absrel, RMSE

## 시뮬레이션 환경 (Simulation Environment)

- ROS2 + Gz Harmonic + PX4 SITL
- rotors_simulator 대신 PX4가 Gz Harmonic 공식 지원
- ros_gz 브릿지로 ROS2 <-> Gz 토픽 연동
- Wide stereo 카메라 모델을 SDF로 정의
- 논문의 25m x 25m 실내 환경을 SDF로 새로 구축

## 개발 단계 (Development Phases)

### Phase 1: 프로젝트 골격 + 오프라인 테스트 환경
- 디렉토리 구조, config 시스템
- BaseMDE 인터페이스
- 데이터셋 다운로드 스크립트
- 오프라인 테스트 프레임워크

### Phase 2: MDE 네트워크
- ConvNeXt v2 encoder
- LWA decoder + PPM Head
- Scaling block
- Scale-Invariant loss
- 학습 파이프라인, KITTI 학습 및 평가

### Phase 3: Depth Refinement
- Rectification, 스테레오 매칭 (SGBM)
- 슈퍼픽셀 + depth 통계 (C++/pybind11)
- Linear regression refinement (C++/pybind11)
- MDE only vs MDE + refinement 비교 평가

### Phase 4: Obstacle Avoidance
- Avoidance command (C++/pybind11)
- Algorithm 1 구현
- 오프라인 테스트 (depth -> command 시각화)

### Phase 5: ROS2 + PX4 + Gz Harmonic 연동
- ROS2 패키지, Gz 월드 + 카메라 모델
- PX4 SITL 연동, 통합 비행 테스트

### Phase 6: 성능 최적화 + 추가 실험
- 연산 시간 프로파일링
- 다양한 환경 실험
- 졸업논문 vs TIE 알고리즘 비교

**Phase 1-4: Mac Mini, Phase 5-6: Ubuntu**
