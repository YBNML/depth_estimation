# Depth Estimation and Obstacle Avoidance for UAV

Wide-FOV 스테레오 카메라용 빠른 단안 깊이 추정(MDE)과 슈퍼픽셀 기반 깊이 보정(Depth Refinement),
그리고 behavior-arbitration 기반 3D 장애물 회피 알고리즘으로 구성된 UAV 자율 비행 시스템이다.

---

## 목차 (Table of Contents)

1. [프로젝트 개요](#1-프로젝트-개요-project-overview)
2. [시스템 아키텍처](#2-시스템-아키텍처-system-architecture)
3. [단안 깊이 추정 네트워크](#3-단안-깊이-추정-네트워크-monocular-depth-estimation-network)
4. [Depth Refinement 알고리즘](#4-depth-refinement-알고리즘)
5. [장애물 회피 알고리즘](#5-장애물-회피-알고리즘-obstacle-avoidance)
6. [실험 결과](#6-실험-결과-results)
7. [설치 및 사용법](#7-설치-및-사용법-installation--usage)
8. [저장소 구조](#8-저장소-구조-repository-structure)
9. [참고 문헌](#9-참고-문헌-references)

---

## 1. 프로젝트 개요 (Project Overview)

### 배경 (Motivation)

쿼드로터 UAV 의 자율 비행에는 **정확한 depth 추정**과 **빠른 회피 명령**이 필수다.
기존 접근법은 다음과 같은 한계를 가진다:

| 기존 방식 | 한계 |
|----------|------|
| LiDAR / Event Camera | 비싸고 무거움, 소형 UAV 탑재 어려움 |
| 단안 학습 기반 (DroNet, Chakravarty) | 좁은 FOV 로 인한 충돌 위험 |
| Wide-stereo SLAM (Mueller 등) | 복잡한 mapping + VIO 필요 |

### 접근 (Approach)

본 프로젝트는 두 편의 논문을 기반으로 한다:

- **Master's Thesis (2022)** — *"Refined Depth Estimation and Safety Navigation with a Binocular Camera"* (KNU, 저자: 김형진)
- **IEEE TIE (2025)** — *"Obstacle Avoidance of a UAV Using Fast Monocular Depth Estimation for a Wide Stereo Camera"* (저자: E. Cho, H. Kim, P. Kim, H. Lee)

핵심 아이디어는 다음 세 가지이다:

1. **Wide-FOV 스테레오 카메라** (110° × 2 = 160°) 로 넓은 시야 확보
2. **경량 단안 깊이 추정 네트워크** (ConvNeXt v2 + LWA decoder, 15M params) 로 실시간 추론
3. **슈퍼픽셀 기반 선형 회귀 refinement** 로 unseen 환경에서도 정확한 metric depth

### 목표 사양

| 항목 | 목표 |
|------|------|
| Depth 추정 지연시간 | ≤ 50 ms (Jetson NX) |
| End-to-end (depth + refinement + avoidance) | ≤ 100 ms |
| KITTI δ₁ (seen) | ≥ 0.96 |
| 도메인 일반화 (unseen) | δ₁ 향상 (refinement 효과) |
| 실내 dynamic 장애물 회피 | 가능 |
| 실외 static 장애물 + target-point 주행 | 가능 |

---

## 2. 시스템 아키텍처 (System Architecture)

### 전체 파이프라인

```
Stereo RGB (Left, Right)
     │
     ├─────────────────────────────────────────────┐
     │                                             │
     ▼                                             ▼
 ┌───────────────┐                         ┌──────────────┐
 │ Monocular     │                         │ Rectification│
 │ Depth         │                         │ + SGBM/Libelas│
 │ Estimation    │                         │ Stereo Match │
 │ (ConvNeXt v2  │                         └──────┬───────┘
 │  + LWA)       │                                │
 └──────┬────────┘                                │
        │                                         │
        │ dense depth d_hat                       │ sparse reliable depth d_stereo
        │                                         │
        ▼                                         ▼
 ┌─────────────────────────────────────────────────────┐
 │  Depth Refinement                                   │
 │  (Superpixel + Outlier 제거 + Linear Regression)    │
 │  → d = s·d_hat + t                                  │
 └─────────────────┬───────────────────────────────────┘
                   │
                   │ refined depth image
                   │
                   ▼
 ┌──────────────────────────────────────┐
 │  Obstacle Avoidance                  │
 │  (Behavior arbitration + Algorithm 1)│
 └─────────────────┬────────────────────┘
                   │
                   ▼
       (heading ψ, altitude v_z, velocity v) → UAV flight controller
```

### 모듈 분리 원칙

- **코어 라이브러리** (`src/`): ROS 의존 없음. Python + C++ (pybind11 으로 가속)
- **ROS2 래퍼** (Phase 5): 코어를 import 해서 topic 처리만 담당
- **개발 환경**: Mac Mini 에서 알고리즘 개발/테스트, Ubuntu + RTX 5070 에서 학습 및 실기기 배포

---

## 3. 단안 깊이 추정 네트워크 (Monocular Depth Estimation Network)

### 3.1 설계 철학

TIE 논문의 네트워크는 **단일 metric 최대화** 가 아닌 **시스템 레벨 제약 조건 하의 적정 정확도** 를 목표로 설계되었다.

**세 가지 설계 목표 (모두 동시 만족):**

1. **실시간성** — UAV onboard (Jetson NX) 10 Hz 이상, 전체 파이프라인 ≤ 100 ms
2. **일반화** — KITTI 로만 학습해도 vKITTI2, ApolloScape, DDAD 에서 동작
3. **Refinement 친화** — 후단 stereo 기반 선형 회귀와 결합 용이

### 3.2 전체 구조

```
Input RGB (B, 3, H, W)
   │
   ▼
 Encoder: ConvNeXt v2 Tiny (timm, 15M params)
   │
   ├─ stage 0 feature (H/4,  96 ch)  ──┐
   ├─ stage 1 feature (H/8,  192 ch) ─┤
   ├─ stage 2 feature (H/16, 384 ch) ─┤
   └─ stage 3 feature (H/32, 768 ch) ─┤
                  │                    │
                  ▼                    │
          PPM Head (pool 1,2,3,6)     │ skip
          → global feature (H/32,128ch)
                  │                    │
                  ▼                    │
          LWA Block 1 (7×7 DW) ◄──────┤
          → H/8, 128 ch               │
                  │                    │
                  ▼                    │
          LWA Block 2 (3×3 DW) ◄──────┤
          → H/4, 128 ch               │
                  │                    │
                  ▼                    │
          LWA Block 3 (3×3 DW) ◄──────┘
          → H/2, 128 ch
                  │
                  ▼
          Scaling Block (hard sigmoid × max_depth)
                  │
                  ▼
          Bilinear upsample → (B, 1, H, W) depth map
```

### 3.3 구성요소별 상세 설명

#### (a) Encoder — ConvNeXt v2 Tiny

**역할:** RGB 이미지에서 의미있는 multi-scale feature 를 추출.

**왜 ConvNeXt v2 를 선택했는가:**

| 대안 | 단점 |
|------|------|
| ViT / Swin | ImageNet-22k 대량 pretrain 필요, patching 연산 overhead |
| ResNet / DenseNet | ViT 급 정확도 미달, unseen 일반화 약함 |
| DINOv2 | 86~335M params 로 무거움, 실시간 불가 |

**ConvNeXt v2 의 장점:**
- ViT 설계 원칙 (7×7 depthwise conv, LayerNorm, GELU) 을 CNN 에 적용
- Stem block (4×4 conv, stride 4) 이 transformer 의 patching 연산 대체 → 속도 이득
- 15M params 로 AdaBins (78M) 의 1/5 수준, 정확도는 유사

**출력:** 4 개 stage feature list
- stage 0: `(B, 96, H/4, W/4)`   — 낮은 레벨 (edge, texture)
- stage 1: `(B, 192, H/8, W/8)`  — 중간 레벨 (부분 객체)
- stage 2: `(B, 384, H/16, W/16)` — 높은 레벨 (구조)
- stage 3: `(B, 768, H/32, W/32)` — 최고 레벨 (장면 의미)

#### (b) PPM Head (Pyramid Pooling Module)

**역할:** Encoder 최상위 stage 에 global context 를 주입.

**왜 필요한가:**
- Decoder 가 depthwise conv 기반이라 receptive field 가 좁음
- Pool size 1, 2, 3, 6 으로 multi-scale average pooling → 여러 스케일의 context 융합

**구조:**
```
input (B, 768, H/32, W/32)
  ├─ pool(1) → conv1x1 → upsample
  ├─ pool(2) → conv1x1 → upsample
  ├─ pool(3) → conv1x1 → upsample
  ├─ pool(6) → conv1x1 → upsample
  └─ 원본
     → concat → conv3x3 → (B, 128, H/32, W/32)
```

#### (c) LWA Block (Lightweight Attention Decoder)

**역할:** Encoder feature + 이전 decoder feature 를 fusion 해 2× 해상도로 upsample.

**왜 Lightweight 인가:**

기존 일반 conv 연산량:
```
f_conv = C·H·W · K² · (2.5C + 1) + 3·C·H·W
       = C·H·W · 1452   (K=3, C=64 기준)
```

LWA 연산량:
```
f_our = C·H·W · (2K₁² + 1.5K² + 2.5C + 0.5) + 2·C·H·W
      = C·H·W · 274      (K₁=7, K=3, C=64 기준)
```

→ **약 5.3× 빠름.** 정확도는 거의 동등 (TIE 논문 Fig. 3 참조).

**블록 내부:**
```python
local_reduced = Conv1x1(local_feat)           # 채널 축소
global_up = Bilinear_Upsample(global_feat)    # 해상도 맞춤
fused = Concat(local_reduced, global_up)
fused = DW_SeparableConv(fused, kernel=7 or 3)  # depthwise + pointwise
fused = DW_SeparableConv(fused, kernel=3)
attn_map = Sigmoid(DW_Conv(fused))            # self-attention gate
out = fused * attn_map
out = Bilinear_Upsample(out, 2x)              # 2× upsample
```

**3 단 stacking:**
- LWA Block 1: `kernel=7` (큰 receptive field)
- LWA Block 2, 3: `kernel=3`

#### (d) Scaling Block — Adaptive max_depth

**역할:** Decoder feature → `[0, max_depth]` 범위 depth 값 생성.

**왜 중요한가:**
- NewCRFs, GLPDepth 등 기존 모델은 `sigmoid × 80m` 로 고정됨 → KITTI 전용
- Scaling Block 은 `max_depth` 를 외부 인자로 받음 → **한 네트워크로 KITTI (80m) 와 NYU (10m) 모두 학습 가능**

**구조:**
```
feature (B, 128, H/2, W/2)
  ↓ 7x7 depthwise conv + BN + GELU
  ↓ 3x3 depthwise separable conv
  ↓ 1x1 conv (→ 1 channel)
  ↓ hard sigmoid
  ↓ × max_depth
output depth (B, 1, H/2, W/2)
```

**Hard sigmoid:** `clamp((x+3)/6, 0, 1)`  
- 일반 sigmoid 보다 계산 빠름
- extreme 구간에서도 gradient 살아있음

#### (e) Loss — Scale-Invariant Loss

TIE 논문 eq. (1):
```
L = α · √( (1/T) Σ gᵢ²  −  (λ/T²) · (Σ gᵢ)²  )
```

- `gᵢ = log(pred_i) − log(gt_i)`
- `T`: 유효 픽셀 수
- `α = 10.0` (스케일 factor)
- `λ = 0.85` (scale invariance 정도; 1.0 이면 완전 scale invariant, 0 이면 일반 log MSE)

**λ = 0.85** 를 선택한 이유: scale error 도 약간 penalize. 완전 scale invariant (λ=1.0) 면 네트워크가 depth 상대값만 맞추고 절대값은 틀릴 수 있음.

---

## 4. Depth Refinement 알고리즘

### 4.1 왜 필요한가

| 단독 사용 시 한계 |
|-------------------|
| **MDE only** — 학습 분포와 다른 환경(복도, 단색 벽)에서 스케일 오차 큼 |
| **Stereo only** — sparse 해서 dense depth 못 만듦, texture 없으면 불안정 |

→ MDE 의 **dense 구조** + Stereo 의 **신뢰할 수 있는 metric scale reference** 결합.

### 4.2 알고리즘 단계 (TIE 논문 방식)

```
Step 1. Rectification
   좌/우 이미지를 epipolar line 이 수평이 되도록 정렬.

Step 2. Stereo Matching (SGBM 또는 Libelas)
   겹치는 영역에서 sparse stereo depth d_stereo 획득.

Step 3. Superpixel 분할 (SEEDS)
   RGB 이미지를 시각적 유사 픽셀끼리 그룹화.
   → 각 그룹 g 안에서는 depth 가 거의 일정하다고 가정.

Step 4. Outlier 제거 (2 단계)
   ① Superpixel 일관성: 그룹 내 stereo depth 변동이 크면 제외
   ② MDE-Stereo 일관성: 그룹 median 이 50% 이상 차이나면 제외

Step 5. Linear Regression
   남은 신뢰 샘플 n 개에 대해 OLS:
      minimize Σ ( d_stereo_g - (s·d_hat_g + t) )²
   → (s_d, t_d) 구함.

Step 6. Temporal Smoothing (1차 필터)
   s_f ← α · s_f_prev + (1-α) · s_d
   t_f ← α · t_f_prev + (1-α) · t_d
   (프레임 간 flicker 제거)

Step 7. 전체 MDE depth 에 적용
   d_refined = s_f · d_hat + t_f
```

### 4.3 선형 회귀 수식 (OLS)

```
         Σ d_stereo · Σ d_hat²  −  Σ d_hat · Σ (d_stereo · d_hat)
 s_d = ─────────────────────────────────────────────────────────
                 n · Σ d_hat²  −  (Σ d_hat)²

         n · Σ (d_hat · d_stereo)  −  Σ d_hat · Σ d_stereo
 t_d = ─────────────────────────────────────────────────────
                 n · Σ d_hat²  −  (Σ d_hat)²
```

**조건부 업데이트:** `n < threshold` 일 때는 이전 `s_f`, `t_f` 유지 → 불안정한 regression 방지.

### 4.4 Thesis (2022) vs TIE (2025) 비교

| 항목 | Thesis | TIE |
|------|--------|-----|
| 보정 모델 | `d = k · d_hat` (scale only) | `d = s·d_hat + t` (scale + bias) |
| Outlier 처리 | binary flag `r^g ∈ {0,1}` | 2 단계 필터링 후 누적 regression |
| 시간적 smoothing | 없음 | 1차 필터 (α smoothing) |
| 표현력 | scale 만 보정 | scale + offset 동시 보정 |

### 4.5 Refinement 효과 (TIE 논문 Table II)

| Dataset | MDE only δ₁ | MDE + Refinement δ₁ | 향상 |
|---------|-------------|---------------------|------|
| KITTI (seen) | 0.959 | **0.973** | +1.5% |
| vKITTI2 (unseen) | 0.860 | **0.910** | +5.8% |
| ApolloScape (unseen) | 0.079 | **0.797** | **+900%** |
| DDAD (unseen) | 0.790 | 0.843 | +6.7% |
| Outdoor close-range (≤15m) | 0.390 | **0.645** | +65% |

→ **unseen domain 에서 refinement 가 결정적.** 특히 ApolloScape 처럼 학습 분포와 완전히 다른 환경에서 10배 향상.

---

## 5. 장애물 회피 알고리즘 (Obstacle Avoidance)

### 5.1 공통 철학 — Behavior Arbitration

- 복잡한 mapping / VIO / trajectory optimization **불필요**
- Refined depth image 만으로 명령 직접 생성
- 각 장애물이 밀어내는 "힘(potential)" 합성 → 방향 결정 (Althaus & Christensen 2002)

**출력:** heading `ψ`, altitude rate `v_z`, velocity `v`

### 5.2 Steering / Thrust 계산 (TIE 논문 eq. 4-6)

이미지를 여러 superpixel 로 나눈 뒤, 각 그룹 `g` 에서:

```
Exponential depth (가까울수록 큰 가중치):
   D_g = exp(d_c − d_g) · N_g · S_n / (H · W)

    - d_c: 기준 거리 (user-defined)
    - N_g: 그룹 픽셀 수 (큰 장애물 = 큰 가중)
    - S_n: 전체 superpixel 개수 (정규화)

Horizontal 명령 집계 (이미지 중심 근처 강조):
   φ_f^h = Σ D_g · exp(−(u_g / HFOV)²)

Vertical 명령 집계:
   φ_f^v = Σ D_g · exp(−(v_g / VFOV)²)

Smoothing (명령 폭주 방지):
   δ_h = φ_f^h · exp(−(φ_f^h / G_a)²)   ← steering
   δ_v = φ_f^v · exp(−(φ_f^v / G_a)²)   ← altitude rate
```

### 5.3 Target-point Navigation (TIE 논문 Algorithm 1)

회피 + 목표점 주행을 동시에 수행하는 완전 자율 비행 알고리즘.

#### (a) Collision Probability 계산

```python
def get_steering_and_collision_prob(superpixels):
    δ_h, δ_v     = eq.(6)                      # smoothed 회피 명령
    δ_coll       = (φ_f^h + φ_f^v) / S_n       # 전체 장면의 위험도
    p_coll       = δ_coll · exp((δ_coll · G_c)²)  # → [0, 1]
    return δ_h, δ_v, p_coll
```

- 모든 depth 가 작으면 `p_coll → 1` (충돌 임박)
- 멀리 보이면 `p_coll → 0`

#### (b) 회피 + 목표점 명령 합성

```python
def command_for_desired_location(δ_hl, δ_hr, p_coll_l, p_coll_r, P_goal, P_current):
    # 좌/우 이미지의 collision prob 시간적 smoothing
    p_coll_t = 0.3 · mean(p_coll_l, p_coll_r) + 0.7 · p_coll_t_prev

    # 충돌 위험에 반비례하는 속도
    v_d = V_max · (1 − p_coll_t)

    if distance(P_current, P_goal) > threshold:
        δ_g = desired_heading(P_current, P_goal)   # 목표점 방향
        δ_t = δ_hr − δ_hl                          # 좌우 회피 차이

        # 회피 각도 크기에 따라 블렌딩 비율 결정
        w, tc = get_weight(δ_t)

        # 시간 감쇠 (회피 지속시간 제한)
        w_time = exp(−0.05 · (t_now − tc)) · w

        # 회피와 목표점의 가중 합성
        δ_blend = w_time · δ_t + (1 − w_time) · δ_g

        # heading low-pass filter
        ψ_t = 0.7 · ψ_t_prev + 0.3 · δ_blend
    else:
        v_d, ψ_t = 0, ψ_t_prev  # 도착, 정지

    return ψ_t, v_d
```

#### (c) get-weight — 회피/목표 혼합 비율

```python
def get_weight(δ_t):
    if |δ_t| < 10°:
        return w=0.5, tc: no update     # 장애물 작음 → 회피/목표 반반
    else:
        return w = min(|δ_t|/20, 1), tc = t_now  # 장애물 크면 회피 우선
```

### 5.4 Thesis (2022) vs TIE (2025)

| 항목 | Thesis | TIE |
|------|--------|-----|
| 기능 | 회피만 | 회피 + target-point 주행 |
| 속도 제어 | superpixel density 기반 (heuristic) | collision probability 기반 (물리 해석 명확) |
| 명령 smoothing | 없음 | 1차 low-pass + 시간 감쇠 |
| 검증 환경 | simulation only | 실제 드론 (indoor dynamic + outdoor static) |

---

## 6. 실험 결과 (Results)

### 6.1 KITTI 학습 결과 (7 Epoch)

**학습 환경:** RTX 5070 12GB, batch_size=8, AdamW lr=1e-4, cosine schedule.

| Epoch | Train Loss | Val Loss | Time |
|-------|-----------|----------|------|
| 1 | 1.176 | 0.787 | 42.5 분 |
| 2 | 0.746 | 0.648 | 42.5 분 |
| 3 | 0.615 | 0.583 | 42.5 분 |
| 4 | 0.534 | 0.497 | 42.5 분 |
| 5 | 0.476 | 0.450 | 42.5 분 |
| 6 | 0.437 | 0.427 | 42.5 분 |
| **7** | **0.417** | **0.413** | 42.5 분 |

**KITTI Test Set 정량 평가** (Eigen split, 652 images):

| 메트릭 | 본 프로젝트 (7 epoch) | TIE 논문 (25 epoch) | 해석 |
|--------|----------------------|---------------------|------|
| δ₁ (< 1.25) | **0.9589** | 0.959 | 95.9% 픽셀이 정확 범위 — 논문 수준 |
| δ₂ (< 1.25²) | 0.9940 | — | 99.4% |
| δ₃ (< 1.25³) | 0.9986 | — | 99.9% |
| AbsRel | **0.0614** | 0.065 | 상대 오차 6.1% — 논문보다 약간 우수 |
| RMSE | 3.01 m | 2.44 m | 평균 3m 오차 (25 epoch 완주 시 개선 여지) |

→ **7 epoch 만에 논문 25 epoch 결과 수준 달성.** 모델 및 학습 파이프라인 정상 동작 확인.

### 6.2 NYU 학습 결과 (7 Epoch)

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 1.449 | 1.702 |
| 3 | 0.766 | 1.585 |
| 6 | **0.555** | **1.576** ← 최적 |
| 7 | 0.531 | 1.597 (약간 상승, 경미한 overfitting) |

**NYU Val Set 정량 평가** (표준 654 images, epoch 6):

| 메트릭 | 본 프로젝트 | AdaBins (25 epoch) | BTS | DenseDepth |
|--------|------------|--------------------|----|-----------|
| δ₁ | **0.8592** | 0.903 | 0.885 | 0.846 |
| AbsRel | **0.1203** | 0.103 | 0.110 | 0.123 |
| RMSE | 0.4549 m | — | — | — |

→ **7 epoch 만에 DenseDepth 능가, BTS 근접.** 25 epoch 시 AdaBins 수준 도달 예상.

### 6.3 시각적 비교 — Domain Matching 중요성

동일한 실내 스테레오 이미지에 두 모델 적용:

| 모델 | 출력 depth 범위 | 해석 |
|------|---------------|------|
| KITTI-trained (outdoor, max 80m) | 3.26 ~ 80.0 m | ❌ 일부 픽셀 포화, 도로 장면으로 오해 |
| NYU-trained (indoor, max 10m) | 1.43 ~ 3.48 m | ✅ 실내 스케일 정확 |

→ **같은 네트워크 구조라도 학습 데이터가 domain 일치 여부를 결정.** TIE 논문의 refinement 가 해결하려는 문제 정확히 드러남.

### 6.4 δ₁ 벤치마크 포지션 (최신 논문 대비)

**KITTI Eigen split leaderboard:**

| Rank | Method | δ₁ | AbsRel | RMSE | 파라미터 |
|------|--------|-----|--------|------|---------|
| 1 | SPIdepth (2024) | 0.990 | 0.029 | 1.39 | — |
| 2 | UniK3D (2025) | 0.990 | 0.037 | 1.68 | — |
| 3 | Metric3Dv2 (2024) | 0.989 | 0.039 | 1.77 | ViT-g2 |
| — | Depth Anything V2-L (2024) | 0.982 | 0.045 | 1.90 | 335M |
| — | NeWCRFs (2022) | 0.974 | 0.052 | 2.07 | 270M |
| — | **Ours (ConvNeXt+LWA, 7ep)** | **0.959** | **0.061** | 3.01 | **15M** |
| — | AdaBins (2021) | 0.964 | 0.058 | 2.36 | 78M |
| — | BTS (2019) | 0.956 | 0.059 | 2.76 | 47M |

→ **15M params 로 270M NeWCRFs 대비 δ₁ 1.5% 차이**, 파라미터는 **1/18 수준**. 속도 대비 정확도 우수.

---

## 7. 설치 및 사용법 (Installation & Usage)

### 7.1 환경 설정

```bash
# Conda 가상환경
conda create -n depth_estimation python=3.10 -y
conda activate depth_estimation

# 의존성 설치
pip install -r requirements.txt
```

주요 의존성: PyTorch 2.x, timm, OpenCV, NumPy, PyYAML, h5py.

### 7.2 동작 검증

```bash
# 전체 단위 테스트 (35개, 모두 통과해야 함)
PYTHONPATH=src python -m pytest tests/ -v

# Forward/backward 스모크 테스트
python scripts/smoke_test_mde.py
```

### 7.3 오프라인 파이프라인 테스트

```bash
python scripts/test_offline.py \
    --left path/to/left.png \
    --right path/to/right.png \
    --model convnext_mde
```

### 7.4 KITTI 학습

```bash
# 데이터 다운로드 (Eigen split, ~175GB)
bash scripts/download_kitti_full.sh data/kitti

# 학습
python scripts/train_kitti.py \
    --raw-dir data/kitti/raw \
    --depth-dir data/kitti \
    --epochs 25 --batch-size 8

# 평가
python scripts/evaluate_mde.py --weights weights/convnext_mde_epoch25.pth
```

### 7.5 NYU 학습

```bash
# HuggingFace 에서 FastDepth 전처리판 다운로드 (~35GB)
python scripts/download_nyu_hf.py

# 학습
python scripts/train_nyu.py --epochs 7 --batch-size 8 \
    --train-dir data/nyu/nyudepthv2/train \
    --val-dir data/nyu/nyudepthv2/val

# 평가 (표준 654 val images)
python scripts/evaluate_nyu.py --weights weights/convnext_mde_nyu_epoch06.pth
```

---

## 8. 저장소 구조 (Repository Structure)

```
depth_estimation/
├── config/                 # YAML 설정
│   ├── camera/             # 카메라 파라미터
│   │   ├── wide_stereo_112.yaml   # 졸업논문 구성 (82°×2)
│   │   └── wide_stereo_160.yaml   # TIE 논문 구성 (110°×2)
│   ├── training/
│   │   ├── kitti.yaml
│   │   └── nyu.yaml
│   └── default.yaml
├── src/
│   ├── mde/                # 단안 깊이 추정
│   │   ├── model/
│   │   │   ├── encoder.py         # ConvNeXt v2 encoder
│   │   │   ├── ppm_head.py        # PPM head
│   │   │   ├── lwa_decoder.py     # LWA decoder block
│   │   │   └── scaling_block.py   # Adaptive scaling
│   │   ├── dataset/
│   │   │   ├── kitti.py           # KITTI dataset
│   │   │   ├── nyu_h5.py          # NYU (FastDepth h5) dataset
│   │   │   └── transforms.py      # Augmentation
│   │   ├── convnext_mde.py        # 전체 MDE 모델 (BaseMDE 구현)
│   │   ├── base.py                # BaseMDE 추상 인터페이스
│   │   ├── dummy.py               # DummyMDE (테스트용)
│   │   ├── loss.py                # Scale-invariant loss
│   │   ├── train.py               # KITTI 학습 루프
│   │   ├── train_nyu.py           # NYU 학습 루프
│   │   ├── evaluate.py            # KITTI 평가
│   │   └── evaluate_nyu.py        # NYU 평가
│   ├── stereo/             # Rectification, SGBM (Phase 3 예정)
│   ├── refinement/         # Depth refinement (Phase 3 예정)
│   ├── navigation/         # 장애물 회피 (Phase 4 예정)
│   ├── pipeline.py         # End-to-end 파이프라인
│   ├── evaluation.py       # delta/absrel/rmse 메트릭
│   └── config.py
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
│   ├── algorithms/
│   │   └── refinement_and_avoidance.md   # Refinement + 회피 상세 설명
│   ├── mde_survey/
│   │   ├── MDE_Network_Survey.pptx       # MDE 네트워크 서베이 PPT
│   │   └── build_pptx.py
│   └── superpowers/        # 프로젝트 spec / plan
│       ├── specs/
│       └── plans/
├── tests/                  # pytest 단위 테스트 (35개)
├── requirements.txt
└── README.md
```

---

## 9. 참고 문헌 (References)

### 본 프로젝트 기반 논문

1. **H. Kim** — *"Refined Depth Estimation and Safety Navigation with a Binocular Camera"*, M.S. Thesis, Kyungpook National University, 2022.
2. **E. Cho, H. Kim, P. Kim, H. Lee** — *"Obstacle Avoidance of a UAV Using Fast Monocular Depth Estimation for a Wide Stereo Camera"*, IEEE Transactions on Industrial Electronics, 72(2), Feb 2025.

### MDE 네트워크

3. D. Eigen et al., *Depth Map Prediction from a Single Image using a Multi-Scale Deep Network*, NIPS 2014.
4. I. Alhashim, P. Wonka, *DenseDepth: High Quality Monocular Depth Estimation via Transfer Learning*, arXiv 2018.
5. J.H. Lee et al., *From Big to Small: Multi-Scale Local Planar Guidance (BTS)*, arXiv 2019.
6. C. Godard et al., *Monodepth2: Digging Into Self-Supervised Monocular Depth Estimation*, ICCV 2019.
7. S.F. Bhat et al., *AdaBins: Depth Estimation using Adaptive Bins*, CVPR 2021.
8. R. Ranftl et al., *Vision Transformers for Dense Prediction (DPT)*, ICCV 2021.
9. D. Kim et al., *Global-Local Path Networks for Monocular Depth Estimation (GLPDepth)*, arXiv 2022.
10. W. Yuan et al., *NewCRFs: Neural Window Fully-connected CRFs*, CVPR 2022.
11. S.F. Bhat et al., *ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth*, arXiv 2023.
12. L. Yang et al., *Depth Anything V1/V2*, CVPR/NeurIPS 2024.
13. B. Ke et al., *Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth*, CVPR 2024 (Oral).
14. S. Woo et al., *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*, CVPR 2023.

### 보조 알고리즘

15. L. Di Stefano et al., *ZNCC-based template matching using bounded partial correlation*, Pattern Recognition Letters 2005.
16. H. Hirschmüller, *Stereo processing by semiglobal matching and mutual information (SGM)*, IEEE TPAMI 2008.
17. P. Althaus, H. Christensen, *Behaviour coordination for navigation in office environments*, IROS 2002.

### 데이터셋 / 벤치마크

- KITTI: https://www.cvlibs.net/datasets/kitti/eval_depth.php
- NYU Depth v2: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
- PapersWithCode MDE leaderboard (mirror): https://opencodepapers-b7572d.gitlab.io/benchmarks/

---

## 라이선스

MIT License.
