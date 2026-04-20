# Depth Refinement & Obstacle Avoidance — 두 논문 알고리즘 정리

본 문서는 본 프로젝트의 근간이 되는 두 논문에서 제안한 **Depth Refinement** 알고리즘과 **Obstacle Avoidance (회피 비행)** 알고리즘을 정리한 자료다.

- **Thesis (2022)**: Hyeongjin Kim, *"Refined Depth Estimation and Safety Navigation with a Binocular Camera"*, M.S. Thesis, Kyungpook National Univ.
- **TIE (2025)**: Euihyeon Cho, Hyeongjin Kim, Pyojin Kim, Hyeonbeom Lee, *"Obstacle Avoidance of a UAV Using Fast Monocular Depth Estimation for a Wide Stereo Camera,"* IEEE Trans. Industrial Electronics, Vol. 72, No. 2, Feb 2025.

---

## 0. 공통 문제 정의 (왜 Refinement인가?)

- **MDE만으로는 부족**: 학습 기반 단안 깊이 추정은 학습 데이터와 유사한 환경에서만 정확. 복도·모노톤 벽 같은 texture 약한 장면에서 오차 크다.
- **Stereo만으로는 부족**: Stereo matching은 texture 없는 영역에서 신뢰할 수 없고 sparse해서 dense depth 못 만듦.
- **해법**: 두 장점을 결합.
  - MDE는 **dense한 global 구조**를 제공
  - Stereo는 **sparse하지만 reliable한 metric scale reference** 제공
  - → MDE depth를 stereo depth로 **재스케일(refine)**

두 논문의 차이는 "어떻게 스케일을 맞추는가"와 "reliable한 stereo 픽셀을 어떻게 고르는가"이다.

---

## 1. Depth Refinement

### 1.1 Thesis (2022) — 비율 스케일링 방식

**아이디어:** superpixel 그룹 단위로 (stereo 중앙값 / mono 중앙값) 비율을 계산해 그룹의 mono depth 전체에 곱한다.

#### 1.1.1 전체 파이프라인 (Fig. 2)

```
RGB (좌/우)
    ├─ MDE (Adabins) ─────────→ dense depth d_mono
    └─ ZNCC Stereo Matching ─→ sparse depth d_stereo (겹치는 영역만)
                │
                ↓
       Superpixel 분할 (RGB 기준)
                │
                ↓
        Refinement (eq. 2-3)
                │
                ↓
          refined depth d_ref
```

#### 1.1.2 슈퍼픽셀 기반 통계

각 superpixel 그룹 `g` 내에서:

| 기호 | 의미 |
|------|------|
| `d^g_mono,i` | 그룹 `g`의 `i`-번째 픽셀의 MDE 추정 깊이 |
| `d^g_stereo,i` | 그룹 `g`의 `i`-번째 픽셀의 ZNCC 추정 깊이 |
| `d^g_mono,c` | 그룹 `g` MDE 깊이의 **중앙값(median)** (대표값) |
| `d^g_stereo,c` | 그룹 `g` stereo 깊이의 **중앙값** |
| `d^g_max`, `d^g_min` | 그룹 `g` stereo depth의 90%/10% 분위수 |
| `N^g` | 그룹 `g`의 픽셀 수 |

#### 1.1.3 Refinement 수식 (eq. 2-3)

```
d^g_ref,i = r^g · (Σ d^g_stereo,c / Σ d^g_mono,c) · d^g_mono,i        (2)

         ⎧ 1  if  d^g_max / d^g_min < 1.5
r^g   =  ⎨                                                             (3)
         ⎩ 0  otherwise
```

- `r^g`는 **outlier 제거 플래그**. 그룹 내 stereo depth 분산이 크면 신뢰할 수 없다고 보고 그룹 전체를 0으로 만든다 (= refinement 안 씀, 원 MDE 유지).
- `1.5` 임계값은 경험적으로 선택.
- 중앙값(median)을 쓰는 이유: 평균은 outlier에 민감, 중앙값은 robust.

#### 1.1.4 한계

- `r^g ∈ {0, 1}`의 **binary** 결정 → 부드러운 가중치 없음.
- 프레임 단위 독립 계산 → 시간적 일관성(temporal smoothness) 없음.
- 비율이 frame마다 크게 변동하면 depth가 깜빡거림(flicker).

---

### 1.2 TIE (2025) — Linear Regression 방식

**아이디어:** MDE depth와 stereo depth의 관계를 단순 비율이 아닌 **선형 함수 `d = s·d̂ + t`** 로 모델링.

#### 1.2.1 이론적 근거 (MiDaS 계열)

MiDaS 논문 [Ranftl et al.] 에서 제시된 correction 수식:

```
d = s(d) · d̂ + t(d)

t(d) = median(d)                                               (2)
s(d) = (1/n) Σ | d_i − t(d) |
```

하지만 `s(d)`, `t(d)`는 **ground-truth가 있어야** 계산 가능. 실시간 비행 중엔 GT 없음 → stereo depth를 GT 대신 사용.

#### 1.2.2 파이프라인 (Fig. 4a)

```
RGB (좌/우)
    ├─ Rectification
    ├─ MDE (ConvNeXt+LWA) ──→ d̂ (full image)
    ├─ Stereo Matching (Libelas) → d_stereo (sparse, 겹치는 영역)
    └─ Superpixel
                │
                ↓
       Outlier 제거 (아래 1.2.3)
                │
                ↓
      Linear Regression → (s_f, t_f)
                │
                ↓
  d_i = s_f · d̂_i + t_f  (full image에 적용)
```

#### 1.2.3 Outlier 제거 (그림 4b, 4c)

두 단계 필터링:

**Step 1. Superpixel 기반 reliable stereo 선택**
- 각 superpixel 그룹 내 stereo depth 변화가 큰 그룹 제외 (texture 약해서 stereo 불안정한 경우)

**Step 2. MDE–Stereo 일관성 체크**
- superpixel 그룹의 **MDE 중앙값이 stereo depth와 50% 이상 차이나면 outlier 처리**

결과: `N`개의 "신뢰할 수 있는 (stereo, MDE) 쌍"이 남음 → 이걸로 regression.

#### 1.2.4 Regression 수식 (eq. 3의 변형)

최소제곱(OLS) 해 (선형회귀 공식):

```
         Σ d^g_i · Σ(d̂^g_i)²  −  Σ d̂^g_i · Σ d^g_i · d̂^g_i
s_d = ────────────────────────────────────────────────────
               n · Σ(d̂^g_i)²  −  (Σ d̂^g_i)²

         n · Σ d̂^g_i · d^g_i  −  Σ d̂^g_i · Σ d^g_i
t_d = ───────────────────────────────────────────────
               n · Σ(d̂^g_i)²  −  (Σ d̂^g_i)²
```

**조건부 업데이트:**
- 유효 inlier 개수 `n ≥ threshold` → 위 공식으로 업데이트
- `n < threshold` → 이전 값 유지 (stale update 방지)

#### 1.2.5 시간적 평활화 (1차 필터)

```
s_f ← α · s_f^(prev) + (1−α) · s_d
t_f ← α · t_f^(prev) + (1−α) · t_d
```

- α (time constant) 로 프레임 간 smoothing → **flicker 제거**.
- α 크면 안정적이나 반응 느림. 논문은 실험적 값 사용.

#### 1.2.6 최종 Refined Depth

```
d_i = s_f · d̂_i + t_f  (모든 픽셀에 적용)
```

#### 1.2.7 Thesis 대비 개선점

| 항목 | Thesis | TIE |
|------|--------|-----|
| 보정 모델 | `d = k · d̂` (scaling only) | `d = s·d̂ + t` (scaling + bias) |
| Outlier 처리 | superpixel 내 min/max 비율 | superpixel + MDE/stereo 50% 불일치 |
| Inlier 선택 | binary flag `r^g ∈ {0,1}` | 여러 superpixel의 축적된 샘플 regression |
| 시간적 일관성 | 없음 | 1차 필터 (α smoothing) |
| 업데이트 조건 | 매 프레임 독립 | n ≥ threshold 일 때만 |
| 표현력 | scale만 보정 (offset 못 잡음) | scale + offset 동시 보정 |

---

## 2. Obstacle Avoidance (회피 비행 알고리즘)

### 2.1 공통 철학

- **복잡한 mapping·VIO·trajectory optimization 없이** refined depth image만으로 명령 생성
- **Behavior arbitration** (Althaus & Christensen, 2002): 각 장애물이 밀어내는 "힘(potential)"을 합성해 방향 결정
- 출력: heading angle `ψ`, altitude rate `z`, velocity `v`

### 2.2 Thesis (2022) — Superpixel 기반 Behavior Arbitration

#### 2.2.1 입력

- `d^g_ref,c`: 그룹 `g`의 refined depth 대표값
- `(x^g_c, y^g_c)`: 그룹 대표 픽셀 좌표 (이미지 중심에서의 거리)
- `N^g`: 그룹 크기

#### 2.2.2 명령 생성 (eq. 4-6)

```
δ_ψ = λ_h · Σ ψ_h,c · exp(−ψ_h,c² / σ_h²) · exp(−w_depth · d^g_ref,c · N^g)    (4)
δ_z = λ_v · Σ ψ_v,c · exp(−ψ_v,c² / σ_v²) · exp(−w_depth · d^g_ref,c · N^g)    (5)
δ_v = v_max · C_d / max(C_d)                                                    (6)
```

- `ψ_h,c`, `ψ_v,c`: 이미지 중심 기준 수평/수직 bearing 각도
- `σ_h`, `σ_v`: FOV에 비례하는 가우시안 폭
- 내부 boundary 보정:
  ```
  X^g = x^g_c − H/2
  C_x = A_1 · ((X^g)² − A_2²)           (quadratic weighting)
  C_d = 7 · Σ C_x · d^g_ref,c · N^g / (H·W)
  ```
- 적응형 gain:
  ```
  λ_h = 250 · C_d^{-0.6},  λ_v = 5
  w_depth = 0.0368 · ln(C_d) − 0.125
  ```

#### 2.2.3 해석

- **가까운 장애물일수록** `exp(−w_depth · d)` 커져서 강한 회피
- **중심에서 먼 장애물**은 회피 약함 (충돌 위험 작음)
- **중심에 정확히 있는 장애물**은 회피 방향 애매 → quadratic `C_x`로 완화

#### 2.2.4 최종 desired velocity

```
ψ^d = ψ_o + δ_ψ     (heading 갱신)
v^d_z = δ_z          (수직 속도)
v^d_x = δ_v · cos(ψ^d)
v^d_y = δ_v · sin(ψ^d)
```

#### 2.2.5 특징 & 한계

- **장점**: 복잡 수식처럼 보이지만 superpixel 2000개 미만 기준 수 ms 내 계산
- **한계**:
  - 하이퍼파라미터 많음 (`A_1`, `A_2`, `w_depth` 계수, `λ` 관계식)
  - 정량화 어려움 (heuristic)
  - 목표점까지 **주행(go-to-goal)** 기능 없음 — 회피만 함

---

### 2.3 TIE (2025) — Exponential Depth + Target-point Navigation

#### 2.3.1 Steering/Thrust 계산 (eq. 4-6)

전체 이미지를 여러 superpixel로 나눈 뒤:

```
D^g_i = exp(d_c − d^g_i) · (N^g · S_n) / (H_image · W_image)     (5)

φ_f^h = Σ D^g_i · W^h_i,     W^h_i = exp(−(u^g_i / HFOV)²)       (4 horizontal)
φ_f^v = Σ D^g_i · W^v_i,     W^v_i = exp(−(v^g_i / VFOV)²)       (4 vertical)

δ_h = φ_f^h · exp(−(φ_f^h / G_a)²)                                (6 steering)
δ_v = φ_f^v · exp(−(φ_f^v / G_a)²)                                (6 altitude)
```

여기서:
- `D^g_i`: **exponential depth** — 가까울수록 지수적으로 큰 가중치
- `d_c`: user-defined 기준 거리 (이 값보다 가까우면 가중치 크게 튐)
- `N^g · S_n / (H·W)`: **superpixel 크기 정규화** — 큰 장애물일수록 큰 가중
- `W^h`, `W^v`: **이미지 중심 근처 강조** (바깥은 덜 중요)
- `G_a`: user gain
- 바깥 `exp(−(φ/G_a)²)`: **명령 smoothing** — 너무 큰 명령 방지

#### 2.3.2 Thesis 대비 변화

| 항목 | Thesis | TIE |
|------|--------|-----|
| Depth 가중 | `exp(−w_depth · d)` (자체 gain 조정) | `exp(d_c − d)` (closer=larger 명확) |
| Superpixel 크기 반영 | `N^g`로 단순 곱 | `N^g · S_n / (H·W)`로 정규화 |
| Bearing 가중 | 가우시안 (자체 σ) | FOV 정규화 (`u/HFOV`) |
| Smoothing | 없음 (명령 폭주 가능) | `exp(−(φ/G_a)²)`로 억제 |
| 파라미터 | 많음 | 간소화, 물리적 해석 명확 |

#### 2.3.3 Collision Probability + Target Navigation (Algorithm 1)

**2개 함수 합성으로 자율 비행 완성:**

##### ① `get-steering-and-collision-prob(S)`

```python
def get_steering_and_collision_prob(S):
    φ_f_h, φ_f_v = eq.(4)              # 좌/우 분리된 이미지 각각에서
    δ_h, δ_v = eq.(6)                  # smoothing된 명령
    δ_coll = (φ_f_h + φ_f_v) / S_n     # 중앙 가중 정규화
    p_coll = δ_coll · exp((δ_coll · G_c)²)  # collision probability
    return δ_h, δ_v, p_coll
```

- 모든 depth가 작으면 `p_coll` → 1 (충돌 임박)
- 멀리 보이면 `p_coll` → 0

##### ② `command-for-desired-location(Pc, Pg, ...)`

```python
def command_for_desired_location(δ_hl, δ_hr, p_coll_l, p_coll_r, Pg, Pc, tnow):
    p_coll_t = 0.3·mean(p_coll_l, p_coll_r) + 0.7·p_coll_t_prev  # temporal smoothing
    v_d = V_max · (1 − p_coll_t)                                  # 충돌 위험 높으면 감속

    if distance(Pc, Pg) > threshold:
        δ_g = desired_heading(Pc, Pg)     # 목표점 방향
        δ_t = δ_hr − δ_hl                  # 우/좌 회피 명령 차이
        w, tc = get_weight(δ_t)
        w_time = exp(−0.05·(tnow − tc)) · w

        δ_t_blend = w_time·δ_t + (1 − w_time)·δ_g
        ψ_t = 0.7·ψ_t_prev + 0.3·δ_t_blend
    else:
        v_d = 0, ψ_t = ψ_t_prev

    return ψ_t, v_d
```

##### ③ `get-weight(δ_t)` — 회피 vs 목표 혼합

```python
def get_weight(δ_t):
    if |δ_t| < 10°:
        return w=0.5, tc: no update  # 장애물 작음 → 회피/목표 반반
    else:
        return w = min(|δ_t|/20, 1), tc = tnow  # 장애물 크면 회피 우선
```

#### 2.3.4 핵심 설계 의도

| 기능 | 수식 | 의도 |
|------|------|------|
| Velocity 제어 | `v_d = V_max · (1 − p_coll)` | 충돌 위험 ↑ → 속도 ↓ |
| Heading blending | `w_time · δ_t + (1−w_time) · δ_g` | 회피와 목표 접근의 동적 균형 |
| 시간 감쇠 | `exp(−0.05·(tnow − tc))` | 회피 명령 지속시간 제한 |
| 저역통과 | `0.7 · ψ_prev + 0.3 · δ_blend` | 급격한 heading 변화 방지 |

#### 2.3.5 Thesis 대비 개선

| 항목 | Thesis | TIE |
|------|--------|-----|
| 작동 모드 | 회피만 (go-to-goal 없음) | 회피 + 목표 주행 동시 |
| Velocity 제어 | superpixel density 기반 (heuristic) | collision probability 기반 (물리 해석) |
| 명령 smoothing | 없음 | 1차 low-pass + 시간 감쇠 |
| Algorithm 정립 | 수식만 나열 | Algorithm 1로 명시적 pseudocode |
| 실제 드론 비행 검증 | simulation only | real drone + dynamic/static 장애물 |

---

## 3. 실험 결과 요약

### 3.1 Depth Refinement 정량 평가

**Thesis (자체 실내 데이터셋, 22 images, avg δ)**

| Method | δ (avg) |
|--------|---------|
| Adabins (MDE only) | 0.517 |
| ZNCC (stereo only) | 0.299 |
| **Ours1 (w/ZNCC)** | **0.705** |

→ **MDE only 대비 36% 향상, stereo only 대비 2배 향상**

**TIE (다중 데이터셋, δ₁)**

| Dataset | MDE only (Ours) | Ours + Refinement | 개선 |
|---------|-----------------|-------------------|------|
| KITTI (seen) | 0.959 | **0.973** | +1.5% |
| vKITTI2 (unseen) | 0.860 | **0.910** | +5.8% |
| ApolloScape (unseen) | 0.079 | **0.797** | **+900%** |
| Outdoor ≤15m | 0.390 | **0.645** | +65% |

→ unseen 도메인에서 **refinement가 결정적** (특히 ApolloScape에서 10배 향상)

### 3.2 Obstacle Avoidance 실험

**Thesis — Simulation (Fig. 7)**
- DroNet: 실내 simulation (학습 분포와 달라) 4/N 시도 전부 충돌
- Chakravarty: 복잡 환경 충돌
- **Ours**: 성공 3회, 실패 1회 (95% 환경에서 작동)
- 3D 장애물 (Fig. 7c): 공중에 매달린 장애물도 회피 성공

**TIE — Real Drone (Fig. 15, 16)**
- **Indoor dynamic**: 걸어다니는 사람 2명 회피 성공
- **Outdoor static**: 가상 벽 + 정지 장애물 2개 회피, 목표점 도달
- **10 Hz 이상 실시간 동작** (Jetson NX onboard)

---

## 4. 최종 정리 — 두 논문의 교훈

### 4.1 Refinement 설계
- **절대적 depth 스케일**은 MDE만으로 잡기 어렵다 → reliable한 external reference 필요
- **Stereo matching**은 sparse하지만 metric scale 제공에 유용
- 단순 비율보다 **linear regression**이 bias도 잡아줘서 더 정확
- **Outlier 제거**는 두 단계 (superpixel 일관성 + MDE/stereo 일관성)
- **시간적 smoothing**은 flicker 제거에 필수

### 4.2 회피 비행 설계
- **Depth image → 직접 명령** 생성이 가장 단순하고 빠름 (mapping 불필요)
- **Behavior arbitration**은 수학적 보장은 약하지만 실무에서 매우 효율적
- 회피만으론 부족 — **target-point 주행과 혼합**해야 실제 비행 가능
- **Collision probability**로 속도 제어하면 dynamic 환경에서 안정적
- 모든 명령은 **low-pass filter + 시간 감쇠**로 부드럽게 만들어야 실제 드론 제어기가 따라감

### 4.3 이 프로젝트의 재구현 전략
- **Phase 3**에서 TIE 논문의 linear regression refinement 구현 (superpixel 필터링 + OLS + 1차 필터)
- **Phase 4**에서 TIE Algorithm 1의 obstacle avoidance 구현 (collision probability + target navigation)
- C++/pybind11로 핵심 연산 (superpixel 통계, regression coefficient) 가속
- Thesis 버전(비율 스케일링)도 option으로 유지해 비교 실험 가능
