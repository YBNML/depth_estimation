# Phase 1: 프로젝트 골격 및 오프라인 테스트 환경

> **에이전트 작업자용 안내:** 이 계획을 task 단위로 구현할 때 `superpowers:subagent-driven-development` (권장) 또는 `superpowers:executing-plans` 스킬을 사용한다. 각 step 은 체크박스 (`- [ ]`) 로 진행 상태를 추적한다.

**목표 (Goal):** 프로젝트 디렉토리 구조, Python 환경, config 시스템, BaseMDE 인터페이스, 데이터셋 다운로드, 오프라인 테스트 파이프라인을 구축한다.

**아키텍처 (Architecture):** 순수 Python 코어 라이브러리 (ROS 의존 없음). config YAML 로 카메라 · 모델 파라미터 관리. BaseMDE 추상 클래스로 MDE 모델 교체 가능. `pipeline.py` 가 전체 흐름 오케스트레이션.

**기술 스택 (Tech Stack):** Python 3.10 (conda), PyTorch 2.x (MPS), OpenCV, NumPy, PyYAML, pybind11, CMake

---

## 파일 구조 (File Structure)

```
depth_estimation/
├── config/
│   ├── camera/
│   │   ├── wide_stereo_112.yaml    # 졸업논문 카메라 (82deg x 2)
│   │   └── wide_stereo_160.yaml    # TIE 논문 카메라 (110deg x 2)
│   ├── default.yaml                # 기본 설정 (모델 선택, 경로 등)
│   └── training/
│       └── kitti.yaml              # KITTI 학습 하이퍼파라미터
├── src/
│   ├── __init__.py
│   ├── config.py                   # Config 로더
│   ├── mde/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseMDE 추상 클래스
│   │   └── dummy.py                # DummyMDE (테스트용 더미 모델)
│   ├── stereo/
│   │   ├── __init__.py
│   │   └── rectification.py        # (placeholder, Phase 3에서 구현)
│   ├── refinement/
│   │   ├── __init__.py
│   │   └── refine.py               # (placeholder, Phase 3에서 구현)
│   ├── navigation/
│   │   ├── __init__.py
│   │   └── navigator.py            # (placeholder, Phase 4에서 구현)
│   ├── pipeline.py                 # 전체 파이프라인
│   └── evaluation.py               # 평가 메트릭
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_base_mde.py
│   ├── test_evaluation.py
│   └── test_pipeline.py
├── scripts/
│   └── download_kitti.py           # KITTI 다운로드
├── data/                           # 데이터 디렉토리 (gitignore)
├── weights/                        # 모델 가중치 (gitignore)
├── requirements.txt
├── setup.py
└── .gitignore
```

---

### Task 1: Python 환경 및 프로젝트 초기화

**Files:**
- Create: `requirements.txt`
- Create: `setup.py`
- Create: `.gitignore`
- Create: `src/__init__.py`

- [ ] **Step 1: Conda 설치 및 가상환경 생성**

Conda가 설치되어 있지 않다면 Miniforge를 먼저 설치한다 (Apple Silicon 최적화):
```bash
brew install miniforge
conda init zsh
```
쉘을 재시작한 후 가상환경을 생성한다:
```bash
conda create -n depth_estimation python=3.10 -y
conda activate depth_estimation
```

- [ ] **Step 2: requirements.txt 작성**

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
PyYAML>=6.0
matplotlib>=3.7.0
pybind11>=2.11.0
pytest>=7.4.0
```

- [ ] **Step 3: 의존성 설치**

```bash
conda activate depth_estimation
pip install -r requirements.txt
```

- [ ] **Step 4: setup.py 작성**

```python
from setuptools import setup, find_packages

setup(
    name="depth_estimation",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
```

- [ ] **Step 5: .gitignore 작성**

```
__pycache__/
*.pyc
*.pyo
.DS_Store
data/
weights/
*.egg-info/
build/
dist/
.pytest_cache/
```

- [ ] **Step 6: src/__init__.py 작성**

```python
```

(빈 파일)

- [ ] **Step 7: git 초기화 및 첫 커밋**

```bash
cd /Users/khj/YBNML_macmini/depth_estimation
git init
git add requirements.txt setup.py .gitignore src/__init__.py
git commit -m "feat: initialize project with Python environment"
```

---

### Task 2: Config 시스템

**Files:**
- Create: `config/default.yaml`
- Create: `config/camera/wide_stereo_112.yaml`
- Create: `config/camera/wide_stereo_160.yaml`
- Create: `src/config.py`
- Test: `tests/__init__.py`, `tests/test_config.py`

- [ ] **Step 1: 테스트 작성**

`tests/__init__.py` (빈 파일)

`tests/test_config.py`:
```python
import os
import pytest
from config import Config


def test_load_default_config():
    cfg = Config()
    assert cfg.model_name is not None
    assert cfg.image_height > 0
    assert cfg.image_width > 0


def test_load_camera_config():
    cfg = Config()
    cam = cfg.load_camera("wide_stereo_160")
    assert cam["left"]["hfov"] == 110
    assert cam["right"]["hfov"] == 110
    assert cam["baseline"] > 0


def test_load_camera_config_112():
    cfg = Config()
    cam = cfg.load_camera("wide_stereo_112")
    assert cam["left"]["hfov"] == 82
    assert cam["right"]["hfov"] == 82


def test_config_override():
    cfg = Config(overrides={"image_height": 720, "image_width": 1280})
    assert cfg.image_height == 720
    assert cfg.image_width == 1280
```

- [ ] **Step 2: 테스트 실행하여 실패 확인**

```bash
cd /Users/khj/YBNML_macmini/depth_estimation
python -m pytest tests/test_config.py -v
```
Expected: FAIL (ModuleNotFoundError: No module named 'config')

- [ ] **Step 3: config YAML 파일 작성**

`config/default.yaml`:
```yaml
# Model settings
model_name: "convnext_mde"  # convnext_mde, adabins, dummy
model_weights: null

# Image settings
image_height: 480
image_width: 640

# Camera config name (from config/camera/)
camera: "wide_stereo_160"

# Depth settings
max_depth: 80.0
min_depth: 0.1

# Refinement
refinement_method: "linear_regression"  # linear_regression, ratio_scaling, none

# Paths
data_dir: "data"
weights_dir: "weights"
```

`config/camera/wide_stereo_112.yaml`:
```yaml
# 졸업논문 카메라 구성: 각 82도 FOV, 총 112도
left:
  hfov: 82
  vfov: 50
  fx: 380.0
  fy: 380.0
  cx: 320.0
  cy: 240.0
  distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
  rotation_y_deg: -15.0  # 외향 15도

right:
  hfov: 82
  vfov: 50
  fx: 380.0
  fy: 380.0
  cx: 320.0
  cy: 240.0
  distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
  rotation_y_deg: 15.0

baseline: 0.2  # meters
total_hfov: 112
image_height: 480
image_width: 640
```

`config/camera/wide_stereo_160.yaml`:
```yaml
# TIE 논문 카메라 구성: 각 110도 FOV, 총 160도
left:
  hfov: 110
  vfov: 74
  fx: 256.0
  fy: 256.0
  cx: 320.0
  cy: 240.0
  distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
  rotation_y_deg: -25.0

right:
  hfov: 110
  vfov: 74
  fx: 256.0
  fy: 256.0
  cx: 320.0
  cy: 240.0
  distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
  rotation_y_deg: 25.0

baseline: 0.2
total_hfov: 160
image_height: 480
image_width: 640
```

- [ ] **Step 4: Config 클래스 구현**

`src/config.py`:
```python
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:
    """프로젝트 설정을 관리하는 클래스."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        if config_path is None:
            config_path = str(_PROJECT_ROOT / "config" / "default.yaml")

        with open(config_path, "r") as f:
            self._data = yaml.safe_load(f)

        if overrides:
            self._data.update(overrides)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def load_camera(self, camera_name: str) -> Dict[str, Any]:
        camera_path = _PROJECT_ROOT / "config" / "camera" / f"{camera_name}.yaml"
        with open(camera_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT
```

- [ ] **Step 5: 테스트 실행하여 통과 확인**

```bash
cd /Users/khj/YBNML_macmini/depth_estimation
PYTHONPATH=src python -m pytest tests/test_config.py -v
```
Expected: 4 passed

- [ ] **Step 6: 커밋**

```bash
git add config/ src/config.py tests/
git commit -m "feat: add config system with camera parameter YAML"
```

---

### Task 3: BaseMDE 인터페이스 + DummyMDE

**Files:**
- Create: `src/mde/__init__.py`
- Create: `src/mde/base.py`
- Create: `src/mde/dummy.py`
- Test: `tests/test_base_mde.py`

- [ ] **Step 1: 테스트 작성**

`tests/test_base_mde.py`:
```python
import numpy as np
import pytest

from mde.base import BaseMDE
from mde.dummy import DummyMDE


def test_base_mde_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseMDE()


def test_dummy_mde_predict_shape():
    model = DummyMDE(max_depth=10.0)
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = model.predict(rgb)
    assert depth.shape == (480, 640)
    assert depth.dtype == np.float32


def test_dummy_mde_depth_range():
    model = DummyMDE(max_depth=10.0)
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = model.predict(rgb)
    assert np.all(depth >= 0.0)
    assert np.all(depth <= 10.0)


def test_dummy_mde_max_depth():
    model = DummyMDE(max_depth=80.0)
    assert model.get_max_depth() == 80.0


def test_dummy_mde_different_resolution():
    model = DummyMDE(max_depth=10.0)
    rgb = np.random.randint(0, 255, (352, 704, 3), dtype=np.uint8)
    depth = model.predict(rgb)
    assert depth.shape == (352, 704)
```

- [ ] **Step 2: 테스트 실행하여 실패 확인**

```bash
PYTHONPATH=src python -m pytest tests/test_base_mde.py -v
```
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: BaseMDE 추상 클래스 구현**

`src/mde/__init__.py`:
```python
from mde.base import BaseMDE
```

`src/mde/base.py`:
```python
from abc import ABC, abstractmethod

import numpy as np


class BaseMDE(ABC):
    """Monocular Depth Estimation 추상 인터페이스.

    모든 MDE 모델은 이 클래스를 상속하여 구현한다.
    predict()는 RGB 이미지를 입력받아 depth map을 반환한다.
    """

    @abstractmethod
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """RGB 이미지에서 depth map을 추정한다.

        Args:
            rgb: (H, W, 3) uint8 RGB 이미지.

        Returns:
            (H, W) float32 depth map (미터 단위).
        """
        ...

    @abstractmethod
    def get_max_depth(self) -> float:
        """모델의 최대 추정 깊이를 반환한다."""
        ...
```

- [ ] **Step 4: DummyMDE 구현**

`src/mde/dummy.py`:
```python
import numpy as np

from mde.base import BaseMDE


class DummyMDE(BaseMDE):
    """테스트용 더미 MDE 모델. 랜덤 depth를 생성한다."""

    def __init__(self, max_depth: float = 10.0):
        self._max_depth = max_depth

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        depth = np.random.uniform(0.1, self._max_depth, (h, w)).astype(np.float32)
        return depth

    def get_max_depth(self) -> float:
        return self._max_depth
```

- [ ] **Step 5: 테스트 실행하여 통과 확인**

```bash
PYTHONPATH=src python -m pytest tests/test_base_mde.py -v
```
Expected: 5 passed

- [ ] **Step 6: 커밋**

```bash
git add src/mde/ tests/test_base_mde.py
git commit -m "feat: add BaseMDE interface and DummyMDE for testing"
```

---

### Task 4: 평가 메트릭 모듈

**Files:**
- Create: `src/evaluation.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: 테스트 작성**

`tests/test_evaluation.py`:
```python
import numpy as np
import pytest

from evaluation import compute_metrics


def test_perfect_prediction():
    gt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pred = gt.copy()
    metrics = compute_metrics(pred, gt)
    assert metrics["delta1"] == pytest.approx(1.0)
    assert metrics["delta2"] == pytest.approx(1.0)
    assert metrics["delta3"] == pytest.approx(1.0)
    assert metrics["absrel"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)


def test_skip_zero_gt():
    gt = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    pred = np.array([[999.0, 2.0], [3.0, 999.0]], dtype=np.float32)
    metrics = compute_metrics(pred, gt)
    # gt=0인 픽셀은 무시, 나머지는 완벽 예측
    assert metrics["delta1"] == pytest.approx(1.0)
    assert metrics["absrel"] == pytest.approx(0.0)


def test_scaled_prediction():
    gt = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    pred = gt * 1.2  # 20% 과대 추정
    metrics = compute_metrics(pred, gt)
    # max(1.2, 1/1.2) = 1.2 < 1.25 이므로 delta1 = 1.0
    assert metrics["delta1"] == pytest.approx(1.0)
    assert metrics["absrel"] == pytest.approx(0.2)


def test_bad_prediction():
    gt = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    pred = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    metrics = compute_metrics(pred, gt)
    # max(2/1, 1/2) = 2.0, 2.0 > 1.25 but < 1.5625
    assert metrics["delta1"] == pytest.approx(0.0)
    assert metrics["delta2"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
```

- [ ] **Step 2: 테스트 실행하여 실패 확인**

```bash
PYTHONPATH=src python -m pytest tests/test_evaluation.py -v
```
Expected: FAIL

- [ ] **Step 3: evaluation.py 구현**

`src/evaluation.py`:
```python
from typing import Dict

import numpy as np


def compute_metrics(
    pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.001
) -> Dict[str, float]:
    """depth estimation 정확도 메트릭을 계산한다.

    Args:
        pred: (H, W) 추정 depth.
        gt: (H, W) ground-truth depth.
        min_depth: gt가 이 값 이하인 픽셀은 무시.

    Returns:
        delta1, delta2, delta3, absrel, rmse를 포함하는 dict.
    """
    valid = gt > min_depth
    pred_v = pred[valid]
    gt_v = gt[valid]

    if len(gt_v) == 0:
        return {"delta1": 0.0, "delta2": 0.0, "delta3": 0.0, "absrel": 0.0, "rmse": 0.0}

    thresh = np.maximum(pred_v / gt_v, gt_v / pred_v)

    delta1 = float(np.mean(thresh < 1.25))
    delta2 = float(np.mean(thresh < 1.25 ** 2))
    delta3 = float(np.mean(thresh < 1.25 ** 3))

    absrel = float(np.mean(np.abs(pred_v - gt_v) / gt_v))
    rmse = float(np.sqrt(np.mean((pred_v - gt_v) ** 2)))

    return {
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "absrel": absrel,
        "rmse": rmse,
    }
```

- [ ] **Step 4: 테스트 실행하여 통과 확인**

```bash
PYTHONPATH=src python -m pytest tests/test_evaluation.py -v
```
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/evaluation.py tests/test_evaluation.py
git commit -m "feat: add depth estimation evaluation metrics"
```

---

### Task 5: 파이프라인 골격

**Files:**
- Create: `src/pipeline.py`
- Create: `src/stereo/__init__.py`
- Create: `src/refinement/__init__.py`
- Create: `src/navigation/__init__.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: 테스트 작성**

`tests/test_pipeline.py`:
```python
import numpy as np
import pytest

from pipeline import DepthEstimationPipeline
from config import Config


def test_pipeline_creation():
    cfg = Config(overrides={"model_name": "dummy"})
    pipe = DepthEstimationPipeline(cfg)
    assert pipe is not None


def test_pipeline_run_mono():
    cfg = Config(overrides={"model_name": "dummy", "refinement_method": "none"})
    pipe = DepthEstimationPipeline(cfg)
    left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = pipe.run(left, right)
    assert "left_depth" in result
    assert "right_depth" in result
    assert result["left_depth"].shape == (480, 640)
    assert result["right_depth"].shape == (480, 640)


def test_pipeline_run_returns_float32():
    cfg = Config(overrides={"model_name": "dummy", "refinement_method": "none"})
    pipe = DepthEstimationPipeline(cfg)
    left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = pipe.run(left, right)
    assert result["left_depth"].dtype == np.float32
    assert result["right_depth"].dtype == np.float32
```

- [ ] **Step 2: 테스트 실행하여 실패 확인**

```bash
PYTHONPATH=src python -m pytest tests/test_pipeline.py -v
```
Expected: FAIL

- [ ] **Step 3: placeholder __init__.py 파일 생성**

`src/stereo/__init__.py`:
```python
```

`src/refinement/__init__.py`:
```python
```

`src/navigation/__init__.py`:
```python
```

- [ ] **Step 4: pipeline.py 구현**

`src/pipeline.py`:
```python
from typing import Any, Dict

import numpy as np

from config import Config
from mde.base import BaseMDE
from mde.dummy import DummyMDE


def _create_mde(cfg: Config) -> BaseMDE:
    """config에 따라 MDE 모델을 생성한다."""
    name = cfg.model_name
    max_depth = cfg.max_depth

    if name == "dummy":
        return DummyMDE(max_depth=max_depth)
    # Phase 2에서 추가:
    # if name == "convnext_mde":
    #     from mde.convnext import ConvNeXtMDE
    #     return ConvNeXtMDE(cfg)
    raise ValueError(f"Unknown model: {name}")


class DepthEstimationPipeline:
    """전체 depth estimation 파이프라인.

    Input(left, right RGB) -> MDE -> (Refinement) -> Output(depth maps)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mde = _create_mde(cfg)

    def run(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> Dict[str, Any]:
        """좌우 RGB 이미지에서 depth를 추정한다.

        Args:
            left_rgb: (H, W, 3) 좌측 RGB 이미지.
            right_rgb: (H, W, 3) 우측 RGB 이미지.

        Returns:
            dict with keys: left_depth, right_depth (각 (H, W) float32).
        """
        left_depth = self.mde.predict(left_rgb)
        right_depth = self.mde.predict(right_rgb)

        result = {
            "left_depth": left_depth,
            "right_depth": right_depth,
        }

        if self.cfg.refinement_method != "none":
            result = self._refine(result, left_rgb, right_rgb)

        return result

    def _refine(
        self, result: Dict[str, Any], left_rgb: np.ndarray, right_rgb: np.ndarray
    ) -> Dict[str, Any]:
        """Depth refinement (Phase 3에서 구현)."""
        return result
```

- [ ] **Step 5: 테스트 실행하여 통과 확인**

```bash
PYTHONPATH=src python -m pytest tests/test_pipeline.py -v
```
Expected: 3 passed

- [ ] **Step 6: 커밋**

```bash
git add src/stereo/ src/refinement/ src/navigation/ src/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline skeleton with MDE model factory"
```

---

### Task 6: 오프라인 테스트 스크립트

**Files:**
- Create: `scripts/test_offline.py`

- [ ] **Step 1: 스크립트 작성**

`scripts/test_offline.py`:
```python
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
```

- [ ] **Step 2: 더미 이미지로 실행 테스트**

```bash
cd /Users/khj/YBNML_macmini/depth_estimation
# 테스트용 더미 이미지 생성
python -c "
import numpy as np, cv2
img = np.random.randint(0,255,(480,640,3),dtype=np.uint8)
cv2.imwrite('/tmp/test_left.png', img)
cv2.imwrite('/tmp/test_right.png', img)
"
conda activate depth_estimation
python scripts/test_offline.py --left /tmp/test_left.png --right /tmp/test_right.png
```
Expected: output_depth.png 생성, matplotlib 창 표시

- [ ] **Step 3: 커밋**

```bash
git add scripts/test_offline.py
git commit -m "feat: add offline test script with visualization"
```

---

### Task 7: KITTI 데이터셋 다운로드 스크립트

**Files:**
- Create: `scripts/download_kitti.py`
- Create: `config/training/kitti.yaml`

- [ ] **Step 1: KITTI 학습 config 작성**

`config/training/kitti.yaml`:
```yaml
# KITTI depth estimation training config
dataset: "kitti"
data_dir: "data/kitti"

# Eigen split
train_file: "data/kitti/eigen_train_files.txt"
val_file: "data/kitti/eigen_val_files.txt"
test_file: "data/kitti/eigen_test_files.txt"

# Training hyperparameters
batch_size: 8
learning_rate: 0.0001
weight_decay: 0.01
epochs: 25
optimizer: "adamw"
scheduler: "cosine"

# Loss
loss: "scale_invariant"
si_alpha: 10.0
si_lambda: 0.85

# Augmentation
augmentation:
  horizontal_flip: true
  color_jitter: true
  random_crop: true
  crop_height: 352
  crop_width: 704

# Depth range
max_depth: 80.0
min_depth: 0.001
```

- [ ] **Step 2: 다운로드 스크립트 작성**

`scripts/download_kitti.py`:
```python
#!/usr/bin/env python3
"""KITTI depth estimation 데이터셋을 다운로드한다.

Eigen split 기준으로 raw data와 depth annotation을 다운로드한다.

Usage:
    python scripts/download_kitti.py --output data/kitti
    python scripts/download_kitti.py --output data/kitti --split-only  # split 파일만
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


KITTI_RAW_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
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

    output_dir = args.output

    if args.split_only:
        download_splits(output_dir)
        return

    if args.depth_only:
        download_depth_annotations(output_dir)
        return

    download_splits(output_dir)
    download_depth_annotations(output_dir)

    print("\n=== KITTI raw data ===")
    print("KITTI raw data는 용량이 크므로 수동 다운로드를 권장합니다.")
    print(f"URL: {KITTI_RAW_URL}")
    print("필요한 시퀀스만 선택적으로 다운로드하세요.")
    print(f"다운로드 후 {output_dir}/ 에 압축 해제하세요.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 데이터 디렉토리 생성 및 커밋**

```bash
mkdir -p data
git add scripts/download_kitti.py config/training/kitti.yaml
git commit -m "feat: add KITTI dataset download script and training config"
```

---

### Task 8: 전체 테스트 실행 및 최종 커밋

- [ ] **Step 1: 전체 테스트 실행**

```bash
cd /Users/khj/YBNML_macmini/depth_estimation
conda activate depth_estimation
PYTHONPATH=src python -m pytest tests/ -v
```
Expected: 모든 테스트 통과 (12 tests passed)

- [ ] **Step 2: 프로젝트 구조 확인**

```bash
find . -type f -not -path './venv/*' -not -path './.git/*' -not -path './catkin_ws2/*' -not -path './paper/*' -not -name '.DS_Store' -not -path './영상/*' | sort
```

Expected output:
```
./.gitignore
./config/camera/wide_stereo_112.yaml
./config/camera/wide_stereo_160.yaml
./config/default.yaml
./config/training/kitti.yaml
./docs/superpowers/plans/2026-04-14-phase1-project-skeleton.md
./docs/superpowers/specs/2026-04-14-depth-estimation-reimplementation-design.md
./requirements.txt
./scripts/download_kitti.py
./scripts/test_offline.py
./setup.py
./src/__init__.py
./src/config.py
./src/evaluation.py
./src/mde/__init__.py
./src/mde/base.py
./src/mde/dummy.py
./src/navigation/__init__.py
./src/pipeline.py
./src/refinement/__init__.py
./src/stereo/__init__.py
./tests/__init__.py
./tests/test_base_mde.py
./tests/test_config.py
./tests/test_evaluation.py
./tests/test_pipeline.py
```
