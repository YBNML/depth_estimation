# Phase 2: MDE 네트워크 (ConvNeXt v2 + LWA Decoder)

> **에이전트 작업자용 안내:** 이 계획을 task 단위로 구현할 때 `superpowers:subagent-driven-development` 스킬을 사용한다.

**목표 (Goal):** TIE 논문의 fast MDE 네트워크 (ConvNeXt v2 encoder + LWA decoder + Scaling block) 를 PyTorch 로 구현하고 KITTI 로 학습한다.

**아키텍처 (Architecture):** `timm` 라이브러리의 pretrained ConvNeXt v2 Tiny backbone 을 encoder 로 사용. 커스텀 LWA decoder, PPM Head, Scaling Block 을 구현해 결합. Scale-Invariant loss 로 학습. Mac Mini 에서 batch=8, 5 epoch 스모크 테스트 → RTX 5070 PC 에서 25 epoch 본격 학습.

**기술 스택 (Tech Stack):** PyTorch 2.x (Mac 은 MPS, Ubuntu 는 CUDA), timm, torchvision, PIL

---

## 파일 구조 (File Structure)

```
src/mde/
├── __init__.py                 # 업데이트 (ConvNeXtMDE export)
├── base.py                     # (existing)
├── dummy.py                    # (existing)
├── convnext_mde.py             # 통합 MDE 모델 (BaseMDE 구현)
├── model/
│   ├── __init__.py
│   ├── encoder.py              # ConvNeXt v2 encoder wrapper
│   ├── ppm_head.py             # Pyramid Pooling Module
│   ├── lwa_decoder.py          # LWA decoder block
│   └── scaling_block.py        # Scaling block
├── loss.py                     # Scale-Invariant loss
├── dataset/
│   ├── __init__.py
│   ├── kitti.py                # KITTI Eigen split dataset
│   ├── nyu.py                  # NYU Depth v2 dataset
│   └── transforms.py           # Augmentation
├── train.py                    # Training 스크립트
└── evaluate.py                 # KITTI test set 평가

scripts/
├── download_nyu.py             # NYU 다운로드
└── train_kitti.py              # 학습 entry point

tests/
├── test_mde_model.py           # 모델 구조 테스트
├── test_mde_loss.py            # Loss 테스트
└── test_mde_dataset.py         # Dataset 테스트
```

---

### Task 1: 의존성 추가 및 환경 확인

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: requirements.txt 업데이트**

`requirements.txt`에 추가:
```
timm>=1.0.0
Pillow>=10.0.0
tqdm>=4.65.0
```

- [ ] **Step 2: 설치**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate depth_estimation
pip install -r requirements.txt
```

- [ ] **Step 3: MPS 동작 확인**

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available()); print('MPS built:', torch.backends.mps.is_built())"
```
Expected: 둘 다 True

- [ ] **Step 4: timm 백본 로드 확인**

```bash
python -c "import timm; m = timm.create_model('convnextv2_tiny', pretrained=True, features_only=True); print([f['num_chs'] for f in m.feature_info])"
```
Expected: `[96, 192, 384, 768]` (4 stage의 채널 수)

- [ ] **Step 5: 커밋**

```bash
git add requirements.txt
git commit -m "chore: add timm, Pillow, tqdm for MDE network"
```

---

### Task 2: ConvNeXt v2 Encoder 래퍼

**Files:**
- Create: `src/mde/model/__init__.py` (empty)
- Create: `src/mde/model/encoder.py`
- Test: `tests/test_mde_model.py`

- [ ] **Step 1: 테스트 작성**

`tests/test_mde_model.py`:
```python
import pytest
import torch

from mde.model.encoder import ConvNeXtV2Encoder


def test_encoder_output_shapes():
    enc = ConvNeXtV2Encoder(variant="convnextv2_tiny", pretrained=False)
    x = torch.randn(1, 3, 352, 704)
    features = enc(x)
    assert len(features) == 4
    # 4x, 8x, 16x, 32x 다운샘플링
    assert features[0].shape == (1, 96, 88, 176)   # H/4
    assert features[1].shape == (1, 192, 44, 88)   # H/8
    assert features[2].shape == (1, 384, 22, 44)   # H/16
    assert features[3].shape == (1, 768, 11, 22)   # H/32


def test_encoder_channels():
    enc = ConvNeXtV2Encoder(variant="convnextv2_tiny", pretrained=False)
    assert enc.channels == [96, 192, 384, 768]
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

```bash
PYTHONPATH=src python -m pytest tests/test_mde_model.py -v
```
Expected: FAIL

- [ ] **Step 3: Encoder 구현**

`src/mde/model/__init__.py` (빈 파일)

`src/mde/model/encoder.py`:
```python
from typing import List

import timm
import torch
import torch.nn as nn


class ConvNeXtV2Encoder(nn.Module):
    """ConvNeXt v2 encoder - timm 기반.

    입력: (B, 3, H, W) RGB
    출력: 4개 stage feature list (H/4, H/8, H/16, H/32)
    """

    def __init__(self, variant: str = "convnextv2_tiny", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
        )
        self.channels: List[int] = [f["num_chs"] for f in self.backbone.feature_info]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
PYTHONPATH=src python -m pytest tests/test_mde_model.py -v
```
Expected: 2 passed

- [ ] **Step 5: 커밋**

```bash
git add src/mde/model/ tests/test_mde_model.py
git commit -m "feat: add ConvNeXt v2 encoder wrapper"
```

---

### Task 3: PPM Head

**Files:**
- Create: `src/mde/model/ppm_head.py`
- Modify: `tests/test_mde_model.py`

- [ ] **Step 1: 테스트 추가**

`tests/test_mde_model.py`에 추가:
```python
from mde.model.ppm_head import PPMHead


def test_ppm_head_output_shape():
    ppm = PPMHead(in_channels=768, out_channels=128, pool_sizes=(1, 2, 3, 6))
    x = torch.randn(1, 768, 11, 22)
    out = ppm(x)
    assert out.shape == (1, 128, 11, 22)
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

```bash
PYTHONPATH=src python -m pytest tests/test_mde_model.py::test_ppm_head_output_shape -v
```
Expected: FAIL

- [ ] **Step 3: PPMHead 구현**

`src/mde/model/ppm_head.py`:
```python
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPMHead(nn.Module):
    """Pyramid Pooling Module Head.

    여러 pool size로 context feature를 추출해 concat 후 1x1 conv로 축소.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        self.pool_sizes = pool_sizes
        branch_ch = in_channels // len(pool_sizes)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            )
            for ps in pool_sizes
        ])

        fused_ch = in_channels + branch_ch * len(pool_sizes)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_ch, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [x]
        for branch in self.branches:
            y = branch(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            outs.append(y)
        return self.fuse(torch.cat(outs, dim=1))
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
PYTHONPATH=src python -m pytest tests/test_mde_model.py -v
```
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add src/mde/model/ppm_head.py tests/test_mde_model.py
git commit -m "feat: add PPM Head for global context feature"
```

---

### Task 4: LWA Decoder Block

**Files:**
- Create: `src/mde/model/lwa_decoder.py`
- Modify: `tests/test_mde_model.py`

- [ ] **Step 1: 테스트 추가**

`tests/test_mde_model.py`에 추가:
```python
from mde.model.lwa_decoder import LWABlock


def test_lwa_block_first_stage():
    # 첫 단계: 7x7 depthwise
    block = LWABlock(local_ch=384, global_ch=128, out_ch=128, kernel_size=7)
    local_feat = torch.randn(1, 384, 22, 44)
    global_feat = torch.randn(1, 128, 11, 22)
    out = block(local_feat, global_feat)
    # 2x upscale
    assert out.shape == (1, 128, 44, 88)


def test_lwa_block_later_stage():
    # 이후 단계: 3x3 depthwise
    block = LWABlock(local_ch=192, global_ch=128, out_ch=128, kernel_size=3)
    local_feat = torch.randn(1, 192, 44, 88)
    global_feat = torch.randn(1, 128, 22, 44)
    out = block(local_feat, global_feat)
    assert out.shape == (1, 128, 88, 176)
```

- [ ] **Step 2: LWABlock 구현**

`src/mde/model/lwa_decoder.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=in_ch, bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class LWABlock(nn.Module):
    """Lightweight Attention decoder block.

    Local feature (encoder stage)와 Global feature (이전 decoder stage)를
    결합해 2x upsample된 feature를 출력한다.
    """

    def __init__(
        self,
        local_ch: int,
        global_ch: int,
        out_ch: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.local_reduce = nn.Conv2d(local_ch, out_ch, kernel_size=1, bias=False)

        self.conv_a = DepthwiseSeparableConv(out_ch + global_ch, out_ch, kernel_size=kernel_size)
        self.conv_b = DepthwiseSeparableConv(out_ch, out_ch, kernel_size=3)

        self.attn = nn.Sequential(
            DepthwiseSeparableConv(out_ch, out_ch, kernel_size=3),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        local_reduced = self.local_reduce(local_feat)
        h, w = local_reduced.shape[-2:]
        global_up = F.interpolate(global_feat, size=(h, w), mode="bilinear", align_corners=False)

        fused = torch.cat([local_reduced, global_up], dim=1)
        fused = self.conv_a(fused)
        fused = self.conv_b(fused)

        attn_map = self.attn(fused)
        out = fused * attn_map

        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        return out
```

- [ ] **Step 3: 테스트 통과 확인**

```bash
PYTHONPATH=src python -m pytest tests/test_mde_model.py -v
```
Expected: 5 passed

- [ ] **Step 4: 커밋**

```bash
git add src/mde/model/lwa_decoder.py tests/test_mde_model.py
git commit -m "feat: add LWA decoder block with depthwise separable conv"
```

---

### Task 5: Scaling Block

**Files:**
- Create: `src/mde/model/scaling_block.py`
- Modify: `tests/test_mde_model.py`

- [ ] **Step 1: 테스트 추가**

`tests/test_mde_model.py`에 추가:
```python
from mde.model.scaling_block import ScalingBlock


def test_scaling_block_output_range():
    block = ScalingBlock(in_channels=128, max_depth=80.0)
    x = torch.randn(1, 128, 352, 704)
    depth = block(x)
    assert depth.shape == (1, 1, 352, 704)
    assert torch.all(depth >= 0.0)
    assert torch.all(depth <= 80.0)


def test_scaling_block_different_max_depth():
    block = ScalingBlock(in_channels=128, max_depth=10.0)
    x = torch.randn(1, 128, 352, 704)
    depth = block(x)
    assert torch.all(depth <= 10.0)
```

- [ ] **Step 2: ScalingBlock 구현**

`src/mde/model/scaling_block.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from mde.model.lwa_decoder import DepthwiseSeparableConv


class HardSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + 3.0) / 6.0, 0.0, 1.0)


class ScalingBlock(nn.Module):
    """Adaptive depth scaling block.

    LWA decoder 최종 출력에서 max_depth에 맞는 depth map을 생성한다.
    7x7 depthwise -> GELU -> 3x3 separable conv -> hard sigmoid -> * max_depth
    """

    def __init__(self, in_channels: int, max_depth: float):
        super().__init__()
        self.max_depth = max_depth

        self.dw7 = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, padding=3,
            groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.gelu = nn.GELU()

        self.sep = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3)
        self.out = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.hard_sigmoid = HardSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gelu(self.bn1(self.dw7(x)))
        y = self.sep(y)
        y = self.out(y)
        y = self.hard_sigmoid(y)
        return y * self.max_depth
```

- [ ] **Step 3: 테스트 통과 확인**

```bash
PYTHONPATH=src python -m pytest tests/test_mde_model.py -v
```
Expected: 7 passed

- [ ] **Step 4: 커밋**

```bash
git add src/mde/model/scaling_block.py tests/test_mde_model.py
git commit -m "feat: add scaling block for adaptive max depth"
```

---

### Task 6: ConvNeXtMDE 전체 모델 조립

**Files:**
- Create: `src/mde/convnext_mde.py`
- Modify: `src/mde/__init__.py`
- Modify: `tests/test_mde_model.py`

- [ ] **Step 1: 테스트 추가**

`tests/test_mde_model.py`에 추가:
```python
import numpy as np
from mde.convnext_mde import ConvNeXtMDE


def test_convnext_mde_forward():
    model = ConvNeXtMDE(max_depth=80.0, pretrained=False)
    x = torch.randn(1, 3, 352, 704)
    depth = model(x)
    # 입력과 같은 해상도의 depth map
    assert depth.shape == (1, 1, 352, 704)
    assert torch.all(depth >= 0.0)
    assert torch.all(depth <= 80.0)


def test_convnext_mde_predict_interface():
    model = ConvNeXtMDE(max_depth=80.0, pretrained=False)
    model.eval()
    rgb = np.random.randint(0, 255, (352, 704, 3), dtype=np.uint8)
    depth = model.predict(rgb)
    assert isinstance(depth, np.ndarray)
    assert depth.shape == (352, 704)
    assert depth.dtype == np.float32


def test_convnext_mde_get_max_depth():
    model = ConvNeXtMDE(max_depth=80.0, pretrained=False)
    assert model.get_max_depth() == 80.0
```

- [ ] **Step 2: ConvNeXtMDE 구현**

`src/mde/convnext_mde.py`:
```python
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mde.base import BaseMDE
from mde.model.encoder import ConvNeXtV2Encoder
from mde.model.lwa_decoder import LWABlock
from mde.model.ppm_head import PPMHead
from mde.model.scaling_block import ScalingBlock


class ConvNeXtMDE(nn.Module, BaseMDE):
    """TIE 논문의 fast MDE 모델.

    ConvNeXt v2 encoder + PPM head + 3 LWA decoder blocks + Scaling block.
    """

    def __init__(
        self,
        max_depth: float = 80.0,
        variant: str = "convnextv2_tiny",
        pretrained: bool = True,
        decoder_ch: int = 128,
    ):
        super().__init__()
        self._max_depth = max_depth

        self.encoder = ConvNeXtV2Encoder(variant=variant, pretrained=pretrained)
        e_ch = self.encoder.channels  # [96, 192, 384, 768]

        # Global feature (encoder 최상위 stage에 PPM 적용)
        self.ppm = PPMHead(in_channels=e_ch[3], out_channels=decoder_ch)

        # 3개 LWA block (위에서 아래로)
        # Block 1: H/16 <- PPM(H/32) + encoder stage 2 (384 ch)
        self.lwa1 = LWABlock(local_ch=e_ch[2], global_ch=decoder_ch, out_ch=decoder_ch, kernel_size=7)
        # Block 2: H/8 <- lwa1(up) + encoder stage 1 (192 ch)
        self.lwa2 = LWABlock(local_ch=e_ch[1], global_ch=decoder_ch, out_ch=decoder_ch, kernel_size=3)
        # Block 3: H/4 <- lwa2(up) + encoder stage 0 (96 ch)
        self.lwa3 = LWABlock(local_ch=e_ch[0], global_ch=decoder_ch, out_ch=decoder_ch, kernel_size=3)

        self.scaling = ScalingBlock(in_channels=decoder_ch, max_depth=max_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = self.encoder(x)  # [f0 (H/4), f1 (H/8), f2 (H/16), f3 (H/32)]

        g = self.ppm(feats[3])             # H/32, decoder_ch
        d1 = self.lwa1(feats[2], g)        # H/8 (2x up from H/16)
        d2 = self.lwa2(feats[1], d1)       # H/4 (2x up from H/8)
        d3 = self.lwa3(feats[0], d2)       # H/2 (2x up from H/4)

        depth = self.scaling(d3)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
        return depth

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """BaseMDE 인터페이스. numpy RGB -> numpy depth."""
        device = next(self.parameters()).device
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            depth = self.forward(x)
        return depth.squeeze().cpu().numpy().astype(np.float32)

    def get_max_depth(self) -> float:
        return self._max_depth
```

- [ ] **Step 3: __init__ 업데이트**

`src/mde/__init__.py`:
```python
from mde.base import BaseMDE
from mde.dummy import DummyMDE
from mde.convnext_mde import ConvNeXtMDE
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
PYTHONPATH=src python -m pytest tests/test_mde_model.py -v
```
Expected: 10 passed (pretrained=False로 테스트)

- [ ] **Step 5: Pipeline 통합**

`src/pipeline.py`의 `_create_mde` 함수 업데이트:
```python
def _create_mde(cfg: Config) -> BaseMDE:
    name = cfg.model_name
    max_depth = cfg.max_depth
    if name == "dummy":
        return DummyMDE(max_depth=max_depth)
    if name == "convnext_mde":
        from mde.convnext_mde import ConvNeXtMDE
        model = ConvNeXtMDE(max_depth=max_depth, pretrained=True)
        if cfg.get("model_weights"):
            import torch
            state = torch.load(cfg.model_weights, map_location="cpu")
            model.load_state_dict(state)
        model.eval()
        return model
    raise ValueError(f"Unknown model: {name}")
```

- [ ] **Step 6: 전체 테스트 실행**

```bash
PYTHONPATH=src python -m pytest tests/ -v
```
Expected: 전부 통과

- [ ] **Step 7: 커밋**

```bash
git add src/mde/ tests/test_mde_model.py src/pipeline.py
git commit -m "feat: assemble ConvNeXtMDE model with encoder/PPM/LWA/scaling"
```

---

### Task 7: Scale-Invariant Loss

**Files:**
- Create: `src/mde/loss.py`
- Test: `tests/test_mde_loss.py`

- [ ] **Step 1: 테스트 작성**

`tests/test_mde_loss.py`:
```python
import torch

from mde.loss import ScaleInvariantLoss


def test_si_loss_zero_when_perfect():
    loss_fn = ScaleInvariantLoss(alpha=10.0, lambd=0.85)
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    gt = pred.clone()
    mask = torch.ones_like(pred, dtype=torch.bool)
    loss = loss_fn(pred, gt, mask)
    assert loss.item() == 0.0


def test_si_loss_positive_when_different():
    loss_fn = ScaleInvariantLoss(alpha=10.0, lambd=0.85)
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    gt = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
    mask = torch.ones_like(pred, dtype=torch.bool)
    loss = loss_fn(pred, gt, mask)
    # Scale invariant: 모든 pred = gt * k 같은 비율이므로 loss가 매우 작음
    assert loss.item() < 0.01


def test_si_loss_masks_invalid():
    loss_fn = ScaleInvariantLoss(alpha=10.0, lambd=0.85)
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    gt = torch.tensor([[[[0.0, 2.0], [3.0, 0.0]]]])  # 0은 invalid
    mask = gt > 0
    # mask된 픽셀만 사용 -> 완벽 일치
    loss = loss_fn(pred, gt, mask)
    assert loss.item() == 0.0
```

- [ ] **Step 2: ScaleInvariantLoss 구현**

`src/mde/loss.py`:
```python
import torch
import torch.nn as nn


class ScaleInvariantLoss(nn.Module):
    """Scale-Invariant loss (Eigen et al. 2014).

    L = alpha * sqrt((1/T) * sum(g_i^2) - (lambd/T^2) * (sum g_i)^2)
    where g_i = log(pred) - log(gt).
    """

    def __init__(self, alpha: float = 10.0, lambd: float = 0.85, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.lambd = lambd
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_v = pred[mask].clamp(min=self.eps)
        gt_v = gt[mask].clamp(min=self.eps)

        g = torch.log(pred_v) - torch.log(gt_v)
        n = g.numel()
        if n == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        term1 = (g ** 2).mean()
        term2 = self.lambd * (g.mean() ** 2)
        loss = torch.sqrt((term1 - term2).clamp(min=self.eps))
        return self.alpha * loss
```

- [ ] **Step 3: 테스트 실행**

```bash
PYTHONPATH=src python -m pytest tests/test_mde_loss.py -v
```
Expected: 3 passed

- [ ] **Step 4: 커밋**

```bash
git add src/mde/loss.py tests/test_mde_loss.py
git commit -m "feat: add Scale-Invariant loss for depth estimation"
```

---

### Task 8: KITTI Dataset + Augmentation

**Files:**
- Create: `src/mde/dataset/__init__.py`
- Create: `src/mde/dataset/kitti.py`
- Create: `src/mde/dataset/transforms.py`
- Test: `tests/test_mde_dataset.py`

- [ ] **Step 1: 테스트 작성**

`tests/test_mde_dataset.py`:
```python
import pytest
import torch

from mde.dataset.transforms import DepthAugmentation


def test_augmentation_shapes():
    import numpy as np
    aug = DepthAugmentation(crop_height=352, crop_width=704, training=True)
    rgb = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
    depth = np.random.uniform(0, 80, (375, 1242)).astype(np.float32)
    rgb_t, depth_t = aug(rgb, depth)
    assert rgb_t.shape == (3, 352, 704)
    assert depth_t.shape == (1, 352, 704)


def test_augmentation_eval_no_crop():
    import numpy as np
    aug = DepthAugmentation(crop_height=352, crop_width=704, training=False)
    rgb = np.random.randint(0, 255, (352, 704, 3), dtype=np.uint8)
    depth = np.random.uniform(0, 80, (352, 704)).astype(np.float32)
    rgb_t, depth_t = aug(rgb, depth)
    assert rgb_t.shape == (3, 352, 704)
    assert depth_t.shape == (1, 352, 704)
```

- [ ] **Step 2: Augmentation 구현**

`src/mde/dataset/__init__.py` (빈 파일)

`src/mde/dataset/transforms.py`:
```python
import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF


class DepthAugmentation:
    """RGB + depth 동시 변환."""

    def __init__(
        self,
        crop_height: int = 352,
        crop_width: int = 704,
        training: bool = True,
    ):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.training = training

    def __call__(
        self, rgb: np.ndarray, depth: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()

        if self.training:
            # Random crop
            h, w = rgb_t.shape[-2:]
            if h > self.crop_height and w > self.crop_width:
                top = random.randint(0, h - self.crop_height)
                left = random.randint(0, w - self.crop_width)
                rgb_t = rgb_t[:, top:top + self.crop_height, left:left + self.crop_width]
                depth_t = depth_t[:, top:top + self.crop_height, left:left + self.crop_width]

            # Horizontal flip
            if random.random() < 0.5:
                rgb_t = torch.flip(rgb_t, dims=[-1])
                depth_t = torch.flip(depth_t, dims=[-1])

            # Color jitter (brightness/contrast/saturation)
            if random.random() < 0.5:
                rgb_t = TF.adjust_brightness(rgb_t, random.uniform(0.8, 1.2))
                rgb_t = TF.adjust_contrast(rgb_t, random.uniform(0.8, 1.2))
                rgb_t = TF.adjust_saturation(rgb_t, random.uniform(0.8, 1.2))
                rgb_t = torch.clamp(rgb_t, 0.0, 1.0)
        else:
            # Center crop if larger
            h, w = rgb_t.shape[-2:]
            if h >= self.crop_height and w >= self.crop_width:
                top = (h - self.crop_height) // 2
                left = (w - self.crop_width) // 2
                rgb_t = rgb_t[:, top:top + self.crop_height, left:left + self.crop_width]
                depth_t = depth_t[:, top:top + self.crop_height, left:left + self.crop_width]

        return rgb_t, depth_t
```

- [ ] **Step 3: KITTI Dataset 구현**

`src/mde/dataset/kitti.py`:
```python
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from mde.dataset.transforms import DepthAugmentation


def _read_depth_png(path: str) -> np.ndarray:
    """KITTI depth PNG (uint16) -> meters (float32)."""
    d = np.array(Image.open(path), dtype=np.uint16)
    return d.astype(np.float32) / 256.0


class KITTIDepthDataset(Dataset):
    """KITTI Eigen split depth dataset.

    split_file 형식: "<seq_path> <image_idx> <side>" (e.g. "2011_09_26/2011_09_26_drive_0001_sync 0000000000 l")
    raw_dir: KITTI raw data 루트
    depth_dir: depth annotation 루트 (data_depth_annotated)
    """

    def __init__(
        self,
        split_file: str,
        raw_dir: str,
        depth_dir: str,
        crop_height: int = 352,
        crop_width: int = 704,
        training: bool = True,
    ):
        self.raw_dir = Path(raw_dir)
        self.depth_dir = Path(depth_dir)
        self.transform = DepthAugmentation(crop_height, crop_width, training=training)

        with open(split_file, "r") as f:
            self.samples = [line.strip().split() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def _image_paths(self, seq: str, idx: str, side: str) -> Tuple[str, str]:
        cam = "image_02" if side == "l" else "image_03"
        rgb_path = self.raw_dir / seq / cam / "data" / f"{idx}.png"
        # depth annotation 경로: train/{sequence}/proj_depth/groundtruth/image_02/{idx}.png
        # sequence 경로에서 상위 디렉토리 분리
        seq_parent = Path(seq).parent
        seq_name = Path(seq).name
        depth_path = (
            self.depth_dir / "train" / seq_name /
            "proj_depth" / "groundtruth" / cam / f"{idx}.png"
        )
        # val split에도 있을 수 있음
        if not depth_path.exists():
            depth_path = (
                self.depth_dir / "val" / seq_name /
                "proj_depth" / "groundtruth" / cam / f"{idx}.png"
            )
        return str(rgb_path), str(depth_path)

    def __getitem__(self, idx: int):
        if len(self.samples[idx]) == 3:
            seq, img_idx, side = self.samples[idx]
        else:
            seq, img_idx = self.samples[idx][:2]
            side = "l"

        rgb_path, depth_path = self._image_paths(seq, img_idx, side)

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        depth = _read_depth_png(depth_path)

        rgb_t, depth_t = self.transform(rgb, depth)
        return rgb_t, depth_t
```

- [ ] **Step 4: 테스트 실행**

```bash
PYTHONPATH=src python -m pytest tests/test_mde_dataset.py -v
```
Expected: 2 passed

- [ ] **Step 5: 커밋**

```bash
git add src/mde/dataset/ tests/test_mde_dataset.py
git commit -m "feat: add KITTI dataset loader and depth augmentation"
```

---

### Task 9: NYU Dataset + 다운로드 스크립트 (준비만)

**Files:**
- Create: `src/mde/dataset/nyu.py`
- Create: `scripts/download_nyu.py`

- [ ] **Step 1: NYU Dataset 구현**

`src/mde/dataset/nyu.py`:
```python
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset

from mde.dataset.transforms import DepthAugmentation


class NYUDepthDataset(Dataset):
    """NYU Depth v2 labeled dataset (.mat 또는 h5 파일 기반).

    split_file: 각 줄이 "rgb_path depth_path" 형식.
    root_dir: NYU 데이터셋 루트.
    """

    def __init__(
        self,
        split_file: str,
        root_dir: str,
        crop_height: int = 416,
        crop_width: int = 544,
        training: bool = True,
    ):
        self.root = Path(root_dir)
        self.transform = DepthAugmentation(crop_height, crop_width, training=training)

        with open(split_file, "r") as f:
            self.samples = [line.strip().split() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image

        rgb_path, depth_path = self.samples[idx]
        rgb = np.array(Image.open(self.root / rgb_path).convert("RGB"))

        if depth_path.endswith(".png"):
            depth = np.array(Image.open(self.root / depth_path), dtype=np.uint16).astype(np.float32) / 1000.0
        elif depth_path.endswith(".npy"):
            depth = np.load(self.root / depth_path).astype(np.float32)
        else:
            raise ValueError(f"Unsupported depth format: {depth_path}")

        return self.transform(rgb, depth)
```

- [ ] **Step 2: NYU 다운로드 스크립트**

`scripts/download_nyu.py`:
```python
#!/usr/bin/env python3
"""NYU Depth v2 dataset 다운로드.

Adabins/BTS에서 제공하는 전처리된 버전을 다운로드한다.
원본 mat 파일보다 이미지 단위로 분리되어 있어 학습에 편리.

Usage:
    python scripts/download_nyu.py --output data/nyu
"""
import argparse
import os
import subprocess
import sys


NYU_TRAIN_URL = "https://tinyurl.com/nyu-data-zip"  # placeholder; 실제로는 BTS/Adabins 링크 사용
NYU_TEST_URL = "https://tinyurl.com/nyu-test-zip"


def download_file(url: str, output_path: str) -> None:
    print(f"Downloading: {url} -> {output_path}")
    subprocess.run(["curl", "-L", "-o", output_path, url], check=True)


def main():
    parser = argparse.ArgumentParser(description="Download NYU Depth v2 dataset")
    parser.add_argument("--output", type=str, default="data/nyu", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print("NYU Depth v2 다운로드는 수동으로 진행을 권장합니다.")
    print("\n추천 소스:")
    print("  1. BTS 저장소: https://github.com/cleinc/bts (README의 데이터 링크 참조)")
    print("  2. Adabins 저장소: https://github.com/shariqfarooq123/AdaBins")
    print("  3. 공식 NYU v2: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html")
    print(f"\n다운로드 후 {args.output}/ 에 압축 해제하세요.")
    print("\n예상 디렉토리 구조:")
    print(f"  {args.output}/sync/ (RGB+depth 페어)")
    print(f"  {args.output}/official_splits/train/")
    print(f"  {args.output}/official_splits/test/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: requirements에 h5py 추가**

`requirements.txt`에 추가:
```
h5py>=3.8.0
```

```bash
pip install h5py
```

- [ ] **Step 4: 커밋**

```bash
git add src/mde/dataset/nyu.py scripts/download_nyu.py requirements.txt
git commit -m "feat: add NYU Depth v2 dataset loader and download guide"
```

---

### Task 10: Training Script

**Files:**
- Create: `src/mde/train.py`
- Create: `scripts/train_kitti.py`
- Modify: `config/training/kitti.yaml` (추가 설정)

- [ ] **Step 1: 학습 루프 구현**

`src/mde/train.py`:
```python
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from mde.convnext_mde import ConvNeXtMDE
from mde.dataset.kitti import KITTIDepthDataset
from mde.loss import ScaleInvariantLoss


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: Dict[str, Any]) -> None:
    device = get_device()
    print(f"Device: {device}")

    # Dataset
    train_ds = KITTIDepthDataset(
        split_file=cfg["train_file"],
        raw_dir=cfg["raw_dir"],
        depth_dir=cfg["depth_dir"],
        crop_height=cfg["augmentation"]["crop_height"],
        crop_width=cfg["augmentation"]["crop_width"],
        training=True,
    )
    val_ds = KITTIDepthDataset(
        split_file=cfg["val_file"],
        raw_dir=cfg["raw_dir"],
        depth_dir=cfg["depth_dir"],
        crop_height=cfg["augmentation"]["crop_height"],
        crop_width=cfg["augmentation"]["crop_width"],
        training=False,
    )

    num_workers = cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # Model, loss, optimizer
    model = ConvNeXtMDE(max_depth=cfg["max_depth"], pretrained=True).to(device)
    loss_fn = ScaleInvariantLoss(alpha=cfg["si_alpha"], lambd=cfg["si_lambda"])
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    weights_dir = Path(cfg.get("weights_dir", "weights"))
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        t0 = time.time()

        for rgb, gt_depth in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            rgb = rgb.to(device)
            gt_depth = gt_depth.to(device)

            pred = model(rgb)
            mask = (gt_depth > cfg["min_depth"]) & (gt_depth < cfg["max_depth"])
            if mask.sum() == 0:
                continue

            loss = loss_fn(pred, gt_depth, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = train_loss_sum / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for rgb, gt_depth in val_loader:
                rgb = rgb.to(device)
                gt_depth = gt_depth.to(device)
                pred = model(rgb)
                mask = (gt_depth > cfg["min_depth"]) & (gt_depth < cfg["max_depth"])
                if mask.sum() == 0:
                    continue
                val_loss_sum += loss_fn(pred, gt_depth, mask).item()
                n_val += 1
        val_loss = val_loss_sum / max(n_val, 1)

        dt = time.time() - t0
        print(f"[Epoch {epoch+1}] train={train_loss:.4f} val={val_loss:.4f} time={dt:.1f}s lr={scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        ckpt_path = weights_dir / f"convnext_mde_epoch{epoch+1:02d}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  saved -> {ckpt_path}")
```

- [ ] **Step 2: 학습 entry point**

`scripts/train_kitti.py`:
```python
#!/usr/bin/env python3
"""KITTI 데이터셋으로 ConvNeXtMDE 학습.

Usage:
    python scripts/train_kitti.py
    python scripts/train_kitti.py --epochs 5 --batch-size 8   # Mac Mini 스모크 테스트
    python scripts/train_kitti.py --epochs 25 --batch-size 8  # RTX 5070 본격 학습
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mde.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/training/kitti.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--raw-dir", type=str, default="data/kitti/raw")
    parser.add_argument("--depth-dir", type=str, default="data/kitti")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    cfg["raw_dir"] = args.raw_dir
    cfg["depth_dir"] = args.depth_dir
    cfg["num_workers"] = args.num_workers

    train(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 커밋**

```bash
git add src/mde/train.py scripts/train_kitti.py
git commit -m "feat: add KITTI training loop with AdamW + cosine LR"
```

---

### Task 11: Mac Mini 스모크 테스트 (forward only)

**Files:**
- Create: `scripts/smoke_test_mde.py`

- [ ] **Step 1: 스모크 테스트 스크립트**

`scripts/smoke_test_mde.py`:
```python
#!/usr/bin/env python3
"""ConvNeXtMDE forward pass + backward pass 스모크 테스트.

실제 데이터 없이 random tensor로 한 iteration이 문제없이 돌아가는지 확인.
Mac Mini MPS 에서 메모리/속도 체크용.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mde.convnext_mde import ConvNeXtMDE
from mde.loss import ScaleInvariantLoss
from mde.train import get_device


def main():
    device = get_device()
    print(f"Device: {device}")

    model = ConvNeXtMDE(max_depth=80.0, pretrained=False).to(device)
    loss_fn = ScaleInvariantLoss(alpha=10.0, lambd=0.85)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_size = 8
    h, w = 352, 704

    print(f"Batch size: {batch_size}, Image: {h}x{w}")
    rgb = torch.randn(batch_size, 3, h, w, device=device)
    gt = torch.rand(batch_size, 1, h, w, device=device) * 80.0
    mask = torch.ones_like(gt, dtype=torch.bool)

    # Warmup
    for _ in range(2):
        pred = model(rgb)
        loss = loss_fn(pred, gt, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()

    t0 = time.time()
    n_iter = 5
    for _ in range(n_iter):
        pred = model(rgb)
        loss = loss_fn(pred, gt, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()
    dt = (time.time() - t0) / n_iter

    print(f"Avg iteration time: {dt*1000:.1f}ms")
    print(f"Final loss: {loss.item():.4f}")
    print("Smoke test PASSED.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Mac Mini에서 실행**

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate depth_estimation
python scripts/smoke_test_mde.py
```
Expected: "Smoke test PASSED." 출력 + iteration time 표시 (Mac Mini MPS 기준 약 1-3초/iter 예상)

- [ ] **Step 3: 커밋**

```bash
git add scripts/smoke_test_mde.py
git commit -m "feat: add MDE forward/backward smoke test"
```

---

### Task 12: Evaluation Script

**Files:**
- Create: `src/mde/evaluate.py`
- Create: `scripts/evaluate_mde.py`

- [ ] **Step 1: Evaluation 구현**

`src/mde/evaluate.py`:
```python
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation import compute_metrics
from mde.convnext_mde import ConvNeXtMDE
from mde.dataset.kitti import KITTIDepthDataset
from mde.train import get_device


def evaluate_kitti(cfg: Dict[str, Any], weights_path: str) -> Dict[str, float]:
    device = get_device()

    model = ConvNeXtMDE(max_depth=cfg["max_depth"], pretrained=False).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_ds = KITTIDepthDataset(
        split_file=cfg["test_file"],
        raw_dir=cfg["raw_dir"],
        depth_dir=cfg["depth_dir"],
        crop_height=cfg["augmentation"]["crop_height"],
        crop_width=cfg["augmentation"]["crop_width"],
        training=False,
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    all_metrics = {"delta1": [], "delta2": [], "delta3": [], "absrel": [], "rmse": []}

    with torch.no_grad():
        for rgb, gt in tqdm(loader, desc="Evaluating"):
            rgb = rgb.to(device)
            gt = gt.to(device)
            pred = model(rgb)

            pred_np = pred.squeeze().cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()

            metrics = compute_metrics(pred_np, gt_np, min_depth=cfg["min_depth"])
            for k, v in metrics.items():
                all_metrics[k].append(v)

    avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    return avg
```

- [ ] **Step 2: Entry point**

`scripts/evaluate_mde.py`:
```python
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
```

- [ ] **Step 3: 커밋**

```bash
git add src/mde/evaluate.py scripts/evaluate_mde.py
git commit -m "feat: add KITTI evaluation script for trained MDE"
```

---

### Task 13: 전체 테스트 및 확인

- [ ] **Step 1: 전체 테스트 실행**

```bash
PYTHONPATH=src python -m pytest tests/ -v
```
Expected: 전체 테스트 통과

- [ ] **Step 2: 프로젝트 구조 확인**

```bash
find src scripts tests config -type f | sort
```

- [ ] **Step 3: Phase 2 완료 요약 메시지**

모든 테스트 통과, 스모크 테스트 동작 확인. 실제 학습은 KITTI 데이터 다운로드 후 실행.
