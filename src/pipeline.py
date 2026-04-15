"""전체 depth estimation 파이프라인 오케스트레이션.

Input(좌/우 RGB) → MDE → (Refinement) → Output(depth maps).

현재 Phase 1-2 단계에서는 MDE까지만 구현되어 있으며,
_refine 은 Phase 3에서 Depth Refinement 구현 후 채워넣는다.

사용 예:
    from config import Config
    from pipeline import DepthEstimationPipeline

    cfg = Config(overrides={"model_name": "dummy"})
    pipe = DepthEstimationPipeline(cfg)
    result = pipe.run(left_rgb, right_rgb)
    # result = {"left_depth": ..., "right_depth": ...}
"""

from typing import Any, Dict

import numpy as np

from config import Config
from mde.base import BaseMDE
from mde.dummy import DummyMDE


def _create_mde(cfg: Config) -> BaseMDE:
    """Config.model_name 에 따라 MDE 모델 인스턴스를 생성.

    지원하는 모델:
        - "dummy":         DummyMDE (테스트용 랜덤 depth)
        - "convnext_mde":  ConvNeXtMDE (TIE 논문 모델)

    `cfg.model_weights` 가 지정되어 있으면 체크포인트를 로드하고 eval 모드로 설정.

    Args:
        cfg: Config 객체.

    Returns:
        BaseMDE 인터페이스를 구현한 모델 인스턴스.

    Raises:
        ValueError: 알 수 없는 model_name.
    """
    name = cfg.model_name
    max_depth = cfg.max_depth

    if name == "dummy":
        return DummyMDE(max_depth=max_depth)

    if name == "convnext_mde":
        # 지연 import: ConvNeXtMDE는 torch+timm 의존성이 있어서
        # dummy 모델만 쓰는 테스트에서는 로드하지 않는다.
        from mde.convnext_mde import ConvNeXtMDE

        model = ConvNeXtMDE(max_depth=max_depth, pretrained=True)
        # 학습된 weight가 지정되어 있으면 로드
        if cfg.get("model_weights"):
            import torch
            state = torch.load(cfg.model_weights, map_location="cpu")
            model.load_state_dict(state)
        # inference 모드로 고정 (dropout/BN 통계 고정)
        model.eval()
        return model

    raise ValueError(f"Unknown model: {name}")


class DepthEstimationPipeline:
    """전체 depth estimation 파이프라인.

    단일 MDE 모델을 보유하고, 좌우 RGB 각각에 대해 depth를 추정한다.
    `refinement_method != "none"` 일 때는 (Phase 3 이후 구현될) refinement도 적용.

    Args:
        cfg: Config 객체. 최소 `model_name`, `max_depth`, `refinement_method` 필요.

    Attributes:
        cfg: 원본 Config.
        mde: 선택된 MDE 모델 (BaseMDE).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mde = _create_mde(cfg)

    def run(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> Dict[str, Any]:
        """좌우 RGB 이미지에서 depth 추정 (+선택적 refinement).

        Args:
            left_rgb: (H, W, 3) uint8 RGB 이미지 (좌측 카메라).
            right_rgb: (H, W, 3) uint8 RGB 이미지 (우측 카메라).

        Returns:
            {
                "left_depth":  (H, W) float32 depth (meters),
                "right_depth": (H, W) float32 depth (meters),
                # Phase 3 후 추가 예정: refined depth, stereo depth, etc.
            }
        """
        left_depth = self.mde.predict(left_rgb)
        right_depth = self.mde.predict(right_rgb)
        result = {"left_depth": left_depth, "right_depth": right_depth}

        # refinement_method=="none" 이면 MDE 결과 그대로 반환
        if self.cfg.refinement_method != "none":
            result = self._refine(result, left_rgb, right_rgb)
        return result

    def _refine(
        self, result: Dict[str, Any], left_rgb: np.ndarray, right_rgb: np.ndarray
    ) -> Dict[str, Any]:
        """Depth refinement (Phase 3에서 구현 예정).

        현재는 placeholder — input 그대로 반환.
        Phase 3: rectification + SGBM stereo + superpixel-based refinement.
        """
        return result
