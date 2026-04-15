"""Monocular Depth Estimation (MDE) 모델의 공통 추상 인터페이스.

이 인터페이스를 구현하는 모든 모델 (DummyMDE, ConvNeXtMDE, 향후 추가될
다른 모델)은 pipeline/refinement/navigation 모듈에서 교체 가능하게 사용된다.

원칙:
    - predict(rgb)는 numpy in/out (ROS 노드, 오프라인 스크립트 모두에서 편함).
    - get_max_depth()로 모델이 출력할 수 있는 최대 depth를 알려준다 (refinement 등에서 필요).
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseMDE(ABC):
    """Monocular Depth Estimation 추상 클래스.

    Subclass 예시:
        - DummyMDE: 테스트용 랜덤 depth
        - ConvNeXtMDE: TIE 논문 모델
        - (향후) AdabinsMDE, MiDaSMDE 등
    """

    @abstractmethod
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """RGB 이미지 한 장에서 depth map을 추정.

        Args:
            rgb: (H, W, 3) uint8 RGB 이미지.

        Returns:
            (H, W) float32 depth map (meters).
        """
        ...

    @abstractmethod
    def get_max_depth(self) -> float:
        """모델이 출력할 수 있는 최대 depth (meters).

        예: KITTI 모델 80.0, NYU 모델 10.0.
        """
        ...
