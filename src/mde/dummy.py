"""테스트용 Dummy MDE 모델.

실제 학습이나 네트워크 없이 pipeline 동작을 검증하기 위한 placeholder 구현.
랜덤 depth를 반환하지만 BaseMDE 인터페이스를 준수하므로 pipeline에 꽂아 쓸 수 있다.

테스트와 데모 용도로만 사용하고, 실제 시스템에서는 ConvNeXtMDE 등을 쓴다.
"""

import numpy as np

from mde.base import BaseMDE


class DummyMDE(BaseMDE):
    """랜덤 depth를 생성하는 더미 MDE.

    Args:
        max_depth: 출력할 depth의 상한 (meters).
    """

    def __init__(self, max_depth: float = 10.0):
        self._max_depth = max_depth

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """
        Args:
            rgb: (H, W, 3) uint8 RGB. 값은 무시되고 크기만 사용.

        Returns:
            (H, W) float32 랜덤 depth, 값 범위 [0.1, max_depth].
        """
        h, w = rgb.shape[:2]
        # 0에 가까운 값 방지 위해 0.1부터 시작
        return np.random.uniform(0.1, self._max_depth, (h, w)).astype(np.float32)

    def get_max_depth(self) -> float:
        return self._max_depth
