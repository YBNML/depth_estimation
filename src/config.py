"""프로젝트 전역 설정 관리.

YAML 파일에서 설정을 로드하고, 필요시 runtime override를 허용한다.

디렉토리 규약:
    <project_root>/
    ├── config/
    │   ├── default.yaml            # 공통 설정 (모델 종류, 해상도 등)
    │   ├── camera/
    │   │   ├── wide_stereo_112.yaml  # 졸업논문 카메라 (82° x 2)
    │   │   └── wide_stereo_160.yaml  # TIE 논문 카메라 (110° x 2)
    │   └── training/
    │       └── kitti.yaml          # 학습 하이퍼파라미터
    └── src/

사용 예:
    from config import Config

    cfg = Config()                              # default.yaml 로드
    cfg = Config(overrides={"model_name": "dummy"})
    cam = cfg.load_camera("wide_stereo_160")    # 카메라 YAML 별도 로드

    cfg.model_name   # attribute 접근
    cfg.get("optional_key", default_value)
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# 이 파일 위치 기준으로 프로젝트 루트 결정: src/config.py -> <root>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:
    """YAML 기반 설정 객체.

    Args:
        config_path: 로드할 YAML 파일 경로. None이면 config/default.yaml.
        overrides: YAML 값을 덮어쓸 dict. CLI 인자로 설정을 override할 때 유용.

    Attributes:
        _data: 파싱된 설정 dict (내부용).
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        if config_path is None:
            config_path = str(_PROJECT_ROOT / "config" / "default.yaml")

        with open(config_path, "r") as f:
            self._data = yaml.safe_load(f)

        # overrides 는 YAML 로드 후 적용 (CLI 인자 등이 최우선)
        if overrides:
            self._data.update(overrides)

    def __getattr__(self, name: str) -> Any:
        """cfg.key 형태의 attribute 접근을 dict 조회로 변환.

        `_`로 시작하는 이름은 내부 속성으로 간주해 기본 동작 유지.
        """
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def load_camera(self, camera_name: str) -> Dict[str, Any]:
        """카메라 YAML을 로드해 dict로 반환.

        Args:
            camera_name: config/camera/<name>.yaml 의 basename.

        Returns:
            카메라 파라미터 dict (left, right, baseline, total_hfov 등).
        """
        camera_path = _PROJECT_ROOT / "config" / "camera" / f"{camera_name}.yaml"
        with open(camera_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """존재하지 않는 키일 때 default를 반환하는 안전한 조회."""
        return self._data.get(key, default)

    @property
    def project_root(self) -> Path:
        """프로젝트 루트 경로 (scripts에서 상대 경로 해석용)."""
        return _PROJECT_ROOT
