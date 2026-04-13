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
