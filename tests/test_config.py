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
