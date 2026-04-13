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
