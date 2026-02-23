"""Shared fixtures for registration tests."""

import numpy as np
import pytest
import SimpleITK as sitk
import xarray as xr


@pytest.fixture
def sample_2d_image():
    """2D NumPy array with a bright square in the centre (32x32)."""
    img = np.zeros((32, 32), dtype=np.float32)
    img[12:20, 12:20] = 100.0
    return img


@pytest.fixture
def sample_3d_array():
    """3D NumPy array with a bright cube in the centre (16x16x16)."""
    vol = np.zeros((16, 16, 16), dtype=np.float32)
    vol[6:10, 6:10, 6:10] = 100.0
    return vol


@pytest.fixture
def sample_2d_dataarray_spatial(sample_2d_image):
    """Spatial (y, x) DataArray wrapping sample_2d_image with 0.1 mm spacing."""
    return xr.DataArray(
        sample_2d_image,
        dims=("y", "x"),
        coords={"y": np.arange(32) * 0.1, "x": np.arange(32) * 0.1},
    )


@pytest.fixture
def sample_3d_dataarray_spatial(sample_3d_array):
    """Spatial (z, y, x) DataArray wrapping sample_3d_array with unit spacing."""
    return xr.DataArray(
        sample_3d_array,
        dims=("z", "y", "x"),
        coords={
            "z": np.arange(16) * 1.0,
            "y": np.arange(16) * 1.0,
            "x": np.arange(16) * 1.0,
        },
    )


@pytest.fixture
def sample_2d_dataarray(sample_2d_image):
    """2D+time DataArray (5 frames) for volumewise registration tests."""
    n_frames = 5
    data = np.stack([sample_2d_image] * n_frames, axis=0)
    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": np.arange(n_frames) * 0.1,
            "y": np.arange(32) * 0.1,
            "x": np.arange(32) * 0.1,
        },
    )


@pytest.fixture
def sample_3d_dataarray(sample_3d_array):
    """3D+time DataArray (3 frames) for volumewise registration tests."""
    n_frames = 3
    data = np.stack([sample_3d_array] * n_frames, axis=0)
    return xr.DataArray(
        data,
        dims=("time", "z", "y", "x"),
        coords={
            "time": np.arange(n_frames) * 0.1,
            "z": np.arange(16) * 1.0,
            "y": np.arange(16) * 1.0,
            "x": np.arange(16) * 1.0,
        },
    )


@pytest.fixture
def translation_transform_2d():
    """2D translation transform with known offset (tx=2, ty=3)."""
    t = sitk.TranslationTransform(2)
    t.SetOffset((2.0, 3.0))
    return t


@pytest.fixture
def euler_transform_2d():
    """2D Euler transform with rotation ~5.7° and translation (1.5, 2.5)."""
    t = sitk.Euler2DTransform()
    t.SetAngle(0.1)
    t.SetTranslation((1.5, 2.5))
    return t


@pytest.fixture
def translation_transform_3d():
    """3D translation transform with known offset (tx=1, ty=2, tz=3)."""
    t = sitk.TranslationTransform(3)
    t.SetOffset((1.0, 2.0, 3.0))
    return t


@pytest.fixture
def euler_transform_3d():
    """3D Euler transform with rotations (0.05, 0.1, 0.15) rad and translation (1, 2, 3)."""
    t = sitk.Euler3DTransform()
    t.SetRotation(0.05, 0.1, 0.15)
    t.SetTranslation((1.0, 2.0, 3.0))
    return t
