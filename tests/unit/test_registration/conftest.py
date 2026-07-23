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
def sample_singleton_z_dataarray_spatial(sample_2d_image):
    """Single-slice spatial (1, y, x) DataArray wrapping sample_2d_image.

    Single-slice fUSI data is represented as 3D with a singleton `z` axis (0.2 mm
    thick) and 0.1 mm in-plane spacing.
    """
    return xr.DataArray(
        sample_2d_image[np.newaxis, :, :],
        dims=("z", "y", "x"),
        coords={
            "z": xr.DataArray([0.0], dims=("z",), attrs={"voxdim": 0.2}),
            "y": np.arange(32) * 0.1,
            "x": np.arange(32) * 0.1,
        },
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
def sample_singleton_z_dataarray(sample_2d_image):
    """Single-slice time-varying (time, 1, y, x) DataArray (5 frames).

    Single-slice fUSI recordings are represented as 3D+time data with a singleton
    `z` axis (0.2 mm thick) and 0.1 mm in-plane spacing.
    """
    n_frames = 5
    data = np.stack([sample_2d_image] * n_frames, axis=0)[:, np.newaxis, :, :]
    return xr.DataArray(
        data,
        dims=("time", "z", "y", "x"),
        coords={
            "time": np.arange(n_frames) * 0.1,
            "z": xr.DataArray([0.0], dims=("z",), attrs={"voxdim": 0.2}),
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
