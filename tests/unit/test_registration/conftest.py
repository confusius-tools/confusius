"""Shared fixtures for registration tests."""

import numpy as np
import pytest
import SimpleITK as sitk

from confusius.xarray import create_fusi_dataarray


@pytest.fixture
def sample_2d_image():
    """Single-slice 3D NumPy array with a bright square in the centre (1x32x32)."""
    img = np.zeros((1, 32, 32), dtype=np.float32)
    img[:, 12:20, 12:20] = 100.0
    return img


@pytest.fixture
def sample_3d_array():
    """3D NumPy array with a bright cube in the centre (16x16x16)."""
    vol = np.zeros((16, 16, 16), dtype=np.float32)
    vol[6:10, 6:10, 6:10] = 100.0
    return vol


@pytest.fixture
def sample_2d_dataarray_spatial(sample_2d_image):
    """Singleton-k (k, j, i) DataArray wrapping sample_2d_image with 0.1 mm spacing."""
    return create_fusi_dataarray(
        sample_2d_image,
        dims=("k", "j", "i"),
        spacing=(1.0, 0.1, 0.1),
        origin=(0.0, 0.0, 0.0),
    )


@pytest.fixture
def sample_3d_dataarray_spatial(sample_3d_array):
    """Spatial (k, j, i) DataArray wrapping sample_3d_array with unit spacing."""
    return create_fusi_dataarray(
        sample_3d_array,
        dims=("k", "j", "i"),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )


@pytest.fixture
def sample_3d_texture_array():
    """3D NumPy array with several off-centre cubes of different sizes (32x32x32).

    Unlike `sample_3d_array`'s single centred cube, this scatters asymmetric
    features across the volume so every region -- including each B-spline
    control point's compact support -- has non-flat image content to constrain
    it. A single blob on an otherwise empty background leaves most control
    points with near-zero gradient information, which is what makes
    `mesh_size=(10, 10, 10)` (this project's default) numerically unstable on
    `sample_3d_array` or `sample_2d_image`.
    """
    vol = np.zeros((32, 32, 32), dtype=np.float32)
    vol[4:10, 4:10, 4:10] = 60.0
    vol[20:28, 6:12, 18:24] = 100.0
    vol[8:14, 20:26, 6:12] = 80.0
    vol[18:24, 20:30, 20:26] = 40.0
    vol[14:18, 14:18, 14:18] = 120.0
    return vol


@pytest.fixture
def sample_3d_dataarray_texture_spatial(sample_3d_texture_array):
    """Spatial (k, j, i) DataArray wrapping sample_3d_texture_array with unit spacing."""
    return create_fusi_dataarray(
        sample_3d_texture_array,
        dims=("k", "j", "i"),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )


@pytest.fixture
def sample_2d_dataarray(sample_2d_image):
    """Singleton-k+time DataArray (5 frames) for volumewise registration tests."""
    n_frames = 5
    data = np.stack([sample_2d_image] * n_frames, axis=0)
    return create_fusi_dataarray(
        data,
        dims=("time", "k", "j", "i"),
        time=np.arange(n_frames) * 0.1,
        spacing=(1.0, 0.1, 0.1),
        origin=(0.0, 0.0, 0.0),
    )


@pytest.fixture
def sample_3d_dataarray(sample_3d_array):
    """3D+time DataArray (3 frames) for volumewise registration tests."""
    n_frames = 3
    data = np.stack([sample_3d_array] * n_frames, axis=0)
    return create_fusi_dataarray(
        data,
        dims=("time", "k", "j", "i"),
        time=np.arange(n_frames) * 0.1,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
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
