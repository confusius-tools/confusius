"""Shared fixtures for registration tests."""

import numpy as np
import pytest
import SimpleITK as sitk
import xarray as xr

from confusius._utils.geometry import add_physical_coords_from_voxel_affine


def _add_voxel_affine(
    data: xr.DataArray,
    voxel_dims: tuple[str, ...],
    spacing: tuple[float, ...],
) -> xr.DataArray:
    affine = np.eye(len(voxel_dims) + 1, dtype=np.float64)
    affine[:-1, :-1] = np.diag(spacing)
    physical_names = ("y", "x") if len(voxel_dims) == 2 else ("z", "y", "x")
    return add_physical_coords_from_voxel_affine(
        data,
        affine,
        voxel_dims=voxel_dims,
        physical_coord_names=physical_names,
        physical_coord_attrs={
            name: {"units": "mm", "voxdim": step}
            for name, step in zip(physical_names, spacing, strict=True)
        },
    )


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
    """Spatial (j, i) DataArray wrapping sample_2d_image with 0.1 mm spacing."""
    da = xr.DataArray(
        sample_2d_image,
        dims=("j", "i"),
        coords={"j": np.arange(32), "i": np.arange(32)},
    )
    return _add_voxel_affine(da, ("j", "i"), (0.1, 0.1))


@pytest.fixture
def sample_3d_dataarray_spatial(sample_3d_array):
    """Spatial (k, j, i) DataArray wrapping sample_3d_array with unit spacing."""
    da = xr.DataArray(
        sample_3d_array,
        dims=("k", "j", "i"),
        coords={"k": np.arange(16), "j": np.arange(16), "i": np.arange(16)},
    )
    return _add_voxel_affine(da, ("k", "j", "i"), (1.0, 1.0, 1.0))


@pytest.fixture
def sample_2d_dataarray(sample_2d_image):
    """2D+time DataArray (5 frames) for volumewise registration tests."""
    n_frames = 5
    data = np.stack([sample_2d_image] * n_frames, axis=0)
    da = xr.DataArray(
        data,
        dims=("time", "j", "i"),
        coords={
            "time": np.arange(n_frames) * 0.1,
            "j": np.arange(32),
            "i": np.arange(32),
        },
    )
    return _add_voxel_affine(da, ("j", "i"), (0.1, 0.1))


@pytest.fixture
def sample_3d_dataarray(sample_3d_array):
    """3D+time DataArray (3 frames) for volumewise registration tests."""
    n_frames = 3
    data = np.stack([sample_3d_array] * n_frames, axis=0)
    da = xr.DataArray(
        data,
        dims=("time", "k", "j", "i"),
        coords={
            "time": np.arange(n_frames) * 0.1,
            "k": np.arange(16),
            "j": np.arange(16),
            "i": np.arange(16),
        },
    )
    return _add_voxel_affine(da, ("k", "j", "i"), (1.0, 1.0, 1.0))


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
