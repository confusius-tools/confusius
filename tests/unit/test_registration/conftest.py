"""Shared fixtures for registration tests."""

import numpy as np
import pytest
import SimpleITK as sitk

from confusius.xarray import create_voxeldata


@pytest.fixture
def sample_voxeldata_2d_registration():
    """Singleton-k registration image with a centred square and 0.1 mm in-plane spacing."""
    img = np.zeros((1, 32, 32), dtype=np.float32)
    img[:, 12:20, 12:20] = 100.0
    return create_voxeldata(
        img,
        dims=("k", "j", "i"),
        spacing=(1.0, 0.1, 0.1),
        origin=(0.0, 0.0, 0.0),
    )


@pytest.fixture
def sample_voxeldata_3d_registration():
    """3D registration volume with a centred cube and unit spacing."""
    vol = np.zeros((16, 16, 16), dtype=np.float32)
    vol[6:10, 6:10, 6:10] = 100.0
    return create_voxeldata(
        vol,
        dims=("k", "j", "i"),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )


@pytest.fixture
def sample_voxeldata_3d_feature_registration():
    """3D registration volume with several off-centre features and unit spacing."""
    vol = np.zeros((32, 32, 32), dtype=np.float32)
    vol[4:10, 4:10, 4:10] = 60.0
    vol[20:28, 6:12, 18:24] = 100.0
    vol[8:14, 20:26, 6:12] = 80.0
    vol[18:24, 20:30, 20:26] = 40.0
    vol[14:18, 14:18, 14:18] = 120.0
    return create_voxeldata(
        vol,
        dims=("k", "j", "i"),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )


@pytest.fixture
def sample_voxeldata_2dt_registration(sample_voxeldata_2d_registration):
    """Singleton-k+time registration DataArray with five identical frames."""
    n_frames = 5
    return create_voxeldata(
        np.stack([sample_voxeldata_2d_registration.values] * n_frames, axis=0),
        dims=("time", "k", "j", "i"),
        time=np.arange(n_frames) * 0.1,
        spacing=(1.0, 0.1, 0.1),
        origin=(0.0, 0.0, 0.0),
    )


@pytest.fixture
def sample_voxeldata_3dt_registration(sample_voxeldata_3d_registration):
    """3D+time registration DataArray with three identical frames."""
    n_frames = 3
    return create_voxeldata(
        np.stack([sample_voxeldata_3d_registration.values] * n_frames, axis=0),
        dims=("time", "k", "j", "i"),
        time=np.arange(n_frames) * 0.1,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )


@pytest.fixture
def sample_voxeldata_2d_extra_dim_registration(sample_voxeldata_2d_registration):
    """Singleton-k registration DataArray with a `component` and a `time` extra dim.

    Each `(component, time)` slice has a distinct off-centre square so that
    resampling results can be told apart per slice.
    """
    n_components, n_frames = 3, 2
    base = np.zeros((1, 32, 32), dtype=np.float32)
    slices = []
    for c in range(n_components):
        for _t in range(n_frames):
            frame = base.copy()
            offset = 2 * c
            frame[:, 10 + offset : 18 + offset, 10 + offset : 18 + offset] = 100.0
            slices.append(frame)
    data = np.stack(slices, axis=0).reshape(n_components, n_frames, 1, 32, 32)
    return create_voxeldata(
        data,
        dims=("component", "time", "k", "j", "i"),
        extra_coords={"component": np.arange(n_components)},
        time=np.arange(n_frames) * 0.1,
        spacing=(1.0, 0.1, 0.1),
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
