"""Shared fixtures for IQ processing tests."""

import numpy as np
import pytest
import xarray as xr

from confusius._utils.geometry import add_physical_coords_from_voxel_affine


@pytest.fixture
def sample_iq_block_4d(rng):
    """Create a sample 4D IQ block with shape (time, z, y, x).

    Shape: (10, 4, 6, 8) - small enough for fast tests.
    """
    shape = (10, 4, 6, 8)
    return rng.random(shape) + 1j * rng.random(shape)


@pytest.fixture
def sample_iq_block_4d_small(rng):
    """Create a smaller 4D IQ block for edge case testing.

    Shape: (5, 2, 3, 4).
    """
    shape = (5, 2, 3, 4)
    return rng.random(shape) + 1j * rng.random(shape)


@pytest.fixture
def sample_iq_block_4d_long(rng):
    """Create a 4D IQ block with long time dimension for Butterworth tests.

    Shape: (100, 4, 6, 8) - enough samples for Butterworth filter padding.
    The filter needs at least ntaps * 3 samples (typically ~27 for order 4).
    """
    shape = (100, 4, 6, 8)
    return rng.random(shape) + 1j * rng.random(shape)


@pytest.fixture
def spatial_mask(rng, sample_iq_block_4d):
    """Create a spatial mask with shape (z, y, x) matching sample_iq_block_4d."""
    _, z, y, x = sample_iq_block_4d.shape
    return rng.random((z, y, x)) > 0.5


@pytest.fixture
def spatial_mask_small(rng, sample_iq_block_4d_small):
    """Create a spatial mask matching sample_iq_block_4d_small."""
    _, z, y, x = sample_iq_block_4d_small.shape
    return rng.random((z, y, x)) > 0.5


@pytest.fixture
def sample_iq_dataarray(rng):
    """Create sample xarray DataArray with IQ data.

    Shape: (20, 4, 6, 8) with canonical `(time, k, j, i)` coordinates and linked
    physical `z/y/x` coordinates.
    """
    shape = (20, 4, 6, 8)
    data = rng.random(shape) + 1j * rng.random(shape)

    base = xr.DataArray(
        data,
        dims=("time", "k", "j", "i"),
        coords={
            "time": xr.DataArray(
                np.arange(20) * 0.1,
                dims=("time",),
                attrs={
                    "units": "s",
                    "volume_acquisition_duration": 0.1,
                    "volume_acquisition_reference": "start",
                },
            ),
            "k": xr.DataArray(
                np.arange(4),
                dims=("k",),
                attrs={"voxdim": 1.0},
            ),
            "j": xr.DataArray(
                np.arange(6),
                dims=("j",),
                attrs={"voxdim": 1.0},
            ),
            "i": xr.DataArray(
                np.arange(8),
                dims=("i",),
                attrs={"voxdim": 1.0},
            ),
        },
        attrs={
            "compound_sampling_frequency": 10.0,
            "transmit_frequency": 15.625e6,
            "beamforming_sound_velocity": 1540.0,
        },
    )
    return add_physical_coords_from_voxel_affine(
        base,
        np.diag([0.1, 0.05, 0.05, 1.0]),
        voxel_dims=("k", "j", "i"),
        physical_coord_names=("z", "y", "x"),
        physical_coord_attrs={
            "z": {"units": "mm", "voxdim": 0.1},
            "y": {"units": "mm", "voxdim": 0.05},
            "x": {"units": "mm", "voxdim": 0.05},
        },
    )


@pytest.fixture
def sample_spatial_mask_xarray(rng, sample_iq_dataarray):
    """Create a boolean spatial mask matching sample_iq_dataarray.

    Shape: (k=4, j=6, i=8) with coordinates matching sample_iq_dataarray.
    """
    k = sample_iq_dataarray.sizes["k"]
    j = sample_iq_dataarray.sizes["j"]
    i = sample_iq_dataarray.sizes["i"]
    return xr.DataArray(
        rng.random((k, j, i)) > 0.5,
        dims=("k", "j", "i"),
        coords={
            "k": sample_iq_dataarray.coords["k"],
            "j": sample_iq_dataarray.coords["j"],
            "i": sample_iq_dataarray.coords["i"],
            "z": sample_iq_dataarray.coords["z"],
            "y": sample_iq_dataarray.coords["y"],
            "x": sample_iq_dataarray.coords["x"],
        },
    )
