"""Shared fixtures for IQ processing tests."""

import numpy as np
import pytest
import xarray as xr


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

    Shape: (20, 4, 6, 8) with proper coordinates and required attributes.
    """
    shape = (20, 4, 6, 8)
    data = rng.random(shape) + 1j * rng.random(shape)

    return xr.DataArray(
        data,
        dims=("time", "z", "y", "x"),
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
            "z": xr.DataArray(np.arange(4) * 0.1, dims=("z",), attrs={"units": "mm"}),
            "y": xr.DataArray(np.arange(6) * 0.05, dims=("y",), attrs={"units": "mm"}),
            "x": xr.DataArray(np.arange(8) * 0.05, dims=("x",), attrs={"units": "mm"}),
        },
        attrs={
            "compound_sampling_frequency": 10.0,
            "transmit_frequency": 15.625e6,
            "beamforming_sound_velocity": 1540.0,
        },
    )


@pytest.fixture
def sample_spatial_mask_xarray(rng, sample_iq_dataarray):
    """Create a boolean spatial mask matching sample_iq_dataarray.

    Shape: (z=4, y=6, x=8) with coordinates matching sample_iq_dataarray.
    """
    z = sample_iq_dataarray.sizes["z"]
    y = sample_iq_dataarray.sizes["y"]
    x = sample_iq_dataarray.sizes["x"]
    return xr.DataArray(
        rng.random((z, y, x)) > 0.5,
        dims=("z", "y", "x"),
        coords={
            "z": sample_iq_dataarray.coords["z"],
            "y": sample_iq_dataarray.coords["y"],
            "x": sample_iq_dataarray.coords["x"],
        },
    )
