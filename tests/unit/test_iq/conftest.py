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
def sample_iq_dataset(rng):
    """Create sample xarray Dataset with IQ data.

    Shape: (20, 4, 6, 8) with proper coordinates and required attributes
    on the iq DataArray (consistent with how conversion functions store data).
    """
    import xarray as xr

    shape = (20, 4, 6, 8)
    data = rng.random(shape) + 1j * rng.random(shape)

    iq_data = xr.DataArray(
        data,
        dims=("time", "z", "y", "x"),
        coords={
            "time": xr.DataArray(
                np.arange(20) * 0.1, dims=("time",), attrs={"units": "s"}
            ),
            "z": np.arange(4) * 0.1,
            "y": np.arange(6) * 0.05,
            "x": np.arange(8) * 0.05,
        },
        attrs={
            "compound_sampling_frequency": 10.0,
            "transmit_frequency": 15.625e6,
            "beamforming_sound_velocity": 1540.0,
        },
    )

    return xr.Dataset({"iq": iq_data})


@pytest.fixture
def sample_iq_dataarray(sample_iq_dataset):
    """Return the IQ DataArray from sample_iq_dataset.

    Shape: (20, 4, 6, 8) with proper coordinates and required attributes.
    """
    return sample_iq_dataset["iq"]


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
