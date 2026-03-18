"""Shared fixtures for unit tests."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg", force=True)


@pytest.fixture
def sample_3d_volume(rng):
    """3D spatial volume (z, y, x) with consistent spatial coordinates.

    Shape: (4, 6, 8) - small enough for fast tests.
    Includes time as a scalar coordinate for consistency with 4D volumes.
    Includes metadata attributes for testing labels and units.
    """
    shape = (4, 6, 8)
    data = rng.random(shape)
    da = xr.DataArray(
        data,
        dims=["z", "y", "x"],
        coords={
            "z": xr.DataArray(
                1.0 + np.arange(4) * 0.2,
                dims=["z"],
                attrs={"units": "mm"},
            ),
            "y": xr.DataArray(
                2.0 + np.arange(6) * 0.1,
                dims=["y"],
                attrs={"units": "mm"},
            ),
            "x": xr.DataArray(
                3.0 + np.arange(8) * 0.05,
                dims=["x"],
                attrs={"units": "mm"},
            ),
            "time": 0.0,  # Scalar coord for consistency with 4D volumes.
        },
        attrs={
            "long_name": "Intensity",
            "units": "a.u.",
        },
    )
    return da


@pytest.fixture
def sample_4d_volume(rng):
    """4D volume (time, z, y, x) with consistent coordinates.

    Shape: (10, 4, 6, 8) - small enough for fast tests.
    Spatial coordinates match sample_3d_volume exactly.
    Includes metadata attributes for testing labels and units.
    """
    shape = (10, 4, 6, 8)
    data = rng.random(shape)
    da = xr.DataArray(
        data,
        dims=["time", "z", "y", "x"],
        coords={
            "time": xr.DataArray(
                10.0 + np.arange(10) * 0.5,
                dims=["time"],
                attrs={"units": "s"},
            ),
            "z": xr.DataArray(
                1.0 + np.arange(4) * 0.2,
                dims=["z"],
                attrs={"units": "mm"},
            ),
            "y": xr.DataArray(
                2.0 + np.arange(6) * 0.1,
                dims=["y"],
                attrs={"units": "mm"},
            ),
            "x": xr.DataArray(
                3.0 + np.arange(8) * 0.05,
                dims=["x"],
                attrs={"units": "mm"},
            ),
        },
        attrs={
            "long_name": "Intensity",
            "units": "a.u.",
        },
    )
    return da


@pytest.fixture
def sample_4d_volume_complex(rng):
    """Complex-valued 4D volume (time, z, y, x) for IQ processing tests.

    Shape: (10, 4, 6, 8) - matches sample_4d_volume spatial dimensions.
    Includes metadata attributes for testing labels and units.
    """
    shape = (10, 4, 6, 8)
    data = rng.random(shape) + 1j * rng.random(shape)
    da = xr.DataArray(
        data,
        dims=["time", "z", "y", "x"],
        coords={
            "time": xr.DataArray(
                np.arange(10) * 0.1,
                dims=["time"],
                attrs={"units": "s"},
            ),
            "z": xr.DataArray(
                np.arange(4) * 0.1,
                dims=["z"],
                attrs={"units": "mm"},
            ),
            "y": xr.DataArray(
                np.arange(6) * 0.05,
                dims=["y"],
                attrs={"units": "mm"},
            ),
            "x": xr.DataArray(
                np.arange(8) * 0.05,
                dims=["x"],
                attrs={"units": "mm"},
            ),
        },
        attrs={
            "long_name": "Complex Signal",
            "units": "a.u.",
        },
    )
    return da


@pytest.fixture
def sample_timeseries(rng):
    """Factory fixture for 2D time-series data (time, voxels).

    Creates DataArray with proper time coordinates.
    """

    def _make(
        n_time=100,
        n_voxels=50,
        sampling_rate=100.0,
    ):
        data = rng.normal(size=(n_time, n_voxels))
        return xr.DataArray(
            data,
            dims=["time", "space"],
            coords={
                "time": np.arange(n_time) / sampling_rate,
                "space": np.arange(n_voxels),
            },
        )

    return _make


@pytest.fixture
def spatial_mask(rng, sample_4d_volume):
    """Boolean spatial mask matching (z, y, x) of sample volumes."""
    _, z, y, x = sample_4d_volume.shape
    return rng.random((z, y, x)) > 0.5


@pytest.fixture
def matplotlib_pyplot():
    """Set up and teardown fixture for matplotlib.

    Returns the pyplot module and ensures all figures are closed after the test.
    """
    import matplotlib.pyplot as plt

    plt.close("all")
    yield plt
    plt.close("all")
