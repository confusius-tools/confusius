"""Fixtures for `confusius.decoding` tests."""

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def decoding_volume(rng):
    """`(time, z, y, x)` volume with enough samples for cross-validation.

    Shape is `(40, 2, 5, 6)`. Spatial coordinates are deliberately anisotropic: `z` is
    spaced 1.0 apart while `y` and `x` are spaced 0.2 apart, so a radius between those
    two values selects in-plane neighbors only.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator from the shared test fixtures.

    Returns
    -------
    xarray.DataArray
        Random `(time, z, y, x)` volume with millimeter coordinates.
    """
    n_time = 40
    return xr.DataArray(
        rng.standard_normal((n_time, 2, 5, 6)),
        name="power_doppler",
        dims=["time", "z", "y", "x"],
        coords={
            "time": xr.DataArray(
                np.arange(n_time) * 0.5, dims=["time"], attrs={"units": "s"}
            ),
            "z": xr.DataArray(
                np.array([0.0, 1.0]), dims=["z"], attrs={"units": "mm", "voxdim": 1.0}
            ),
            "y": xr.DataArray(
                np.arange(5) * 0.2, dims=["y"], attrs={"units": "mm", "voxdim": 0.2}
            ),
            "x": xr.DataArray(
                np.arange(6) * 0.2, dims=["x"], attrs={"units": "mm", "voxdim": 0.2}
            ),
        },
        attrs={"long_name": "Intensity", "units": "a.u."},
    )


@pytest.fixture
def full_mask(decoding_volume):
    """All-`True` boolean mask matching `decoding_volume`'s spatial geometry.

    Parameters
    ----------
    decoding_volume : xarray.DataArray
        Volume providing the spatial dimensions and coordinates.

    Returns
    -------
    xarray.DataArray
        Boolean `(z, y, x)` mask, all `True`.
    """
    return xr.ones_like(decoding_volume.isel(time=0, drop=True), dtype=bool)
