"""Fixtures for `confusius.decoding` tests."""

import pytest
import xarray as xr

from confusius.xarray import create_fusi_dataarray


@pytest.fixture
def decoding_volume(rng):
    """Canonical `(time, k, j, i)` fUSI volume with enough samples for cross-validation.

    Shape is `(40, 2, 5, 6)`. Spatial spacing is deliberately anisotropic: `k`
    (elevation, world `z`) is spaced 1.0 mm apart while `j`/`i` (world `y`/`x`) are
    spaced 0.2 mm apart, so a radius between those two values selects in-plane
    neighbors only.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator from the shared test fixtures.

    Returns
    -------
    xarray.DataArray
        Random canonical `(time, k, j, i)` volume with a real `VoxelToWorldIndex`.
    """
    n_time = 40
    return create_fusi_dataarray(
        rng.standard_normal((n_time, 2, 5, 6)),
        name="power_doppler",
        dims=("time", "k", "j", "i"),
        dt=0.5,
        spacing=(1.0, 0.2, 0.2),
        origin=(0.0, 0.0, 0.0),
        attrs={"long_name": "Intensity", "units": "a.u."},
    )


@pytest.fixture
def full_mask(decoding_volume):
    """All-`True` boolean mask matching `decoding_volume`'s spatial geometry.

    Parameters
    ----------
    decoding_volume : xarray.DataArray
        Volume providing the spatial dimensions, coordinates, and voxel-to-world index.

    Returns
    -------
    xarray.DataArray
        Boolean `(k, j, i)` mask, all `True`, carrying `decoding_volume`'s
        `VoxelToWorldIndex`.
    """
    return xr.ones_like(decoding_volume.isel(time=0, drop=True), dtype=bool)
