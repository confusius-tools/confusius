"""Utility functions for the signal module."""

import numpy as np
import xarray as xr


def remove_zero_variance_voxels(
    signals: xr.DataArray,
    variance_tolerance: float = 1e-10,
) -> xr.DataArray:
    """Remove voxels with near-zero variance from signals.

    Zero-variance voxels can cause numerical issues and bias statistical
    computations like PCA/SVD. This function identifies and removes them.

    Parameters
    ----------
    signals : (time, voxels) xarray.DataArray
        Signals with time and voxels dimensions.
    variance_tolerance : float, default: 1e-10
        Voxels with standard deviation <= this value are considered zero-variance.

    Returns
    -------
    xarray.DataArray
        Signals with zero-variance voxels removed. If all voxels have zero
        variance, raises an error.

    Raises
    ------
    ValueError
        If all voxels have variance below tolerance.

    Notes
    -----
    Uses standard deviation (with Bessel's correction) to identify low-variance
    voxels. This is simpler than robust SD (used in DVARS) but sufficient for
    identifying truly constant voxels.
    """
    voxel_std = signals.std(dim="time", ddof=1)

    nonzero_mask = voxel_std.values > variance_tolerance

    if not np.any(nonzero_mask):
        raise ValueError(
            "All voxels have variance below tolerance. Check input data or adjust "
            "variance_tolerance parameter."
        )

    return signals.isel(space=nonzero_mask)
