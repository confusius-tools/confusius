"""Reconstruction through VoxelData with ConfUSIus mask validation."""

import numpy as np
import xarray as xr
from voxeldata.extract import unmask as unmask_generic

from confusius.validation import ensure_voxeldata


def unmask(
    signals: np.ndarray | xr.DataArray,
    mask: xr.DataArray,
    new_dims: list[str] | None = None,
    new_dims_coords: dict[str, np.ndarray] | None = None,
    attrs: dict | None = None,
    fill_value: float = 0.0,
) -> xr.DataArray:
    """Reconstruct the complete mask grid from positional voxel samples.

    ConfUSIus validates the mask's acquisition metadata before delegation.
    Reconstruction is eager, including for Dask-backed signals. It does not
    realign samples by their `space` labels.

    Parameters
    ----------
    signals : numpy.ndarray or xarray.DataArray
        Samples with shape `(..., space)`. A DataArray must have `space` last;
        leading dimension coordinates and their attributes are retained.
    mask : xarray.DataArray
        Boolean or finite binary numeric VoxelData mask. Its native labels,
        extra feature axes, units, geometry dependencies, and named frames
        define the reconstructed grid.
    new_dims : list of str, optional
        Leading dimension names for NumPy signals. Defaults to `dim_0`, `dim_1`,
        and so on. Ignored for DataArray signals.
    new_dims_coords : dict[str, numpy.ndarray], optional
        Leading coordinate vectors for NumPy signals. Missing entries use integer
        positions. Ignored for DataArray signals.
    attrs : dict, optional
        Output array attributes. If not provided, uses an empty dictionary.
    fill_value : float, default: 0.0
        Value for unselected voxels, cast to the signal dtype.

    Returns
    -------
    xarray.DataArray
        Dense reconstructed VoxelData array with leading signal dimensions and
        the mask's complete spatial grid.

    Raises
    ------
    TypeError
        If signals or mask have an unsupported type.
    ValueError
        If sample counts, dimension names/coordinates, geometry, or acquisition
        metadata are invalid.
    """
    mask = ensure_voxeldata(mask, allow_extra_dims=True)
    return unmask_generic(
        signals,
        mask,
        new_dims=new_dims,
        new_dims_coords=new_dims_coords,
        attrs=attrs,
        fill_value=fill_value,
    )
