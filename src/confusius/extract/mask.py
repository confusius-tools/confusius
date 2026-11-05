"""Mask extraction with ConfUSIus acquisition validation."""

import xarray as xr
from voxeldata.extract import extract_with_mask as extract_generic_mask

from confusius.validation import ensure_voxeldata


def extract_with_mask(data: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Extract voxel signals using VoxelData's spatial-mask implementation.

    ConfUSIus canonicalizes inputs and applies acquisition defaults/validation
    before delegating geometry, alignment, and sample selection to VoxelData.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array with arbitrary nonspatial dimensions.
    mask : xarray.DataArray
        VoxelData mask on the same physical voxel grid, optionally including
        extra feature axes. All mask dimensions are flattened into `space`.
        Boolean or finite binary numeric values are accepted. A shared mask
        requires identical geometry across dependencies absent from the mask.

    Returns
    -------
    xarray.DataArray
        Signals with nonspatial dimensions followed by `space`. Its MultiIndex
        contains selected mask-axis labels. Dask signals remain lazy.

    Raises
    ------
    TypeError
        If the mask is not a boolean or binary numeric DataArray.
    ValueError
        If geometry, dimensions, alignment, or acquisition metadata are invalid.
    """
    data = ensure_voxeldata(data, allow_extra_dims=True)
    mask = ensure_voxeldata(mask, allow_extra_dims=True)
    return extract_generic_mask(data, mask)
