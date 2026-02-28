"""Extraction of signals using boolean masks."""

import xarray as xr

from confusius.validation import validate_mask


def extract_with_mask(
    data: xr.DataArray,
    mask: xr.DataArray,
) -> xr.DataArray:
    """Extract signals from fUSI data using a boolean mask.

    This function flattens the spatial dimensions specified by the mask into a
    single `voxels` dimension, while preserving all other dimensions (e.g., time,
    pose).

    Parameters
    ----------
    data : xarray.DataArray
        Input array with spatial dimensions matching the mask. Can have any number
        of non-spatial dimensions (e.g., `time`, `pose`). The spatial dimensions
        must match those in the mask.
    mask : xarray.DataArray
        Boolean mask defining which voxels to extract. Its dimensions define the
        spatial dimensions that will be flattened. Must have identical coordinates
        to the data's spatial dimensions.

    Returns
    -------
    xarray.DataArray
        Array with spatial dimensions flattened into a `voxels` dimension.
        All non-spatial dimensions are preserved. The `voxels` dimension has a
        MultiIndex storing spatial coordinates.

        For example:

        - `(time, z, y, x)` → `(time, voxels)`
        - `(time, pose, z, y, x)` → `(time, pose, voxels)`
        - `(z, y, x)` → `(voxels,)`

        For simple round-trip reconstruction, use `.unstack("voxels")` which
        re-creates the original DataArray using the smallest bounding box containing the
        masked voxels. For full mask shape reconstruction, use
        `confusius.extract.unmask`.

    Raises
    ------
    ValueError
        If `mask` dimensions don't match `data`'s spatial dimensions, or if `data` has
        fewer than 2 spatial dimensions.
    TypeError
        If `mask` is not boolean dtype.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> from confusius.extract import extract_with_mask
    >>>
    >>> # 3D+t data: (time, z, y, x)
    >>> data = xr.DataArray(
    ...     np.random.randn(100, 10, 20, 30),
    ...     dims=["time", "z", "y", "x"],
    ... )
    >>> mask = xr.DataArray(
    ...     np.random.rand(10, 20, 30) > 0.5,
    ...     dims=["z", "y", "x"],
    ... )
    >>> signals = extract_with_mask(data, mask)
    >>> signals.dims
    ("time", "voxels")
    >>>
    >>> # 3D+t data with extra dim: (time, pose, z, y, x)
    >>> pose_data = xr.DataArray(
    ...     np.random.randn(100, 5, 10, 20, 30),
    ...     dims=["time", "pose", "z", "y", "x"],
    ... )
    >>> pose_signals = extract_with_mask(pose_data, mask)
    >>> pose_signals.dims
    ("time", "pose", "voxels")
    """
    validate_mask(mask, data, "mask")

    spatial_dims = list(mask.dims)
    non_spatial_dims = [d for d in data.dims if d not in spatial_dims]

    if non_spatial_dims:
        sel_dict = {d: 0 for d in non_spatial_dims}
        template = data.isel(sel_dict)
    else:
        template = data

    mask_aligned = mask.reindex_like(template)

    data_flat = data.stack(voxels=spatial_dims)

    mask_flat = mask_aligned.values.flatten()
    return data_flat.isel(voxels=mask_flat)
