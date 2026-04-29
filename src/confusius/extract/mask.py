"""Extraction of signals using boolean masks."""

import numpy as np
import xarray as xr

from confusius.validation import validate_mask


def extract_with_mask(data: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Extract signals from fUSI data using a binary mask.

    This function flattens the spatial dimensions specified by the mask into a single
    `space` dimension, while preserving all other dimensions (e.g., time, components).

    Parameters
    ----------
    data : xarray.DataArray
        Input array with spatial dimensions matching the mask. Can have any number of
        non-spatial dimensions (e.g., `time`, `components`). The spatial dimensions must
        match those in the mask.
    mask : xarray.DataArray
        Mask defining which voxels to extract. Its dimensions define the spatial
        dimensions that will be flattened. Must have boolean dtype, or integer dtype
        with exactly one non-zero value (0 = background, one region id =
        foreground). The latter format is produced by
        [`Atlas.get_masks`][confusius.atlas.Atlas.get_masks]. Coordinates must match
        data.

    Returns
    -------
    xarray.DataArray
        Array with spatial dimensions flattened into a `space` dimension. All
        non-spatial dimensions are preserved. The `space` dimension has a MultiIndex
        storing spatial coordinates.

        For example:

        - `(time, z, y, x)` → `(time, space)`
        - `(time, pose, z, y, x)` → `(time, pose, space)`
        - `(z, y, x)` → `(space,)`

        For simple round-trip reconstruction, use `.unstack("space")` which
        re-creates the original DataArray using the smallest bounding box containing the
        masked voxels. For full mask shape reconstruction, use
        [`confusius.extract.unmask`][confusius.extract.unmask].

    Raises
    ------
    ValueError
        If `mask` dimensions don't match `data`'s spatial dimensions.
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
    ("time", "space")
    >>>
    >>> # 3D+t data with extra dim: (time, pose, z, y, x)
    >>> pose_data = xr.DataArray(
    ...     np.random.randn(100, 5, 10, 20, 30),
    ...     dims=["time", "pose", "z", "y", "x"],
    ... )
    >>> pose_signals = extract_with_mask(pose_data, mask)
    >>> pose_signals.dims
    ("time", "pose", "space")
    """
    validate_mask(mask, data, "mask")

    spatial_dims = list(mask.dims)
    non_spatial_dims = [d for d in data.dims if d not in spatial_dims]

    if non_spatial_dims:
        sel_dict = {d: 0 for d in non_spatial_dims}
        template = data.isel(sel_dict)
    else:
        template = data

    coord_updates = {
        dim: template.coords[dim]
        for dim in spatial_dims
        if dim in mask.coords and dim in template.coords
    }
    # validate_mask() already checked that shared spatial coords match within
    # tolerance; snap them onto the template coords here so reindex_like()
    # performs exact label alignment instead of introducing NaNs.
    mask_aligned = mask.assign_coords(coord_updates).reindex_like(template)

    if (
        np.issubdtype(mask_aligned.dtype, np.floating)
        and np.isnan(mask_aligned.values).any()
    ):
        raise ValueError(
            "mask could not be aligned to data coordinates. If coordinates are nearly "
            "equal, ensure they describe the same voxel grid before extraction."
        )

    if "space" in data.dims and set(spatial_dims) == {"space"}:
        mask_flat = mask_aligned.values.astype(bool)
        return data.isel(space=mask_flat)

    data_flat = data.stack(space=spatial_dims)
    mask_flat = mask_aligned.values.ravel().astype(bool)
    # Rebuild the space index from the selected voxel coordinates so unstack() uses the
    # reduced grid implied by the extracted mask.
    return data_flat.isel(space=mask_flat).set_xindex(spatial_dims)
