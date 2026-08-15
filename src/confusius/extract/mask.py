"""Extraction of signals using boolean masks."""

import xarray as xr

from confusius.validation import validate_mask


def extract_with_mask(data: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Extract signals from fUSI data using a binary mask.

    This function flattens `mask`'s native voxel dimensions (`k`/`j`/`i`) into a
    single `space` dimension, while preserving all other dimensions of `data` (e.g.,
    `time`, `pose`).

    Parameters
    ----------
    data : xarray.DataArray
        Input array. Must be a canonical voxel-grid DataArray with native voxel dims
        `k`/`j`/`i` and a `VoxelToWorldIndex`, plus any number of non-spatial
        dimensions (e.g., `time`, `pose`). See
        [`ensure_fusi`][confusius.validation.ensure_fusi].
    mask : xarray.DataArray
        Mask defining which voxels to extract, sharing `data`'s voxel grid. Must have
        boolean dtype, or integer dtype with exactly one non-zero value (0 =
        background, one region id = foreground). The latter format is produced by
        [`get_masks`][confusius.atlas.AtlasAccessor.get_masks].

    Returns
    -------
    xarray.DataArray
        Array with `k`/`j`/`i` flattened into a `space` dimension. All non-spatial
        dimensions are preserved. The `space` dimension has a MultiIndex storing
        spatial coordinates.

        - `(time, k, j, i)` → `(time, space)`
        - `(time, pose, k, j, i)` → `(time, pose, space)`
        - `(k, j, i)` → `(space,)`

        For simple round-trip reconstruction, use `.unstack("space")` which
        re-creates the original DataArray using the smallest bounding box containing the
        masked voxels. For full mask shape reconstruction, use
        [`confusius.extract.unmask`][confusius.extract.unmask].

    Raises
    ------
    ValueError
        If `mask` or `data` isn't a canonical voxel-grid DataArray, or if `mask`'s
        voxel grid doesn't match `data`'s.
    TypeError
        If `mask` is not boolean dtype (or a single-label integer dtype).

    Examples
    --------
    >>> import numpy as np
    >>> from confusius.extract import extract_with_mask
    >>> from confusius.xarray import create_fusi_dataarray
    >>>
    >>> # 3D+t data: (time, k, j, i)
    >>> data = create_fusi_dataarray(
    ...     np.random.randn(100, 10, 20, 30),
    ...     dims=("time", "k", "j", "i"),
    ...     dt=0.5,
    ...     spacing=(1.0, 1.0, 1.0),
    ... )
    >>> mask = create_fusi_dataarray(
    ...     np.random.rand(10, 20, 30) > 0.5,
    ...     dims=("k", "j", "i"),
    ...     spacing=(1.0, 1.0, 1.0),
    ... )
    >>> signals = extract_with_mask(data, mask)
    >>> signals.dims
    ('time', 'space')
    >>>
    >>> # 3D+t data with extra dim: (time, pose, k, j, i)
    >>> pose_data = create_fusi_dataarray(
    ...     np.random.randn(100, 5, 10, 20, 30),
    ...     dims=("time", "pose", "k", "j", "i"),
    ...     dt=0.5,
    ...     spacing=(1.0, 1.0, 1.0),
    ... )
    >>> pose_signals = extract_with_mask(pose_data, mask)
    >>> pose_signals.dims
    ('time', 'pose', 'space')
    """
    mask = validate_mask(mask, data, "mask")

    # validate_mask() already checked mask and data share the same voxel grid (same
    # k/j/i coordinates and voxel_to_world affine), so mask.values.ravel() is already
    # positionally aligned with data.stack(space=spatial_dims) below -- both iterate
    # spatial_dims (mask's own dim order) the same way.
    spatial_dims = list(mask.dims)
    data_flat = data.stack(space=spatial_dims)
    mask_flat = mask.values.ravel()
    # Rebuild the space index from the selected voxel coordinates so unstack() uses the
    # reduced grid implied by the extracted mask.
    return data_flat.isel(space=mask_flat).set_xindex(spatial_dims)
