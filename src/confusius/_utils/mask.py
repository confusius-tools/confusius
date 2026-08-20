"""Shared mask handling for consumers that accept VoxelData or extracted signals.

Some consumers (`decomposition/_base.py`, `signal/confounds.py`,
`stats/thresholding.py`, `plotting/image.py`) operate on either a canonical
VoxelData array or an already-extracted, non-spatial signals array (e.g.
`extract_with_labels` output). This module holds the shared dispatch logic so each
consumer doesn't reimplement the branch on
[`has_voxel_to_world_index`][confusius._utils.geometry.has_voxel_to_world_index].
"""

import xarray as xr

from confusius._utils.geometry import has_voxel_to_world_index
from confusius.extract.mask import extract_with_mask
from confusius.validation.coordinates import validate_matching_coordinates
from confusius.validation.mask import check_mask_dtype, ensure_mask


def _validate_feature_mask(
    mask: xr.DataArray,
    data: xr.DataArray,
    mask_name: str,
    require_exact_dims: bool = False,
) -> xr.DataArray:
    """Validate a mask against data that isn't a canonical VoxelData array.

    Used by
    [`validate_spatial_or_feature_mask`][confusius._utils.mask.validate_spatial_or_feature_mask]
    when `data` doesn't carry a `VoxelToWorldIndex` -- e.g. an already-extracted
    signals array such as [`extract_with_labels`][confusius.extract.extract_with_labels]
    output. There is no spatial grid to validate against here, so only dimension
    names, sizes, and coordinate labels are checked (mirroring
    `xarray.align(join="exact")`'s own behavior: a dimension missing a
    coordinate/index on either side is not checked, only dimensions where both sides
    carry one).

    Parameters
    ----------
    mask : xarray.DataArray
        Mask to validate. Must have boolean dtype, or integer dtype with exactly one
        non-zero value (0 = background, one region id = foreground).
    data : xarray.DataArray
        Data array to validate `mask` against. May carry dimensions `mask` doesn't
        (e.g. `time`), same as [`ensure_mask`][confusius.validation.ensure_mask].
    mask_name : str
        Label used for `mask` in error messages.
    require_exact_dims : bool, default: False
        Whether `mask`'s dimensions must match all non-`time` dimensions of `data` in
        the same order, mirroring `ensure_mask`'s parameter of the same name.

    Returns
    -------
    xarray.DataArray
        The validated `mask`, coerced to boolean dtype.

    Raises
    ------
    TypeError
        If `mask` is not a boolean or single-label integer DataArray.
    ValueError
        If `mask`'s dimensions aren't a subset of `data`'s (or don't match exactly
        when `require_exact_dims` is set), or if their sizes or coordinates disagree.
    """
    check_mask_dtype(mask, mask_name)

    if not set(mask.dims).issubset(set(data.dims)):
        missing = set(mask.dims) - set(data.dims)
        raise ValueError(
            f"Data is missing dimensions from {mask_name}: {missing}. "
            f"Data dims: {data.dims}, {mask_name} dims: {mask.dims}."
        )

    if require_exact_dims:
        expected_dims = tuple(str(d) for d in data.dims if d != "time")
        mask_dims = tuple(str(d) for d in mask.dims)
        if mask_dims != expected_dims:
            raise ValueError(
                f"{mask_name} dimensions must match all non-time dimensions of data "
                f"in the same order. Expected {expected_dims}, got {mask_dims}."
            )

    for dim in mask.dims:
        if mask.sizes[dim] != data.sizes[dim]:
            raise ValueError(
                f"{mask_name} dimension {dim!r} has size {mask.sizes[dim]}, "
                f"expected {data.sizes[dim]}."
            )
    validate_matching_coordinates(mask, data, left_name=mask_name, right_name="data")

    return mask.astype(bool)


def validate_spatial_or_feature_mask(
    data: xr.DataArray,
    mask: xr.DataArray,
    mask_name: str = "mask",
    require_exact_dims: bool = False,
) -> xr.DataArray:
    """Validate a mask against `data`, whether it's VoxelData or extracted signals.

    Dispatches on [`has_voxel_to_world_index`][confusius._utils.geometry.has_voxel_to_world_index]:
    if `data` is a canonical VoxelData array, `mask` is validated against it via
    [`ensure_mask`][confusius.validation.ensure_mask] (full grid semantics:
    `VoxelToWorldIndex`, `voxel_to_world` affine, and voxel-space coordinates must all
    match). Otherwise `data` is treated as an already-extracted, non-spatial signals
    array (e.g. [`extract_with_labels`][confusius.extract.extract_with_labels]
    output): `mask` is checked only for dtype, matching dimensions/sizes, and matching
    coordinate labels where both sides carry them -- no `VoxelToWorldIndex` is
    required, since there is no spatial grid.

    Parameters
    ----------
    data : xarray.DataArray
        Data array to validate `mask` against.
    mask : xarray.DataArray
        Mask to validate. Must have boolean dtype, or integer dtype with exactly one
        non-zero value (0 = background, one region id = foreground).
    mask_name : str, default: "mask"
        Name of the mask parameter (used in error messages).
    require_exact_dims : bool, default: False
        Whether `mask`'s dimensions must match all non-`time` dimensions of `data` in
        the same order, in either case.

    Returns
    -------
    xarray.DataArray
        The validated `mask`, coerced to boolean dtype.

    Raises
    ------
    TypeError
        If `mask` is not a boolean or single-label integer DataArray.
    ValueError
        If `mask` doesn't match `data`'s grid (VoxelData case) or dimensions/sizes/
        coordinates (already-extracted case).
    """
    if has_voxel_to_world_index(data):
        return ensure_mask(mask, data, mask_name, require_exact_dims=require_exact_dims)
    return _validate_feature_mask(
        mask, data, mask_name, require_exact_dims=require_exact_dims
    )


def select_masked_features(
    data: xr.DataArray,
    mask: xr.DataArray,
    mask_name: str = "mask",
    require_exact_dims: bool = False,
) -> xr.DataArray:
    """Select masked elements from `data`, whether it's VoxelData or extracted signals.

    Dispatches on [`has_voxel_to_world_index`][confusius._utils.geometry.has_voxel_to_world_index]:
    for a canonical VoxelData `data` (native voxel dims `k`/`j`/`i` and a
    `VoxelToWorldIndex`), this is [`extract_with_mask`][confusius.extract.extract_with_mask]
    with full grid validation. For an already-extracted, non-spatial signals array
    (e.g. [`extract_with_labels`][confusius.extract.extract_with_labels] output),
    `mask` is validated against `data`'s dimensions, sizes, and coordinate labels only
    (see [`validate_spatial_or_feature_mask`][confusius._utils.mask.validate_spatial_or_feature_mask]),
    and the selected elements are flattened into a `space` dimension the same way, but
    without `extract_with_mask`'s spatial round-trip guarantees (no `space`
    `MultiIndex` is built, so `.unstack("space")` is not supported on the result).

    Parameters
    ----------
    data : xarray.DataArray
        Input array: a canonical VoxelData array, or an already-extracted signals
        array.
    mask : xarray.DataArray
        Mask defining which elements to select, sharing `data`'s dimensions. Must have
        boolean dtype, or integer dtype with exactly one non-zero value (0 =
        background, one region id = foreground).
    mask_name : str, default: "mask"
        Name of the mask parameter (used in error messages).
    require_exact_dims : bool, default: False
        Whether `mask`'s dimensions must match all non-`time` dimensions of `data` in
        the same order, in either case.

    Returns
    -------
    xarray.DataArray
        Array with `mask`'s dimensions flattened into a `space` dimension. All other
        dimensions of `data` are preserved.

    Raises
    ------
    ValueError
        If `mask` doesn't match `data`'s grid (VoxelData case) or dimensions/sizes/
        coordinates (already-extracted case).
    TypeError
        If `mask` is not boolean dtype (or a single-label integer dtype).
    """
    mask = validate_spatial_or_feature_mask(
        data, mask, mask_name, require_exact_dims=require_exact_dims
    )
    if has_voxel_to_world_index(data):
        return extract_with_mask(data, mask)
    spatial_dims = list(mask.dims)
    return data.stack(space=spatial_dims).isel(space=mask.values.ravel())
