"""Mask validation utilities."""

import numpy as np
import xarray as xr

from confusius.validation.voxeldata import ensure_voxeldata, validate_voxeldata


def _check_spatial_alignment(
    spatial_da: xr.DataArray, data: xr.DataArray, name: str
) -> None:
    """Check that `spatial_da` and `data` share the same VoxelData grid.

    Assumes both are already canonical VoxelData arrays (see
    [`validate_voxeldata`][confusius.validation.validate_voxeldata]) -- callers that
    may not be should canonicalize first via
    [`ensure_voxeldata`][confusius.validation.ensure_voxeldata]. `xarray.align` with
    `join="exact"` dispatches to the `VoxelToWorldIndex`'s `equals`, which compares
    both the voxel-space `k`/`j`/`i` coordinates (what `reindex_like`/`stack` actually
    key alignment on) and the underlying `voxel_to_world` affine — so two arrays
    sharing the same voxel-space coordinate labels but different affines are correctly
    rejected, not silently treated as aligned.

    Parameters
    ----------
    spatial_da : xarray.DataArray
        Canonical VoxelData array to check (mask or labels).
    data : xarray.DataArray
        Canonical reference VoxelData array.
    name : str
        Label used for `spatial_da` in error messages.

    Raises
    ------
    ValueError
        If `spatial_da`'s voxel-space coordinates or `voxel_to_world` affine don't
        match `data`'s.
    """
    # `xr.align` groups indexes to compare by (coordinate name, dims order) before ever
    # calling `Index.equals()`, so two VoxelData arrays sharing the same
    # VoxelToWorldIndex grid but a differently-transposed k/j/i order fall into separate
    # groups and raise a spurious `AlignmentError`, even though the grids genuinely
    # match. Aligning on a shared dim order sidesteps that grouping, without changing
    # what's returned.
    common_dims = [dim for dim in data.dims if dim in spatial_da.dims]
    try:
        xr.align(
            spatial_da.transpose(*common_dims, ...),
            data.transpose(*common_dims, ...),
            join="exact",
        )
    except xr.AlignmentError as error:
        raise ValueError(f"{name} does not share data's voxel grid: {error}") from error


def check_mask_dtype(mask: xr.DataArray, mask_name: str) -> None:
    """Check that `mask` has boolean dtype or binary numeric dtype.

    A binary numeric mask (int or float) contains at most one distinct non-zero
    value -- e.g. `{0, 1}` or `{0.0, 5.0}` -- which lets masks produced by tools that
    don't support a boolean dtype (e.g. FSL/NiBabel-written NIfTI masks) pass
    through unchanged instead of requiring an upfront `.astype(bool)`.

    Parameters
    ----------
    mask : xarray.DataArray
        Mask to check.
    mask_name : str
        Label used for `mask` in error messages.

    Raises
    ------
    TypeError
        If `mask` is not a boolean or binary numeric DataArray.
    """
    if mask.dtype == bool:
        return
    if np.issubdtype(mask.dtype, np.integer) or np.issubdtype(mask.dtype, np.floating):
        non_zero = np.unique(mask.values[mask.values != 0])
        if len(non_zero) > 1:
            raise TypeError(
                f"{mask_name} has {mask.dtype} dtype with {len(non_zero)} distinct "
                f"non-zero values. A mask must be boolean or binary (0 = background, "
                f"one non-zero value = foreground). "
                f"For multi-region extraction use extract_with_labels instead."
            )
        return
    raise TypeError(
        f"{mask_name} must be boolean dtype or a binary numeric dtype, "
        f"got {mask.dtype}."
    )


def validate_mask(
    mask: xr.DataArray,
    data: xr.DataArray,
    mask_name: str = "mask",
    require_exact_dims: bool = False,
) -> None:
    """Validate that a mask shares data's VoxelData grid.

    `mask` and `data` must already be canonical VoxelData arrays (see
    [`validate_voxeldata`][confusius.validation.validate_voxeldata]) -- this does not
    canonicalize either. For a `mask`/`data` pair that may not already be canonical
    (e.g. a scalar-reduced voxel dim), use
    [`ensure_mask`][confusius.validation.ensure_mask] instead.

    Parameters
    ----------
    mask : xarray.DataArray
        Mask to validate. Must have boolean dtype, or binary numeric dtype (0 =
        background, at most one non-zero value = foreground). The latter format
        is produced by [`get_masks`][confusius.atlas.AtlasAccessor.get_masks], and also
        covers masks written by tools without a boolean dtype (e.g. FSL/NiBabel NIfTI
        masks with values in `{0, 1}` or `{0.0, 1.0}`).
    data : xarray.DataArray
        VoxelData array to validate mask against.
    mask_name : str, default: "mask"
        Name of the mask parameter (used in error messages).
    require_exact_dims : bool, default: False
        Whether `mask.dims` must match all non-`time` dimensions of `data` in the same
        order.

    Raises
    ------
    TypeError
        If `mask` is not a boolean or binary numeric DataArray.
    ValueError
        If `mask` or `data` isn't a valid VoxelData array, if `mask`'s voxel
        grid doesn't match `data`'s, or if `require_exact_dims` is set and `mask`'s
        dimensions don't match `data`'s.
    """
    if not isinstance(mask, xr.DataArray):
        raise TypeError(
            f"{mask_name} must be an xarray.DataArray, got {type(mask).__name__}."
        )

    check_mask_dtype(mask, mask_name)
    validate_voxeldata(mask, allow_extra_dims=True)
    validate_voxeldata(data, allow_extra_dims=True)
    _check_spatial_alignment(mask, data, mask_name)

    if require_exact_dims:
        expected_dims = tuple(str(d) for d in data.dims if d != "time")
        mask_dims = tuple(str(d) for d in mask.dims)
        if mask_dims != expected_dims:
            raise ValueError(
                f"{mask_name} dimensions must match all non-time dimensions of data "
                f"in the same order. Expected {expected_dims}, got {mask_dims}."
            )


def ensure_mask(
    mask: xr.DataArray,
    data: xr.DataArray,
    mask_name: str = "mask",
    require_exact_dims: bool = False,
    coerce_bool: bool = True,
) -> xr.DataArray:
    """Canonicalize `mask` and `data`, then validate that `mask` shares data's grid.

    Both `mask` and `data` are canonicalized via
    [`ensure_voxeldata`][confusius.validation.ensure_voxeldata] (restoring any
    scalar-reduced voxel dims) before
    [`validate_mask`][confusius.validation.validate_mask] checks them.

    Parameters
    ----------
    mask : xarray.DataArray
        Mask to validate. Must have boolean dtype, or binary numeric dtype (0 =
        background, at most one non-zero value = foreground). The latter format
        is produced by [`get_masks`][confusius.atlas.AtlasAccessor.get_masks], and also
        covers masks written by tools without a boolean dtype (e.g. FSL/NiBabel NIfTI
        masks with values in `{0, 1}` or `{0.0, 1.0}`).
    data : xarray.DataArray
        VoxelData array to validate mask against.
    mask_name : str, default: "mask"
        Name of the mask parameter (used in error messages).
    require_exact_dims : bool, default: False
        Whether `mask.dims` must match all non-`time` dimensions of `data` in the same
        order.
    coerce_bool : bool, default: True
        Whether to coerce the returned `mask` to boolean dtype. Binary numeric masks
        (`{0, x}`) become `{False, True}` so callers can index with the result without
        the non-zero value being misread as a positional index. When False, `mask` is
        returned with its original dtype unchanged.

    Returns
    -------
    xarray.DataArray
        The canonicalized `mask`, coerced to boolean dtype when `coerce_bool` is `True`
        (the default), otherwise returned with its original dtype.

    Raises
    ------
    TypeError
        If `mask` is not a boolean or binary numeric DataArray.
    ValueError
        If `mask` or `data` isn't a VoxelData array, if `mask`'s voxel
        grid doesn't match `data`'s, or if `require_exact_dims` is set and `mask`'s
        dimensions don't match `data`'s.
    """
    mask = ensure_voxeldata(mask, allow_extra_dims=True)
    data = ensure_voxeldata(data, allow_extra_dims=True)
    validate_mask(mask, data, mask_name, require_exact_dims=require_exact_dims)

    # Coerce after validation so callers never index with a raw binary numeric
    # mask (which xarray.isel would treat as positional indices). See PR #197.
    if coerce_bool:
        return mask.astype(bool)
    return mask


def validate_labels(
    labels: xr.DataArray,
    data: xr.DataArray,
    labels_name: str = "labels",
) -> None:
    """Validate that a label map shares data's VoxelData grid.

    `labels` and `data` must already be canonical VoxelData arrays (see
    [`validate_voxeldata`][confusius.validation.validate_voxeldata]) -- this does not
    canonicalize either. For a `labels`/`data` pair that may not already be canonical
    (e.g. a scalar-reduced voxel dim), use
    [`ensure_labels`][confusius.validation.ensure_labels] instead.

    Parameters
    ----------
    labels : xarray.DataArray
        Label map to validate. Must have integer dtype. Accepts two formats:

        - **Flat label map**: Spatial dims only, e.g. `(k, j, i)`. Background voxels
          labeled `0`; each unique non-zero integer identifies a distinct,
          non-overlapping region. The `regions` coordinate of the output holds the
          integer label values.
        - **Stacked mask format**: Has a leading `mask` dimension followed by
          spatial dims, e.g. `(mask, k, j, i)`. Each layer has values in `{0,
          region_id}` and regions may overlap. The `region` coordinate of the
          output holds the `mask` coordinate values (e.g., region label).

    data : xarray.DataArray
        VoxelData array to validate labels against.
    labels_name : str, default: "labels"
        Name of the labels parameter (used in error messages).

    Raises
    ------
    TypeError
        If `labels` is not an integer dtype DataArray.
    ValueError
        If `labels` or `data` isn't a valid VoxelData array, or if `labels`'s
        voxel grid doesn't match `data`'s.
    """
    if not isinstance(labels, xr.DataArray):
        raise TypeError(
            f"{labels_name} must be an xarray.DataArray, got {type(labels).__name__}."
        )

    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"{labels_name} must be integer dtype, got {labels.dtype}.")

    validate_voxeldata(labels, allow_extra_dims=True)
    validate_voxeldata(data, allow_extra_dims=True)
    _check_spatial_alignment(labels, data, labels_name)


def ensure_labels(
    labels: xr.DataArray,
    data: xr.DataArray,
    labels_name: str = "labels",
) -> xr.DataArray:
    """Canonicalize `labels` and `data`, then validate labels share data's grid.

    Both `labels` and `data` are canonicalized via
    [`ensure_voxeldata`][confusius.validation.ensure_voxeldata] (restoring any
    scalar-reduced voxel dims) before
    [`validate_labels`][confusius.validation.validate_labels] checks them.

    Parameters
    ----------
    labels : xarray.DataArray
        Label map to validate. Must have integer dtype. Accepts two formats:

        - **Flat label map**: Spatial dims only, e.g. `(k, j, i)`. Background voxels
          labeled `0`; each unique non-zero integer identifies a distinct,
          non-overlapping region. The `regions` coordinate of the output holds the
          integer label values.
        - **Stacked mask format**: Has a leading `mask` dimension followed by
          spatial dims, e.g. `(mask, k, j, i)`. Each layer has values in `{0,
          region_id}` and regions may overlap. The `region` coordinate of the
          output holds the `mask` coordinate values (e.g., region label).

    data : xarray.DataArray
        VoxelData array to validate labels against.
    labels_name : str, default: "labels"
        Name of the labels parameter (used in error messages).

    Returns
    -------
    xarray.DataArray
        The canonicalized `labels`.

    Raises
    ------
    TypeError
        If `labels` is not an integer dtype DataArray.
    ValueError
        If `labels` or `data` isn't a VoxelData array, or if `labels`'s
        voxel grid doesn't match `data`'s.
    """
    labels = ensure_voxeldata(labels, allow_extra_dims=True)
    data = ensure_voxeldata(data, allow_extra_dims=True)
    validate_labels(labels, data, labels_name)
    return labels
