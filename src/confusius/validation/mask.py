"""Acquisition-aware validation layered on VoxelData's mask and label helpers."""

import xarray as xr
from voxeldata.masks import check_mask_dtype, check_spatial_alignment
from voxeldata.masks import ensure_labels as ensure_generic_labels
from voxeldata.masks import ensure_mask as ensure_generic_mask
from voxeldata.masks import validate_labels as validate_generic_labels
from voxeldata.masks import validate_mask as validate_generic_mask

from confusius.validation.voxeldata import ensure_voxeldata, validate_voxeldata

__all__ = [
    "check_mask_dtype",
    "check_spatial_alignment",
    "ensure_labels",
    "ensure_mask",
    "validate_labels",
    "validate_mask",
]


def validate_mask(
    mask: xr.DataArray,
    data: xr.DataArray,
    mask_name: str = "mask",
    require_exact_dims: bool = False,
) -> None:
    """Validate a mask's voxel grid and ConfUSIus acquisition metadata.

    Parameters
    ----------
    mask : xarray.DataArray
        Canonical VoxelData mask with boolean or finite binary numeric values.
    data : xarray.DataArray
        Canonical reference VoxelData array.
    mask_name : str, default: "mask"
        Name of the mask parameter in errors.
    require_exact_dims : bool, default: False
        Whether to require all non-time dimensions in the same order as `data`.

    Raises
    ------
    TypeError
        If `mask` is not a boolean or binary numeric DataArray.
    ValueError
        If model, grid, dimensions, or acquisition metadata are invalid.
    """
    validate_generic_mask(mask, data, mask_name, require_exact_dims=require_exact_dims)
    validate_voxeldata(mask, allow_extra_dims=True)
    validate_voxeldata(data, allow_extra_dims=True)


def ensure_mask(
    mask: xr.DataArray,
    data: xr.DataArray,
    mask_name: str = "mask",
    require_exact_dims: bool = False,
    coerce_bool: bool = True,
) -> xr.DataArray:
    """Canonicalize and validate a mask, including acquisition defaults.

    Geometry, binary-mask validation, alignment, and coercion belong to VoxelData.
    ConfUSIus adds its acquisition validation and defaults before delegation.

    Parameters
    ----------
    mask : xarray.DataArray
        VoxelData mask with boolean or finite binary numeric values.
    data : xarray.DataArray
        Reference VoxelData array.
    mask_name : str, default: "mask"
        Name of the mask parameter in errors.
    require_exact_dims : bool, default: False
        Whether to require all non-time dimensions in the same order as `data`.
    coerce_bool : bool, default: True
        Whether to coerce validated numeric foreground values to booleans.

    Returns
    -------
    xarray.DataArray
        Canonical mask, optionally coerced to boolean dtype.

    Raises
    ------
    TypeError
        If `mask` is not a boolean or binary numeric DataArray.
    ValueError
        If model, grid, dimensions, or acquisition metadata are invalid.
    """
    mask = ensure_voxeldata(mask, allow_extra_dims=True)
    data = ensure_voxeldata(data, allow_extra_dims=True)
    return ensure_generic_mask(
        mask,
        data,
        mask_name,
        require_exact_dims=require_exact_dims,
        coerce_bool=coerce_bool,
    )


def validate_labels(
    labels: xr.DataArray,
    data: xr.DataArray,
    labels_name: str = "labels",
) -> None:
    """Validate label geometry and ConfUSIus acquisition metadata.

    Parameters
    ----------
    labels : xarray.DataArray
        Canonical integer VoxelData labels, flat or stacked along `mask`.
    data : xarray.DataArray
        Canonical reference VoxelData array.
    labels_name : str, default: "labels"
        Name of the labels parameter in errors.

    Raises
    ------
    TypeError
        If labels are not an integer DataArray.
    ValueError
        If model, voxel grid, or acquisition metadata are invalid.
    """
    validate_generic_labels(labels, data, labels_name)
    validate_voxeldata(labels, allow_extra_dims=True)
    validate_voxeldata(data, allow_extra_dims=True)


def ensure_labels(
    labels: xr.DataArray,
    data: xr.DataArray,
    labels_name: str = "labels",
) -> xr.DataArray:
    """Canonicalize and validate labels, including acquisition defaults.

    Parameters
    ----------
    labels : xarray.DataArray
        Integer VoxelData labels, flat or stacked along `mask`.
    data : xarray.DataArray
        Reference VoxelData array.
    labels_name : str, default: "labels"
        Name of the labels parameter in errors.

    Returns
    -------
    xarray.DataArray
        Canonical integer label array on the reference voxel grid.

    Raises
    ------
    TypeError
        If labels are not an integer DataArray.
    ValueError
        If model, voxel grid, or acquisition metadata are invalid.
    """
    labels = ensure_voxeldata(labels, allow_extra_dims=True)
    data = ensure_voxeldata(data, allow_extra_dims=True)
    return ensure_generic_labels(labels, data, labels_name)
