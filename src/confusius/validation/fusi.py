"""Validation helpers for ConfUSIus-style fUSI DataArrays."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import xarray as xr

from confusius._dims import POSE_DIM, SPATIAL_DIMS, TIME_DIM
from confusius._utils.coordinates import get_coordinate_spacing_info

_ALLOWED_CORE_DIMS = (TIME_DIM, POSE_DIM, *SPATIAL_DIMS)
"""Core dimension names recognized by ConfUSIus fUSI validators."""

_CANONICAL_DIM_ORDER = (TIME_DIM, POSE_DIM, *SPATIAL_DIMS)
"""Canonical relative ordering of ConfUSIus core dimensions."""


def _validate_dimension_coordinate(da: xr.DataArray, dim: Hashable) -> None:
    """Validate a single dimension coordinate.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose coordinate should be validated.
    dim : Hashable
        Dimension name whose matching coordinate is required.

    Raises
    ------
    ValueError
        If the coordinate is missing, is not one-dimensional along `dim`, has the
        wrong length, contains non-finite numeric values, or is not strictly
        increasing when numeric.
    """
    if dim not in da.coords:
        raise ValueError(f"Missing required coordinate for dimension {dim!r}.")

    coord = da.coords[dim]
    if coord.dims != (dim,):
        raise ValueError(
            f"Coordinate {dim!r} must be a 1D dimension coordinate with dims "
            f"({dim!r},), got {coord.dims!r}."
        )

    if coord.sizes[dim] != da.sizes[dim]:
        raise ValueError(
            f"Coordinate {dim!r} has length {coord.sizes[dim]}, but dimension "
            f"{dim!r} has size {da.sizes[dim]}."
        )

    if np.issubdtype(coord.dtype, np.number):
        values = np.asarray(coord.values)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Coordinate {dim!r} contains non-finite numeric values.")
        if values.size > 1 and not np.all(np.diff(values) > 0):
            raise ValueError(
                f"Coordinate {dim!r} must be strictly monotonic-increasing."
            )


def _validate_core_dimension_names(da: xr.DataArray, allow_extra_dims: bool) -> None:
    """Validate that core ConfUSIus dimensions are named consistently.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose dimensions should be checked.
    allow_extra_dims : bool
        Whether dimensions outside the ConfUSIus core set are allowed.

    Raises
    ------
    ValueError
        If any dimension name is not a string or if extra dimensions are present when
        `allow_extra_dims` is `False`.
    """
    invalid_dims = [dim for dim in da.dims if not isinstance(dim, str)]
    if invalid_dims:
        raise ValueError(
            f"All dimensions must be strings, got invalid dimensions: {invalid_dims!r}."
        )

    if not allow_extra_dims:
        unexpected_dims = [dim for dim in da.dims if dim not in _ALLOWED_CORE_DIMS]
        if unexpected_dims:
            raise ValueError(
                f"Unexpected dimensions {unexpected_dims!r}. ConfUSIus fUSI DataArrays "
                f"may only use dimensions {_ALLOWED_CORE_DIMS!r}."
            )


def _validate_canonical_core_dim_order(da: xr.DataArray) -> None:
    """Validate the relative order of ConfUSIus core dimensions.

    Extra dimensions are ignored. Only the subsequence formed by the ConfUSIus core
    dimensions present in `da.dims` is checked.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose dimension order should be checked.

    Raises
    ------
    ValueError
        If the relative order of the present core dimensions differs from the canonical
        ConfUSIus order.
    """
    present_core_dims = tuple(dim for dim in da.dims if dim in _ALLOWED_CORE_DIMS)
    expected_order = tuple(dim for dim in _CANONICAL_DIM_ORDER if dim in da.dims)
    if present_core_dims != expected_order:
        raise ValueError(
            f"Core dimensions {present_core_dims!r} are not in canonical ConfUSIus "
            f"order {expected_order!r}."
        )


def _validate_required_coordinate_attrs(
    da: xr.DataArray,
    dims: tuple[str, ...],
    attr_name: str,
) -> None:
    """Validate that selected dimension coordinates carry a required attribute.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose dimension coordinates should be checked.
    dims : tuple[str, ...]
        Coordinate dimensions that must carry the attribute when present in `da`.
    attr_name : str
        Required coordinate attribute name.

    Raises
    ------
    ValueError
        If a required attribute is missing from any selected coordinate.
    """
    for dim in dims:
        if dim in da.coords and attr_name not in da.coords[dim].attrs:
            raise ValueError(
                f"Coordinate {dim!r} is missing required {attr_name!r} metadata."
            )


def _validate_regular_spacing(
    da: xr.DataArray,
    regular_spacing_tolerance: float,
) -> None:
    """Validate that numeric dimension coordinates have regular spacing.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose numeric dimension coordinates should be checked.
    regular_spacing_tolerance : float
        Relative tolerance used to assess regularity.

    Raises
    ------
    ValueError
        If a numeric dimension coordinate has non-uniform or undefined spacing.
    """
    for dim in da.dims:
        coord = da.coords[dim]
        if not np.issubdtype(coord.dtype, np.number):
            continue
        spacing = get_coordinate_spacing_info(str(dim), da, regular_spacing_tolerance)
        if spacing.value is None:
            raise ValueError(
                f"Coordinate {dim!r} must have regular spacing, but spacing is "
                "non-uniform or undefined."
            )


def validate_fusi_dataarray(
    data: xr.DataArray,
    *,
    require_time: bool = False,
    allow_pose: bool = True,
    allow_extra_dims: bool = True,
    minimum_spatial_dims: int = 2,
    require_regular_spacing: bool = False,
    regular_spacing_tolerance: float = 1e-2,
    require_canonical_dim_order: bool = False,
    require_spatial_voxdim: bool = False,
    require_spatial_units: bool = False,
    require_time_units: bool = False,
) -> None:
    """Validate that a DataArray follows ConfUSIus fUSI conventions.

    This validator is intentionally broader than
    [`validate_iq_dataarray`][confusius.validation.validate_iq_dataarray]: it accepts
    2D, 3D, 2D+t, 3D+t, and multi-pose variants as long as dimensions, coordinates,
    and optional metadata follow ConfUSIus conventions.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to validate.
    require_time : bool, default: False
        Whether to require a `time` dimension.
    allow_pose : bool, default: True
        Whether to allow a `pose` dimension.
    allow_extra_dims : bool, default: True
        Whether dimensions outside the ConfUSIus core set (`time`, `pose`, `z`, `y`,
        `x`) are allowed.
    minimum_spatial_dims : int, default: 2
        Minimum number of spatial dimensions from `("z", "y", "x")` required in the
        DataArray.
    require_regular_spacing : bool, default: False
        Whether numeric dimension coordinates must have regular spacing.
    regular_spacing_tolerance : float, default: 1e-2
        Relative tolerance used to assess coordinate regularity.
    require_canonical_dim_order : bool, default: False
        Whether the ConfUSIus core dimensions present in the DataArray must appear in
        canonical relative order `(time, pose, z, y, x)`.
    require_spatial_voxdim : bool, default: False
        Whether present spatial coordinates must define a `voxdim` attribute.
    require_spatial_units : bool, default: False
        Whether present spatial coordinates must define a `units` attribute.
    require_time_units : bool, default: False
        Whether the `time` coordinate must define a `units` attribute when present.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    ValueError
        If dimension names are invalid, required dimensions or coordinates are missing,
        there are too few spatial dimensions, numeric coordinates are not strictly
        increasing, optional stricter checks fail, or required metadata is missing.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"data must be an xarray.DataArray, got {type(data).__name__}.")

    if minimum_spatial_dims < 0 or minimum_spatial_dims > len(SPATIAL_DIMS):
        raise ValueError(
            "minimum_spatial_dims must be between 0 and 3 inclusive, got "
            f"{minimum_spatial_dims}."
        )

    _validate_core_dimension_names(data, allow_extra_dims=allow_extra_dims)

    if require_time and TIME_DIM not in data.dims:
        raise ValueError("DataArray must have a 'time' dimension.")

    if not allow_pose and POSE_DIM in data.dims:
        raise ValueError("DataArray must not have a 'pose' dimension.")

    spatial_dims_present = [dim for dim in SPATIAL_DIMS if dim in data.dims]
    if len(spatial_dims_present) < minimum_spatial_dims:
        raise ValueError(
            f"DataArray must have at least {minimum_spatial_dims} spatial dimensions "
            f"from {SPATIAL_DIMS!r}, got {tuple(spatial_dims_present)!r}."
        )

    for dim in data.dims:
        _validate_dimension_coordinate(data, dim)

    if require_regular_spacing:
        _validate_regular_spacing(data, regular_spacing_tolerance)

    if require_canonical_dim_order:
        _validate_canonical_core_dim_order(data)

    if require_spatial_voxdim:
        _validate_required_coordinate_attrs(data, SPATIAL_DIMS, "voxdim")

    if require_spatial_units:
        _validate_required_coordinate_attrs(data, SPATIAL_DIMS, "units")

    if require_time_units and TIME_DIM in data.dims:
        _validate_required_coordinate_attrs(data, (TIME_DIM,), "units")
