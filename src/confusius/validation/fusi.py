"""Validation helpers for ConfUSIus-style fUSI DataArrays."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr

from confusius._dims import CORE_DIMS, POSE_DIM, TIME_DIM, VOXEL_DIMS
from confusius._utils.coordinates import get_coordinate_spacing_info
from confusius._utils.geometry import (
    get_voxel_affine_spatial_dims,
    get_voxel_affine_world_coord_names,
    get_voxel_to_world_affine,
    get_voxel_world_spacing,
    has_voxel_world_geometry,
)
from confusius._utils.validation import require_dataarray
from confusius.validation.time_series import (
    validate_required_time_dimension,
    validate_timepoint_count,
    validate_unchunked_time,
    validate_uniform_time,
)

RegularSpacingDims = Literal["space", "core", "all"] | str | Sequence[str]
"""Selector for dimensions that must satisfy regular-spacing checks."""


def _get_spatial_dims(da: xr.DataArray) -> tuple[str, ...]:
    """Return present voxel-space spatial dimensions.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    tuple[str, ...]
        Present voxel-space spatial dimensions in canonical order.
    """
    return tuple(dim for dim in VOXEL_DIMS if dim in da.dims)


def _validate_voxel_affine_geometry(da: xr.DataArray) -> None:
    """Validate voxel-affine metadata.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to validate.

    Raises
    ------
    ValueError
        If voxel-affine metadata is missing or inconsistent.
    """
    if not has_voxel_world_geometry(da):
        raise ValueError(
            "DataArray must use native voxel dimensions `k/j/i` with "
            "VoxelToWorldIndex-backed world coordinates and defined spatial spacing."
        )

    voxel_dims = get_voxel_affine_spatial_dims(da)
    expected_shape = (len(voxel_dims) + 1, len(voxel_dims) + 1)
    affine = get_voxel_to_world_affine(da)
    if affine.shape != expected_shape:
        raise ValueError(
            "voxel_to_world must have shape "
            f"{expected_shape} for voxel-affine dimensions {voxel_dims!r}, got "
            f"{affine.shape}."
        )

    world_coord_names = get_voxel_affine_world_coord_names(da)
    for name, dim in zip(world_coord_names, voxel_dims, strict=True):
        if name not in da.coords:
            raise ValueError(
                f"Voxel-affine data is missing physical coordinate {name!r}."
            )
        coord = da.coords[name]
        if set(coord.dims) not in ({dim}, set(voxel_dims)):
            raise ValueError(
                f"Voxel-affine coordinate {name!r} must have dims {voxel_dims!r} "
                f"(in any order) or {(dim,)!r}, got {coord.dims!r}."
            )


def _validate_dimension_coordinate(
    da: xr.DataArray, dim: Hashable, *, require_numeric: bool
) -> None:
    """Validate a single dimension coordinate.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose coordinate should be validated.
    dim : hashable
        Dimension name whose matching coordinate is required.
    require_numeric : bool
        Whether the coordinate must be numeric, finite, and strictly increasing.

    Raises
    ------
    ValueError
        If the coordinate is missing, malformed, non-numeric, non-finite, or not
        strictly increasing.
    """
    if dim not in da.coords:
        if dim in CORE_DIMS:
            raise ValueError(f"Missing required coordinate for dimension {dim!r}.")
        return

    coord = da.coords[dim]
    if coord.dims != (dim,):
        raise ValueError(
            f"Coordinate {dim!r} must be a 1D dimension coordinate with dims "
            f"({dim!r},), got {coord.dims!r}."
        )

    if not require_numeric:
        return
    if not np.issubdtype(coord.dtype, np.number):
        raise ValueError(f"Coordinate {dim!r} must be numeric.")

    values = np.asarray(coord.values)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Coordinate {dim!r} contains non-finite numeric values.")
    if values.size > 1 and not np.all(np.diff(values) > 0):
        raise ValueError(f"Coordinate {dim!r} must be strictly monotonic-increasing.")


def _validate_core_dimension_names(da: xr.DataArray, allow_extra_dims: bool) -> None:
    """Validate core dimension names.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose dimensions should be checked.
    allow_extra_dims : bool
        Whether dimensions outside the ConfUSIus core set are allowed.

    Raises
    ------
    ValueError
        If dimensions are not strings or unexpected dimensions are present.
    """
    invalid_dims = [dim for dim in da.dims if not isinstance(dim, str)]
    if invalid_dims:
        raise ValueError(
            f"All dimensions must be strings, got invalid dimensions: {invalid_dims!r}."
        )

    if not allow_extra_dims:
        unexpected_dims = [dim for dim in da.dims if dim not in CORE_DIMS]
        if unexpected_dims:
            raise ValueError(
                f"Unexpected dimensions {unexpected_dims!r}. ConfUSIus fUSI DataArrays "
                f"may only use dimensions {CORE_DIMS!r} and at most 3 spatial dimensions."
            )


def _validate_canonical_core_dim_order(da: xr.DataArray) -> None:
    """Validate the relative order of ConfUSIus core dimensions.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose dimension order should be checked.

    Raises
    ------
    ValueError
        If core dimensions are not in canonical order.
    """
    present_core_dims = tuple(dim for dim in da.dims if dim in CORE_DIMS)
    expected_order = tuple(dim for dim in CORE_DIMS if dim in da.dims)
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
    """Validate that selected coordinates carry a required attribute.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose coordinates should be checked.
    dims : tuple[str, ...]
        Coordinates that must carry the attribute when present in `da`.
    attr_name : str
        Required coordinate attribute name.

    Raises
    ------
    ValueError
        If a required attribute is missing.
    """
    for dim in dims:
        if dim in da.coords and attr_name not in da.coords[dim].attrs:
            raise ValueError(
                f"Coordinate {dim!r} is missing required {attr_name!r} metadata."
            )


def _validate_regular_spacing(
    da: xr.DataArray,
    regular_spacing_tolerance: float,
    regular_spacing_dims: RegularSpacingDims,
    spatial_dims: tuple[str, ...],
) -> None:
    """Validate that selected numeric coordinates have regular spacing.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose coordinates should be checked.
    regular_spacing_tolerance : float
        Relative tolerance used to assess regularity.
    regular_spacing_dims : {"space", "core", "all"} or str or sequence[str]
        Dimensions to validate.
    spatial_dims : tuple[str, ...]
        Present voxel-space spatial dimensions.

    Raises
    ------
    ValueError
        If selected coordinate spacing is missing, invalid, or non-uniform.
    """
    if regular_spacing_dims == "space":
        dims_to_check = list(spatial_dims)
    elif regular_spacing_dims == "core":
        dims_to_check = [dim for dim in CORE_DIMS if dim in da.dims]
    elif regular_spacing_dims == "all":
        dims_to_check = [str(dim) for dim in da.dims]
    elif isinstance(regular_spacing_dims, str):
        dims_to_check = [regular_spacing_dims]
    else:
        dims_to_check = [str(dim) for dim in regular_spacing_dims]

    missing_dims = [dim for dim in dims_to_check if dim not in da.dims]
    if missing_dims:
        raise ValueError(
            "regular_spacing_dims contains dimensions not present in data: "
            f"{missing_dims!r}. Present dims: {tuple(str(dim) for dim in da.dims)!r}."
        )

    voxel_spacing = get_voxel_world_spacing(da)
    for dim in dims_to_check:
        if dim not in da.coords:
            raise ValueError(
                f"Missing required coordinate for dimension {dim!r} when checking "
                "for regular spacing."
            )
        coord = da.coords[dim]
        if not np.issubdtype(coord.dtype, np.number):
            continue
        spacing_value = (
            voxel_spacing[dim]
            if dim in spatial_dims
            else get_coordinate_spacing_info(dim, da, regular_spacing_tolerance).value
        )
        if spacing_value is None:
            raise ValueError(
                f"Coordinate {dim!r} must have regular spacing, but spacing is "
                "non-uniform or undefined."
            )


def canonicalize_fusi(data: xr.DataArray) -> xr.DataArray:
    """Return `data` with scalar voxel-space spatial coordinates restored as dims.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to canonicalize.

    Returns
    -------
    xarray.DataArray
        DataArray with missing scalar-indexed voxel dimensions restored.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    ValueError
        If a missing voxel dimension has no scalar coordinate to restore.
    """
    require_dataarray(data)

    result = data
    for dim in VOXEL_DIMS:
        if dim in result.dims:
            continue
        if dim not in result.coords:
            continue
        if result.coords[dim].shape != ():
            raise ValueError(
                f"DataArray is missing voxel dimension {dim!r}, but coordinate "
                f"{dim!r} is not scalar."
            )
        coord = result.coords[dim]
        attrs = coord.attrs.copy()
        spatial_index = VOXEL_DIMS.index(dim)
        next_dims = [d for d in VOXEL_DIMS[spatial_index + 1 :] if d in result.dims]
        previous_dims = [d for d in VOXEL_DIMS[:spatial_index] if d in result.dims]
        if next_dims:
            axis = result.dims.index(next_dims[0])
        elif previous_dims:
            axis = result.dims.index(previous_dims[-1]) + 1
        else:
            axis = len(result.dims)
        result = result.expand_dims({dim: [coord.item()]}, axis=axis)
        result.coords[dim].attrs.update(attrs)

    from confusius._utils.geometry import restore_world_coords_from_voxel_affine

    return restore_world_coords_from_voxel_affine(result)


def ensure_fusi(data: xr.DataArray, **validate_kwargs: Any) -> xr.DataArray:
    """Canonicalize and validate a ConfUSIus fUSI DataArray.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to canonicalize and validate.
    **validate_kwargs : Any
        Keyword arguments forwarded to
        [validate_fusi][confusius.validation.validate_fusi].

    Returns
    -------
    xarray.DataArray
        Canonicalized DataArray that passed fUSI validation.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    ValueError
        If canonicalization or validation fails.
    """
    require_dataarray(data)
    result = canonicalize_fusi(data)
    validate_fusi(result, **validate_kwargs)
    return result


def validate_fusi(
    data: xr.DataArray,
    *,
    require_time: bool = False,
    require_unchunked_time: bool = False,
    require_uniform_time: bool = False,
    uniformity_tolerance: float = 1e-2,
    allow_pose: bool = True,
    allow_extra_dims: bool = True,
    minimum_spatial_dims: int = 2,
    require_regular_spacing: bool = False,
    regular_spacing_tolerance: float = 1e-2,
    regular_spacing_dims: RegularSpacingDims = "space",
    require_canonical_dim_order: bool = False,
    require_spatial_voxdim: bool = False,
    require_spatial_units: bool = False,
    require_time_units: bool = False,
) -> None:
    """Validate that a DataArray follows ConfUSIus fUSI conventions.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to validate. It must use native voxel dimensions `k`, `j`, `i` and
        world coordinates `z`, `y`, `x`.
    require_time : bool, default: False
        Whether to require a `time` dimension with more than one timepoint.
    require_unchunked_time : bool, default: False
        Whether to require the time dimension to occupy one Dask chunk.
    require_uniform_time : bool, default: False
        Whether to require uniformly sampled time coordinates.
    uniformity_tolerance : float, default: 1e-2
        Maximum relative variation allowed between consecutive time intervals.
    allow_pose : bool, default: True
        Whether to allow a `pose` dimension.
    allow_extra_dims : bool, default: True
        Whether dimensions outside the ConfUSIus core set are allowed.
    minimum_spatial_dims : int, default: 2
        Minimum number of voxel spatial dimensions required.
    require_regular_spacing : bool, default: False
        Whether selected numeric dimension coordinates must have regular spacing.
    regular_spacing_tolerance : float, default: 1e-2
        Relative tolerance used to assess coordinate regularity.
    regular_spacing_dims : {"space", "core", "all"} or str or sequence[str], default: "space"
        Dimensions that must satisfy regular-spacing checks.
    require_canonical_dim_order : bool, default: False
        Whether core dimensions must appear in canonical order.
    require_spatial_voxdim : bool, default: False
        Whether present physical spatial coordinates must define `voxdim` metadata.
    require_spatial_units : bool, default: False
        Whether present physical spatial coordinates must define `units` metadata.
    require_time_units : bool, default: False
        Whether the `time` coordinate must define `units` metadata when present.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    ValueError
        If dimension, coordinate, timing, or metadata validation fails.
    """
    require_dataarray(data)

    if minimum_spatial_dims < 0 or minimum_spatial_dims > len(VOXEL_DIMS):
        raise ValueError(
            "minimum_spatial_dims must be between 0 and 3 inclusive, got "
            f"{minimum_spatial_dims}."
        )

    _validate_core_dimension_names(data, allow_extra_dims=allow_extra_dims)
    _validate_voxel_affine_geometry(data)

    if require_time or require_unchunked_time or require_uniform_time:
        validate_required_time_dimension(data)
        validate_timepoint_count(data, "fUSI validation")
    if require_unchunked_time:
        validate_unchunked_time(data, "fUSI validation")
    if require_uniform_time:
        validate_uniform_time(data, "fUSI validation", uniformity_tolerance)

    if not allow_pose and POSE_DIM in data.dims:
        raise ValueError("DataArray must not have a 'pose' dimension.")

    spatial_dims = _get_spatial_dims(data)
    if len(spatial_dims) < minimum_spatial_dims:
        raise ValueError(
            f"DataArray must have at least {minimum_spatial_dims} spatial dimensions "
            f"from {VOXEL_DIMS!r}, got {spatial_dims!r}."
        )

    for dim in data.dims:
        _validate_dimension_coordinate(data, dim, require_numeric=dim in CORE_DIMS)

    if require_regular_spacing:
        _validate_regular_spacing(
            data,
            regular_spacing_tolerance,
            regular_spacing_dims,
            spatial_dims,
        )

    if require_canonical_dim_order:
        _validate_canonical_core_dim_order(data)

    physical_coords = get_voxel_affine_world_coord_names(data)
    singleton_spatial_dims = tuple(
        name
        for name, dim in zip(physical_coords, spatial_dims, strict=True)
        if data.sizes[dim] == 1
    )
    if require_spatial_voxdim:
        _validate_required_coordinate_attrs(data, spatial_dims, "voxdim")
    else:
        _validate_required_coordinate_attrs(data, singleton_spatial_dims, "voxdim")
    if require_spatial_units:
        _validate_required_coordinate_attrs(data, physical_coords, "units")
    if require_time_units and TIME_DIM in data.dims:
        _validate_required_coordinate_attrs(data, (TIME_DIM,), "units")


validate_fusi_dataarray = validate_fusi
"""Deprecated internal alias for [validate_fusi][confusius.validation.fusi.validate_fusi]."""
