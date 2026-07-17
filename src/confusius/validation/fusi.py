"""Validation helpers for ConfUSIus-style fUSI DataArrays."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Literal

import numpy as np
import xarray as xr

from confusius._dims import POSE_DIM, TIME_DIM
from confusius._utils.coordinates import get_coordinate_spacing_info
from confusius._utils.geometry import (
    get_voxel_affine_physical_coord_names,
    get_voxel_affine_spatial_dims,
    has_voxel_affine_geometry,
)

RegularSpacingDims = Literal["space", "core", "all"] | str | Sequence[str]
"""Selector for dimensions that must satisfy regular-spacing checks."""

_PHYSICAL_CORE_DIMS = (TIME_DIM, POSE_DIM, "z", "y", "x")
"""Physical-grid dimension names recognized by ConfUSIus fUSI validators."""

_VOXEL_CORE_DIMS = (TIME_DIM, POSE_DIM, "k", "j", "i")
"""Voxel-affine dimension names recognized by ConfUSIus fUSI validators."""

_VOXEL_SPATIAL_DIMS = ("k", "j", "i")
"""Voxel-space spatial dimension names used by ConfUSIus geometry."""

_PHYSICAL_SPATIAL_DIMS = ("z", "y", "x")
"""Physical-grid spatial dimension names used by ConfUSIus geometry."""


def _get_allowed_core_dims(da: xr.DataArray) -> tuple[str, ...]:
    """Return the core dimension names valid for a DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    tuple[str, ...]
        Allowed core dimensions for `da`.
    """
    return _VOXEL_CORE_DIMS if has_voxel_affine_geometry(da) else _PHYSICAL_CORE_DIMS


def _get_spatial_dims(da: xr.DataArray) -> tuple[str, ...]:
    """Return the spatial dimensions to validate for a DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    tuple[str, ...]
        Spatial dimensions for `da`, using `k/j/i` for voxel-affine geometry and
        `z/y/x` otherwise.
    """
    spatial_dims = (
        _VOXEL_SPATIAL_DIMS if has_voxel_affine_geometry(da) else _PHYSICAL_SPATIAL_DIMS
    )
    return tuple(dim for dim in spatial_dims if dim in da.dims)


def _validate_voxel_affine_geometry(da: xr.DataArray) -> None:
    """Validate voxel-affine metadata when present.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to validate.

    Raises
    ------
    ValueError
        If stored voxel-affine metadata is incomplete or inconsistent with the DataArray
        shape.
    """
    if not has_voxel_affine_geometry(da):
        return

    voxel_dims = get_voxel_affine_spatial_dims(da)
    expected_shape = (len(voxel_dims) + 1, len(voxel_dims) + 1)
    affine = np.asarray(da.attrs["voxel_to_physical"])
    if affine.shape != expected_shape:
        raise ValueError(
            "voxel_to_physical must have shape "
            f"{expected_shape} for voxel-affine dimensions {voxel_dims!r}, got "
            f"{affine.shape}."
        )

    physical_coord_names = get_voxel_affine_physical_coord_names(da)
    for name, dim in zip(physical_coord_names, voxel_dims, strict=True):
        if name not in da.coords:
            raise ValueError(
                f"Voxel-affine data is missing physical coordinate {name!r}."
            )
        coord = da.coords[name]
        if coord.dims not in {voxel_dims, (dim,)}:
            raise ValueError(
                f"Voxel-affine coordinate {name!r} must have dims {voxel_dims!r} "
                f"or {(dim,)!r}, got {coord.dims!r}."
            )


def _validate_dimension_coordinate(
    da: xr.DataArray,
    dim: Hashable,
    *,
    require_numeric: bool,
    allowed_core_dims: tuple[str, ...],
) -> None:
    """Validate a single dimension coordinate.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose coordinate should be validated.
    dim : Hashable
        Dimension name whose matching coordinate is required.
    require_numeric : bool
        Whether the coordinate must be numeric, finite, and strictly increasing.
    allowed_core_dims : tuple[str, ...]
        Core dimensions whose matching coordinates are required.

    Raises
    ------
    ValueError
        If the coordinate is missing, is not one-dimensional along `dim`, or (when
        `require_numeric=True`) is non-numeric, non-finite, or not strictly
        increasing.
    """
    if dim not in da.coords:
        if dim in allowed_core_dims:
            raise ValueError(f"Missing required coordinate for dimension {dim!r}.")
        return

    coord = da.coords[dim]
    if coord.dims != (dim,):
        raise ValueError(
            f"Coordinate {dim!r} must be a 1D dimension coordinate with dims "
            f"({dim!r},), got {coord.dims!r}."
        )

    if require_numeric and not np.issubdtype(coord.dtype, np.number):
        raise ValueError(f"Coordinate {dim!r} must be numeric.")

    if require_numeric:
        values = np.asarray(coord.values)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Coordinate {dim!r} contains non-finite numeric values.")
        if values.size > 1 and not np.all(np.diff(values) > 0):
            raise ValueError(
                f"Coordinate {dim!r} must be strictly monotonic-increasing."
            )


def _validate_core_dimension_names(
    da: xr.DataArray,
    allow_extra_dims: bool,
    allowed_core_dims: tuple[str, ...],
) -> None:
    """Validate that core ConfUSIus dimensions are named consistently.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose dimensions should be checked.
    allow_extra_dims : bool
        Whether dimensions outside the ConfUSIus core set are allowed.
    allowed_core_dims : tuple[str, ...]
        Core dimensions valid for the current geometry model.

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
        unexpected_dims = [dim for dim in da.dims if dim not in allowed_core_dims]
        if unexpected_dims:
            raise ValueError(
                f"Unexpected dimensions {unexpected_dims!r}. ConfUSIus fUSI DataArrays "
                f"may only use dimensions {allowed_core_dims!r}."
            )


def _validate_canonical_core_dim_order(
    da: xr.DataArray, allowed_core_dims: tuple[str, ...]
) -> None:
    """Validate the relative order of ConfUSIus core dimensions.

    Extra dimensions are ignored. Only the subsequence formed by the ConfUSIus core
    dimensions present in `da.dims` is checked.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose dimension order should be checked.
    allowed_core_dims : tuple[str, ...]
        Core dimensions valid for the current geometry model.

    Raises
    ------
    ValueError
        If the relative order of the present core dimensions differs from the canonical
        ConfUSIus order.
    """
    present_core_dims = tuple(dim for dim in da.dims if dim in allowed_core_dims)
    expected_order = tuple(dim for dim in allowed_core_dims if dim in da.dims)
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
    regular_spacing_dims: RegularSpacingDims,
    spatial_dims: tuple[str, ...],
) -> None:
    """Validate that selected numeric coordinates have regular spacing.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose numeric dimension coordinates should be checked.
    regular_spacing_tolerance : float
        Relative tolerance used to assess regularity.
    regular_spacing_dims : {"space", "core", "all"} or str or sequence[str]
        Dimensions to validate. `"space"` checks present spatial dimensions, `"core"`
        checks present core dimensions, and `"all"` checks all present dimensions. Any
        other string is treated as a single dimension name. A sequence checks only
        listed dimensions. Non-numeric dimensions are ignored.
    spatial_dims : tuple[str, ...]
        Spatial dimensions for the current geometry model.

    Raises
    ------
    ValueError
        If `regular_spacing_dims` is invalid, references missing dimensions, or if a
        selected numeric coordinate has non-uniform or undefined spacing.
    """
    if regular_spacing_dims == "space":
        dims_to_check = list(spatial_dims)
    elif regular_spacing_dims == "core":
        dims_to_check = [dim for dim in _get_allowed_core_dims(da) if dim in da.dims]
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

    for dim in dims_to_check:
        if dim not in da.coords:
            raise ValueError(
                f"Missing required coordinate for dimension {dim!r} when checking "
                "for regular spacing."
            )
        coord = da.coords[dim]
        if not np.issubdtype(coord.dtype, np.number):
            continue
        spacing = get_coordinate_spacing_info(dim, da, regular_spacing_tolerance)
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
    regular_spacing_dims: RegularSpacingDims = "space",
    require_canonical_dim_order: bool = False,
    require_spatial_voxdim: bool = False,
    require_spatial_units: bool = False,
    require_time_units: bool = False,
) -> None:
    """Validate that a DataArray follows ConfUSIus fUSI conventions.

    A valid fUSI DataArray must:

    - Use ConfUSIus native voxel dimensions drawn from `(time, pose, k, j, i)`, with
      optional extra dimensions if `allow_extra_dims` is `True` (e.g., `region`,
      `component`, `mask`, etc.).
    - Store linked physical coordinates through a `voxel_to_physical` affine and the
      corresponding physical `z/y/x` coordinates.
    - Have matching 1D coordinates for all core dimensions. Extra-dimension
      coordinates are optional.
    - Have numeric, finite, and strictly increasing core dimension coordinates.

    Additional requirements can be enforced using the function parameters.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to validate.
    require_time : bool, default: False
        Whether to require a `time` dimension.
    allow_pose : bool, default: True
        Whether to allow a `pose` dimension.
    allow_extra_dims : bool, default: True
        Whether dimensions outside the ConfUSIus core set (`time`, `pose`, `k`, `j`,
        `i`) are allowed.
    minimum_spatial_dims : int, default: 2
        Minimum number of voxel spatial dimensions required in the DataArray. This
        counts present dimensions from `k/j/i`.
    require_regular_spacing : bool, default: False
        Whether numeric dimension coordinates must have regular spacing.
    regular_spacing_tolerance : float, default: 1e-2
        Relative tolerance used to assess coordinate regularity.
    regular_spacing_dims : {"space", "core", "all"} or str or sequence[str], default: "space"
        Dimensions that must satisfy regular-spacing checks when
        `require_regular_spacing=True`. Use `"space"` for present `k`, `j`, `i`
        dimensions, `"core"` for present core dimensions (`time`, `pose`, `k`, `j`,
        `i`), `"all"` for all present dimensions, a string for one explicit dimension
        name, or a sequence for multiple explicit dimension names. Non-numeric
        coordinates are ignored.
    require_canonical_dim_order : bool, default: False
        Whether the ConfUSIus core dimensions present in the DataArray must appear in
        canonical relative order `(time, pose, k, j, i)`.
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
        there are too few spatial dimensions, core numeric coordinate constraints fail,
        optional stricter checks fail, or required metadata is missing.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"data must be an xarray.DataArray, got {type(data).__name__}.")

    if minimum_spatial_dims < 0 or minimum_spatial_dims > len(_VOXEL_SPATIAL_DIMS):
        raise ValueError(
            "minimum_spatial_dims must be between 0 and 3 inclusive, got "
            f"{minimum_spatial_dims}."
        )

    allowed_core_dims = _get_allowed_core_dims(data)
    spatial_dims = _get_spatial_dims(data)

    _validate_core_dimension_names(
        data,
        allow_extra_dims=allow_extra_dims,
        allowed_core_dims=allowed_core_dims,
    )

    _validate_voxel_affine_geometry(data)

    if require_time and TIME_DIM not in data.dims:
        raise ValueError("DataArray must have a 'time' dimension.")

    if not allow_pose and POSE_DIM in data.dims:
        raise ValueError("DataArray must not have a 'pose' dimension.")

    spatial_dims_present = list(spatial_dims)
    if len(spatial_dims_present) < minimum_spatial_dims:
        raise ValueError(
            f"DataArray must have at least {minimum_spatial_dims} spatial dimensions "
            f"from {_VOXEL_SPATIAL_DIMS!r}, got {tuple(spatial_dims_present)!r}."
        )

    for dim in data.dims:
        _validate_dimension_coordinate(
            data,
            dim,
            require_numeric=dim in allowed_core_dims,
            allowed_core_dims=allowed_core_dims,
        )

    if require_regular_spacing:
        _validate_regular_spacing(
            data,
            regular_spacing_tolerance,
            regular_spacing_dims,
            spatial_dims,
        )

    if require_canonical_dim_order:
        _validate_canonical_core_dim_order(data, allowed_core_dims)

    if require_spatial_voxdim:
        _validate_required_coordinate_attrs(data, spatial_dims, "voxdim")

    if require_spatial_units:
        spatial_unit_coords = get_voxel_affine_physical_coord_names(data)
        _validate_required_coordinate_attrs(data, spatial_unit_coords, "units")

    if require_time_units and TIME_DIM in data.dims:
        _validate_required_coordinate_attrs(data, (TIME_DIM,), "units")
