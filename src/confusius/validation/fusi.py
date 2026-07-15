"""Validation helpers for ConfUSIus-style fUSI DataArrays."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr

from confusius._dims import CORE_DIMS, POSE_DIM, SPATIAL_DIMS, TIME_DIM
from confusius._utils.coordinates import get_coordinate_spacing_info

RegularSpacingDims = Literal["space", "core", "all"] | str | Sequence[str]
"""Selector for dimensions that must satisfy regular-spacing checks."""

_ALLOWED_CORE_DIMS = CORE_DIMS
"""Core dimension names recognized by ConfUSIus fUSI validators."""


def _validate_dimension_coordinate(
    da: xr.DataArray, dim: Hashable, *, require_numeric: bool
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

    Raises
    ------
    ValueError
        If the coordinate is missing, is not one-dimensional along `dim`, or (when
        `require_numeric=True`) is non-numeric, non-finite, or not strictly
        increasing.
    """
    if dim not in da.coords:
        if dim in _ALLOWED_CORE_DIMS:
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
    expected_order = tuple(dim for dim in _ALLOWED_CORE_DIMS if dim in da.dims)
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
        checks present core dimensions (`time`, `pose`, `z`, `y`, `x`), and `"all"`
        checks all present dimensions. Any other string is treated as a single
        dimension name. A sequence checks only listed dimensions. Non-numeric
        dimensions are ignored.

    Raises
    ------
    ValueError
        If `regular_spacing_dims` is invalid, references missing dimensions, or if a
        selected numeric coordinate has non-uniform or undefined spacing.
    """
    if regular_spacing_dims == "space":
        dims_to_check = [dim for dim in SPATIAL_DIMS if dim in da.dims]
    elif regular_spacing_dims == "core":
        dims_to_check = [dim for dim in _ALLOWED_CORE_DIMS if dim in da.dims]
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


def canonicalize_fusi_dataarray(data: xr.DataArray) -> xr.DataArray:
    """Return `data` with scalar spatial coordinates restored as dimensions.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to canonicalize. It may be a canonical fUSI DataArray already, or an
        xarray scalar-indexed slice such as `da.isel(z=0)` where `z` remains as a
        scalar coordinate but no longer appears in `dims`.

    Returns
    -------
    xarray.DataArray
        DataArray with any missing spatial dimensions restored from scalar coordinates.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    ValueError
        If a missing spatial dimension has no scalar coordinate to restore from.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"data must be an xarray.DataArray, got {type(data).__name__}.")

    result = data
    for dim in SPATIAL_DIMS:
        if dim in result.dims:
            continue
        if dim not in result.coords or result.coords[dim].shape != ():
            raise ValueError(
                f"DataArray must contain all spatial dimensions {SPATIAL_DIMS!r}, but "
                f"is missing spatial dimension {dim!r}. If this came from scalar "
                f"indexing, keep the scalar {dim!r} coordinate so ConfUSIus can "
                "restore it as a singleton dimension."
            )
        coord = result.coords[dim]
        attrs = coord.attrs.copy()
        spatial_index = SPATIAL_DIMS.index(dim)
        next_dims = [d for d in SPATIAL_DIMS[spatial_index + 1 :] if d in result.dims]
        previous_dims = [d for d in SPATIAL_DIMS[:spatial_index] if d in result.dims]
        if next_dims:
            axis = result.dims.index(next_dims[0])
        elif previous_dims:
            axis = result.dims.index(previous_dims[-1]) + 1
        else:
            axis = len(result.dims)
        result = result.expand_dims({dim: [coord.item()]}, axis=axis)
        result.coords[dim].attrs.update(attrs)

    return result


def ensure_fusi_dataarray(data: xr.DataArray, **validate_kwargs: Any) -> xr.DataArray:
    """Canonicalize and validate a ConfUSIus fUSI DataArray.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to canonicalize and validate.
    **validate_kwargs : Any
        Keyword arguments forwarded to
        [validate_fusi_dataarray][confusius.validation.validate_fusi_dataarray].

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
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"data must be an xarray.DataArray, got {type(data).__name__}.")

    _validate_core_dimension_names(
        data,
        allow_extra_dims=validate_kwargs.get("allow_extra_dims", True),
    )
    if validate_kwargs.get("allow_pose") is False and POSE_DIM in data.dims:
        raise ValueError("DataArray must not have a 'pose' dimension.")
    if validate_kwargs.get("require_time") is True and TIME_DIM not in data.dims:
        raise ValueError("DataArray must have a 'time' dimension.")

    result = canonicalize_fusi_dataarray(data)
    validate_fusi_dataarray(result, **validate_kwargs)
    return result


def validate_fusi_dataarray(
    data: xr.DataArray,
    *,
    require_time: bool = False,
    allow_pose: bool = True,
    allow_extra_dims: bool = True,
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

    - Contain all three spatial dimensions `z`, `y`, and `x`. Single-slice
      acquisitions are represented as 3D data with a singleton `z` axis (for example
      `(1, y, x)` or `(time, 1, y, x)`), never as bare `(y, x)` or `(time, y, x)`.
    - Have dimension names from the set `(time, pose, z, y, x)`, with optional extra
      dimensions if `allow_extra_dims` is `True` (e.g., `region`, `component`, `mask`,
      etc.).
    - Have matching 1D coordinates for all core dimensions (`time`, `pose`, `z`, `y`,
      `x`). Extra-dimension coordinates are optional.
    - Have numeric, finite, and strictly increasing core dimension coordinates (`time`,
      `pose`, `z`, `y`, `x`).

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
        Whether dimensions outside the ConfUSIus core set (`time`, `pose`, `z`, `y`,
        `x`) are allowed.
    require_regular_spacing : bool, default: False
        Whether numeric dimension coordinates must have regular spacing.
    regular_spacing_tolerance : float, default: 1e-2
        Relative tolerance used to assess coordinate regularity.
    regular_spacing_dims : {"space", "core", "all"} or str or sequence[str], default: "space"
        Dimensions that must satisfy regular-spacing checks when
        `require_regular_spacing=True`. Use `"space"` for present `z`, `y`, `x`
        dimensions, `"core"` for present core dimensions (`time`, `pose`, `z`, `y`,
        `x`), `"all"` for all present dimensions, a string for one explicit dimension
        name, or a sequence for multiple explicit dimension names. Non-numeric
        coordinates are ignored.
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
        there are too few spatial dimensions, core numeric coordinate constraints fail,
        optional stricter checks fail, or required metadata is missing.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"data must be an xarray.DataArray, got {type(data).__name__}.")

    _validate_core_dimension_names(data, allow_extra_dims=allow_extra_dims)

    if require_time and TIME_DIM not in data.dims:
        raise ValueError("DataArray must have a 'time' dimension.")

    if not allow_pose and POSE_DIM in data.dims:
        raise ValueError("DataArray must not have a 'pose' dimension.")

    missing_spatial_dims = [dim for dim in SPATIAL_DIMS if dim not in data.dims]
    if missing_spatial_dims:
        raise ValueError(
            f"DataArray must contain all spatial dimensions {SPATIAL_DIMS!r}, but is "
            f"missing {tuple(missing_spatial_dims)!r}. Single-slice acquisitions must "
            "be stored as 3D data with a singleton 'z' axis (for example (1, y, x) or "
            "(time, 1, y, x)), not as bare (y, x) or (time, y, x)."
        )

    for dim in data.dims:
        _validate_dimension_coordinate(
            data, dim, require_numeric=dim in _ALLOWED_CORE_DIMS
        )

    if require_regular_spacing:
        _validate_regular_spacing(data, regular_spacing_tolerance, regular_spacing_dims)

    if require_canonical_dim_order:
        _validate_canonical_core_dim_order(data)

    if require_spatial_voxdim:
        _validate_required_coordinate_attrs(data, SPATIAL_DIMS, "voxdim")

    if require_spatial_units:
        _validate_required_coordinate_attrs(data, SPATIAL_DIMS, "units")

    if require_time_units and TIME_DIM in data.dims:
        _validate_required_coordinate_attrs(data, (TIME_DIM,), "units")
