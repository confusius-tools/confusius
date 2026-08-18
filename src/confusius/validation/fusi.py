"""Validation helpers for the ConfUSIus VoxelData model and fUSI recordings."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr

from confusius._dims import CORE_DIMS, POSE_DIM, TIME_DIM, VOXEL_DIMS
from confusius._utils.coordinates import get_coordinate_spacing_info
from confusius._utils.geometry import (
    get_voxel_to_world_coord_names,
    get_voxel_to_world_index_spacing,
    get_voxel_to_world_spatial_dims,
    has_voxel_to_world_index,
    update_voxel_to_world_coord_attrs,
)
from confusius._utils.stack import find_stack_level
from confusius._utils.validation import require_dataarray
from confusius.timing import TIMING_REFERENCE_FACTORS, ensure_time_acquisition_attrs
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


def _validate_voxel_to_world_geometry(da: xr.DataArray) -> None:
    """Validate that a DataArray has voxel-to-world index consistent with its dims.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to validate.

    Raises
    ------
    ValueError
        If voxel-to-world index is missing or inconsistent.
    """
    if not has_voxel_to_world_index(da):
        raise ValueError(
            "DataArray must use native voxel dimensions `k/j/i` with "
            "VoxelToWorldIndex-backed world coordinates and defined spatial spacing."
        )

    # Set membership only, not order: dim order is a separate concern covered by the
    # opt-in `require_canonical_dim_order` check below.
    voxel_dims = get_voxel_to_world_spatial_dims(da)
    if set(voxel_dims) != set(VOXEL_DIMS):
        raise ValueError(
            f"Voxel-to-world index must cover native voxel dimensions {VOXEL_DIMS!r}, "
            f"got {voxel_dims!r}."
        )

    # World coordinate existence/dims are not checked here: they are guaranteed by
    # construction (VoxelToWorldIndex registers them via xr.Coordinates.from_xindex)
    # and by xarray's own index-corruption protection (dropping an index-linked
    # coordinate independently of its siblings raises before this function is ever
    # reached). No public construction path can desync them from `voxel_dims`.


def _validate_dimension_coordinate(
    da: xr.DataArray, dim: Hashable, *, require_numeric: bool, require_ascending: bool
) -> None:
    """Validate a single dimension coordinate.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose coordinate should be validated.
    dim : hashable
        Dimension name whose matching coordinate is required.
    require_numeric : bool
        Whether the coordinate must be numeric, finite, and strictly monotonic.
    require_ascending : bool
        Whether strict monotonicity must be increasing specifically. Ignored if
        `require_numeric` is `False`. Voxel dims (`k`/`j`/`i`) may run in either
        direction — the voxel-to-world affine, not coordinate direction, encodes
        orientation, so a flipped axis (e.g. `da.isel(i=slice(None, None, -1))`) is
        still valid VoxelData. `time`/`pose` must be increasing.

    Raises
    ------
    ValueError
        If the coordinate is missing, malformed, non-numeric, non-finite, or not
        strictly monotonic (in the required direction, if any).
    """
    if dim not in da.coords:
        if dim in CORE_DIMS:
            raise ValueError(f"Missing required coordinate for dimension {dim!r}.")
        return

    coord = da.coords[dim]
    if dim == TIME_DIM and POSE_DIM in da.dims and coord.dims == (TIME_DIM, POSE_DIM):
        # A pose-dependent array's "time" coordinate may be genuinely
        # (time, pose)-shaped, holding each pose's own real timestamp directly
        # (poses acquired sequentially rather than simultaneously) -- see
        # confusius.multipose.stack_poses. There is no single answer for "the" time
        # of a (pose, k, j, i) voxel any more than there is a single answer for its
        # z/y/x position, so this validates every pose's own column independently
        # instead of requiring one shared 1D dimension coordinate.
        if not require_numeric:
            return
        if not np.issubdtype(coord.dtype, np.number):
            raise ValueError(f"Coordinate {dim!r} must be numeric.")
        values = np.asarray(coord.values)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Coordinate {dim!r} contains non-finite numeric values.")
        if values.shape[0] <= 1:
            return
        diffs = np.diff(values, axis=0)
        if require_ascending:
            if not np.all(diffs > 0):
                raise ValueError(
                    f"Coordinate {dim!r} must be strictly monotonic-increasing for "
                    "every pose."
                )
        elif not (np.all(diffs > 0) or np.all(diffs < 0)):
            raise ValueError(
                f"Coordinate {dim!r} must be strictly monotonic for every pose."
            )
        return

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
    if values.size <= 1:
        return
    diffs = np.diff(values)
    if require_ascending:
        if not np.all(diffs > 0):
            raise ValueError(
                f"Coordinate {dim!r} must be strictly monotonic-increasing."
            )
    elif not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(f"Coordinate {dim!r} must be strictly monotonic.")


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

    voxel_spacing = get_voxel_to_world_index_spacing(da)
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

    from confusius._utils.geometry import restore_voxel_to_world_index

    result = restore_voxel_to_world_index(result)
    result = _ensure_spatial_metadata_attrs(result)
    return ensure_time_acquisition_attrs(result)


def _ensure_spatial_metadata_attrs(data: xr.DataArray) -> xr.DataArray:
    """Fill in default `units` metadata on the world spatial coordinates.

    Every VoxelData-compatible DataArray carries `units` on its world (`z`/`y`/`x`)
    coordinates. Both `voxdim` and `units` are already guaranteed there by
    [attach_voxel_to_world_index][confusius._utils.geometry.attach_voxel_to_world_index]
    itself for freshly-attached data, so this function is mainly a safety net for
    data whose index predates that default (e.g. hand-built `VoxelToWorldIndex`
    objects, or data deserialized from a format that stored one). It fills in `"mm"`
    (the project-wide world-coordinate unit) for anything still missing it.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to fill in.

    Returns
    -------
    xarray.DataArray
        `data` unchanged when it does not carry a voxel-to-world index or its world
        coordinates already carry `units`. Otherwise a copy with `units` filled in.

    Warns
    -----
    UserWarning
        If any world coordinate was missing `units` and defaulted to `"mm"`.
    """
    if not has_voxel_to_world_index(data):
        return data

    world_coords = get_voxel_to_world_coord_names(data)
    missing = [name for name in world_coords if "units" not in data.coords[name].attrs]
    if not missing:
        return data

    result = update_voxel_to_world_coord_attrs(
        data, {name: {"units": "mm"} for name in missing}
    )
    warnings.warn(
        f"World coordinate(s) {missing} missing 'units'; defaulting to 'mm'.",
        stacklevel=find_stack_level(),
    )
    return result


def ensure_fusi(data: xr.DataArray, **validate_kwargs: Any) -> xr.DataArray:
    """Canonicalize and validate a DataArray against the ConfUSIus VoxelData model.

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
        Canonicalized DataArray that passed VoxelData validation.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    ValueError
        If canonicalization or validation fails.
    """
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
    require_regular_spacing: bool = False,
    regular_spacing_tolerance: float = 1e-2,
    regular_spacing_dims: RegularSpacingDims = "space",
    require_canonical_dim_order: bool = False,
) -> None:
    """Validate that a DataArray follows the ConfUSIus VoxelData model.

    This is the general-purpose VoxelData checker: by default it enforces only the
    universal `k`/`j`/`i` + `VoxelToWorldIndex` structure required of any
    VoxelData-compatible DataArray, so it is the right tool for any VoxelData array,
    not only fUSI recordings. The optional flags below layer on genuine
    fUSI-recording-specific requirements (e.g. acquisition timing) and should only be
    enabled for actual fUSI recordings.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to validate. It must use native voxel dimensions `k`, `j`, `i` and
        world coordinates `z`, `y`, `x`.
    require_time : bool, default: False
        Whether to require a `time` dimension with more than one timepoint, as in an
        actual fUSI recording.
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
    require_regular_spacing : bool, default: False
        Whether selected numeric dimension coordinates must have regular spacing.
    regular_spacing_tolerance : float, default: 1e-2
        Relative tolerance used to assess coordinate regularity.
    regular_spacing_dims : {"space", "core", "all"} or str or sequence[str], default: "space"
        Dimensions that must satisfy regular-spacing checks.
    require_canonical_dim_order : bool, default: False
        Whether core dimensions must appear in canonical order.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    ValueError
        If dimension, coordinate, timing, or metadata validation fails.
    """
    require_dataarray(data)

    _validate_core_dimension_names(data, allow_extra_dims=allow_extra_dims)

    spatial_dims = _get_spatial_dims(data)
    if spatial_dims != VOXEL_DIMS:
        raise ValueError(
            f"DataArray must include all native voxel dimensions {VOXEL_DIMS!r}, "
            f"got {spatial_dims!r}."
        )

    _validate_voxel_to_world_geometry(data)

    if require_time or require_unchunked_time or require_uniform_time:
        validate_required_time_dimension(data)
        validate_timepoint_count(data, "fUSI validation")
    if require_unchunked_time:
        validate_unchunked_time(data, "fUSI validation")
    if require_uniform_time:
        validate_uniform_time(data, "fUSI validation", uniformity_tolerance)

    if TIME_DIM in data.dims and TIME_DIM in data.coords:
        time_attrs = data.coords[TIME_DIM].attrs
        if "volume_acquisition_reference" not in time_attrs:
            raise ValueError(
                "'time' coordinate is missing 'volume_acquisition_reference'."
            )
        if time_attrs["volume_acquisition_reference"] not in TIMING_REFERENCE_FACTORS:
            raise ValueError(
                "'time' coordinate 'volume_acquisition_reference' must be one of "
                f"{tuple(TIMING_REFERENCE_FACTORS)!r}, got "
                f"{time_attrs['volume_acquisition_reference']!r}."
            )
        if "volume_acquisition_duration" not in time_attrs:
            raise ValueError(
                "'time' coordinate is missing 'volume_acquisition_duration'."
            )

    if not allow_pose and POSE_DIM in data.dims:
        raise ValueError("DataArray must not have a 'pose' dimension.")

    for dim in data.dims:
        _validate_dimension_coordinate(
            data,
            dim,
            require_numeric=dim in CORE_DIMS,
            require_ascending=dim not in VOXEL_DIMS,
        )

    if require_regular_spacing:
        _validate_regular_spacing(
            data,
            regular_spacing_tolerance,
            regular_spacing_dims,
            spatial_dims,
        )

    if require_canonical_dim_order:
        _validate_canonical_core_dim_order(data)

    world_coords = get_voxel_to_world_coord_names(data)
    _validate_required_coordinate_attrs(data, world_coords, "voxdim")
    _validate_required_coordinate_attrs(data, world_coords, "units")
    if TIME_DIM in data.dims:
        _validate_required_coordinate_attrs(data, (TIME_DIM,), "units")
