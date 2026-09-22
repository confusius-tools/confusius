"""Validation helpers for the ConfUSIus VoxelData model and fUSI recordings."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Literal, SupportsFloat, SupportsIndex

import numpy as np
import xarray as xr

from confusius._dims import CORE_DIMS, POSE_DIM, TIME_DIM, VOXEL_DIMS
from confusius._utils.coordinates import get_coordinate_spacing_info
from confusius._utils.geometry import (
    get_voxel_to_world_index_spacing,
    get_voxel_to_world_spatial_dims,
    has_voxel_to_world_index,
)
from confusius._utils.validation import require_dataarray
from confusius.timing import (
    TIMING_REFERENCE_FACTORS,
    convert_time_reference,
    convert_time_units,
    ensure_slice_time_acquisition_attrs,
    ensure_time_acquisition_attrs,
)
from confusius.validation.time_series import (
    validate_required_time_dimension,
    validate_timepoint_count,
    validate_unchunked_time,
    validate_uniform_time,
)

RegularSpacingDims = Literal["space", "core", "all"] | str | Sequence[str]
"""Selector for dimensions that must satisfy regular-spacing checks."""

_VELOCITY_ATTRS = ("transmit_frequency", "beamforming_sound_velocity")
"""Attributes required for velocity estimation from IQ VoxelData."""


def require_positive_finite(
    value: str | SupportsFloat | SupportsIndex, name: str
) -> float:
    """Return a finite positive numeric value.

    Parameters
    ----------
    value : str or typing.SupportsFloat or typing.SupportsIndex
        Candidate value.
    name : str
        Name used in the validation error.

    Returns
    -------
    float
        Validated value.

    Raises
    ------
    ValueError
        If `value` is not numeric, finite, and positive.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


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

    # Dim order is validated separately after geometry consistency.
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
        # instead of requiring one shared 1D dimension coordinate. This branch is
        # only entered for dim == TIME_DIM, which is always in CORE_DIMS, so
        # require_numeric is always True here (the single call site sets it to
        # `dim in CORE_DIMS`) -- unlike the generic 1D path below, which is also
        # reached for non-core extra dims where require_numeric can be False.
        if not np.issubdtype(coord.dtype, np.number):
            raise ValueError(f"Coordinate {dim!r} must be numeric.")
        values = np.asarray(coord.values)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Coordinate {dim!r} contains non-finite numeric values.")
        if values.shape[0] <= 1:
            return
        diffs = np.diff(values, axis=0)
        # require_ascending is likewise always True here (the single call site sets
        # it to `dim not in VOXEL_DIMS`, and TIME_DIM is never a voxel dim), so only
        # the strictly-increasing direction is ever required for a pose-dependent
        # time coordinate.
        if not np.all(diffs > 0):
            raise ValueError(
                f"Coordinate {dim!r} must be strictly monotonic-increasing for "
                "every pose."
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


def _validate_no_zero_length_dims(da: xr.DataArray) -> None:
    """Validate that no dimension has zero length.

    A zero-length dimension means `da` has no data at all, regardless of the size
    of its other dimensions.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose dimension sizes should be checked.

    Raises
    ------
    ValueError
        If any dimension has size 0.
    """
    empty_dims = [str(dim) for dim in da.dims if da.sizes[dim] == 0]
    if empty_dims:
        raise ValueError(
            f"DataArray must not have zero-length dimensions, got empty dimensions "
            f"{empty_dims!r} (shape {da.shape!r})."
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


def _validate_slice_time_coordinate(da: xr.DataArray) -> None:
    """Validate optional slice acquisition timestamps.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose `slice_time` coordinate should be validated.

    Raises
    ------
    ValueError
        If `slice_time` is present but does not describe numeric finite acquisition
        times with `time` as its leading dimension for time-series data, is missing
        required acquisition metadata, or its acquisition windows fall outside their
        volume's repetition interval.
    """
    if "slice_time" not in da.coords:
        return

    coord = da.coords["slice_time"]
    if not np.issubdtype(coord.dtype, np.number):
        raise ValueError("Coordinate 'slice_time' must be numeric.")
    if not np.all(np.isfinite(coord.values)):
        raise ValueError("Coordinate 'slice_time' contains non-finite numeric values.")
    if "units" not in coord.attrs:
        raise ValueError(
            "Coordinate 'slice_time' is missing required 'units' metadata."
        )
    if "volume_acquisition_reference" not in coord.attrs:
        raise ValueError(
            "Coordinate 'slice_time' is missing 'volume_acquisition_reference'."
        )
    if coord.attrs["volume_acquisition_reference"] not in TIMING_REFERENCE_FACTORS:
        raise ValueError(
            "Coordinate 'slice_time' 'volume_acquisition_reference' must be one of "
            f"{tuple(TIMING_REFERENCE_FACTORS)!r}, got "
            f"{coord.attrs['volume_acquisition_reference']!r}."
        )
    if "volume_acquisition_duration" not in coord.attrs:
        raise ValueError(
            "Coordinate 'slice_time' is missing 'volume_acquisition_duration'."
        )

    if TIME_DIM in da.dims:
        if len(coord.dims) != 2 or coord.dims[0] != TIME_DIM:
            raise ValueError(
                "Coordinate 'slice_time' must have dims ('time', <sweep_dim>) "
                f"when `time` is a dimension, got {coord.dims!r}."
            )
        sweep_dim = coord.dims[1]
        if sweep_dim not in da.dims:
            raise ValueError(
                "Coordinate 'slice_time' sweep dimension must be present in data, "
                f"got {sweep_dim!r}."
            )
        _validate_slice_time_within_volume_window(da, coord)
        return

    if len(coord.dims) != 1:
        raise ValueError(
            "Coordinate 'slice_time' must be 1D when `time` is scalar or absent, "
            f"got {coord.dims!r}."
        )
    if coord.dims[0] not in da.dims:
        raise ValueError(
            "Coordinate 'slice_time' dimension must be present in data, "
            f"got {coord.dims[0]!r}."
        )


_SLICE_TIME_TOLERANCE = 1e-6
"""Relative tolerance for `slice_time` consistency checks against floating drift."""


def _validate_slice_time_within_volume_window(
    da: xr.DataArray, coord: xr.DataArray
) -> None:
    """Check that `slice_time` stays within its own volume's acquisition window.

    `time.attrs["volume_acquisition_duration"]` is the time to acquire the whole
    `(k, j, i)` volume, so every one of its slices must fall within
    `[onset, onset + duration]` (`time` and `slice_time` both converted to `"start"`
    reference) -- never before the volume starts, never after it finishes.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray carrying both `slice_time` and a dimension `time` coordinate.
    coord : xarray.DataArray
        The `slice_time` coordinate, with dims `(time, <sweep_dim>)`.

    Raises
    ------
    ValueError
        If any slice's acquisition window (from `slice_time` and its own
        `volume_acquisition_duration`/`volume_acquisition_reference`) falls outside
        its volume's own `[onset, onset + volume_acquisition_duration]` window.
    """
    if TIME_DIM not in da.coords:
        return
    time_coord = da.coords[TIME_DIM]
    time_attrs = time_coord.attrs
    if (
        "volume_acquisition_duration" not in time_attrs
        or "volume_acquisition_reference" not in time_attrs
    ):
        return

    time_duration = convert_time_units(
        float(time_attrs["volume_acquisition_duration"]), time_attrs.get("units"), "s"
    )
    time_onset = convert_time_reference(
        convert_time_units(time_coord.values, time_attrs.get("units"), "s"),
        time_duration,
        from_reference=time_attrs["volume_acquisition_reference"],
        to_reference="start",
    )

    slice_duration = convert_time_units(
        float(coord.attrs["volume_acquisition_duration"]), coord.attrs.get("units"), "s"
    )
    slice_onset = convert_time_reference(
        convert_time_units(coord.values, coord.attrs.get("units"), "s"),
        slice_duration,
        from_reference=coord.attrs["volume_acquisition_reference"],
        to_reference="start",
    )

    tolerance = _SLICE_TIME_TOLERANCE * max(float(np.max(time_duration)), 1.0)
    window_start = time_onset[:, np.newaxis]
    window_end = window_start + time_duration
    slice_start = slice_onset
    slice_end = slice_onset + slice_duration
    if np.any(slice_start < window_start - tolerance) or np.any(
        slice_end > window_end + tolerance
    ):
        raise ValueError(
            "Coordinate 'slice_time' acquisition windows fall outside their own "
            "volume's [onset, onset + volume_acquisition_duration] window."
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


def canonicalize_voxeldata(data: xr.DataArray) -> xr.DataArray:
    """Restore a scalar-indexed VoxelData dimension and its geometry.

    Scalar indexing such as `data.isel(j=0)` removes `j` from the array dimensions
    but retains it as a scalar coordinate. This function restores every missing
    native voxel dimension (`k`, `j`, `i`) as a length-one dimension, then orders all
    dimensions as `(...extra_dims, time, pose, k, j, i)`. It preserves coordinate
    values and their order. When scalar indexing fixed a dimension in a
    `VoxelToWorldIndex`,
    its geometry is rebuilt from the untouched affine, including its `units`
    (`VoxelToWorldIndex` always carries one, defaulting to `"mm"` since
    [attach_voxel_to_world_index][confusius._utils.geometry.attach_voxel_to_world_index]
    itself defaults it). Missing time acquisition metadata also defaults where
    possible. This function does not otherwise validate the VoxelData model; use
    [ensure_voxeldata][confusius.validation.ensure_voxeldata] to canonicalize and
    validate.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to canonicalize.

    Returns
    -------
    xarray.DataArray
        Canonicalized DataArray with all native voxel dimensions present.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    ValueError
        If a native voxel dimension is absent and has no scalar coordinate from which
        to restore it, or its same-named coordinate is not scalar.

    Warns
    -----
    UserWarning
        If missing `time` or `slice_time` acquisition metadata is defaulted.
    """
    require_dataarray(data)

    result = data
    for dim in VOXEL_DIMS:
        if dim in result.dims:
            continue
        if dim not in result.coords:
            raise ValueError(
                f"DataArray is missing voxel dimension {dim!r}, and has no scalar "
                f"coordinate {dim!r} to restore it from."
            )
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
    result = result.transpose(
        *(dim for dim in result.dims if dim not in CORE_DIMS),
        *(dim for dim in CORE_DIMS if dim in result.dims),
    )
    result = ensure_time_acquisition_attrs(result)
    return ensure_slice_time_acquisition_attrs(result)


def ensure_voxeldata(data: xr.DataArray, **validate_kwargs: Any) -> xr.DataArray:
    """Return a canonical, validated VoxelData array.

    This is the normal entry point for spatial inputs: it restores scalar-indexed
    native voxel dimensions and geometry, and orders dimensions as
    `(...extra_dims, time, pose, k, j, i)` with
    [canonicalize_voxeldata][confusius.validation.canonicalize_voxeldata], then checks
    the resulting DataArray against the VoxelData model. Use
    [validate_voxeldata][confusius.validation.validate_voxeldata] when the input must
    already follow that model.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to canonicalize and validate.
    **validate_kwargs : Any
        Keyword arguments forwarded to
        [validate_voxeldata][confusius.validation.validate_voxeldata].

    Returns
    -------
    xarray.DataArray
        Canonicalized VoxelData array that satisfies the requested validation checks.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    ValueError
        If canonicalization or validation fails.
    """
    result = canonicalize_voxeldata(data)
    validate_voxeldata(result, **validate_kwargs)
    return result


def validate_voxeldata(
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
    require_velocity_attrs: bool = False,
    require_dtype: Any | None = None,
) -> None:
    """Validate a DataArray against the VoxelData model without modifying it.

    This requires non-empty native voxel dimensions (`k`, `j`, `i`), their
    coordinates, and a matching `VoxelToWorldIndex` (which carries the world-space
    `units` shared by `z`/`y`/`x`, exposed via `.fusi.affine.units`), and dimensions
    ordered as `(...extra_dims, time, pose, k, j, i)`. It also validates every
    dimension coordinate. When `time` is present, its acquisition metadata and
    `units` are required. The optional flags add requirements
    needed by a particular consumer. Use
    [ensure_voxeldata][confusius.validation.ensure_voxeldata] when scalar indexing
    may have removed a voxel dimension and the input should be canonicalized first.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to validate.
    require_time : bool, default: False
        Whether to require `time` with more than one coordinate value.
    require_unchunked_time : bool, default: False
        Whether to require `time` with more than one coordinate value in a single Dask
        chunk.
    require_uniform_time : bool, default: False
        Whether to require `time` with more than one uniformly spaced coordinate
        value.
    uniformity_tolerance : float, default: 1e-2
        Maximum relative variation allowed between consecutive time intervals.
    allow_pose : bool, default: True
        Whether to allow a `pose` dimension.
    allow_extra_dims : bool, default: True
        Whether to allow dimensions outside `time`, `pose`, `k`, `j`, and `i`.
    require_regular_spacing : bool, default: False
        Whether to require regular spacing for selected numeric dimension coordinates.
    regular_spacing_tolerance : float, default: 1e-2
        Relative tolerance used to assess coordinate regularity.
    regular_spacing_dims : {"space", "core", "all"} or str or sequence[str], default: "space"
        Dimensions to check for regular spacing. `"space"` checks `k`, `j`, and `i`;
        `"core"` checks present core dimensions; `"all"` checks every dimension.
    require_velocity_attrs : bool, default: False
        Whether to require positive, finite `transmit_frequency` and
        `beamforming_sound_velocity` DataArray attributes.
    require_dtype : Any, optional
        Required data dtype or dtype class, passed to `numpy.issubdtype`.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray` or its dtype does not satisfy
        `require_dtype`.
    ValueError
        If VoxelData geometry, dimensions, coordinates, timing, spacing, or metadata
        validation fails.
    """
    require_dataarray(data)
    _validate_no_zero_length_dims(data)

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
        validate_timepoint_count(data, "VoxelData validation")
    if require_unchunked_time:
        validate_unchunked_time(data, "VoxelData validation")
    if require_uniform_time:
        validate_uniform_time(data, "VoxelData validation", uniformity_tolerance)

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

    _validate_slice_time_coordinate(data)

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

    _validate_canonical_core_dim_order(data)

    if require_dtype is not None and not np.issubdtype(data.dtype, require_dtype):
        raise TypeError(
            f"Expected data dtype compatible with {require_dtype}, got {data.dtype}."
        )

    if TIME_DIM in data.dims:
        _validate_required_coordinate_attrs(data, (TIME_DIM,), "units")

    if require_velocity_attrs:
        missing_attrs = sorted(set(_VELOCITY_ATTRS) - set(data.attrs))
        if missing_attrs:
            raise ValueError(
                f"Missing required DataArray attributes: {missing_attrs}. "
                f"Velocity estimation requires attributes: {_VELOCITY_ATTRS}."
            )
        for attr in _VELOCITY_ATTRS:
            require_positive_finite(data.attrs[attr], attr)
