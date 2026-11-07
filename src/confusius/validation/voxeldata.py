"""ConfUSIus acquisition validation layered on the standalone VoxelData model."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from voxeldata import canonicalize_voxeldata as canonicalize_generic_voxeldata
from voxeldata import require_positive_finite
from voxeldata import validate_voxeldata as validate_generic_voxeldata
from voxeldata.validation import RegularSpacingDims

from confusius._dims import CORE_DIMS, TIME_DIM
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

_VELOCITY_ATTRS = ("transmit_frequency", "beamforming_sound_velocity")
"""Attributes required for velocity estimation from IQ VoxelData."""

_SLICE_TIME_TOLERANCE = 1e-6
"""Relative tolerance for `slice_time` consistency checks against floating drift."""


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


def canonicalize_voxeldata(data: xr.DataArray) -> xr.DataArray:
    """Restore a scalar-indexed VoxelData dimension and its geometry.

    Scalar indexing such as `data.isel(j=0)` removes `j` from the array dimensions
    but retains it as a scalar coordinate. This function restores every missing
    native voxel dimension (`k`, `j`, `i`) as a length-one dimension, then orders all
    dimensions as `(...extra_dims, grid, time, k, j, i)`. It preserves coordinate
    values and their order. When scalar indexing fixed a dimension in a
    `VoxelToWorldIndex`, its geometry is rebuilt from the unchanged voxel-to-world
    affine, preserving the world-space units. Missing time acquisition metadata
    also defaults where possible. This function does not otherwise validate the
    VoxelData model; use
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
    result = canonicalize_generic_voxeldata(data)
    result = result.transpose(
        *(d for d in result.dims if d not in CORE_DIMS),
        *(d for d in CORE_DIMS if d in result.dims),
    )
    result = ensure_time_acquisition_attrs(result)
    return ensure_slice_time_acquisition_attrs(result)


def ensure_voxeldata(data: xr.DataArray, **validate_kwargs: Any) -> xr.DataArray:
    """Return a canonical, validated VoxelData array.

    This is the normal entry point for spatial inputs: it restores scalar-indexed
    native voxel dimensions and geometry, and orders dimensions as
    `(...extra_dims, grid, time, k, j, i)` with
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
    allow_grid: bool = True,
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
    `units` shared by `z`/`y`/`x`, exposed via `.vd.units`), and dimensions
    ordered as `(*nonspatial_dims, k, j, i)`, with any leading order. It also validates every
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
    allow_grid : bool, default: True
        Whether to allow a `grid` dimension.
    allow_extra_dims : bool, default: True
        Whether to allow dimensions outside `time`, `grid`, `k`, `j`, and `i`.
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
    validate_generic_voxeldata(
        data,
        allow_extra_dims=True,
        require_regular_spacing=require_regular_spacing,
        regular_spacing_tolerance=regular_spacing_tolerance,
        regular_spacing_dims=regular_spacing_dims,
        require_dtype=require_dtype,
    )
    if not allow_extra_dims:
        extra_dims = sorted(str(d) for d in data.dims if d not in CORE_DIMS)
        if extra_dims:
            raise ValueError(f"Unexpected dimensions {extra_dims!r}.")
    if not allow_grid and "grid" in data.dims:
        raise ValueError("DataArray must not have a 'grid' dimension.")
    if require_time or require_unchunked_time or require_uniform_time:
        validate_required_time_dimension(data)
        validate_timepoint_count(data, "VoxelData validation")
    if require_unchunked_time:
        validate_unchunked_time(data, "VoxelData validation")
    if require_uniform_time:
        validate_uniform_time(data, "VoxelData validation", uniformity_tolerance)
    if TIME_DIM in data.dims:
        time_attrs = data.coords[TIME_DIM].attrs
        if "units" not in time_attrs:
            raise ValueError("Coordinate 'time' is missing required 'units' metadata.")
        for key in ("volume_acquisition_reference", "volume_acquisition_duration"):
            if key not in time_attrs:
                raise ValueError(f"'time' coordinate is missing '{key}'.")
        if time_attrs["volume_acquisition_reference"] not in TIMING_REFERENCE_FACTORS:
            raise ValueError(
                "'time' coordinate 'volume_acquisition_reference' must be one of "
                f"{tuple(TIMING_REFERENCE_FACTORS)!r}, got "
                f"{time_attrs['volume_acquisition_reference']!r}."
            )
        require_positive_finite(
            time_attrs["volume_acquisition_duration"], "volume_acquisition_duration"
        )
    _validate_slice_time_coordinate(data)
    if require_velocity_attrs:
        missing_attrs = sorted(set(_VELOCITY_ATTRS) - set(data.attrs))
        if missing_attrs:
            raise ValueError(
                f"Missing required DataArray attributes: {missing_attrs}. "
                f"Velocity estimation requires attributes: {_VELOCITY_ATTRS}."
            )
        for attr in _VELOCITY_ATTRS:
            require_positive_finite(data.attrs[attr], attr)
