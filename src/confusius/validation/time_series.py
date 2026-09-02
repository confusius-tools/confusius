"""Time series validation utilities."""

import warnings
from typing import Literal, overload

import numpy as np
import pandas as pd
import xarray as xr

from confusius._dims import POSE_DIM, TIME_DIM
from confusius._utils.coordinates import get_coordinate_spacing_info
from confusius._utils.stack import find_stack_level
from confusius.validation.coordinates import validate_matching_coordinates


def validate_required_time_dimension(data: xr.DataArray) -> None:
    """Validate that a DataArray has a `time` dimension.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray whose dimensions should be checked.

    Raises
    ------
    ValueError
        If `data` has no `time` dimension.
    """
    if TIME_DIM not in data.dims:
        raise ValueError("DataArray must have a 'time' dimension.")


def validate_timepoint_count(data: xr.DataArray, operation_name: str) -> None:
    """Validate that a DataArray has more than one timepoint.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray with a `time` dimension.
    operation_name : str
        Name of the operation used in error messages.

    Raises
    ------
    ValueError
        If `data` has fewer than two timepoints.
    """
    if data.sizes[TIME_DIM] <= 1:
        raise ValueError(
            f"{operation_name.capitalize()} requires more than 1 timepoint, "
            f"got {data.sizes[TIME_DIM]}"
        )


def validate_unchunked_time(data: xr.DataArray, operation_name: str) -> None:
    """Validate that `time` occupies one Dask chunk.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray with a `time` dimension.
    operation_name : str
        Name of the operation used in error messages.

    Raises
    ------
    ValueError
        If `data` is chunked along `time` into multiple chunks.
    """
    time_axis = data.get_axis_num(TIME_DIM)
    if hasattr(data.data, "chunks"):
        time_chunks = data.data.chunks[time_axis]
        if len(time_chunks) > 1:
            raise ValueError(
                f"Data is chunked along the 'time' dimension ({len(time_chunks)} "
                f"chunks), but {operation_name} requires the full time series. "
                f"Rechunk your data so 'time' is not chunked: "
                f"data.chunk({{'time': -1}})"
            )


def validate_sorted_time(data: xr.DataArray, operation_name: str) -> None:
    """Validate that `time` coordinates are strictly increasing.

    For a pose-dependent `(time, pose)`-shaped `time` coordinate, checks each pose's
    own timestamps along the `time` axis, not across poses.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray with a `time` dimension.
    operation_name : str
        Name of the operation used in error messages.

    Raises
    ------
    ValueError
        If `time` coordinates are not strictly increasing.
    """
    time_coord = data.coords[TIME_DIM]
    time_axis = time_coord.dims.index(TIME_DIM)
    if np.any(np.diff(time_coord.values, axis=time_axis) <= 0):
        raise ValueError(
            f"time coordinates must be strictly increasing for {operation_name}."
        )


def validate_one_dimensional_time(data: xr.DataArray, operation_name: str) -> None:
    """Validate that `time` is a plain 1D dimension coordinate.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray with a `time` dimension.
    operation_name : str
        Name of the operation used in error messages.

    Raises
    ------
    ValueError
        If the `time` coordinate has more than one dimension (e.g. a pose-dependent
        `(time, pose)`-shaped `time` coordinate from unconsolidated multi-pose data).
    """
    if data.coords[TIME_DIM].ndim > 1:
        raise ValueError(
            f"{operation_name.capitalize()} requires a 1D 'time' coordinate, got a "
            f"{data.coords[TIME_DIM].dims}-shaped 'time' coordinate. Consolidate "
            "multi-pose data first with confusius.multipose.consolidate_poses."
        )


def validate_uniform_time(
    data: xr.DataArray,
    operation_name: str,
    uniformity_tolerance: float = 1e-2,
) -> float:
    """Validate uniformly sampled `time` coordinates and return spacing.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray with a `time` dimension.
    operation_name : str
        Name of the operation used in error messages.
    uniformity_tolerance : float, default: 1e-2
        Maximum allowed relative range of consecutive time intervals.

    Returns
    -------
    float
        Representative time spacing.

    Raises
    ------
    ValueError
        If `time` coordinates are not uniformly sampled.
    """
    time_spacing = get_coordinate_spacing_info(
        TIME_DIM, data, uniformity_tolerance
    ).value
    if time_spacing is None:
        raise ValueError(
            "Non-uniform 'time' coordinates detected. "
            f"{operation_name.capitalize()} requires uniformly sampled data. "
            "Consider interpolating your data to a regular time grid first."
        )
    return time_spacing


@overload
def validate_time_series(  # numpydoc ignore=GL08,PR01,RT01
    time_series: xr.DataArray,
    operation_name: str,
    require_unchunked_time: bool = True,
    require_sorted_time: bool = False,
    require_uniform_time: Literal[False] = False,
    require_1d_time: bool = False,
    uniformity_tolerance: float = 1e-2,
) -> tuple[int, None]: ...


@overload
def validate_time_series(  # numpydoc ignore=GL08,PR01,RT01
    time_series: xr.DataArray,
    operation_name: str,
    require_unchunked_time: bool = True,
    require_sorted_time: bool = False,
    require_uniform_time: Literal[True] = True,
    require_1d_time: bool = False,
    uniformity_tolerance: float = 1e-2,
) -> tuple[int, float]: ...


def validate_time_series(
    time_series: xr.DataArray,
    operation_name: str,
    require_unchunked_time: bool = True,
    require_sorted_time: bool = False,
    require_uniform_time: bool = False,
    require_1d_time: bool = False,
    uniformity_tolerance: float = 1e-2,
) -> tuple[int, float | None]:
    """Validate time series for time series processing operations.

    Performs common validation checks:

    1. Time series have a `time` dimension.
    2. Time dimension has more than 1 timepoint.
    3. Time dimension is not chunked for Dask arrays (optional).
    4. Time coordinate is 1D, not pose-dependent (optional).
    5. Time coordinate is strictly increasing (optional).
    6. Time coordinate is uniformly sampled (optional).

    Parameters
    ----------
    time_series : xarray.DataArray
        Input time series to validate. Must have a `time` dimension.
    operation_name : str
        Name of the operation (used in error/warning messages).
    require_unchunked_time : bool, default=True
        Whether to require the time dimension to occupy one Dask chunk. Set to `False`
        for operations that can process chunked time (e.g.,
        `confusius.signal.standardize`).
    require_sorted_time : bool, default: False
        Whether to require strictly increasing `time` coordinates.
    require_uniform_time : bool, default: False
        Whether to require uniformly sampled `time` coordinates and return their spacing.
    require_1d_time : bool, default: False
        Whether to require a 1D `time` coordinate, rejecting pose-dependent
        `(time, pose)`-shaped `time` coordinates from unconsolidated multi-pose data.
    uniformity_tolerance : float, default: 1e-2
        Maximum allowed relative range of consecutive time intervals, defined as
        `(max_interval - min_interval) / median_interval`. Raise a `ValueError` if the
        time coordinate exceeds this threshold.

    Returns
    -------
    time_axis : int
        Axis number for the `time` dimension.
    time_spacing : float or None
        Time spacing when `require_uniform_time=True`, otherwise `None`.

    Raises
    ------
    ValueError
        If `time_series` has no `time` dimension, if the `time` dimension has only 1
        timepoint, if the `time` dimension is chunked in a Dask array (when
        `require_unchunked_time=True`), if `require_1d_time=True` and the `time`
        coordinate is not 1D, if `require_sorted_time=True` and the `time` coordinate is
        not strictly increasing, or if `require_uniform_time=True` and the `time`
        coordinate is not uniformly sampled.
    """
    validate_required_time_dimension(time_series)
    validate_timepoint_count(time_series, operation_name)

    if require_1d_time:
        validate_one_dimensional_time(time_series, operation_name)

    time_axis = time_series.get_axis_num(TIME_DIM)

    if require_unchunked_time:
        validate_unchunked_time(time_series, operation_name)

    if require_sorted_time:
        validate_sorted_time(time_series, operation_name)

    if not require_uniform_time:
        return time_axis, None

    return time_axis, validate_uniform_time(
        time_series, operation_name, uniformity_tolerance
    )


def ensure_time_aligned(
    signals: xr.DataArray,
    value: xr.DataArray | np.ndarray | pd.DataFrame,
    name: str,
    *,
    ndim: Literal[1, 2],
    allow_dataframe: bool = True,
) -> xr.DataArray:
    """Return `value` as a `(time, ...)` DataArray aligned with `signals`.

    `value` is validated against `signals` and returned as a `(time,)` DataArray
    (`ndim=1`) or a `(time, name)` DataArray (`ndim=2`, a 1D `value` becoming one
    column) with `time` as its first dimension:

    - A DataArray must have a `time` dimension.
    - A DataFrame must have a `time` column and at least one other, numeric column; the
      other columns become a `name` dimension named after them.
    - A NumPy array is wrapped with dims `(time, name)`.

    `value` must have as many timepoints as `signals`. When both carry `time`
    coordinates these must match within the default coordinate-comparison tolerance
    (`rtol=1e-5`, `atol=1e-8`). When only `signals` does, `value` is assumed to be
    ordered like `signals` along `time` and takes its `time` coordinates, with a warning
    since alignment cannot be verified.

    `signals` with a pose-dependent `(time, pose)` `time` coordinate are represented
    by their consolidated time (see
    [`consolidate_poses`][confusius.multipose.consolidate_poses]).

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Signals defining the `time` grid.
    value : (time, ...) xarray.DataArray, numpy.ndarray, or pandas.DataFrame
        Array to align.
    name : str
        Name of `value` used in error and warning messages. If `value` is a DataFrame,
        its columns become the `name` dimension.
    ndim : {1, 2}
        Number of dimensions of the result: `1` for a single series such as a sample
        mask, `2` for a set of regressors such as confounds.
    allow_dataframe : bool, default: True
        Whether to accept a DataFrame `value`.

    Returns
    -------
    (time,) or (time, confound) xarray.DataArray
        `value` with `time` as its first dimension and `ndim` dimensions.

    Raises
    ------
    TypeError
        If `value` is not one of the accepted types.
    ValueError
        If `value` has no `time` dimension or column, has more than `ndim`
        dimensions, or does not have as many timepoints as `signals`; if a DataFrame
        `value` has duplicate or
        non-numeric columns or no column besides `time`; or if `time` coordinates do
        not match those of `signals`.

    Warns
    -----
    UserWarning
        If `value` has no `time` coordinates while `signals` does, since alignment
        cannot be verified.
    UserWarning
        If the per-pose timing metadata of pose-dependent `signals` are insufficient
        to infer their whole-volume time, in which case the first pose's timestamps
        are used.
    """
    allowed_ndim = "1D" if ndim == 1 else "1D or 2D"
    if allow_dataframe and isinstance(value, pd.DataFrame):
        if TIME_DIM not in value.columns:
            raise ValueError(f"{name} DataFrame must have a 'time' column")
        duplicated = value.columns[value.columns.duplicated()]
        if len(duplicated):
            raise ValueError(
                f"{name} DataFrame has duplicate columns: "
                f"{list(dict.fromkeys(duplicated))}"
            )
        columns = [column for column in value.columns if column != TIME_DIM]
        if not columns:
            raise ValueError(
                f"{name} DataFrame must have at least one column besides 'time'"
            )
        try:
            values = value[columns].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} DataFrame columns must be numeric") from exc
        value = xr.DataArray(
            values,
            dims=(TIME_DIM, name),
            coords={
                TIME_DIM: value[TIME_DIM].to_numpy(),
                name: [str(column) for column in columns],
            },
        )
    elif isinstance(value, np.ndarray):
        if value.ndim > ndim:
            raise ValueError(f"{name} must be {allowed_ndim}, got {value.ndim}D")
        value = xr.DataArray(value, dims=(TIME_DIM, name)[: value.ndim])
    elif not isinstance(value, xr.DataArray):
        accepted = (
            "xarray.DataArray, numpy.ndarray, or pandas.DataFrame"
            if allow_dataframe
            else "xarray.DataArray or numpy.ndarray"
        )
        raise TypeError(f"{name} must be an {accepted}, got {type(value).__name__}")

    if TIME_DIM not in value.dims:
        raise ValueError(f"{name} must have a 'time' dimension")
    if value.ndim > ndim:
        raise ValueError(f"{name} must be {allowed_ndim}, got {value.ndim}D")
    value = value.transpose(TIME_DIM, ...)
    if value.ndim < ndim:
        value = value.expand_dims(name, axis=1)

    signals_time = signals.coords.get(TIME_DIM)
    if signals_time is not None and POSE_DIM in signals_time.dims:
        # Imported here: confusius.timing (used by multipose.timing) imports this
        # module, so a module-level import would be circular.
        from confusius.multipose.timing import consolidate_time_coordinate

        # Per-volume values are timed by the whole volume, not by one pose.
        signals_time = consolidate_time_coordinate(signals_time)
        signals_time = signals_time.assign_coords({TIME_DIM: signals_time.variable})
    if signals_time is not None and TIME_DIM in value.coords:
        try:
            validate_matching_coordinates(signals_time, value, TIME_DIM)
        except ValueError as exc:
            raise ValueError(
                f"{name} time coordinates do not match signals time coordinates"
            ) from exc
        return value
    if value.sizes[TIME_DIM] != signals.sizes[TIME_DIM]:
        raise ValueError(
            f"{name} length ({value.sizes[TIME_DIM]}) must match number of timepoints "
            f"({signals.sizes[TIME_DIM]})"
        )
    if signals_time is None:
        return value

    warnings.warn(
        f"{name} has no 'time' coordinates, so its alignment with signals cannot be "
        "verified; assuming it is ordered like signals along 'time' and using the "
        "'time' coordinates of signals.",
        stacklevel=find_stack_level(),
    )
    return value.assign_coords({TIME_DIM: signals_time.variable})
