"""Time series validation utilities."""

from typing import Literal, overload

import numpy as np
import xarray as xr

from confusius._dims import TIME_DIM
from confusius._utils.coordinates import get_coordinate_spacing_info


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
    if np.any(np.diff(data.coords[TIME_DIM].values) <= 0):
        raise ValueError(
            f"time coordinates must be strictly increasing for {operation_name}."
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
    uniformity_tolerance: float = 1e-2,
) -> tuple[int, None]: ...


@overload
def validate_time_series(  # numpydoc ignore=GL08,PR01,RT01
    time_series: xr.DataArray,
    operation_name: str,
    require_unchunked_time: bool = True,
    require_sorted_time: bool = False,
    require_uniform_time: Literal[True] = True,
    uniformity_tolerance: float = 1e-2,
) -> tuple[int, float]: ...


def validate_time_series(
    time_series: xr.DataArray,
    operation_name: str,
    require_unchunked_time: bool = True,
    require_sorted_time: bool = False,
    require_uniform_time: bool = False,
    uniformity_tolerance: float = 1e-2,
) -> tuple[int, float | None]:
    """Validate time series for time series processing operations.

    Performs common validation checks:

    1. Time series have a `time` dimension.
    2. Time dimension has more than 1 timepoint.
    3. Time dimension is not chunked for Dask arrays (optional).
    4. Time coordinate is strictly increasing (optional).
    5. Time coordinate is uniformly sampled (optional).

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
        `require_unchunked_time=True`), if `require_sorted_time=True` and the `time`
        coordinate is not strictly increasing, or if `require_uniform_time=True` and the
        `time` coordinate is not uniformly sampled.
    """
    validate_required_time_dimension(time_series)
    validate_timepoint_count(time_series, operation_name)

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
