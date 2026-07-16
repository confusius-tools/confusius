"""Time series validation utilities."""

import warnings
from typing import Literal, overload

import xarray as xr

from confusius._utils.coordinates import get_coordinate_spacings


@overload
def validate_time_series(  # numpydoc ignore=GL08,PR01,RT01
    time_series: xr.DataArray,
    operation_name: str,
    check_time_chunks: bool = True,
    require_uniform_time: Literal[False] = False,
    uniformity_tolerance: float = 1e-2,
) -> tuple[int, None]: ...


@overload
def validate_time_series(  # numpydoc ignore=GL08,PR01,RT01
    time_series: xr.DataArray,
    operation_name: str,
    check_time_chunks: bool = True,
    require_uniform_time: Literal[True] = True,
    uniformity_tolerance: float = 1e-2,
) -> tuple[int, float]: ...


def validate_time_series(
    time_series: xr.DataArray,
    operation_name: str,
    check_time_chunks: bool = True,
    require_uniform_time: bool = False,
    uniformity_tolerance: float = 1e-2,
) -> tuple[int, float | None]:
    """Validate time series for time series processing operations.

    Performs common validation checks:

    1. Time series have a `time` dimension.
    2. Time dimension has more than 1 timepoint.
    3. Time dimension is not chunked for Dask arrays (optional).
    4. Time coordinate is uniformly sampled (optional).

    Parameters
    ----------
    time_series : xarray.DataArray
        Input time series to validate. Must have a `time` dimension.
    operation_name : str
        Name of the operation (used in error/warning messages).
    check_time_chunks : bool, default=True
        Whether to raise an error when time dimension is chunked in a Dask array. Set to
        `False` for operations that can handle chunked time (e.g.,
        `confusius.signal.standardize`).
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
        `check_time_chunks=True`), or if `require_uniform_time=True` and the `time`
        coordinate is not uniformly sampled.
    """
    if "time" not in time_series.dims:
        raise ValueError("time_series must have a 'time' dimension")

    if time_series.sizes["time"] <= 1:
        raise ValueError(
            f"{operation_name.capitalize()} requires more than 1 timepoint, "
            f"got {time_series.sizes['time']}"
        )

    time_axis = time_series.get_axis_num("time")

    if check_time_chunks and hasattr(time_series.data, "chunks"):
        time_chunks = time_series.data.chunks[time_axis]
        if len(time_chunks) > 1:
            raise ValueError(
                f"Data is chunked along the 'time' dimension ({len(time_chunks)} "
                f"chunks), but {operation_name} requires the full time series. "
                f"Rechunk your data so 'time' is not chunked: "
                f"data.chunk({{'time': -1}})"
            )

    if not require_uniform_time:
        return time_axis, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        spacing = get_coordinate_spacings(
            time_series, uniformity_tolerance=uniformity_tolerance
        )

    time_spacing = spacing["time"]
    if time_spacing is None:
        raise ValueError(
            "Non-uniform 'time' coordinates detected. "
            f"{operation_name.capitalize()} requires uniformly sampled data. "
            "Consider interpolating your data to a regular time grid first."
        )

    return time_axis, time_spacing
