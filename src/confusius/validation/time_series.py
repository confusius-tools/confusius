"""Time series validation utilities."""

import xarray as xr


def validate_time_series(
    time_series: xr.DataArray,
    operation_name: str,
    check_time_chunks: bool = True,
) -> int:
    """Validate time series for time series processing operations.

    Performs common validation checks:

    1. Time series have a `time` dimension.
    2. Time dimension has more than 1 timepoint.
    3. Time dimension is not chunked for Dask arrays (optional).

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

    Returns
    -------
    int
        Axis number for the `time` dimension.

    Raises
    ------
    ValueError
        If `time_series` has no `time` dimension, if the `time` dimension has only 1
        timepoint, or if the `time` dimension is chunked in a Dask array (when
        `check_time_chunks=True`).
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

    return time_axis
