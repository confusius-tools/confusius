"""Detrending functions for signal preprocessing."""

import numpy as np
import scipy.signal
import xarray as xr

from confusius.validation import validate_time_series


def _polynomial_detrend_wrapper(data, axis, order):
    """Wrapper for polynomial detrending that works with xr.apply_ufunc."""
    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    n_timepoints = data.shape[0]
    time_vals = np.arange(n_timepoints)

    if data.ndim > 2:
        original_shape = data.shape
        data_2d = data.reshape(n_timepoints, -1)
        poly_coeffs = np.polyfit(time_vals, data_2d, order)
        poly_trend = np.polyval(poly_coeffs, time_vals[:, np.newaxis])
        poly_trend = poly_trend.reshape(original_shape)
    else:
        poly_coeffs = np.polyfit(time_vals, data, order)
        poly_trend = np.polyval(poly_coeffs, time_vals[:, np.newaxis])

    result = data - poly_trend

    if axis != 0:
        result = np.moveaxis(result, 0, axis)

    return result


def detrend(signals: xr.DataArray, order: int = 1) -> xr.DataArray:
    """Remove trends from signals across time.

    This function operates along the `time` dimension and works with arrays of any
    shape, making it flexible for both extracted signals and full fUSI data.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Array to detrend. Must have a `time` dimension. Can be any shape, e.g.,
        extracted signals `(time, space)`, full 3D+t imaging data `(time, z, y,
        x)`, or regional signals `(time, region)`.

        !!! warning "Chunking along time is not supported"
            The `time` dimension must NOT be chunked. Chunk only spatial dimensions:
            `data.chunk({'time': -1})`.

    order : int, default: 1
        Polynomial order for detrending:

        - `0`: Remove mean (constant detrending).
        - `1`: Remove linear trend using least squares regression (default).
        - `2+`: Remove polynomial trend of specified order.

    Returns
    -------
    xarray.DataArray
        Detrended signals with same shape and coordinates as input.

    Raises
    ------
    ValueError
        If `signals` does not have a `time` dimension. If `order` is negative.

    Warns
    -----
    UserWarning
        If `signals` have only one timepoint (detrending is skipped).

    Notes
    -----
    - For `order=0` and `order=1`, uses `scipy.signal.detrend`.
    - For `order >= 2`, fits a polynomial of the specified order and subtracts it.

    Examples
    --------
    Remove mean (constant detrending):

    >>> import xarray as xr
    >>> import numpy as np
    >>> signals = xr.DataArray(
    ...     np.random.randn(100, 50) + 10,  # Noise with offset.
    ...     dims=["time", "space"]
    ... )
    >>> detrended = detrend(signals, order=0)  # Remove mean.

    Remove linear trend (default):

    >>> time = np.arange(100)
    >>> drift = time[:, None] * 0.5  # Linear drift.
    >>> noise = np.random.randn(100, 50)
    >>> signals = xr.DataArray(drift + noise, dims=["time", "space"])
    >>> detrended = detrend(signals)  # order=1 by default.

    Remove quadratic trend:

    >>> quadratic_drift = (time[:, None] ** 2) * 0.01
    >>> signals_quad = xr.DataArray(quadratic_drift + noise, dims=["time", "space"])
    >>> detrended_quad = detrend(signals_quad, order=2)

    Works on 3D+t data:

    >>> imaging_3dt = xr.DataArray(
    ...     np.random.randn(100, 10, 20, 30),
    ...     dims=["time", "z", "y", "x"]
    ... )
    >>> detrended_3dt = detrend(imaging_3dt, order=1)
    """
    if "time" not in signals.dims:
        raise ValueError("signals must have a 'time' dimension")

    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}")

    time_axis = validate_time_series(signals, "detrending")

    if order == 0:
        result = xr.apply_ufunc(
            scipy.signal.detrend,
            signals,
            kwargs={"axis": time_axis, "type": "constant"},
            dask="parallelized",
        )
    elif order == 1:
        result = xr.apply_ufunc(
            scipy.signal.detrend,
            signals,
            kwargs={"axis": time_axis, "type": "linear"},
            dask="parallelized",
        )
    else:
        result = xr.apply_ufunc(
            _polynomial_detrend_wrapper,
            signals,
            kwargs={"axis": time_axis, "order": order},
            dask="parallelized",
            output_dtypes=[signals.dtype],
        )

    return result
