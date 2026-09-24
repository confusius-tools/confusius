"""Xarray accessor for scaling operations."""

import numpy as np
import xarray as xr


def _log_output_dtype(dtype: np.dtype) -> np.dtype:
    """Infer the output dtype `numpy.log`/`numpy.log10` would produce for `dtype`.

    Parameters
    ----------
    dtype : numpy.dtype
        Input dtype.

    Returns
    -------
    numpy.dtype
        `dtype` unchanged if floating-point, else `numpy.float64` (matches
        `numpy.log`/`numpy.log10` promotion rules, which keep float32 as float32 but
        promote integer/bool inputs to float64).
    """
    return dtype if np.issubdtype(dtype, np.floating) else np.dtype(np.float64)


def _log10_ignore_errors(x: np.ndarray) -> np.ndarray:
    """Compute `log10`, suppressing divide-by-zero/invalid-value warnings.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        `log10(x)`, with zero/negative inputs mapped to `-inf`/`nan`.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(x)


def _log_ignore_errors(x: np.ndarray) -> np.ndarray:
    """Compute natural `log`, suppressing divide-by-zero/invalid-value warnings.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        `log(x)`, with zero/negative inputs mapped to `-inf`/`nan`.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(x)


def db_scale(data: xr.DataArray, factor: int | None = None) -> xr.DataArray:
    """Convert data to decibel scale relative to maximum value.

    Parameters
    ----------
    data : xarray.DataArray
        Input `DataArray`.
    factor : int, optional
        Scaling factor for decibel conversion. Use 10 for power quantities, 20 for
        amplitude quantities. If not provided, defaults to 20 for complex-valued
        (typically beamformed IQ signals) data and 10 otherwise (typically power Doppler
        signals).

    Returns
    -------
    xarray.DataArray
        Data in decibel scale. Values are in range `[factor * log(min/max), 0]` dB.

    Notes
    -----
    Warnings are suppressed for zero/negative values, which are set to `-inf`.

    If the input data is backed by Dask (lazily loaded), the global maximum is computed
    eagerly when this function is called. This avoids re-triggering a full array scan on
    each frame access (e.g. during napari playback), at the cost of a one-time upfront
    computation.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> data = xr.DataArray([1, 10, 100, 1000])
    >>> db_scale(data, factor=20)
    """
    if factor is None:
        factor = 20 if np.issubdtype(data.dtype, np.complexfloating) else 10

    abs_data = xr.ufuncs.abs(data)
    # We compute the max value non-lazily to avoid re-triggering the entire computation
    # graph for each chunk when visualizing with napari or similar tools. See
    # https://github.com/confusius-tools/confusius/issues/18.
    max_val = abs_data.max().compute()

    # np.errstate only suppresses warnings raised while the context is active. For
    # Dask-backed data, log10 executes lazily on worker threads long after this
    # function returns, so the errstate must be set inside the deferred call itself.
    db_data = factor * xr.apply_ufunc(
        _log10_ignore_errors,
        abs_data / max_val,
        dask="parallelized",
        output_dtypes=[_log_output_dtype(abs_data.dtype)],
    )

    db_data.attrs["units"] = "dB"
    db_data.attrs["scaling"] = f"{factor}*log10(x/max)"

    return db_data


def log_scale(data: xr.DataArray) -> xr.DataArray:
    """Apply natural logarithm to data.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array.

    Returns
    -------
    xarray.DataArray
        Natural logarithm of the data.

    Notes
    -----
    Warnings are suppressed for zero/negative values, which are set to `-inf/nan`.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> data = xr.DataArray([1, np.e, np.e**2])
    >>> log_scale(data)
    """
    log_data = xr.apply_ufunc(
        _log_ignore_errors,
        data,
        dask="parallelized",
        output_dtypes=[_log_output_dtype(data.dtype)],
    )

    log_data.attrs["scaling"] = "log(x)"

    return log_data


def power_scale(data: xr.DataArray, exponent: float = 0.5) -> xr.DataArray:
    """Apply power scaling to data.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array.
    exponent : float, default: 0.5
        Power exponent to apply. Default is 0.5 (square root). Use 2.0 for
        squaring, etc.

    Returns
    -------
    xarray.DataArray
        Power-scaled data.

    Examples
    --------
    >>> import xarray as xr
    >>> data = xr.DataArray([1, 4, 9, 16])
    >>> power_scale(data, exponent=0.5)  # Square root
    """
    # Apply power to absolute value to handle complex data.
    scaled_data = xr.ufuncs.abs(data) ** exponent

    scaled_data.attrs["scaling"] = f"|x|^{exponent}"

    return scaled_data


class FUSIScaleAccessor:
    """Accessor for scaling operations on fUSI data.

    This accessor provides various scaling transformations commonly used
    in functional ultrasound imaging analysis.

    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The DataArray to wrap.

    Examples
    --------
    >>> import xarray as xr
    >>> data = xr.DataArray([1, 10, 100, 1000])
    >>> data.fusi.scale.db(factor=20)
    <xarray.DataArray (dim_0: 4)>
    array([-60., -40., -20.,   0.])
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def db(self, factor: int | None = None) -> xr.DataArray:
        """Convert data to decibel scale relative to maximum value.

        Parameters
        ----------
        factor : int, optional
            Scaling factor for decibel conversion. Use 10 for power quantities, 20 for
            amplitude quantities. If not provided, defaults to 20 for complex-valued
            (typically beamformed IQ signals) data and 10 otherwise (typically power
            Doppler signals).

        Returns
        -------
        xarray.DataArray
            Data in decibel scale. Values are in range `[factor * log(min/max), 0]`
            dB.

        Notes
        -----
        Warnings are suppressed for zero/negative values, which are set to `-inf`.

        If the input data is backed by Dask (lazily loaded), the global maximum is
        computed eagerly when this method is called. This avoids re-triggering a full
        array scan on each frame access (e.g. during napari playback), at the cost of a
        one-time upfront computation.

        Examples
        --------
        >>> data = xr.DataArray([1, 10, 100, 1000])
        >>> data.fusi.scale.db(factor=20)
        <xarray.DataArray (dim_0: 4)>
        array([-60., -40., -20.,   0.])
        """
        return db_scale(self._obj, factor=factor)

    def log(self) -> xr.DataArray:
        """Apply natural logarithm to data.

        Returns
        -------
        xarray.DataArray
            Natural logarithm of the data.

        Examples
        --------
        >>> import numpy as np
        >>> data = xr.DataArray([1, np.e, np.e**2])
        >>> data.fusi.scale.log()
        <xarray.DataArray (dim_0: 3)>
        array([0., 1., 2.])
        """
        return log_scale(self._obj)

    def power(self, exponent: float = 0.5) -> xr.DataArray:
        """Apply power scaling to data.

        Parameters
        ----------
        exponent : float, default: 0.5
            Power exponent to apply. Default is 0.5 (square root). Use 2.0 for
            squaring, etc.

        Returns
        -------
        xarray.DataArray
            Power-scaled data.

        Examples
        --------
        >>> data = xr.DataArray([1, 4, 9, 16])
        >>> data.fusi.scale.power(exponent=0.5)  # Square root
        <xarray.DataArray (dim_0: 4)>
        array([1., 2., 3., 4.])
        """
        return power_scale(self._obj, exponent=exponent)
