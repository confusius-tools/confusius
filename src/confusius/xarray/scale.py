"""Xarray accessor for scaling operations."""

import numpy as np
import xarray as xr


def db_scale(data: xr.DataArray, factor: int = 10) -> xr.DataArray:
    """Convert data to decibel scale relative to maximum value.

    Parameters
    ----------
    data : xarray.DataArray
        Input `DataArray`.
    factor : int, default: 10
        Scaling factor for decibel conversion. Use 10 for power quantities (default),
        20 for amplitude quantities.

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
    abs_data = xr.ufuncs.abs(data)
    # We compute the max value non-lazily to avoid re-triggering the entire computation
    # graph for each chunk when visualizing with napari or similar tools. See
    # https://github.com/confusius-tools/confusius/issues/18.
    max_val = abs_data.max().compute()

    with np.errstate(divide="ignore", invalid="ignore"):
        db_data = factor * xr.ufuncs.log10(abs_data / max_val)

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
    with np.errstate(divide="ignore", invalid="ignore"):
        log_data = xr.ufuncs.log(data)

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

    def db(self, factor: int = 10) -> xr.DataArray:
        """Convert data to decibel scale relative to maximum value.

        Parameters
        ----------
        factor : int, default: 10
            Scaling factor for decibel conversion. Use 10 for power quantities
            (default), 20 for amplitude quantities.

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
