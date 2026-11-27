"""Filtering functions for signal preprocessing."""

import numpy as np
import numpy.typing as npt
import scipy.signal
import xarray as xr

from confusius._utils.filtering import (
    design_butterworth_filter,
    make_cosine_drift_regressors,
    validate_sos_filter_length,
)
from confusius.signal.confounds import regress_confounds
from confusius.validation import validate_time_series


def _butterworth_filter_wrapper(
    data: npt.NDArray,
    axis: int,
    sos: npt.NDArray,
    padtype: str | None,
    padlen: int | None,
) -> npt.NDArray:
    """Apply a predesigned filter through `xarray.apply_ufunc`.

    Parameters
    ----------
    data : numpy.ndarray
        Samples to filter.
    axis : int
        Time axis.
    sos : numpy.ndarray
        Second-order-section filter coefficients.
    padtype : {"odd", "even", "constant", None}
        Padding extension type.
    padlen : int, optional
        Padding length. If not provided, uses SciPy's default.

    Returns
    -------
    numpy.ndarray
        Filtered samples with the original shape and axis order.

    Raises
    ------
    ValueError
        If the padding settings are invalid or the input is too short.
    """
    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    if data.ndim > 2:
        original_shape = data.shape
        n_timepoints = data.shape[0]
        data_2d = data.reshape(n_timepoints, -1)
        filtered_2d = scipy.signal.sosfiltfilt(
            sos, data_2d, axis=0, padtype=padtype, padlen=padlen
        )
        result = filtered_2d.reshape(original_shape)
    else:
        result = scipy.signal.sosfiltfilt(
            sos, data, axis=0, padtype=padtype, padlen=padlen
        )

    if axis != 0:
        result = np.moveaxis(result, 0, axis)

    return result


def filter_butterworth(
    signals: xr.DataArray,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
    order: int = 5,
    padtype: str | None = "odd",
    padlen: int | None = None,
    uniformity_tolerance: float = 1e-2,
) -> xr.DataArray:
    """Apply a low-pass, high-pass, or band-pass Butterworth digital filter to signals.

    This function filters along the `time` dimension and works with arrays of any
    shape, making it flexible for both extracted signals and full fUSI data.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Array to filter. Must have a `time` dimension. Can be any shape, e.g., extracted
        signals `(time, space)` or VoxelData array `(time, k, j, i)`.

        !!! warning "Chunking along time is not supported"
            The `time` dimension must NOT be chunked. Chunk only spatial dimensions:
            `data.chunk({'time': -1})`.

    low_cutoff : float, optional
        Low cutoff frequency in hertz. Frequencies below this are attenuated (acts as
        high-pass filter). If not provided, no high-pass filtering is applied.
    high_cutoff : float, optional
        High cutoff frequency in hertz. Frequencies above this are attenuated (acts as
        low-pass filter). If not provided, no low-pass filtering is applied.
    order : int, default: 5
        Filter order. Higher orders give steeper roll-off but may be less stable.
    padtype : {"odd", "even", "constant", None}, default: "odd"
        Type of padding to use. See `scipy.signal.sosfiltfilt` for details.
        `"odd"` pads with odd-extension (reduces edge effects), `"even"` pads
        with even-extension, and `"constant"` extends the endpoint values.
        Pass `None` to disable padding.
    padlen : int, optional
        Number of elements to pad at each end. If not provided, uses scipy's default:
        `3 * (2 * n_sections + 1 - compensation)`, based on the designed coefficients.
    uniformity_tolerance : float, default: 1e-2
        Maximum allowed relative range of consecutive time intervals, defined as
        `(max_interval - min_interval) / median_interval`. Raise a `ValueError` if
        the time coordinate exceeds this threshold. Increase this value to tolerate
        slight timestamp jitter (e.g. from acquisition clocks or dropped volumes).

    Returns
    -------
    xarray.DataArray
        Filtered signals with same shape and coordinates as input.

    Raises
    ------
    ValueError
        If `signals` does not have a `time` dimension or `time` coordinates, if
        time coordinates are not uniformly sampled within the specified tolerance, or if
        insufficient timepoints for the effective padding length.
    ValueError
        If both `low_cutoff` and `high_cutoff` are `None` (no filtering), or if cutoff
        frequencies are invalid (negative, above Nyquist, or if high_cutoff <=
        low_cutoff for band-pass).
    ValueError
        If `order` is not positive.

    Notes
    -----
    - Uses `scipy.signal.butter` with second-order sections (SOS) for numerical
      stability.
    - Uses `scipy.signal.sosfiltfilt` for zero-phase filtering (forward-backward).
    - Edge effects are handled automatically using padding.

    Examples
    --------
    Low-pass filter to remove high-frequency noise:

    >>> import xarray as xr
    >>> import numpy as np
    >>> from confusius.signal import filter_butterworth
    >>> # Create signals with time coordinates (500 Hz sampling).
    >>> signals = xr.DataArray(
    ...     np.random.randn(1000, 50),
    ...     dims=["time", "space"],
    ...     coords={"time": np.arange(1000) / 500}
    ... )
    >>> # Keep frequencies below 0.1 Hz (sampling rate computed from time coords).
    >>> filtered = filter_butterworth(signals, high_cutoff=0.1)

    High-pass filter to remove slow drift:

    >>> # Keep frequencies above 0.01 Hz (remove lower frequencies).
    >>> filtered = filter_butterworth(signals, low_cutoff=0.01)

    Band-pass filter:

    >>> # Keep only frequencies between 0.01 and 0.1 Hz.
    >>> filtered = filter_butterworth(
    ...     signals, low_cutoff=0.01, high_cutoff=0.1, order=5
    ... )
    """
    time_axis, time_spacing = validate_time_series(
        signals,
        "filtering",
        require_uniform_time=True,
        uniformity_tolerance=uniformity_tolerance,
    )

    sampling_rate = 1.0 / time_spacing
    if order <= 0:
        raise ValueError(f"'order' must be positive, got {order}.")
    sos = design_butterworth_filter(sampling_rate, low_cutoff, high_cutoff, order)
    validate_sos_filter_length(signals.sizes["time"], sos, padtype, padlen)

    result = xr.apply_ufunc(
        _butterworth_filter_wrapper,
        signals,
        kwargs={
            "axis": time_axis,
            "sos": sos,
            "padtype": padtype,
            "padlen": padlen,
        },
        dask="parallelized",
        # SciPy promotes samples against the designed SOS coefficients.
        output_dtypes=[np.result_type(signals.dtype, sos.dtype)],
    )

    return result


def filter_cosine(
    signals: xr.DataArray,
    low_cutoff: float,
    uniformity_tolerance: float = 1e-2,
) -> xr.DataArray:
    """Apply cosine high-pass filtering along `time`.

    This removes low-frequency drift by regressing each time series onto the same
    discrete-cosine basis used by the GLM `drift_model="cosine"` design matrix.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Array to filter. Must have a `time` dimension. Can be any shape, e.g., extracted
        signals `(time, space)` or VoxelData array `(time, k, j, i)`.

        !!! warning "Chunking along time is not supported"
            The `time` dimension must NOT be chunked. Chunk only spatial dimensions:
            `data.chunk({'time': -1})`.

    low_cutoff : float
        High-pass cutoff frequency in hertz. Frequencies below this are removed by
        projection onto the cosine drift basis.
    uniformity_tolerance : float, default: 1e-2
        Maximum allowed relative range of consecutive time intervals, defined as
        `(max_interval - min_interval) / median_interval`. Raise a `ValueError` if
        the time coordinate exceeds this threshold. Increase this value to tolerate
        slight timestamp jitter (e.g. from acquisition clocks or dropped volumes).

    Returns
    -------
    xarray.DataArray
        Filtered signals with same shape and coordinates as input.

    Raises
    ------
    ValueError
        If `signals` does not have a `time` dimension or `time` coordinates, if
        time coordinates are not uniformly sampled within the specified tolerance, or if
        `low_cutoff` is not positive.
    """
    _, time_spacing = validate_time_series(
        signals,
        "filtering",
        require_uniform_time=True,
        uniformity_tolerance=uniformity_tolerance,
    )

    regressors, names = make_cosine_drift_regressors(
        signals.sizes["time"],
        low_cutoff,
        time_spacing,
    )
    cosine_confounds = xr.DataArray(
        regressors,
        dims=["time", "confound"],
        coords={"time": signals.coords["time"], "confound": names},
    )

    return regress_confounds(
        signals,
        cosine_confounds,
        standardize_confounds=False,
    )
