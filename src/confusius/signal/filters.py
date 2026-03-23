"""Filtering functions for signal preprocessing."""

import math
import warnings

import numpy as np
import scipy.signal
import xarray as xr

from confusius._utils import _compute_spacing
from confusius.validation import validate_time_series


def _validate_cutoff_frequencies(
    low_cutoff: float | None, high_cutoff: float | None, nyquist: float
) -> None:
    """Validate cutoff frequencies against Nyquist and each other.

    Parameters
    ----------
    low_cutoff : float or None
        Low cutoff frequency in hertz.
    high_cutoff : float or None
        High cutoff frequency in hertz.
    nyquist : float
        Nyquist frequency (`sampling_rate / 2`).

    Raises
    ------
    ValueError
        If cutoff frequencies are invalid.
    """
    if low_cutoff is None and high_cutoff is None:
        raise ValueError(
            "At least one of 'low_cutoff' or 'high_cutoff' must be specified."
        )

    if low_cutoff is not None:
        if low_cutoff <= 0:
            raise ValueError(f"'low_cutoff' must be positive, got {low_cutoff}.")
        if low_cutoff >= nyquist:
            raise ValueError(
                f"'low_cutoff' ({low_cutoff} Hz) must be less than Nyquist "
                f"frequency ({nyquist} Hz)."
            )

    if high_cutoff is not None:
        if high_cutoff <= 0:
            raise ValueError(f"'high_cutoff' must be positive, got {high_cutoff}.")
        if high_cutoff >= nyquist:
            raise ValueError(
                f"'high_cutoff' ({high_cutoff} Hz) must be less than Nyquist "
                f"frequency ({nyquist} Hz)."
            )

    if low_cutoff is not None and high_cutoff is not None:
        if high_cutoff <= low_cutoff:
            raise ValueError(
                f"'high_cutoff' ({high_cutoff} Hz) must be greater than "
                f"'low_cutoff' ({low_cutoff} Hz) for band-pass filtering."
            )


def _validate_filter_order(n_timepoints: int, order: int) -> None:
    """Validate filter order and check minimum samples requirement.

    Parameters
    ----------
    n_timepoints : int
        Number of time points in the signal.
    order : int
        Filter order.

    Raises
    ------
    ValueError
        If order is invalid or insufficient timepoints.
    """
    if order <= 0:
        raise ValueError(f"'order' must be positive, got {order}.")

    # scipy.signal.sosfiltfilt requires x.shape[axis] > padlen.
    # Default padlen = 3 * (2 * n_sections + 1) where n_sections = ceil(order/2).
    # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html
    n_sections = math.ceil(order / 2)
    min_samples = 3 * (2 * n_sections + 1) + 1

    if n_timepoints < min_samples:
        raise ValueError(
            f"Filtering requires at least {min_samples} timepoints for order={order}, "
            f"but got {n_timepoints}. Use a lower order or more data."
        )


def _butterworth_filter_wrapper(
    data, axis, low_cutoff, high_cutoff, sampling_rate, order, padtype, padlen
):
    """Wrapper for Butterworth filtering that works with xr.apply_ufunc."""
    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    if low_cutoff is not None and high_cutoff is not None:
        btype = "bandpass"
        critical_freqs = [low_cutoff, high_cutoff]
    elif low_cutoff is not None:
        btype = "highpass"
        critical_freqs = low_cutoff
    elif high_cutoff is not None:
        btype = "lowpass"
        critical_freqs = high_cutoff
    else:
        if axis != 0:
            data = np.moveaxis(data, 0, axis)
        return data

    # Use second-order sections (SOS) for numerical stability.
    sos = scipy.signal.butter(
        order, critical_freqs, btype=btype, fs=sampling_rate, output="sos"
    )

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
    padtype: str = "odd",
    padlen: int | None = None,
    uniformity_tolerance: float = 1e-2,
) -> xr.DataArray:
    """Apply a low-pass, high-pass, or band-pass Butterworth digital filter to signals.

    This function filters along the `time` dimension and works with arrays of any
    shape, making it flexible for both extracted signals and full fUSI data.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Array to filter. Must have a `time` dimension. Can be any shape, e.g.,
        extracted signals `(time, space)`, full 3D+t imaging data `(time, z, y,
        x)`, or regional signals `(time, region)`.

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
        with even-extension, `"constant"` pads with zeros. If not provided, no
        padding is applied.
    padlen : int, optional
        Number of elements to pad at each end. If not provided, uses scipy's default:
        `3 * (2 * n_sections + 1 - compensation)` where `n_sections = ceil(order/2)`.
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
        insufficient timepoints for the filter order (needs more than `3 * (2 *
        ceil(order / 2) + 1)`).
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
    time_axis = validate_time_series(signals, "filtering")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        spacing = _compute_spacing(signals, uniformity_tolerance=uniformity_tolerance)

    time_spacing = spacing["time"]
    if time_spacing is None:
        raise ValueError(
            "Non-uniform 'time' coordinates detected. Butterworth filtering requires "
            "uniformly sampled data. Consider interpolating your data to a regular "
            "time grid first."
        )

    sampling_rate = 1.0 / time_spacing
    _validate_cutoff_frequencies(
        low_cutoff=low_cutoff, high_cutoff=high_cutoff, nyquist=sampling_rate / 2.0
    )
    _validate_filter_order(signals.sizes["time"], order)

    result = xr.apply_ufunc(
        _butterworth_filter_wrapper,
        signals,
        kwargs={
            "axis": time_axis,
            "low_cutoff": low_cutoff,
            "high_cutoff": high_cutoff,
            "sampling_rate": sampling_rate,
            "order": order,
            "padtype": padtype,
            "padlen": padlen,
        },
        dask="parallelized",
        output_dtypes=[signals.dtype],
    )

    return result
