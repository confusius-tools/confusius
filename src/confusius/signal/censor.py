"""Sample censoring and interpolation for motion scrubbing."""

import warnings

import numpy as np
import xarray as xr
from xarray.core.types import InterpOptions

from confusius._utils import find_stack_level


def _validate_sample_mask(
    signals: xr.DataArray, sample_mask: xr.DataArray
) -> np.ndarray:
    """Validate `sample_mask` DataArray and convert to boolean Numpy array.

    Parameters
    ----------
    signals : xarray.DataArray
        Input signals with time dimension.
    sample_mask : xarray.DataArray
        Boolean sample mask (`True` = keep, `False` = censor). Must have a `time`
        dimension matching signals.

    Returns
    -------
    numpy.ndarray
        Boolean mask with same length as time dimension, where `True` = keep.

    Raises
    ------
    TypeError
        If `sample_mask` is not an xarray.DataArray.
    ValueError
        If `sample_mask` has wrong dtype, length, missing time dimension, or mismatched
        coordinates.
    """
    if not isinstance(sample_mask, xr.DataArray):
        raise TypeError(
            f"sample_mask must be an xarray.DataArray, got {type(sample_mask).__name__}"
        )

    if "time" not in sample_mask.dims:
        raise ValueError("sample_mask must have a 'time' dimension")

    n_timepoints = signals.sizes["time"]
    mask_values = sample_mask.values

    mask_values = np.asarray(mask_values)

    if "time" in signals.coords and "time" in sample_mask.coords:
        if not signals.coords["time"].equals(sample_mask.coords["time"]):
            raise ValueError(
                "sample_mask time coordinates do not match signals time coordinates"
            )

    if mask_values.dtype != bool:
        raise ValueError(
            f"sample_mask must be boolean DataArray, got dtype {mask_values.dtype}"
        )

    if sample_mask.ndim != 1:
        raise ValueError(
            f"Boolean sample_mask must be 1D, got shape {sample_mask.shape}"
        )
    if len(mask_values) != n_timepoints:
        raise ValueError(
            f"sample_mask length ({len(mask_values)}) must match number of "
            f"timepoints ({n_timepoints})"
        )
    return mask_values


def interpolate_samples(
    signals: xr.DataArray,
    sample_mask: xr.DataArray,
    method: InterpOptions = "linear",
    **kwargs,
) -> xr.DataArray:
    """Interpolate censored samples from signals.

    This function interpolates values at censored (bad) timepoints using samples marked
    as good. The typical use case is to fill in high-motion volumes before temporal
    filtering, then remove them afterward with `censor_samples`. This allows retaining
    regular time sampling during filtering.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Array to interpolate. Must have a `time` dimension and `time` coordinates.
        Can be any shape, e.g., extracted signals `(time, voxels)`, full 3D+t imaging
        data `(time, z, y, x)`, or confounds `(time, n_confounds)`.
    sample_mask : (time,) xarray.DataArray
        Boolean sample mask indicating which timepoints to keep (`True`) vs.
        interpolate (`False`). Must have a `time` dimension matching `signals`.
        If both `signals` and `sample_mask` have `time` coordinates, they must match
        exactly.
    method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "quintic", \
            "polynomial", "pchip", "barycentric", "krogh", "akima", "makima"}, \
            default: "linear"
        Interpolation method passed to `xarray.DataArray.interp`. Common options:

        - `"nearest"`: Nearest-neighbor interpolation (fastest, least smooth).
        - `"linear"`: Linear interpolation (faster, less smooth).
        - `"cubic"`: Cubic spline interpolation (slower, smooth).

        See `xarray.DataArray.interp` for all available methods.
    **kwargs
        Additional keyword arguments passed to `xarray.DataArray.interp`.

    Returns
    -------
    xarray.DataArray
        Signals with interpolated values at censored positions. Same shape and
        coordinates as input.

    Raises
    ------
    ValueError
        If `signals` doesn't have a `time` dimension or `time` coordinates.
    TypeError
        If `sample_mask` is not an xarray.DataArray.
    ValueError
        If `sample_mask` has wrong dtype, length, missing time dimension, or mismatched
        coordinates.
    ValueError
        If all samples are censored (cannot interpolate).

    Warns
    -----
    UserWarning
        If all samples are marked as good (no interpolation needed).

    Notes
    -----
    - Kept samples (`sample_mask=True`) are unchanged; only censored samples
      (`sample_mask=False`) are replaced with interpolated values.
    - Uses `xarray.DataArray.interp` which handles coordinates and Dask arrays
      automatically.

    Examples
    --------
    Interpolate high-motion volumes before filtering:

    >>> import xarray as xr
    >>> import numpy as np
    >>> from confusius.signal import interpolate_samples, filter_butterworth, censor_samples
    >>> # Create signals with time coordinates.
    >>> signals = xr.DataArray(
    ...     np.random.randn(100, 50),
    ...     dims=["time", "space"],
    ...     coords={"time": np.arange(100) / 500}  # 500 Hz.
    ... )
    >>> # Mark high-motion frames (e.g., frames 10, 25, 60 are bad).
    >>> motion_outliers = np.array([10, 25, 60])
    >>> mask_values = np.ones(100, dtype=bool)
    >>> mask_values[motion_outliers] = False  # False = censor.
    >>> sample_mask = xr.DataArray(
    ...     mask_values, dims=["time"], coords={"time": signals.coords["time"]}
    ... )

    Pre-scrubbing workflow (recommended):

    >>> # 1. Interpolate censored samples.
    >>> interpolated = interpolate_samples(signals, sample_mask, method="cubic")
    >>> # 2. Apply temporal filter to complete signal.
    >>> filtered = filter_butterworth(interpolated, high_cutoff=0.1)
    >>> # 3. Remove censored samples after filtering.
    >>> cleaned = censor_samples(filtered, sample_mask)

    Control boundary behavior:

    >>> # Default: extrapolate at boundaries.
    >>> interpolated = interpolate_samples(signals, sample_mask, fill_value="extrapolate")
    >>> # Or fill with NaN outside kept range.
    >>> interpolated_nan = interpolate_samples(signals, sample_mask, fill_value=np.nan)
    """
    if "time" not in signals.dims:
        raise ValueError("signals must have a 'time' dimension.")

    if "time" not in signals.coords:
        raise ValueError(
            "signals must have 'time' coordinates to perform interpolation."
        )

    boolean_mask = _validate_sample_mask(signals, sample_mask)

    if not np.any(boolean_mask):
        raise ValueError("All samples are censored, cannot interpolate.")

    if np.all(boolean_mask):
        warnings.warn(
            "All samples are marked as good, so no interpolation was performed.",
            stacklevel=find_stack_level(),
        )
        return signals

    kept_signals = signals.isel(time=boolean_mask)
    result = kept_signals.interp(
        time=signals.coords["time"], method=method, kwargs=kwargs
    )

    return result


def censor_samples(
    signals: xr.DataArray,
    sample_mask: xr.DataArray,
) -> xr.DataArray:
    """Remove censored samples from signals.

    This function removes timepoints marked as censored (bad) from the signals. After
    censoring, the time series becomes irregular (non-uniform time steps). Be cautious
    with subsequent time-domain analyses that assume uniform sampling.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Array to censor. Must have a `time` dimension. Can be any shape, e.g.,
        extracted signals `(time, space)`, full 3D+t imaging data `(time, z, y,
        x)`, or confounds `(time, n_confounds)`.
    sample_mask : (time,) xarray.DataArray
        Boolean sample mask indicating which timepoints to keep (`True`) vs. remove
        (`False`). Must have a `time` dimension matching `signals`. If both
        `signals` and `sample_mask` have `time` coordinates, they must match exactly.

    Returns
    -------
    xarray.DataArray
        Signals with censored timepoints removed. Shape is `(n_kept, ...)` where
        `n_kept` is the number of `True` values in `sample_mask`. Time coordinates
        are subsetted to kept samples.

    Raises
    ------
    ValueError
        If `signals` doesn't have a `time` dimension or `time` coordinates.
    TypeError
        If `sample_mask` is not an xarray.DataArray.
    ValueError
        If `sample_mask` has wrong dtype, length, missing time dimension, or mismatched
        coordinates.
    ValueError
        If all samples are censored (cannot interpolate).

    Warns
    -----
    UserWarning
        If all samples are kept (no censoring performed).

    Examples
    --------
    Remove high-motion volumes:

    >>> import xarray as xr
    >>> import numpy as np
    >>> from confusius.signal import censor_samples
    >>> # Create signals.
    >>> signals = xr.DataArray(
    ...     np.random.randn(100, 50),
    ...     dims=["time", "space"],
    ...     coords={"time": np.arange(100) / 500}
    ... )
    >>> # Mark frames to keep (False = remove).
    >>> mask_values = np.ones(100, dtype=bool)
    >>> mask_values[[10, 25, 60]] = False  # Remove these frames.
    >>> sample_mask = xr.DataArray(
    ...     mask_values, dims=["time"], coords={"time": signals.coords["time"]}
    ... )
    >>> # Remove censored samples.
    >>> censored = censor_samples(signals, sample_mask)
    >>> censored.sizes["time"]  # 97 timepoints (3 removed).
    97

    Complete pre-scrubbing workflow:

    >>> from confusius.signal import interpolate_samples, filter_butterworth
    >>> # 1. Interpolate censored samples.
    >>> interpolated = interpolate_samples(signals, sample_mask)
    >>> # 2. Apply temporal filter.
    >>> filtered = filter_butterworth(interpolated, high_cutoff=0.1)
    >>> # 3. Remove censored samples.
    >>> cleaned = censor_samples(filtered, sample_mask)
    """
    if "time" not in signals.dims:
        raise ValueError("signals must have a 'time' dimension.")

    boolean_mask = _validate_sample_mask(signals, sample_mask)

    if not np.any(boolean_mask):
        raise ValueError("All samples are censored, cannot remove all timepoints.")

    if np.all(boolean_mask):
        warnings.warn(
            "All samples are marked as good, so no censoring was performed.",
            stacklevel=find_stack_level(),
        )
        return signals

    result = signals.isel(time=boolean_mask)

    return result
