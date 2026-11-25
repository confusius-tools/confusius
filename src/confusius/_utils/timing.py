"""Time-series helpers used across the package."""

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d


def infer_per_pose_duration(
    per_pose_time: npt.NDArray[np.float64], dt_fallback: float | None
) -> float | None:
    """Infer one pose's own acquisition duration from a `(time, pose)` array.

    `dt` (the shared spacing between successive samples of the *same* pose, i.e. the
    repetition period) is not a safe default: it describes a completely different
    quantity. Instead, default from the smallest nonzero gap between poses' own
    timestamps, per time point -- poses sharing the same timestamp (e.g. a
    stacked-linear-probe pose acquired simultaneously with another) don't count
    toward the gap.

    Parameters
    ----------
    per_pose_time : (n_time, n_pose) numpy.ndarray
        Each pose's own acquisition timestamp, per time point.
    dt_fallback : float, optional
        Value to fall back to when every pose shares the same timestamp at every time
        point, leaving no gap to infer from.

    Returns
    -------
    float or None
        Inferred duration, `dt_fallback`, or `None` when neither is available.
    """
    diffs = np.diff(np.sort(per_pose_time, axis=-1), axis=-1)
    nonzero_diffs = diffs[diffs > 0]
    if nonzero_diffs.size:
        return float(nonzero_diffs.min())
    return dt_fallback


def interpolate_timeseries(
    ts: npt.NDArray[np.floating],
    acq_times: npt.NDArray[np.floating],
    *,
    target_times: npt.NDArray[np.floating],
    method: Literal[
        "linear",
        "nearest",
        "nearest-up",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "previous",
        "next",
    ] = "linear",
    fill_value: float
    | tuple[float, float]
    | Literal["extrapolate", "nan"] = "extrapolate",
) -> npt.NDArray[np.floating]:
    """Interpolate a 1D time series from acquisition times to target times.

    Parameters
    ----------
    ts : (time,) numpy.ndarray
        Signal values at acquisition times.
    acq_times : (time,) numpy.ndarray
        Acquisition timestamps.
    target_times : (time,) numpy.ndarray
        Target timestamps to interpolate to.
    method : {"linear", "nearest", "nearest-up", "zero", "slinear", "quadratic", "cubic", "previous", "next"}, default: "linear"
        Interpolation method passed to `scipy.interpolate.interp1d`.
    fill_value : float or tuple[float, float] or {"extrapolate", "nan"}, default: "extrapolate"
        How to handle target times that fall outside the range of the input
        time coordinates.

    Returns
    -------
    numpy.ndarray
        Interpolated signal at `target_times`.
    """
    output_dtype = (
        ts.dtype
        if np.issubdtype(ts.dtype, np.floating)
        else np.result_type(ts.dtype, np.float64)
    )

    interp_fill_value: float | tuple[float, float] | Literal["extrapolate"]
    if fill_value == "nan":
        interp_fill_value = np.nan
    else:
        interp_fill_value = fill_value

    try:
        result = interp1d(
            acq_times,
            ts,
            kind=method,
            bounds_error=False,
            fill_value=interp_fill_value,
            copy=False,
            assume_sorted=True,
        )(target_times)
        return result.astype(output_dtype, copy=False)
    except ValueError as e:
        if "derivatives at boundaries" in str(e):
            warnings.warn(
                f"{e}; falling back to 'linear'.",
                stacklevel=2,
            )
            result = interp1d(
                acq_times,
                ts,
                kind="linear",
                bounds_error=False,
                fill_value=interp_fill_value,
                copy=False,
                assume_sorted=True,
            )(target_times)
            return result.astype(output_dtype, copy=False)
        raise
