"""Time-series helpers used across the package."""

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d


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
    interp_fill_value: float | tuple[float, float] | Literal["extrapolate"]
    if fill_value == "nan":
        interp_fill_value = np.nan
    else:
        interp_fill_value = fill_value

    try:
        return interp1d(
            acq_times,
            ts,
            kind=method,
            bounds_error=False,
            fill_value=interp_fill_value,
        )(target_times)
    except ValueError as e:
        if "derivatives at boundaries" in str(e):
            warnings.warn(
                f"{e}; falling back to 'linear'.",
                stacklevel=2,
            )
            return interp1d(
                acq_times,
                ts,
                kind="linear",
                bounds_error=False,
                fill_value=interp_fill_value,
            )(target_times)
        raise
