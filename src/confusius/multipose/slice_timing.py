"""Slice timing correction for multi-pose fUSI data."""

import warnings
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.interpolate import interp1d

from confusius.validation.time_series import validate_time_series


def _interpolate_timeseries(
    ts: npt.NDArray[np.floating],
    acq_times: npt.NDArray[np.floating],
    *,
    target_times: npt.NDArray[np.floating],
    method: str,
    fill_value: Any,
) -> npt.NDArray[np.floating]:
    """Interpolate a 1D time series from acquisition times to target times.

    Parameters
    ----------
    ts : (time,) numpy.ndarray
        Signal values at acquisition times.
    acq_times : (time,) numpy.ndarray
        Acquisition timestamps for this sweep position.
    target_times : (time,) numpy.ndarray
        Target timestamps to interpolate to.
    method : {"linear", "nearest", "nearest-up", "zero", "slinear", "quadratic", "cubic", "previous", "next"}
        Interpolation kind passed to
        [`scipy.interpolate.interp1d`][scipy.interpolate.interp1d].
    fill_value : float or tuple[float, float] or {"extrapolate"}
        Fill value for out-of-range targets, passed to
        [`scipy.interpolate.interp1d`][scipy.interpolate.interp1d].

    Returns
    -------
    (time,) numpy.ndarray
        Interpolated signal at `target_times`.
    """
    try:
        return interp1d(
            acq_times, ts, kind=method, bounds_error=False, fill_value=fill_value
        )(target_times)
    except ValueError as e:
        if "derivatives at boundaries" in str(e):
            warnings.warn(
                f"{e}; falling back to 'linear'.",
                stacklevel=2,
            )
            return interp1d(
                acq_times, ts, kind="linear", bounds_error=False, fill_value=fill_value
            )(target_times)
        raise


def correct_slice_timings(
    da: xr.DataArray,
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
) -> xr.DataArray:
    """Resample each sweep position to the volume's reference time.

    In multi-pose fUSI acquisitions, each sweep position is acquired at a different time
    within the volume period. This function resamples each position's time series so
    that all positions appear to have been acquired at the time stored in the `time`
    coordinate.

    This function works on both:

    - Consolidated data: dims `(time, <sweep_dim>, ...)` with a `slice_time` coordinate
      with dims `(time, <sweep_dim>)`, typically produced by
      [`consolidate_poses`][confusius.multipose.consolidate_poses].
    - Unconsolidated data: dims `(time, pose, ...)` with a `pose_time` coordinate with
      dims `(time, pose)`.

    The sweep dimension is inferred from the second dim of whichever timing coordinate
    is present.

    If the input is Dask-backed, the function stays lazy: computation is deferred until
    `.compute()` is called. The time dimension must not be chunked; spatial dimensions
    may be freely chunked.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with a `slice_time` or `pose_time` coordinate whose dims are
        `(time, <sweep_dim>)`.
    method : {"linear", "nearest", "nearest-up", "zero", "slinear", "quadratic", "cubic", "previous", "next"}, default: "linear"
        Interpolation method passed to `scipy.interpolate.interp1d`:
        - `"linear"`: linear interpolation.
        - `"nearest"`: nearest-neighbour interpolation.
        - `"quadratic"`: spline of degree 2.
        - `"cubic"`: spline of degree 3.
    fill_value : float or tuple[float, float] or {"extrapolate"}, default: "extrapolate"
        How to handle target times that fall outside the range of a position's
        acquisition times. `"extrapolate"` allows linear extrapolation. Use a float for
        a constant fill value, or a tuple `(left, right)` for different values on each
        side.

    Returns
    -------
    xarray.DataArray
        New DataArray with the same dims and coordinates as the input, but with each
        position's time series resampled to `da.coords["time"].values`. The timing
        coordinate (`slice_time` or `pose_time`) is dropped to avoid accidental
        double-correction.

    Raises
    ------
    ValueError
        If `da` has no `time` dimension or only one time point, if neither `slice_time`
        nor `pose_time` coordinate is present, if the timing coordinate does not have
        dims `(time, <sweep_dim>)`, or if the `time` dimension is chunked.

    Warns
    -----
    UserWarning
        If a spline method fails due to too few points and falls back to `"linear"`.
    """
    validate_time_series(da, "slice timing correction")

    timing_coord_name = next(
        (name for name in ("slice_time", "pose_time") if name in da.coords),
        None,
    )
    if timing_coord_name is None:
        raise ValueError(
            "DataArray has neither 'slice_time' nor 'pose_time' coordinate. "
            "Slice timing correction requires per-pose or per-slice acquisition timestamps."
        )

    timing_coord = da.coords[timing_coord_name]
    if len(timing_coord.dims) != 2 or timing_coord.dims[0] != "time":
        raise ValueError(
            f"{timing_coord_name!r} coordinate must have dims ('time', <sweep_dim>), "
            f"got {timing_coord.dims!r}."
        )

    target_times = da.coords["time"].values

    # apply_ufunc vectorizes over all dims except "time" (the core dim), calling
    # _interpolate_timeseries once per (sweep_pos, *other_dims) element.
    # dask="parallelized" keeps the computation lazy when da is dask-backed. The time
    # dimension must not be chunked for interp1d to receive full series;
    # validate_time_series enforces this above.
    result = xr.apply_ufunc(
        _interpolate_timeseries,
        da,
        timing_coord,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype],
        kwargs={
            "target_times": target_times,
            "method": method,
            "fill_value": fill_value,
        },
    )
    # apply_ufunc appends core dims to the end; restore the original dim order.
    result = result.transpose(*da.dims)

    corrected_coords = {k: v for k, v in da.coords.items() if k != timing_coord_name}
    return xr.DataArray(
        result.data, dims=da.dims, coords=corrected_coords, attrs=da.attrs, name=da.name
    )
