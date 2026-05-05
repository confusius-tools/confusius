"""Slice timing correction for multi-pose fUSI data."""

from typing import Literal

import xarray as xr

from confusius._utils import interpolate_timeseries
from confusius.validation.time_series import validate_time_series


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
        - `"nearest"`: nearest-neighbour interpolation; rounds down at half-integers.
        - `"nearest-up"`: nearest-neighbour interpolation; rounds up at half-integers.
        - `"zero"`: zeroth-order spline (step function).
        - `"slinear"`: first-order spline.
        - `"quadratic"`: second-order spline.
        - `"cubic"`: third-order spline.
        - `"previous"`: use previous point's value.
        - `"next"`: use next point's value.

    fill_value : float or tuple[float, float] or {"extrapolate", "nan"}, default: "extrapolate"
        How to handle target times that fall outside the range of a position's
        acquisition times. `"extrapolate"` allows linear extrapolation. `"nan"`
        inserts NaNs out of bounds. Use a float for a constant fill value, or a tuple
        `(left, right)` for different values on each side.

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
    # interpolate_timeseries once per (sweep_pos, *other_dims) element.
    # dask="parallelized" keeps the computation lazy when da is dask-backed. The time
    # dimension must not be chunked for interp1d to receive full series;
    # validate_time_series enforces this above.
    result = xr.apply_ufunc(
        interpolate_timeseries,
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
