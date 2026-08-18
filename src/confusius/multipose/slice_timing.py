"""Slice timing correction for multi-pose fUSI data."""

from typing import Literal

import numpy as np
import xarray as xr

from confusius._dims import POSE_DIM
from confusius._utils.geometry import (
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
    get_voxel_to_world_coord_names,
)
from confusius._utils.timing import interpolate_timeseries
from confusius.multipose._utils import build_consolidated_time_coordinate
from confusius.validation import ensure_fusi


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
    - Unconsolidated data: dims `(time, pose, ...)` with a pose-dependent
      `(time, pose)`-shaped `time` coordinate (see
      [stack_poses][confusius.multipose.stack_poses]), holding each pose's own real
      timestamp directly. The result's `time` becomes a genuine 1D coordinate,
      computed the same way [`consolidate_poses`][confusius.multipose.consolidate_poses]
      derives its whole-array `time` from per-pose timestamps -- after correction,
      every pose really is simultaneous, so there is no more reason for `time` to
      stay pose-dependent.

    The sweep dimension is inferred from the second dim of whichever timing coordinate
    is present.

    If the input is Dask-backed, the function stays lazy: computation is deferred until
    `.compute()` is called. The time dimension must not be chunked; spatial dimensions
    may be freely chunked.

    Parameters
    ----------
    da : xarray.DataArray
        VoxelData-compatible DataArray with a `slice_time` coordinate, or a
        pose-dependent `(time, pose)`-shaped `time` coordinate, with dims
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
        New VoxelData-compatible DataArray with the same dims as the input, resampled so every sweep
        position appears simultaneous. For already-consolidated input, `time` is
        unchanged and `slice_time` is dropped (avoiding accidental
        double-correction). For pose-dependent input, `time` becomes a genuine 1D
        coordinate (see [stack_poses][confusius.multipose.stack_poses] for what it
        was before correction).

    Raises
    ------
    ValueError
        If `da` has no `time` dimension or only one time point, if `da` has neither a
        `slice_time` coordinate nor a pose-dependent `time` coordinate, if the timing
        coordinate does not have dims `(time, <sweep_dim>)`, or if the `time`
        dimension is chunked.

    Warns
    -----
    UserWarning
        If a spline method fails due to too few points and falls back to `"linear"`.
    """
    if "time" not in da.dims:
        raise ValueError("DataArray must have a 'time' dimension.")
    da = ensure_fusi(da, require_time=True, require_unchunked_time=True)

    time_coord = da.coords["time"]
    is_pose_dependent_time = time_coord.dims == ("time", POSE_DIM)

    if "slice_time" in da.coords:
        timing_coord_name = "slice_time"
        timing_coord = da.coords["slice_time"]
        target_time_coord = time_coord
    elif is_pose_dependent_time:
        # "time" is itself the per-pose timing source (see
        # confusius.multipose.stack_poses); the target to resample onto is the
        # whole-array time this array would have if every pose were simultaneous,
        # computed the same way consolidate_poses derives its own "time".
        timing_coord_name = "time"
        timing_coord = time_coord
        base_time_coord = xr.DataArray(
            np.asarray(time_coord.isel(pose=0).values),
            dims=["time"],
            attrs=dict(time_coord.attrs),
        )
        target_time_coord = build_consolidated_time_coordinate(
            base_time_coord, np.asarray(time_coord.values), dict(time_coord.attrs)
        )
    else:
        raise ValueError(
            "DataArray has neither a 'slice_time' coordinate nor a pose-dependent "
            "(time, pose)-shaped 'time' coordinate. Slice timing correction requires "
            "per-pose or per-slice acquisition timestamps."
        )

    if len(timing_coord.dims) != 2 or timing_coord.dims[0] != "time":
        raise ValueError(
            f"{timing_coord_name!r} coordinate must have dims ('time', <sweep_dim>), "
            f"got {timing_coord.dims!r}."
        )

    target_times = target_time_coord.values

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

    out = da.copy(data=result.data)
    del out.coords[timing_coord_name]
    world_coord_names = tuple(get_voxel_to_world_coord_names(da))
    # `pose` is its own plain, independently indexed coordinate (not owned by the
    # VoxelToWorldIndex -- see its docstring), so dropping the world coordinates here
    # leaves it untouched.
    out = out.drop_vars(world_coord_names, errors="ignore")
    out = attach_voxel_to_world_index(
        out,
        get_voxel_to_world_affine(da),
        world_coord_attrs={
            name: dict(da.coords[name].attrs) for name in world_coord_names
        },
    )
    if timing_coord_name == "time":
        out = out.assign_coords(time=target_time_coord)
    return out
