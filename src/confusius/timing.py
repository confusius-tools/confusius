"""Timing utilities for ConfUSIus."""

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._dims import TIME_DIM
from confusius._utils.coordinates import get_representative_step
from confusius._utils.stack import find_stack_level
from confusius._utils.timing import interpolate_timeseries
from confusius.validation.time_series import validate_time_series

_TIME_UNIT_TO_SECONDS: dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "msec": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "us": 1e-6,
    "usec": 1e-6,
    "microsecond": 1e-6,
    "microseconds": 1e-6,
}
"""Mapping from common time-unit strings to seconds."""

_TIMING_REF_FACTORS: dict[str, float] = {"start": 0.0, "center": 0.5, "end": 1.0}
"""Mapping from timing reference name to fractional position within a volume."""


def _get_time_unit_to_seconds_factor(
    unit: str | None,
    *,
    raise_on_missing: bool = False,
    raise_on_unknown: bool = False,
) -> float:
    """Return the multiplicative factor that converts a time unit to seconds.

    Parameters
    ----------
    unit : str or None
        Time unit string.
    raise_on_missing : bool, default: False
        Whether to raise when `unit` is missing.
    raise_on_unknown : bool, default: False
        Whether to raise when `unit` is not recognized.

    Returns
    -------
    float
        Multiplicative factor that converts values expressed in `unit` to seconds.
        Returns `1.0` when units are missing or unknown and the corresponding
        `raise_on_*` flag is `False`.

    Raises
    ------
    ValueError
        If `unit` is missing or unknown and the corresponding `raise_on_*` flag is
        `True`.
    """
    if unit is None:
        if raise_on_missing:
            raise ValueError("Time unit is missing.")
        return 1.0

    normalized_unit = str(unit).lower()
    if normalized_unit not in _TIME_UNIT_TO_SECONDS:
        if raise_on_unknown:
            raise ValueError(
                f"Unknown time unit: {unit}. Supported units: {list(_TIME_UNIT_TO_SECONDS.keys())}"
            )
        return 1.0

    return _TIME_UNIT_TO_SECONDS[normalized_unit]


def get_time_coord_to_seconds_factor(data: xr.DataArray) -> float:
    """Return the factor that converts a time coordinate to seconds.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray containing a `time` coordinate.

    Returns
    -------
    float
        Multiplicative factor that converts values expressed in the time coordinate's
        units to seconds. Returns `1.0` when units are missing or unknown.

    Raises
    ------
    ValueError
        If `data` does not have a `time` coordinate.

    Warns
    -----
    UserWarning
        If the time coordinate has missing or unknown units. In that case, seconds are
        assumed.
    """
    if "time" not in data.coords:
        raise ValueError("DataArray must have a `time` coordinate.")

    units = data.coords["time"].attrs.get("units")
    if units is None:
        warnings.warn(
            "Time coordinate has no `units` attribute. Assuming seconds.",
            stacklevel=find_stack_level(),
        )
        return 1.0

    normalized_unit = str(units).lower()
    if normalized_unit not in _TIME_UNIT_TO_SECONDS:
        warnings.warn(
            f"Time coordinate uses unknown units {units!r}. Assuming seconds.",
            stacklevel=find_stack_level(),
        )
        return 1.0

    return _get_time_unit_to_seconds_factor(normalized_unit, raise_on_unknown=True)


def convert_time_units(
    values: npt.ArrayLike,
    from_unit: str | None,
    to_unit: str | None = "s",
    *,
    raise_on_missing: bool = False,
    raise_on_unknown: bool = False,
) -> npt.NDArray[np.floating]:
    """Convert time values between units.

    Parameters
    ----------
    values : array_like
        Time values to convert.
    from_unit : str or None
        Current unit of the values.
    to_unit : str or None, default: "s"
        Target unit of the values.
    raise_on_missing : bool, default: False
        Whether to raise when either unit is missing.
    raise_on_unknown : bool, default: False
        Whether to raise when either unit is not recognized.

    Returns
    -------
    numpy.ndarray
        Time values converted to `to_unit`.
    """
    values_array = np.asarray(values)
    if from_unit == to_unit:
        return values_array

    from_factor = _get_time_unit_to_seconds_factor(
        from_unit,
        raise_on_missing=raise_on_missing,
        raise_on_unknown=raise_on_unknown,
    )
    to_factor = _get_time_unit_to_seconds_factor(
        to_unit,
        raise_on_missing=raise_on_missing,
        raise_on_unknown=raise_on_unknown,
    )
    return values_array * (from_factor / to_factor)


def get_representative_time_step(
    data: xr.DataArray,
    *,
    unit: str | None = None,
    uniformity_tolerance: float = 1e-2,
) -> tuple[float | None, bool]:
    """Return a representative time step for the time coordinate.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray containing the time coordinate.
    unit : str or None, default: None
        Unit in which to evaluate the representative step. If `None`, use the native
        units of the time coordinate.
    uniformity_tolerance : float, default: 1e-2
        Maximum allowed per-interval relative deviation from the median consecutive
        difference for the time coordinate to be considered uniform.

    Returns
    -------
    step : float or None
        Representative time step in `unit` when provided, otherwise in the native units
        of the time coordinate. Returns `None` if fewer than two time points are
        available.
    approximate : bool
        Whether the returned step is a median approximation derived from non-uniform
        sampling.
    """
    if "time" not in data.coords:
        raise ValueError("DataArray must have a `time` coordinate.")

    time_values = np.asarray(data.coords["time"].values)
    if unit is not None:
        if unit == "s":
            time_values = time_values * get_time_coord_to_seconds_factor(data)
        else:
            source_in_seconds = get_time_coord_to_seconds_factor(data)
            time_values = convert_time_units(
                time_values * source_in_seconds,
                "s",
                unit,
                raise_on_unknown=True,
            )

    return get_representative_step(
        time_values, uniformity_tolerance=uniformity_tolerance
    )


def convert_time_reference(
    time: npt.ArrayLike,
    volume_duration: float | npt.ArrayLike,
    from_reference: Literal["start", "center", "end"],
    to_reference: Literal["start", "center", "end"],
) -> npt.NDArray[np.floating]:
    """Convert timings from one volume acquisition reference to another.

    Parameters
    ----------
    time : array_like
        Input timings in any physical time unit.
    volume_duration : float or array_like
        Duration of one volume in the same units as `time`. May be a scalar or one
        duration per input timing.
    from_reference : {"start", "center", "end"}
        Reference point used by the input timings.
    to_reference : {"start", "center", "end"}
        Reference point to convert the timings to.

    Returns
    -------
    numpy.ndarray
        Converted timings in the same units as `time`.
    """
    time_values = np.asarray(time)
    volume_duration_values = np.asarray(volume_duration)
    if from_reference == to_reference:
        return time_values

    if from_reference not in _TIMING_REF_FACTORS:
        raise ValueError(
            f"Unknown from_reference: {from_reference!r}. Must be 'start', 'center', or 'end'."
        )
    if to_reference not in _TIMING_REF_FACTORS:
        raise ValueError(
            f"Unknown to_reference: {to_reference!r}. Must be 'start', 'center', or 'end'."
        )

    from_factor = _TIMING_REF_FACTORS[from_reference]
    to_factor = _TIMING_REF_FACTORS[to_reference]
    offset = (to_factor - from_factor) * volume_duration_values

    return time_values + offset


def resample_time(
    data: xr.DataArray,
    new_time: npt.ArrayLike,
    *,
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
    """Resample data to new time coordinates.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray with a `time` coordinate.
    new_time : array_like
        New time coordinates to resample to.
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
        How to handle target times that fall outside the range of the input time
        coordinates. `"extrapolate"` allows linear extrapolation. Use a float for
        a constant fill value, or a tuple `(left, right)` for different values on each
        side.

    Returns
    -------
    xarray.DataArray
        DataArray resampled to `new_time` coordinates.

    Raises
    ------
    ValueError
        If `data` does not have a `time` dimension or has only 1 timepoint.
    """
    validate_time_series(data, "time resampling")

    time_coord = data.coords[TIME_DIM].values
    new_time_arr = np.asarray(new_time)
    output_dtype = np.result_type(data.dtype, np.float64)

    result = xr.apply_ufunc(
        interpolate_timeseries,
        data,
        time_coord,
        input_core_dims=[[TIME_DIM], [TIME_DIM]],
        output_core_dims=[[TIME_DIM]],
        exclude_dims={TIME_DIM},
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {TIME_DIM: new_time_arr.size}},
        output_dtypes=[output_dtype],
        keep_attrs=True,
        kwargs={
            "target_times": new_time_arr,
            "method": method,
            "fill_value": fill_value,
        },
    )

    time_coord_attrs = dict(data.coords[TIME_DIM].attrs)
    new_time_coord = xr.DataArray(
        new_time_arr,
        dims=(TIME_DIM,),
        attrs=time_coord_attrs,
    )
    result = result.assign_coords({TIME_DIM: new_time_coord})
    result = result.transpose(*data.dims)
    result.attrs.update(data.attrs)

    return result


def resample_to_uniform_time(
    data: xr.DataArray,
    *,
    start: float | None = None,
    stop: float | None = None,
    step: float | None = None,
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
    """Resample data to a uniform time grid.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray with a `time` coordinate.
    start : float or None, default: None
        Start of the new time grid. If not provided, use the first time point.
    stop : float or None, default: None
        Stop of the new time grid. If not provided, use the last time point.
    step : float or None, default: None
        Time step for the uniform grid. If not provided, derive it from the input
        coordinate by estimating a representative interval from consecutive time
        differences. For non-uniform input, the median interval is used and a warning is
        emitted. The generated grid always starts at `start` and includes `stop` only
        when it falls on the `start + n * step` lattice.
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
        How to handle target times that fall outside the range of the input time
        coordinates. `"extrapolate"` allows linear extrapolation. Use a float for
        a constant fill value, or a tuple `(left, right)` for different values on each
        side.

    Returns
    -------
    xarray.DataArray
        DataArray resampled to a uniform time grid.

    Raises
    ------
    ValueError
        If `data` does not have a `time` dimension or has only 1 timepoint.
    ValueError
        If `start` is greater than or equal to `stop`, if `step` is not positive,
        or if no valid representative step can be derived.

    Warns
    -----
    UserWarning
        If the original time coordinate is not uniform and `step` is not provided.
        In that case the median step is used.
    """
    validate_time_series(data, "uniform time resampling")

    time_values = data.coords[TIME_DIM].values

    if start is None:
        new_start = time_values[0]
    else:
        new_start = start

    if stop is None:
        new_stop = time_values[-1]
    else:
        new_stop = stop

    if new_start >= new_stop:
        raise ValueError(f"start ({new_start}) must be less than stop ({new_stop}).")

    if step is None:
        step, approximate = get_representative_step(time_values)
        if step is None or not np.isfinite(step) or step <= 0.0:
            raise ValueError(
                "Cannot compute representative time step from data. "
                "Provide `step` explicitly."
            )
        if approximate:
            warnings.warn(
                "Time coordinate is non-uniform; using the median step to build "
                "a uniform target grid.",
                stacklevel=find_stack_level(),
            )
    else:
        step = float(step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError(f"step must be a finite positive value, got {step!r}.")

    span = float(new_stop - new_start)
    n_steps = int(np.floor(span / step + 1e-12)) + 1
    new_time_values = new_start + float(step) * np.arange(n_steps)

    result = resample_time(data, new_time_values, method=method, fill_value=fill_value)

    return result
