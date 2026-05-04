"""Timing utilities for ConfUSIus."""

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils import find_stack_level, get_representative_step

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


def convert_time_values(
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
        difference for the time coordinate to be considered uniform (see
        [`get_representative_step`][confusius._utils.get_representative_step]).

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
            time_values = convert_time_values(
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
