"""Timing helpers shared across the multipose module and other pose-aware consumers."""

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils.stack import find_stack_level
from confusius.timing import convert_time_reference


def build_consolidated_time_coordinate(
    time_coord: xr.DataArray,
    slice_time_values: npt.NDArray[np.floating],
    slice_time_attrs: dict[str, Any],
) -> xr.DataArray:
    """Build one whole-array time coordinate from per-(time, pose) timing metadata.

    Shared by [consolidate_poses][confusius.multipose.consolidate_poses] (reducing a
    pose-dependent `(time, pose)`-shaped `time` coordinate into a consolidated
    `slice_time`), [stack_poses][confusius.multipose.stack_poses] (building a
    fresh whole-array `time` coordinate from independently loaded poses' own
    timestamps), and other pose-aware consumers that need to summarize
    per-`(time, pose)` timestamps into one whole-array time value per time point
    (e.g. [compute_compcor_confounds][confusius.signal.compute_compcor_confounds]).

    Parameters
    ----------
    time_coord : xarray.DataArray
        Reference time coordinate to inherit `units` and, when
        `slice_time_attrs` does not specify its own, `volume_acquisition_reference`
        from.
    slice_time_values : numpy.ndarray
        Per-`(time, pose)` timestamps, shape `(time, pose)`.
    slice_time_attrs : dict[str, Any]
        Attributes carried by the per-pose timing coordinate (`time` or
        `slice_time`), used for `volume_acquisition_duration` and
        `volume_acquisition_reference`.

    Returns
    -------
    xarray.DataArray
        Whole-array `time` coordinate. If per-pose timing metadata are insufficient
        to infer a whole-array duration, `time_coord` is returned unchanged.

    Warns
    -----
    UserWarning
        If per-pose timing metadata do not include a usable
        `volume_acquisition_duration`. In that case, `time_coord` is kept unchanged.
    UserWarning
        If inferred whole-array durations vary across time points. In that case, the
        returned coordinate omits `volume_acquisition_duration`.
    """
    slice_duration = slice_time_attrs.get("volume_acquisition_duration")
    if not isinstance(slice_duration, int | float) or slice_duration <= 0:
        warnings.warn(
            "Cannot infer whole-array timing from per-pose timestamps because "
            "`volume_acquisition_duration` is missing or non-positive. Keeping the "
            "original `time` coordinate.",
            stacklevel=find_stack_level(),
        )
        return time_coord

    slice_reference = slice_time_attrs.get(
        "volume_acquisition_reference",
        time_coord.attrs.get("volume_acquisition_reference", "start"),
    )
    volume_reference = time_coord.attrs.get(
        "volume_acquisition_reference", slice_reference
    )
    slice_onsets = convert_time_reference(
        slice_time_values,
        float(slice_duration),
        from_reference=slice_reference,
        to_reference="start",
    )
    volume_onsets = slice_onsets.min(axis=1)
    volume_durations = slice_onsets.max(axis=1) - volume_onsets + float(slice_duration)
    volume_times = convert_time_reference(
        volume_onsets,
        volume_durations,
        from_reference="start",
        to_reference=volume_reference,
    )

    time_attrs = dict(time_coord.attrs)
    time_attrs["volume_acquisition_reference"] = volume_reference
    if np.allclose(volume_durations, volume_durations[0], rtol=1e-5, atol=0):
        time_attrs["volume_acquisition_duration"] = float(volume_durations[0])
    else:
        time_attrs.pop("volume_acquisition_duration", None)
        warnings.warn(
            "Whole-array acquisition durations vary across time points. Omitting "
            "`time.attrs['volume_acquisition_duration']`.",
            stacklevel=find_stack_level(),
        )

    return xr.DataArray(volume_times, dims=["time"], attrs=time_attrs)
