"""Constructor helper for canonical ConfUSIus fUSI DataArrays."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from confusius._dims import TIME_DIM
from confusius.validation import validate_fusi_dataarray

if TYPE_CHECKING:
    import numpy.typing as npt

_SPATIAL_UNITS = "mm"
"""Physical units attached to the `z`, `y`, and `x` coordinates."""

_TIME_UNITS = "s"
"""Physical units attached to the `time` coordinate."""

VolumeAcquisitionReference = Literal["start", "center", "end"]
"""Where within its acquisition window each frame's `time` coordinate is anchored."""

_VOLUME_ACQUISITION_REFERENCES = ("start", "center", "end")
"""Accepted values for `volume_acquisition_reference`."""


def create_fusi_dataarray(
    data: npt.ArrayLike,
    *,
    dims: Sequence[str],
    dt: float | None = None,
    dz: float | None = None,
    dy: float | None = None,
    dx: float | None = None,
    t0: float = 0.0,
    z0: float = 0.0,
    y0: float = 0.0,
    x0: float = 0.0,
    volume_acquisition_reference: VolumeAcquisitionReference = "start",
    volume_acquisition_duration: float | None = None,
    name: str | None = None,
    attrs: dict | None = None,
) -> xr.DataArray:
    """Build a canonical ConfUSIus fUSI DataArray from a raw array.

    The returned DataArray follows ConfUSIus conventions: it carries the spatial
    dimensions `z`, `y`, and `x` (single-slice acquisitions use a singleton `z` axis),
    an optional leading `time` dimension, regularly spaced physical coordinates in
    millimetres, and `voxdim`/`units` metadata on each spatial coordinate. The result
    is validated with [validate_fusi_dataarray][confusius.validation.validate_fusi_dataarray]
    before being returned.

    Parameters
    ----------
    data : numpy.typing.ArrayLike
        Raw array whose rank matches the length of `dims`.
    dims : sequence[str]
        Explicit dimension names for each axis of `data`, in order. Must include the
        spatial trio `z`, `y`, and `x`, may include a leading `time` dimension, and may
        include extra non-core dimensions. No dimensions are inferred from the array
        rank.
    dt : float, optional
        Spacing of the `time` coordinate, in seconds. If not provided, defaults to
        `1.0`. Ignored when `dims` has no `time` dimension.
    dz : float, optional
        Spacing of the `z` coordinate, in millimetres. If not provided, defaults to
        `1.0`.
    dy : float, optional
        Spacing of the `y` coordinate, in millimetres. If not provided, defaults to
        `1.0`.
    dx : float, optional
        Spacing of the `x` coordinate, in millimetres. If not provided, defaults to
        `1.0`.
    t0 : float, default: 0.0
        Origin of the `time` coordinate, in seconds.
    z0 : float, default: 0.0
        Origin of the `z` coordinate, in millimetres.
    y0 : float, default: 0.0
        Origin of the `y` coordinate, in millimetres.
    x0 : float, default: 0.0
        Origin of the `x` coordinate, in millimetres.
    volume_acquisition_reference : {"start", "center", "end"}, default: "start"
        Where within its acquisition window each frame's `time` coordinate is anchored.
        Stored on the `time` coordinate attributes and used by downstream timing
        helpers. Only applied when a `time` dimension is present.
    volume_acquisition_duration : float, optional
        Duration of a single volume's acquisition window, in seconds. Stored on the
        `time` coordinate attributes. If not provided, defaults to the `time`
        coordinate spacing (the coordinate is always regularly spaced here). Only
        applied when a `time` dimension is present.
    name : str, optional
        Name assigned to the resulting DataArray.
    attrs : dict, optional
        DataArray-level attributes. Acquisition metadata that describes the whole
        recording rather than a coordinate — for example `beamforming_sound_velocity`
        and `transmit_frequency` (consumed by IQ processing) — belongs here.

    Returns
    -------
    xarray.DataArray
        Canonical ConfUSIus fUSI DataArray with regularly spaced physical coordinates
        and spatial metadata.

    Raises
    ------
    ValueError
        If `dims` contains duplicate names, if its length does not match the rank of
        `data`, if `volume_acquisition_reference` is not one of `"start"`, `"center"`,
        `"end"`, if `volume_acquisition_duration` is given without a `time` dimension,
        or if the resulting DataArray fails fUSI validation (for example when the
        spatial trio `z`, `y`, `x` is not present).

    Examples
    --------
    Build a single-slice recording as singleton-`z` 3D+time data:

    >>> import numpy as np
    >>> from confusius.xarray import create_fusi_dataarray
    >>> data = np.random.default_rng(0).standard_normal((20, 1, 64, 96))
    >>> recording = create_fusi_dataarray(
    ...     data, dims=("time", "z", "y", "x"), dt=0.5, dz=0.4, dy=0.1, dx=0.1
    ... )
    >>> recording.dims
    ('time', 'z', 'y', 'x')
    """
    dims = tuple(dims)
    # np.shape reads the array's `shape` without materializing lazy (e.g. dask) arrays.
    shape = np.shape(data)

    if len(set(dims)) != len(dims):
        raise ValueError(f"dims must not contain duplicate names, got {dims!r}.")

    if len(dims) != len(shape):
        raise ValueError(
            f"Length of dims {dims!r} ({len(dims)}) must match the number of array "
            f"dimensions ({len(shape)})."
        )

    if volume_acquisition_reference not in _VOLUME_ACQUISITION_REFERENCES:
        raise ValueError(
            f"volume_acquisition_reference must be one of "
            f"{_VOLUME_ACQUISITION_REFERENCES!r}, got {volume_acquisition_reference!r}."
        )

    if TIME_DIM not in dims and volume_acquisition_duration is not None:
        raise ValueError("volume_acquisition_duration requires a 'time' dimension.")

    # The `time` coordinate is always regularly spaced here, so the acquisition-window
    # duration defaults to that spacing when the caller does not provide one.
    time_step = 1.0 if dt is None else float(dt)
    time_duration = (
        time_step
        if volume_acquisition_duration is None
        else float(volume_acquisition_duration)
    )

    spacings = {TIME_DIM: dt, "z": dz, "y": dy, "x": dx}
    origins = {TIME_DIM: t0, "z": z0, "y": y0, "x": x0}

    coords: dict[str, xr.DataArray] = {}
    for dim, size in zip(dims, shape):
        if dim in spacings:
            spacing = spacings[dim]
            step = 1.0 if spacing is None else float(spacing)
            values = origins[dim] + np.arange(size) * step
            if dim == TIME_DIM:
                coord_attrs: dict[str, object] = {
                    "units": _TIME_UNITS,
                    "volume_acquisition_reference": volume_acquisition_reference,
                    "volume_acquisition_duration": time_duration,
                }
            else:
                coord_attrs = {"units": _SPATIAL_UNITS, "voxdim": step}
            coords[dim] = xr.DataArray(values, dims=(dim,), attrs=coord_attrs)
        else:
            coords[dim] = xr.DataArray(np.arange(size), dims=(dim,))

    result = xr.DataArray(data, dims=dims, coords=coords, name=name, attrs=attrs)

    validate_fusi_dataarray(
        result,
        require_regular_spacing=True,
        regular_spacing_dims="core",
        require_canonical_dim_order=True,
        require_spatial_voxdim=True,
        require_spatial_units=True,
        require_time_units=True,
    )

    return result
