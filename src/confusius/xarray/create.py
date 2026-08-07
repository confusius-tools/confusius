"""Constructor helper for canonical ConfUSIus fUSI DataArrays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, SupportsFloat, SupportsIndex

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._dims import CORE_DIMS, SPATIAL_DIMS, TIME_DIM
from confusius.timing import TIMING_REFERENCE_FACTORS, VolumeAcquisitionReference
from confusius.validation import canonicalize_fusi, validate_fusi, validate_iq

_SPATIAL_UNITS = "mm"
"""Physical units attached to the `z`, `y`, and `x` coordinates."""

_TIME_UNITS = "s"
"""Physical units attached to the `time` coordinate."""


def _require_positive_finite(
    value: str | SupportsFloat | SupportsIndex, name: str
) -> float:
    """Return a finite positive numeric value.

    Parameters
    ----------
    value : str or typing.SupportsFloat or typing.SupportsIndex
        Candidate value.
    name : str
        Name used in the validation error.

    Returns
    -------
    float
        Validated value.

    Raises
    ------
    ValueError
        If `value` is not numeric, finite, and positive.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _require_spacing(dim: str, spacing: float | None) -> float:
    """Return a finite positive coordinate spacing.

    Parameters
    ----------
    dim : str
        Dimension whose spacing is required.
    spacing : float, optional
        Candidate spacing value.

    Returns
    -------
    float
        Validated spacing.

    Raises
    ------
    ValueError
        If `spacing` is not provided or is not positive and finite.
    """
    if spacing is None:
        # The time spacing parameter is named dt, not dtime.
        hint = "dt" if dim == TIME_DIM else f"d{dim}"
        raise ValueError(
            f"Spacing for dimension {dim!r} is required. Provide {hint} or an "
            f"explicit {dim!r} coordinate with enough information to infer spacing."
        )
    return _require_positive_finite(spacing, f"Spacing for dimension {dim!r}")


def _regular_step(values: np.ndarray) -> float | None:
    """Return the regular spacing in `values`, or None if it cannot be inferred.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional coordinate values.

    Returns
    -------
    float or None
        Regular positive step, or None when `values` has fewer than two entries or is
        not regularly spaced.
    """
    if values.size < 2:
        return None
    diffs = np.diff(values.astype(float))
    if not np.all(np.isfinite(diffs)) or not np.all(diffs > 0):
        return None
    step = float(np.median(diffs))
    if not np.allclose(diffs, step, rtol=1e-6, atol=1e-12):
        return None
    return step


def _representative_positive_step(values: np.ndarray, dim: str) -> float | None:
    """Return the median positive step for an increasing coordinate.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional coordinate values.
    dim : str
        Dimension name used in validation errors.

    Returns
    -------
    float or None
        Median positive step, or None when `values` has fewer than two entries.

    Raises
    ------
    ValueError
        If coordinate intervals are not finite and strictly positive.
    """
    if values.size < 2:
        return None
    diffs = np.diff(values.astype(float))
    if not np.all(np.isfinite(diffs)) or not np.all(diffs > 0):
        raise ValueError(f"Coordinate {dim!r} must be strictly increasing.")
    return float(np.median(diffs))


def _coordinate_dataarray(
    dim: str,
    size: int,
    *,
    coords: Mapping[str, npt.ArrayLike | xr.DataArray],
    spacings: Mapping[str, float | None],
    origins: Mapping[str, float],
    volume_acquisition_reference: VolumeAcquisitionReference,
    volume_acquisition_duration: float | None,
) -> xr.DataArray:
    """Build one dimension coordinate.

    Parameters
    ----------
    dim : str
        Dimension name.
    size : int
        Expected coordinate length.
    coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray]
        Explicit coordinates provided by the caller.
    spacings : mapping[str, float or None]
        Per-core-dimension spacings.
    origins : mapping[str, float]
        Per-core-dimension origins used with `spacings`.
    volume_acquisition_reference : {"start", "center", "end"}
        Time-coordinate acquisition reference metadata.
    volume_acquisition_duration : float, optional
        Time-coordinate acquisition duration metadata.

    Returns
    -------
    xarray.DataArray
        One-dimensional dimension coordinate.

    Raises
    ------
    ValueError
        If an explicit coordinate has the wrong shape or required spacing metadata is
        missing.
    """
    generated_from_spacing = False
    if dim in coords:
        coord = coords[dim]
        if isinstance(coord, xr.DataArray):
            coord_values = np.asarray(coord.values)
            attrs = coord.attrs.copy()
        else:
            coord_values = np.asarray(coord)
            attrs = {}
        coord_values = np.atleast_1d(coord_values)
        if coord_values.ndim != 1 or coord_values.size != size:
            raise ValueError(
                f"Coordinate {dim!r} must be 1D with length {size}, got shape "
                f"{coord_values.shape}."
            )
    elif dim in spacings:
        step = _require_spacing(dim, spacings[dim])
        coord_values = origins[dim] + np.arange(size) * step
        attrs = {}
        generated_from_spacing = True
    else:
        coord_values = np.arange(size)
        attrs = {}

    if dim == TIME_DIM:
        if "units" not in attrs:
            attrs["units"] = _TIME_UNITS
        step = _regular_step(coord_values)
        if step is None:
            step = spacings[dim]
        if step is None and dim in coords:
            step = _representative_positive_step(coord_values, dim)
        if "volume_acquisition_reference" not in attrs:
            attrs["volume_acquisition_reference"] = volume_acquisition_reference
        elif attrs["volume_acquisition_reference"] not in TIMING_REFERENCE_FACTORS:
            raise ValueError(
                f"volume_acquisition_reference must be one of "
                f"{tuple(TIMING_REFERENCE_FACTORS)!r}, got "
                f"{attrs['volume_acquisition_reference']!r}."
            )
        if "volume_acquisition_duration" in attrs:
            duration = attrs["volume_acquisition_duration"]
        else:
            duration = volume_acquisition_duration
        if duration is None:
            duration = _require_spacing(dim, step)
        else:
            duration = _require_positive_finite(duration, "volume_acquisition_duration")
        attrs["volume_acquisition_duration"] = duration
    elif dim in SPATIAL_DIMS:
        if "units" not in attrs:
            attrs["units"] = _SPATIAL_UNITS
        if "voxdim" not in attrs:
            step = (
                spacings[dim] if generated_from_spacing else _regular_step(coord_values)
            )
            if step is None:
                step = spacings[dim]
            attrs["voxdim"] = _require_spacing(dim, step)
        else:
            attrs["voxdim"] = _require_positive_finite(
                attrs["voxdim"], f"voxdim for dimension {dim!r}"
            )

    return xr.DataArray(coord_values, dims=(dim,), attrs=attrs)


def _canonicalize_created_dataarray(
    data: xr.DataArray,
    coords: Mapping[str, npt.ArrayLike | xr.DataArray],
    spacings: Mapping[str, float | None],
    origins: Mapping[str, float],
    volume_acquisition_reference: VolumeAcquisitionReference,
    volume_acquisition_duration: float | None,
    canonical_order: bool,
) -> xr.DataArray:
    """Add missing singleton spatial dimensions and optionally transpose.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray built from the caller's raw data and present dimensions.
    coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray]
        Explicit coordinates provided by the caller.
    spacings : mapping[str, float or None]
        Per-core-dimension spacings.
    origins : mapping[str, float]
        Per-core-dimension origins used with `spacings`.
    volume_acquisition_reference : {"start", "center", "end"}
        Time-coordinate acquisition reference metadata.
    volume_acquisition_duration : float, optional
        Time-coordinate acquisition duration metadata.
    canonical_order : bool
        Whether to transpose core dimensions to `(time, z, y, x)` order.

    Returns
    -------
    xarray.DataArray
        DataArray with all spatial dimensions present.
    """
    result = data
    for dim in SPATIAL_DIMS:
        if dim in result.dims:
            continue
        coord = _coordinate_dataarray(
            dim,
            1,
            coords=coords,
            spacings=spacings,
            origins=origins,
            volume_acquisition_reference=volume_acquisition_reference,
            volume_acquisition_duration=volume_acquisition_duration,
        )
        # Attach the missing dimension as a scalar coordinate so that
        # canonicalize_fusi restores it at its canonical position
        # among the present dims instead of prepending it.
        result = result.assign_coords({dim: coord.isel({dim: 0})})
    result = canonicalize_fusi(result)

    if canonical_order:
        ordered_core = [dim for dim in CORE_DIMS if dim in result.dims]
        extra_dims = [dim for dim in result.dims if dim not in CORE_DIMS]
        result = result.transpose(*ordered_core, *extra_dims)

    return result


def create_fusi_dataarray(
    data: npt.ArrayLike,
    *,
    dims: Sequence[str],
    coords: Mapping[str, npt.ArrayLike | xr.DataArray] | None = None,
    dt: float | None = None,
    dz: float | None = None,
    dy: float | None = None,
    dx: float | None = None,
    t0: float = 0.0,
    z0: float = 0.0,
    y0: float = 0.0,
    x0: float = 0.0,
    canonical_order: bool = True,
    volume_acquisition_reference: VolumeAcquisitionReference = "start",
    volume_acquisition_duration: float | None = None,
    name: str | None = None,
    attrs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Build a ConfUSIus fUSI DataArray from a raw array.

    The returned DataArray carries the spatial dimensions `z`, `y`, and `x`.
    Dimensions may be supplied in any order and may omit spatial singleton dimensions;
    omitted spatial dimensions are added with length 1. Coordinates can be supplied
    explicitly with `coords`, or generated from the matching spacing/origin pairs.
    ConfUSIus never guesses physical spacings: every core coordinate must either contain
    enough values to infer regular spacing or receive the corresponding `dt`, `dz`,
    `dy`, or `dx` value.

    Parameters
    ----------
    data : numpy.typing.ArrayLike
        Raw array whose rank matches the length of `dims`.
    dims : sequence[str]
        Explicit dimension names for each axis of `data`, in order. May include any
        subset/order of `time`, `z`, `y`, and `x`, plus extra non-core dimensions.
        Missing spatial dimensions are added as singleton axes.
    coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray], optional
        Explicit 1D coordinates. Spatial coordinates receive `units="mm"` and
        `voxdim` metadata when missing; `voxdim` is inferred from regularly spaced
        coordinates with at least two points, otherwise the matching `d*` spacing must
        be provided. Time coordinates receive `units="s"` when missing.
    dt : float, optional
        Spacing of the `time` coordinate, in seconds. Required when `time` is present
        and the time coordinate spacing cannot be inferred from `coords`.
    dz : float, optional
        Spacing of the `z` coordinate, in millimetres. Required when `z` spacing cannot
        be inferred from `coords`.
    dy : float, optional
        Spacing of the `y` coordinate, in millimetres. Required when `y` spacing cannot
        be inferred from `coords`.
    dx : float, optional
        Spacing of the `x` coordinate, in millimetres. Required when `x` spacing cannot
        be inferred from `coords`.
    t0 : float, default: 0.0
        Origin of the generated `time` coordinate, in seconds.
    z0 : float, default: 0.0
        Origin of the generated `z` coordinate, in millimetres.
    y0 : float, default: 0.0
        Origin of the generated `y` coordinate, in millimetres.
    x0 : float, default: 0.0
        Origin of the generated `x` coordinate, in millimetres.
    canonical_order : bool, default: True
        Whether to transpose the result to canonical core order `(time, z, y, x)` with
        extra dimensions appended afterwards. If `False`, the input dimension order is
        preserved and missing spatial dimensions are added as singleton axes.
    volume_acquisition_reference : {"start", "center", "end"}, default: "start"
        Where within its acquisition window each frame's `time` coordinate is anchored.
        Stored on the `time` coordinate attributes and used by downstream timing
        helpers. Only applied when a `time` dimension is present and ignored when an
        explicit `time` coordinate already carries the attribute.
    volume_acquisition_duration : float, optional
        Duration of a single volume's acquisition window, in seconds. Stored on the
        `time` coordinate attributes. If not provided, defaults to the inferred,
        provided, or median exact time-coordinate spacing. Only applied when a `time`
        dimension is present and ignored when an explicit `time` coordinate already
        carries the attribute.
    name : str, optional
        Name assigned to the resulting DataArray.
    attrs : dict, optional
        DataArray-level attributes. Acquisition metadata that describes the whole
        recording rather than a coordinate — for example `beamforming_sound_velocity`
        and `transmit_frequency` (consumed by IQ processing) — belongs here.

    Returns
    -------
    xarray.DataArray
        ConfUSIus fUSI DataArray with physical coordinates and spatial metadata.

    Raises
    ------
    ValueError
        If `dims` contains duplicate names, if its length does not match the rank of
        `data`, if required spacing/coordinate information is missing, if
        `volume_acquisition_reference` is invalid, if `volume_acquisition_duration` is
        given without a `time` dimension, or if the resulting DataArray fails fUSI
        validation.

    Examples
    --------
    Build a single-slice recording as singleton-`z` 3D+time data:

    >>> import numpy as np
    >>> from confusius.xarray import create_fusi_dataarray
    >>> data = np.random.default_rng(0).standard_normal((20, 64, 96))
    >>> recording = create_fusi_dataarray(
    ...     data, dims=("time", "y", "x"), dt=0.5, dz=0.4, dy=0.1, dx=0.1
    ... )
    >>> recording.dims
    ('time', 'z', 'y', 'x')
    """
    dims = tuple(dims)
    coords = {} if coords is None else dict(coords)
    # np.shape reads the array's `shape` without materializing lazy (e.g. dask) arrays.
    shape = np.shape(data)

    if len(set(dims)) != len(dims):
        raise ValueError(f"dims must not contain duplicate names, got {dims!r}.")

    if len(dims) != len(shape):
        raise ValueError(
            f"Length of dims {dims!r} ({len(dims)}) must match the number of array "
            f"dimensions ({len(shape)})."
        )

    if volume_acquisition_reference not in TIMING_REFERENCE_FACTORS:
        raise ValueError(
            f"volume_acquisition_reference must be one of "
            f"{tuple(TIMING_REFERENCE_FACTORS)!r}, got {volume_acquisition_reference!r}."
        )

    if TIME_DIM not in dims and volume_acquisition_duration is not None:
        raise ValueError("volume_acquisition_duration requires a 'time' dimension.")

    spacings = {TIME_DIM: dt, "z": dz, "y": dy, "x": dx}
    origins = {TIME_DIM: t0, "z": z0, "y": y0, "x": x0}

    data_coords: dict[str, xr.DataArray] = {}
    for dim, size in zip(dims, shape):
        data_coords[dim] = _coordinate_dataarray(
            dim,
            size,
            coords=coords,
            spacings=spacings,
            origins=origins,
            volume_acquisition_reference=volume_acquisition_reference,
            volume_acquisition_duration=volume_acquisition_duration,
        )

    result = xr.DataArray(data, dims=dims, coords=data_coords, name=name, attrs=attrs)
    result = _canonicalize_created_dataarray(
        result,
        coords=coords,
        spacings=spacings,
        origins=origins,
        volume_acquisition_reference=volume_acquisition_reference,
        volume_acquisition_duration=volume_acquisition_duration,
        canonical_order=canonical_order,
    )

    regular_spacing_dims = tuple(
        dim
        for dim in CORE_DIMS
        if dim in result.dims
        and not (dim == TIME_DIM and result.sizes[dim] == 1)
        and not (dim == TIME_DIM and TIME_DIM in coords)
    )
    validate_fusi(
        result,
        require_regular_spacing=True,
        regular_spacing_dims=regular_spacing_dims,
        require_canonical_dim_order=canonical_order,
    )

    return result


def create_iq_dataarray(
    data: npt.ArrayLike,
    *,
    dims: Sequence[str],
    coords: Mapping[str, npt.ArrayLike | xr.DataArray] | None = None,
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
    transmit_frequency: float | None = None,
    beamforming_sound_velocity: float | None = None,
    name: str | None = "iq",
    attrs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Build a canonical ConfUSIus IQ DataArray from a raw complex array.

    Parameters
    ----------
    data : numpy.typing.ArrayLike
        Raw complex IQ array whose rank matches the length of `dims`.
    dims : sequence[str]
        Explicit dimension names for each axis of `data`, in order. Must include
        `time`; spatial singleton dimensions may be omitted and are added from the
        corresponding spacing/origin arguments.
    coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray], optional
        Explicit 1D coordinates. See
        [create_fusi_dataarray][confusius.xarray.create_fusi_dataarray].
    dt : float, optional
        Spacing of the `time` coordinate, in seconds.
    dz : float, optional
        Spacing of the `z` coordinate, in millimetres.
    dy : float, optional
        Spacing of the `y` coordinate, in millimetres.
    dx : float, optional
        Spacing of the `x` coordinate, in millimetres.
    t0 : float, default: 0.0
        Origin of the generated `time` coordinate, in seconds.
    z0 : float, default: 0.0
        Origin of the generated `z` coordinate, in millimetres.
    y0 : float, default: 0.0
        Origin of the generated `y` coordinate, in millimetres.
    x0 : float, default: 0.0
        Origin of the generated `x` coordinate, in millimetres.
    volume_acquisition_reference : {"start", "center", "end"}, default: "start"
        Where within its acquisition window each frame's `time` coordinate is anchored.
    volume_acquisition_duration : float, optional
        Duration of a single volume's acquisition window, in seconds.
    transmit_frequency : float, optional
        Ultrasound transmit frequency in hertz. Stored in DataArray attrs when
        provided.
    beamforming_sound_velocity : float, optional
        Speed of sound assumed during beamforming, in metres per second. Stored in
        DataArray attrs when provided.
    name : str, default: "iq"
        Name assigned to the resulting DataArray.
    attrs : dict, optional
        Additional DataArray-level attributes. Explicit `transmit_frequency` and
        `beamforming_sound_velocity` arguments override same-named keys in `attrs`.

    Returns
    -------
    xarray.DataArray
        Canonical IQ DataArray with dimensions `(time, z, y, x)`.

    Raises
    ------
    TypeError
        If `data` is not complex-valued.
    ValueError
        If coordinate construction or IQ validation fails.
    """
    attrs = {} if attrs is None else dict(attrs)
    if transmit_frequency is not None:
        attrs["transmit_frequency"] = _require_positive_finite(
            transmit_frequency, "transmit_frequency"
        )
    if beamforming_sound_velocity is not None:
        attrs["beamforming_sound_velocity"] = _require_positive_finite(
            beamforming_sound_velocity, "beamforming_sound_velocity"
        )

    result = create_fusi_dataarray(
        data,
        dims=dims,
        coords=coords,
        dt=dt,
        dz=dz,
        dy=dy,
        dx=dx,
        t0=t0,
        z0=z0,
        y0=y0,
        x0=x0,
        canonical_order=True,
        volume_acquisition_reference=volume_acquisition_reference,
        volume_acquisition_duration=volume_acquisition_duration,
        name=name,
        attrs=attrs,
    )
    validate_iq(result)
    return result
