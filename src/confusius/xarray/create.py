"""Constructor helpers for canonical ConfUSIus fUSI DataArrays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, SupportsFloat, SupportsIndex

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._dims import CORE_DIMS, SPATIAL_DIMS, TIME_DIM, VOXEL_DIMS
from confusius._utils.geometry import add_physical_coords_from_voxel_affine
from confusius.timing import TIMING_REFERENCE_FACTORS, VolumeAcquisitionReference
from confusius.validation import validate_fusi, validate_iq

_SPATIAL_UNITS = "mm"
"""Physical units attached to the `z`, `y`, and `x` coordinates."""

_TIME_UNITS = "s"
"""Physical units attached to the `time` coordinate."""

_SPATIAL_TO_VOXEL = dict(zip(SPATIAL_DIMS, VOXEL_DIMS, strict=True))
"""Mapping from public physical spatial axis names to native voxel dimension names."""

_VOXEL_TO_SPATIAL = dict(zip(VOXEL_DIMS, SPATIAL_DIMS, strict=True))
"""Mapping from native voxel dimension names to physical spatial coordinate names."""


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


def _coordinate_values_and_attrs(
    dim: str, coords: Mapping[str, npt.ArrayLike | xr.DataArray]
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Return explicit coordinate values and attrs for `dim` when provided.

    Parameters
    ----------
    dim : str
        Coordinate name.
    coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray]
        Explicit coordinate mapping.

    Returns
    -------
    values : numpy.ndarray
        Coordinate values.
    attrs : dict[str, Any]
        Coordinate attributes.
    """
    if dim not in coords:
        return None
    coord = coords[dim]
    if isinstance(coord, xr.DataArray):
        return np.atleast_1d(np.asarray(coord.values)), coord.attrs.copy()
    return np.atleast_1d(np.asarray(coord)), {}


def _validate_coordinate_shape(dim: str, values: np.ndarray, size: int) -> None:
    """Validate a one-dimensional coordinate length.

    Parameters
    ----------
    dim : str
        Coordinate name.
    values : numpy.ndarray
        Coordinate values.
    size : int
        Expected length.

    Raises
    ------
    ValueError
        If `values` is not one-dimensional with length `size`.
    """
    if values.ndim != 1 or values.size != size:
        raise ValueError(
            f"Coordinate {dim!r} must be 1D with length {size}, got shape "
            f"{values.shape}."
        )


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
    """Build one non-spatial dimension coordinate.

    Parameters
    ----------
    dim : str
        Dimension name.
    size : int
        Expected coordinate length.
    coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray]
        Explicit coordinates provided by the caller.
    spacings : mapping[str, float or None]
        Per-dimension spacings.
    origins : mapping[str, float]
        Per-dimension origins used with `spacings`.
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
        If an explicit coordinate has the wrong shape or required timing metadata is
        missing.
    """
    explicit = _coordinate_values_and_attrs(dim, coords)
    if explicit is not None:
        coord_values, attrs = explicit
        _validate_coordinate_shape(dim, coord_values, size)
    elif dim in spacings:
        step = _require_spacing(dim, spacings[dim])
        coord_values = origins[dim] + np.arange(size) * step
        attrs = {}
    else:
        coord_values = np.arange(size)
        attrs = {}

    if dim == TIME_DIM:
        attrs.setdefault("units", _TIME_UNITS)
        step = _regular_step(coord_values) or spacings[dim]
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
        duration = attrs.get("volume_acquisition_duration", volume_acquisition_duration)
        attrs["volume_acquisition_duration"] = (
            _require_spacing(dim, step)
            if duration is None
            else _require_positive_finite(duration, "volume_acquisition_duration")
        )

    return xr.DataArray(coord_values, dims=(dim,), attrs=attrs)


def _spatial_geometry(
    physical_dim: str,
    voxel_dim: str,
    size: int,
    *,
    coords: Mapping[str, npt.ArrayLike | xr.DataArray],
    spacing: float | None,
    origin: float,
) -> tuple[xr.DataArray, float, float, dict[str, Any]]:
    """Build a voxel coordinate and physical affine parameters for one axis.

    Parameters
    ----------
    physical_dim : str
        Physical coordinate name (`z`, `y`, or `x`).
    voxel_dim : str
        Native voxel dimension name (`k`, `j`, or `i`).
    size : int
        Axis length.
    coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray]
        Explicit coordinate mapping.
    spacing : float, optional
        Physical spacing for the axis.
    origin : float
        Physical origin for the axis.

    Returns
    -------
    voxel_coord : xarray.DataArray
        Native voxel-space dimension coordinate.
    physical_origin : float
        Physical coordinate origin.
    physical_spacing : float
        Physical coordinate spacing per voxel index.
    physical_attrs : dict[str, Any]
        Attributes for the derived physical coordinate.

    Raises
    ------
    ValueError
        If coordinate shape, spacing, or monotonicity is invalid.
    """
    voxel_explicit = _coordinate_values_and_attrs(voxel_dim, coords)
    if voxel_explicit is None:
        voxel_values = np.arange(size, dtype=float)
        voxel_attrs: dict[str, Any] = {}
    else:
        voxel_values, voxel_attrs = voxel_explicit
        _validate_coordinate_shape(voxel_dim, voxel_values, size)

    physical_explicit = _coordinate_values_and_attrs(physical_dim, coords)
    if physical_explicit is None:
        physical_origin = origin
        physical_spacing = _require_spacing(physical_dim, spacing)
        physical_attrs: dict[str, Any] = {}
    else:
        physical_values, physical_attrs = physical_explicit
        _validate_coordinate_shape(physical_dim, physical_values, size)
        step = _regular_step(physical_values)
        if step is None:
            step = spacing
        if step is None and "voxdim" in physical_attrs:
            physical_spacing = _require_positive_finite(
                physical_attrs["voxdim"], f"voxdim for dimension {physical_dim!r}"
            )
        else:
            physical_spacing = _require_spacing(physical_dim, step)
        physical_origin = float(physical_values[0])

    physical_attrs.setdefault("units", _SPATIAL_UNITS)
    physical_attrs["voxdim"] = _require_positive_finite(
        physical_attrs.get("voxdim", physical_spacing),
        f"voxdim for dimension {physical_dim!r}",
    )
    return (
        xr.DataArray(voxel_values, dims=(voxel_dim,), attrs=voxel_attrs),
        physical_origin,
        physical_spacing,
        physical_attrs,
    )


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

    Parameters
    ----------
    data : numpy.typing.ArrayLike
        Raw array whose rank matches the length of `dims`.
    dims : sequence[str]
        Explicit dimension names for each axis. Spatial axes may be supplied as public
        physical names `z/y/x` or native voxel names `k/j/i`; the returned DataArray
        always uses native voxel dimensions and CTI-backed physical coordinates.
    coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray], optional
        Explicit 1D coordinates. Spatial physical coordinates `z/y/x` define the
        output affine; native voxel coordinates `k/j/i` define the sampled voxel axis.
    dt : float, optional
        Spacing of the `time` coordinate, in seconds.
    dz : float, optional
        Spacing of the `z` physical coordinate, in millimetres.
    dy : float, optional
        Spacing of the `y` physical coordinate, in millimetres.
    dx : float, optional
        Spacing of the `x` physical coordinate, in millimetres.
    t0 : float, default: 0.0
        Origin of the generated `time` coordinate, in seconds.
    z0 : float, default: 0.0
        Origin of the generated `z` physical coordinate, in millimetres.
    y0 : float, default: 0.0
        Origin of the generated `y` physical coordinate, in millimetres.
    x0 : float, default: 0.0
        Origin of the generated `x` physical coordinate, in millimetres.
    canonical_order : bool, default: True
        Whether to transpose core dimensions to `(time, pose, k, j, i)` order.
    volume_acquisition_reference : {"start", "center", "end"}, default: "start"
        Where within its acquisition window each frame's `time` coordinate is anchored.
    volume_acquisition_duration : float, optional
        Duration of a single volume's acquisition window, in seconds.
    name : str, optional
        Name assigned to the resulting DataArray.
    attrs : dict, optional
        DataArray-level attributes.

    Returns
    -------
    xarray.DataArray
        ConfUSIus fUSI DataArray with native voxel dimensions and CTI-backed physical
        coordinates.

    Raises
    ------
    ValueError
        If dimensions, coordinates, spacing, timing metadata, or fUSI validation fail.
    """
    dims = tuple(str(dim) for dim in dims)
    coords = {} if coords is None else dict(coords)
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

    data_dims = tuple(_SPATIAL_TO_VOXEL.get(dim, dim) for dim in dims)
    if len(set(data_dims)) != len(data_dims):
        raise ValueError(
            "dims must not mix physical and voxel names for the same axis; got "
            f"{dims!r}."
        )

    spacings = {TIME_DIM: dt, "z": dz, "y": dy, "x": dx}
    origins = {TIME_DIM: t0, "z": z0, "y": y0, "x": x0}

    data_coords: dict[str, xr.DataArray] = {}
    spatial_sizes = {voxel_dim: 1 for voxel_dim in VOXEL_DIMS}
    for dim, size in zip(data_dims, shape, strict=True):
        if dim in VOXEL_DIMS:
            spatial_sizes[dim] = int(size)
            continue
        data_coords[dim] = _coordinate_dataarray(
            dim,
            int(size),
            coords=coords,
            spacings=spacings,
            origins=origins,
            volume_acquisition_reference=volume_acquisition_reference,
            volume_acquisition_duration=volume_acquisition_duration,
        )

    voxel_coords: dict[str, xr.DataArray] = {}
    physical_origins: list[float] = []
    physical_spacings: list[float] = []
    physical_attrs: dict[str, dict[str, Any]] = {}
    for physical_dim, voxel_dim in zip(SPATIAL_DIMS, VOXEL_DIMS, strict=True):
        voxel_coord, physical_origin, physical_spacing, attrs_for_physical = (
            _spatial_geometry(
                physical_dim,
                voxel_dim,
                spatial_sizes[voxel_dim],
                coords=coords,
                spacing=spacings[physical_dim],
                origin=origins[physical_dim],
            )
        )
        voxel_coords[voxel_dim] = voxel_coord
        physical_origins.append(physical_origin)
        physical_spacings.append(physical_spacing)
        physical_attrs[physical_dim] = attrs_for_physical

    result = xr.DataArray(
        data,
        dims=data_dims,
        coords={
            **data_coords,
            **{d: voxel_coords[d] for d in data_dims if d in VOXEL_DIMS},
        },
        name=name,
        attrs={} if attrs is None else dict(attrs),
    )

    for dim in VOXEL_DIMS:
        if dim not in result.dims:
            result = result.expand_dims(
                {dim: voxel_coords[dim].values}, axis=len(result.dims)
            )
            result.coords[dim].attrs.update(voxel_coords[dim].attrs)

    voxel_to_physical = np.eye(len(VOXEL_DIMS) + 1, dtype=np.float64)
    voxel_to_physical[:-1, :-1] = np.diag(physical_spacings)
    voxel_to_physical[:-1, -1] = physical_origins
    result = add_physical_coords_from_voxel_affine(
        result,
        voxel_to_physical,
        voxel_dims=VOXEL_DIMS,
        physical_coord_names=SPATIAL_DIMS,
        physical_coord_attrs=physical_attrs,
        force_transform_index=True,
    )

    if canonical_order:
        ordered_core = [dim for dim in CORE_DIMS if dim in result.dims]
        extra_dims = [dim for dim in result.dims if dim not in CORE_DIMS]
        result = result.transpose(*ordered_core, *extra_dims)

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
        Explicit dimension names for each axis of `data`.
    coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray], optional
        Explicit coordinates. See
        [create_fusi_dataarray][confusius.xarray.create_fusi_dataarray].
    dt : float, optional
        Spacing of the `time` coordinate, in seconds.
    dz : float, optional
        Spacing of the `z` physical coordinate, in millimetres.
    dy : float, optional
        Spacing of the `y` physical coordinate, in millimetres.
    dx : float, optional
        Spacing of the `x` physical coordinate, in millimetres.
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
        Ultrasound transmit frequency in hertz.
    beamforming_sound_velocity : float, optional
        Speed of sound assumed during beamforming, in metres per second.
    name : str, default: "iq"
        Name assigned to the resulting DataArray.
    attrs : dict, optional
        Additional DataArray-level attributes.

    Returns
    -------
    xarray.DataArray
        Canonical IQ DataArray with dimensions `(time, k, j, i)`.

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
