"""Constructor helpers for building ConfUSIus VoxelData."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, SupportsFloat, SupportsIndex

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._dims import CORE_DIMS, SPATIAL_DIMS, TIME_DIM, VOXEL_DIMS
from confusius._utils.coordinates import get_probe_surface_origin
from confusius._utils.geometry import attach_voxel_to_world_index
from confusius.timing import TIMING_REFERENCE_FACTORS, VolumeAcquisitionReference
from confusius.validation import validate_fusi, validate_iq

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
    origins: Mapping[str, float | None],
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
    origins : mapping[str, float or None]
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
    elif dim in spacings and not (
        dim == TIME_DIM and size == 1 and spacings[dim] is None
    ):
        step = _require_spacing(dim, spacings[dim])
        origin = origins[dim]
        if origin is None:
            raise ValueError(f"Origin for dimension {dim!r} must be provided.")
        coord_values = origin + np.arange(size) * step
        attrs = {}
    elif dim == TIME_DIM and size == 1:
        coord_values = np.array([origins[dim]])
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
        if duration is not None:
            attrs["volume_acquisition_duration"] = _require_positive_finite(
                duration, "volume_acquisition_duration"
            )
        elif step is not None:
            attrs["volume_acquisition_duration"] = _require_spacing(dim, step)

    return xr.DataArray(coord_values, dims=(dim,), attrs=attrs)


def _validate_spatial_tuple(
    values: Sequence[float] | None, *, name: str
) -> tuple[float, float, float]:
    """Validate a positive finite `z/y/x` tuple argument.

    Parameters
    ----------
    values : sequence[float], optional
        Candidate tuple values in `z/y/x` order.
    name : str
        Argument name used in validation errors.

    Returns
    -------
    tuple[float, float, float]
        Validated tuple.

    Raises
    ------
    ValueError
        If values are missing when required, have the wrong length, or are not positive
        and finite.
    """
    if values is None:
        raise ValueError(f"{name} must be provided.")
    if len(values) != len(SPATIAL_DIMS):
        raise ValueError(f"{name} must have length 3 in z/y/x order.")
    z, y, x = values
    return (
        _require_positive_finite(z, name),
        _require_positive_finite(y, name),
        _require_positive_finite(x, name),
    )


def _resolve_voxel_to_world(
    *,
    spatial_sizes: Mapping[str, int],
    spacing: Sequence[float] | None,
    origin: Sequence[float] | None,
    direction: npt.ArrayLike | None,
    voxel_to_world: npt.ArrayLike | None,
) -> npt.NDArray[np.float64]:
    """Resolve constructor geometry to one canonical 4x4 affine.

    Parameters
    ----------
    spatial_sizes : mapping[str, int]
        Spatial voxel sizes keyed by native `k/j/i` dimension name.
    spacing : sequence[float], optional
        World spacing in `z/y/x` order.
    origin : sequence[float], optional
        World origin in `z/y/x` order. If not provided, ConfUSIus probe defaults are
        used.
    direction : numpy.typing.ArrayLike, optional
        3x3 world direction matrix in `z/y/x` row and `k/j/i` column order.
    voxel_to_world : numpy.typing.ArrayLike, optional
        4x4 homogeneous affine in `z/y/x` row and `k/j/i` column order.

    Returns
    -------
    (4, 4) numpy.ndarray
        Homogeneous voxel-to-world affine.

    Raises
    ------
    ValueError
        If geometry inputs are ambiguous or invalid.
    """
    if voxel_to_world is not None:
        if spacing is not None or origin is not None or direction is not None:
            raise ValueError(
                "voxel_to_world is mutually exclusive with spacing, origin, and "
                "direction."
            )
        affine = np.asarray(voxel_to_world, dtype=np.float64)
        if affine.shape != (4, 4):
            raise ValueError(
                f"voxel_to_world must have shape (4, 4), got {affine.shape}."
            )
        if not np.allclose(affine[-1], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("voxel_to_world must be a homogeneous affine.")
        return affine

    resolved_spacing = _validate_spatial_tuple(spacing, name="spacing")
    if origin is None:
        resolved_origin = get_probe_surface_origin(spatial_sizes, resolved_spacing)
    else:
        if len(origin) != len(SPATIAL_DIMS):
            raise ValueError("origin must have length 3 in z/y/x order.")
        resolved_origin = tuple(float(value) for value in origin)
        if not np.all(np.isfinite(resolved_origin)):
            raise ValueError("origin must contain finite values.")

    resolved_direction = (
        np.eye(3, dtype=np.float64)
        if direction is None
        else np.asarray(direction, dtype=np.float64)
    )
    if resolved_direction.shape != (3, 3):
        raise ValueError(
            f"direction must have shape (3, 3), got {resolved_direction.shape}."
        )
    if not np.all(np.isfinite(resolved_direction)):
        raise ValueError("direction must contain finite values.")

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = resolved_direction @ np.diag(resolved_spacing)
    affine[:3, 3] = resolved_origin
    return affine


def create_fusi_dataarray(
    data: npt.ArrayLike,
    *,
    dims: Sequence[str],
    time: npt.ArrayLike | xr.DataArray | None = None,
    pose: npt.ArrayLike | xr.DataArray | None = None,
    extra_coords: Mapping[str, npt.ArrayLike | xr.DataArray] | None = None,
    dt: float | None = None,
    t0: float = 0.0,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    direction: npt.ArrayLike | None = None,
    voxdim: Sequence[float] | None = None,
    volume_acquisition_reference: VolumeAcquisitionReference = "start",
    volume_acquisition_duration: float | None = None,
    name: str | None = None,
    attrs: dict[str, Any] | None = None,
    voxel_to_world: npt.ArrayLike | None = None,
    world_coord_attrs: Mapping[str, Mapping[str, Any]] | None = None,
) -> xr.DataArray:
    """Build a VoxelData-compatible ConfUSIus fUSI DataArray from a raw array.

    Parameters
    ----------
    data : numpy.typing.ArrayLike
        Raw array whose rank matches `dims`.
    dims : sequence[str]
        Input dimension names. Spatial dimensions must be native voxel names
        (`k`/`j`/`i`). The returned DataArray uses native voxel dimensions in canonical
        core order.
    time : numpy.typing.ArrayLike or xarray.DataArray, optional
        Coordinate for the `time` dimension.
    pose : numpy.typing.ArrayLike or xarray.DataArray, optional
        Coordinate for the `pose` dimension.
    extra_coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray], optional
        Coordinates for non-core dimensions only.
    dt : float, optional
        Time spacing in seconds, used when `time` is not provided.
    t0 : float, default: 0.0
        First time coordinate value when `dt` is used.
    spacing : sequence[float], optional
        World spacing in `z/y/x` order. Mutually exclusive with `voxel_to_world`.
    origin : sequence[float], optional
        World origin in `z/y/x` order. If not provided, ConfUSIus probe defaults are
        used.
    direction : numpy.typing.ArrayLike, optional
        3x3 direction matrix in world `z/y/x` row and voxel `k/j/i` column order.
    voxdim : sequence[float], optional
        `voxdim` metadata in `z/y/x` order. If not provided, affine column norms are
        used.
    volume_acquisition_reference : {"start", "center", "end"}, default: "start"
        Time reference stored on generated `time` coordinates.
    volume_acquisition_duration : float, optional
        Acquisition duration stored on generated `time` coordinates.
    name : str, optional
        DataArray name.
    attrs : dict, optional
        DataArray attributes.
    voxel_to_world : numpy.typing.ArrayLike, optional
        4x4 homogeneous affine in world `z/y/x` row and voxel `k/j/i` column order.
    world_coord_attrs : mapping[str, mapping[str, Any]], optional
        Attributes to merge onto the derived world coordinates, keyed by world
        coordinate name (`z`/`y`/`x`). Overrides the auto-computed `units`/`voxdim`
        entries for any key present in the given mapping; other auto-computed entries
        are kept.

    Returns
    -------
    xarray.DataArray
        VoxelData-compatible fUSI DataArray with native voxel dimensions and world
        coordinates.

    Raises
    ------
    ValueError
        If `dims` uses world `z`/`y`/`x` names instead of native voxel names, or if
        dimensions, coordinates, geometry, timing, or fUSI validation otherwise fail.
    """
    dims = tuple(str(dim) for dim in dims)
    shape = np.shape(data)
    extra_coords = {} if extra_coords is None else dict(extra_coords)

    if len(set(dims)) != len(dims):
        raise ValueError(f"dims must not contain duplicate names, got {dims!r}.")
    if len(dims) != len(shape):
        raise ValueError(
            f"Length of dims {dims!r} ({len(dims)}) must match the number of array "
            f"dimensions ({len(shape)})."
        )
    invalid_spatial = sorted(set(dims) & set(SPATIAL_DIMS))
    if invalid_spatial:
        raise ValueError(
            f"dims must use native voxel names {VOXEL_DIMS!r}, not world coordinate "
            f"names {invalid_spatial!r}. World z/y/x coordinates are always derived "
            "from the voxel-to-world index, never passed as dims."
        )
    if volume_acquisition_reference not in TIMING_REFERENCE_FACTORS:
        raise ValueError(
            f"volume_acquisition_reference must be one of "
            f"{tuple(TIMING_REFERENCE_FACTORS)!r}, got {volume_acquisition_reference!r}."
        )
    if TIME_DIM not in dims and (
        volume_acquisition_duration is not None or time is not None
    ):
        raise ValueError(
            "time and volume_acquisition_duration require a 'time' dimension."
        )

    forbidden_extra = set(CORE_DIMS) | set(SPATIAL_DIMS)
    overlap = sorted(forbidden_extra & set(extra_coords))
    if overlap:
        raise ValueError(
            f"extra_coords must not include core coordinates: {overlap!r}."
        )

    data_array: Any = data if hasattr(data, "shape") else np.asarray(data)
    data_dims = dims
    for dim in VOXEL_DIMS:
        if dim not in data_dims:
            data_array = np.expand_dims(data_array, axis=-1)
            data_dims = (*data_dims, dim)

    spatial_sizes = {voxel_dim: 1 for voxel_dim in VOXEL_DIMS}
    for dim, size in zip(data_dims, data_array.shape, strict=True):
        if dim in VOXEL_DIMS:
            spatial_sizes[dim] = int(size)

    coord_inputs: dict[str, npt.ArrayLike | xr.DataArray] = dict(extra_coords)
    if time is not None:
        coord_inputs[TIME_DIM] = time
    if pose is not None:
        coord_inputs["pose"] = pose

    data_coords: dict[str, xr.DataArray] = {}
    spacings = {TIME_DIM: dt}
    origins = {TIME_DIM: t0}
    for dim, size in zip(data_dims, data_array.shape, strict=True):
        if dim in VOXEL_DIMS:
            continue
        data_coords[dim] = _coordinate_dataarray(
            dim,
            int(size),
            coords=coord_inputs,
            spacings=spacings,
            origins=origins,
            volume_acquisition_reference=volume_acquisition_reference,
            volume_acquisition_duration=volume_acquisition_duration,
        )

    resolved_voxel_to_world = _resolve_voxel_to_world(
        spatial_sizes=spatial_sizes,
        spacing=spacing,
        origin=origin,
        direction=direction,
        voxel_to_world=voxel_to_world,
    )
    if voxdim is None:
        resolved_voxdim = tuple(
            _require_positive_finite(value, f"voxdim for dimension {dim!r}")
            for dim, value in zip(
                SPATIAL_DIMS,
                np.linalg.norm(resolved_voxel_to_world[:3, :3], axis=0),
                strict=True,
            )
        )
    else:
        resolved_voxdim = _validate_spatial_tuple(voxdim, name="voxdim")

    voxel_coords = {
        dim: xr.DataArray(np.arange(spatial_sizes[dim], dtype=float), dims=(dim,))
        for dim in VOXEL_DIMS
        if dim in data_dims
    }
    world_attrs = {
        dim: {"units": _SPATIAL_UNITS, "voxdim": value}
        for dim, value in zip(SPATIAL_DIMS, resolved_voxdim, strict=True)
    }

    result = xr.DataArray(
        data_array,
        dims=data_dims,
        coords={
            **data_coords,
            **{dim: voxel_coords[dim] for dim in data_dims if dim in VOXEL_DIMS},
        },
        name=name,
        attrs={} if attrs is None else dict(attrs),
    )

    present_voxel_dims = tuple(dim for dim in VOXEL_DIMS if dim in result.dims)
    present_world_names = tuple(
        world
        for voxel, world in zip(VOXEL_DIMS, SPATIAL_DIMS, strict=True)
        if voxel in present_voxel_dims
    )
    present_indices = [VOXEL_DIMS.index(dim) for dim in present_voxel_dims]
    index_affine = np.eye(len(present_voxel_dims) + 1, dtype=np.float64)
    index_affine[:-1, :-1] = resolved_voxel_to_world[:3, :3][
        np.ix_(present_indices, present_indices)
    ]
    index_affine[:-1, -1] = resolved_voxel_to_world[:3, -1][present_indices]

    result = attach_voxel_to_world_index(
        result,
        index_affine,
        world_coord_attrs={
            name: {**world_attrs[name], **(world_coord_attrs or {}).get(name, {})}
            for name in present_world_names
        },
    )

    extra_dims = [dim for dim in result.dims if dim not in CORE_DIMS]
    ordered_core = [dim for dim in CORE_DIMS if dim in result.dims]
    result = result.transpose(*extra_dims, *ordered_core)

    regular_spacing_dims = tuple(
        dim
        for dim in CORE_DIMS
        if dim in result.dims
        and result.sizes[dim] > 1
        and not (dim == TIME_DIM and time is not None)
    )
    validate_fusi(
        result,
        require_regular_spacing=True,
        regular_spacing_dims=regular_spacing_dims,
        require_canonical_dim_order=True,
    )
    return result


def create_iq_dataarray(
    data: npt.ArrayLike,
    *,
    dims: Sequence[str],
    time: npt.ArrayLike | xr.DataArray | None = None,
    pose: npt.ArrayLike | xr.DataArray | None = None,
    extra_coords: Mapping[str, npt.ArrayLike | xr.DataArray] | None = None,
    dt: float | None = None,
    t0: float = 0.0,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    direction: npt.ArrayLike | None = None,
    voxdim: Sequence[float] | None = None,
    volume_acquisition_reference: VolumeAcquisitionReference = "start",
    volume_acquisition_duration: float | None = None,
    transmit_frequency: float | None = None,
    beamforming_sound_velocity: float | None = None,
    name: str | None = "iq",
    attrs: dict[str, Any] | None = None,
    voxel_to_world: npt.ArrayLike | None = None,
    world_coord_attrs: Mapping[str, Mapping[str, Any]] | None = None,
) -> xr.DataArray:
    """Build a VoxelData-compatible ConfUSIus IQ DataArray from a raw complex array.

    Parameters
    ----------
    data : numpy.typing.ArrayLike
        Raw complex IQ array whose rank matches `dims`.
    dims : sequence[str]
        Input dimension names. Spatial dimensions must be native voxel names
        (`k`/`j`/`i`).
    time : numpy.typing.ArrayLike or xarray.DataArray, optional
        Coordinate for the `time` dimension.
    pose : numpy.typing.ArrayLike or xarray.DataArray, optional
        Coordinate for the `pose` dimension.
    extra_coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray], optional
        Coordinates for non-core dimensions only.
    dt : float, optional
        Time spacing in seconds, used when `time` is not provided.
    t0 : float, default: 0.0
        First time coordinate value when `dt` is used.
    spacing : sequence[float], optional
        World spacing in `z/y/x` order. Mutually exclusive with `voxel_to_world`.
    origin : sequence[float], optional
        World origin in `z/y/x` order.
    direction : numpy.typing.ArrayLike, optional
        3x3 direction matrix in world `z/y/x` row and voxel `k/j/i` column order.
    voxdim : sequence[float], optional
        `voxdim` metadata in `z/y/x` order.
    volume_acquisition_reference : {"start", "center", "end"}, default: "start"
        Time reference stored on generated `time` coordinates.
    volume_acquisition_duration : float, optional
        Acquisition duration stored on generated `time` coordinates.
    transmit_frequency : float, optional
        Ultrasound transmit frequency in hertz.
    beamforming_sound_velocity : float, optional
        Speed of sound assumed during beamforming, in metres per second.
    name : str, default: "iq"
        DataArray name.
    attrs : dict, optional
        DataArray attributes.
    voxel_to_world : numpy.typing.ArrayLike, optional
        4x4 homogeneous affine in world `z/y/x` row and voxel `k/j/i` column order.
    world_coord_attrs : mapping[str, mapping[str, Any]], optional
        Attributes to merge onto the derived world coordinates, keyed by world
        coordinate name (`z`/`y`/`x`). Overrides the auto-computed `units`/`voxdim`
        entries for any key present in the given mapping; other auto-computed entries
        are kept.

    Returns
    -------
    xarray.DataArray
        VoxelData-compatible IQ DataArray with native voxel dimensions and world
        coordinates.

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
        time=time,
        pose=pose,
        extra_coords=extra_coords,
        dt=dt,
        t0=t0,
        spacing=spacing,
        origin=origin,
        direction=direction,
        voxdim=voxdim,
        volume_acquisition_reference=volume_acquisition_reference,
        volume_acquisition_duration=volume_acquisition_duration,
        name=name,
        attrs=attrs,
        voxel_to_world=voxel_to_world,
        world_coord_attrs=world_coord_attrs,
    )
    validate_iq(result)
    return result
