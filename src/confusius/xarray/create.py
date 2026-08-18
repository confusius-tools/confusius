"""Constructor helpers for building VoxelData arrays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._dims import CORE_DIMS, POSE_DIM, SPATIAL_DIMS, TIME_DIM, VOXEL_DIMS
from confusius._utils.coordinates import get_probe_surface_origin
from confusius._utils.geometry import attach_voxel_to_world_index
from confusius.timing import TIMING_REFERENCE_FACTORS, VolumeAcquisitionReference
from confusius.validation.fusi import require_positive_finite, validate_voxeldata

_SPATIAL_UNITS = "mm"
"""Physical units attached to the `z`, `y`, and `x` coordinates."""

_TIME_UNITS = "s"
"""Physical units attached to the `time` coordinate."""


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
    return require_positive_finite(spacing, f"Spacing for dimension {dim!r}")


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
    elif dim in spacings:
        step = _require_spacing(dim, spacings[dim])
        origin = origins[dim]
        if origin is None:
            raise ValueError(f"Origin for dimension {dim!r} must be provided.")
        coord_values = origin + np.arange(size) * step
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
            attrs["volume_acquisition_duration"] = require_positive_finite(
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
        require_positive_finite(z, name),
        require_positive_finite(y, name),
        require_positive_finite(x, name),
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


def create_voxeldata(
    data: npt.ArrayLike,
    *,
    dims: Sequence[str],
    time: npt.ArrayLike | xr.DataArray | None = None,
    pose: npt.ArrayLike | xr.DataArray | None = None,
    extra_coords: Mapping[str, npt.ArrayLike | xr.DataArray] | None = None,
    dt: float | None = None,
    t0: float | npt.ArrayLike = 0.0,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    direction: npt.ArrayLike | None = None,
    volume_acquisition_reference: VolumeAcquisitionReference = "start",
    volume_acquisition_duration: float | None = None,
    name: str | None = None,
    attrs: dict[str, Any] | None = None,
    voxel_to_world: npt.ArrayLike | None = None,
    world_coord_attrs: Mapping[str, Mapping[str, Any]] | None = None,
) -> xr.DataArray:
    """Build a VoxelData array from a raw array.

    Parameters
    ----------
    data : numpy.typing.ArrayLike
        Raw array.
    dims : sequence[str]
        Input dimension names. Core dimensions are:

        - `i`/`j`/`k`: native voxel dimensions,
        - `pose`: probe pose dimension,
        - `time`: time dimension.

        Any other extra dimensions are allowed. The returned DataArray will have
        dimensions reordered following the VoxelData model: `(extra_dims, time, pose, k,
        j, i)`.
    time : numpy.typing.ArrayLike or xarray.DataArray, optional
        Floating coordinates for the `time` dimension. A 2D `(n_time, npose)` array or
        DataArray gives each pose its own real timestamps directly (poses acquired
        sequentially rather than simultaneously) rather than a single shared `time`
        axis: there is no single answer for "the" time of a `(pose, k, j, i)` voxel
        any more than there is a single answer for its `z`/`y`/`x` position, so
        `time` requires a scalar `pose` selection first, exactly like world
        coordinates already do. Requires a `pose` dimension in `dims` with a
        matching length. A 2D `time` is not itself an index (xarray dimension
        coordinates must be 1D), so `.sel(time=...)` is unavailable until a pose is
        selected; after that, `.set_xindex("time")` promotes the resulting 1D `time`
        back into a real, selectable index.
    pose : numpy.typing.ArrayLike or xarray.DataArray, optional
        Integer coordinates for the `pose` dimension.
    extra_coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray], optional
        Coordinates for non-core dimensions only.
    dt : float, optional
        Time spacing in seconds, used when `time` is not provided. For multi-pose
        arrays, `dt` is shared across poses.
    t0 : float or numpy.typing.ArrayLike, default: 0.0
        First time coordinate value when `dt` is used. For multi-pose arrays, a 1D
        `t0` with one value per pose generates a pose-dependent `(time, pose)` time
        coordinate using the shared `dt`.
    spacing : sequence[float], optional
        World spacing in `z/y/x` order. Mutually exclusive with `voxel_to_world`.
    origin : sequence[float], optional
        World origin in `z/y/x` order. If not provided, ConfUSIus probe defaults are
        used.
    direction : numpy.typing.ArrayLike, optional
        3x3 direction matrix in world `z/y/x` row and voxel `k/j/i` column order.
    volume_acquisition_reference : {"start", "center", "end"}, default: "start"
        Time reference stored on generated `time` coordinates.
    volume_acquisition_duration : float, optional
        Acquisition duration stored on generated `time` coordinates.
    name : str, optional
        DataArray name.
    attrs : dict, optional
        DataArray attributes.
    voxel_to_world : numpy.typing.ArrayLike, optional
        4x4 homogeneous affine in world `z/y/x` row and voxel `k/j/i` column order,
        or an `(npose, 4, 4)` stack of one such affine per pose. A stack requires a
        `pose` dimension in `dims` with a matching length, and is mutually exclusive
        with `spacing`/`origin`/`direction` — per-pose geometry can only be supplied
        this way, there is no parallel per-pose `spacing`/`origin`/`direction` API.
    world_coord_attrs : mapping[str, mapping[str, Any]], optional
        Attributes to merge onto the derived world coordinates, keyed by world
        coordinate name (`z`/`y`/`x`). Overrides the auto-computed `units` entry for
        any key present in the given mapping; other auto-computed entries are kept.

    Returns
    -------
    xarray.DataArray
        VoxelData array with native voxel dimensions and world
        coordinates.

    Raises
    ------
    ValueError
        If `dims` uses world `z`/`y`/`x` names instead of native voxel names, if a
        pose-stacked `voxel_to_world` is given without a matching `pose` dimension,
        or if dimensions, coordinates, geometry, timing, or VoxelData validation otherwise
        fail.
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

    per_pose_time: npt.NDArray[np.float64] | None = None
    per_pose_time_attrs: dict[str, Any] = {}
    if time is not None:
        time_array = time.values if isinstance(time, xr.DataArray) else np.asarray(time)
        if time_array.ndim == 2:
            if POSE_DIM not in dims:
                raise ValueError(
                    "A 2D time array (one column per pose) requires a 'pose' "
                    "dimension in dims."
                )
            pose_size = shape[dims.index(POSE_DIM)]
            if time_array.shape[1] != pose_size:
                raise ValueError(
                    f"time has {time_array.shape[1]} pose columns, but the 'pose' "
                    f"dimension size is {pose_size}."
                )
            per_pose_time = np.asarray(time_array, dtype=np.float64)
            per_pose_time_attrs = (
                dict(time.attrs) if isinstance(time, xr.DataArray) else {}
            )
            # 1D placeholder: satisfies construction below, replaced at the end.
            time = time_array[:, 0]
    elif TIME_DIM in dims:
        t0_array = np.asarray(t0)
        if t0_array.ndim > 0:
            if POSE_DIM not in dims:
                raise ValueError(
                    "A 1D t0 array (one value per pose) requires a 'pose' "
                    "dimension in dims."
                )
            if t0_array.ndim != 1:
                raise ValueError(
                    f"t0 must be scalar or 1D, got shape {t0_array.shape}."
                )
            pose_size = shape[dims.index(POSE_DIM)]
            if t0_array.size != pose_size:
                raise ValueError(
                    f"t0 has length {t0_array.size}, but the 'pose' dimension size "
                    f"is {pose_size}."
                )
            time_size = shape[dims.index(TIME_DIM)]
            dt_value = _require_spacing(TIME_DIM, dt)
            t0_values = np.asarray(t0_array, dtype=np.float64)
            if not np.all(np.isfinite(t0_values)):
                raise ValueError("t0 must contain finite values.")
            per_pose_time = (
                t0_values[None, :]
                + np.arange(time_size, dtype=np.float64)[:, None] * dt_value
            )
            per_pose_time_attrs = {
                "units": _TIME_UNITS,
                "volume_acquisition_reference": volume_acquisition_reference,
                "volume_acquisition_duration": dt_value,
            }
            time = per_pose_time[:, 0]

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

    scalar_t0: float | None
    if t0 is None or np.asarray(t0).ndim > 0:
        scalar_t0 = None
    else:
        scalar_t0 = float(np.asarray(t0).item())

    data_coords: dict[str, xr.DataArray] = {}
    spacings = {TIME_DIM: dt}
    origins = {TIME_DIM: scalar_t0}
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

    voxel_to_world_array = (
        None if voxel_to_world is None else np.asarray(voxel_to_world, dtype=np.float64)
    )
    is_pose_stacked = (
        voxel_to_world_array is not None and voxel_to_world_array.ndim == 3
    )
    if is_pose_stacked:
        if spacing is not None or origin is not None or direction is not None:
            raise ValueError(
                "voxel_to_world is mutually exclusive with spacing, origin, and "
                "direction."
            )
        if POSE_DIM not in dims:
            raise ValueError(
                "A pose-stacked voxel_to_world (one affine per pose) requires a "
                f"{POSE_DIM!r} dimension in dims."
            )
        if voxel_to_world_array.shape[1:] != (4, 4):
            raise ValueError(
                "voxel_to_world must have shape (npose, 4, 4), got "
                f"{voxel_to_world_array.shape}."
            )
        if not np.allclose(voxel_to_world_array[:, -1], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("Each pose affine must be a homogeneous affine.")
        pose_size = shape[dims.index(POSE_DIM)]
        if voxel_to_world_array.shape[0] != pose_size:
            raise ValueError(
                f"voxel_to_world pose stack length {voxel_to_world_array.shape[0]} "
                f"does not match the {POSE_DIM!r} dimension size {pose_size}."
            )
        resolved_voxel_to_world = voxel_to_world_array
    else:
        resolved_voxel_to_world = _resolve_voxel_to_world(
            spatial_sizes=spatial_sizes,
            spacing=spacing,
            origin=origin,
            direction=direction,
            voxel_to_world=voxel_to_world,
        )
    voxel_coords = {
        dim: xr.DataArray(np.arange(spatial_sizes[dim]), dims=(dim,))
        for dim in VOXEL_DIMS
        if dim in data_dims
    }
    world_attrs = {dim: {"units": _SPATIAL_UNITS} for dim in SPATIAL_DIMS}

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
    if is_pose_stacked:
        npose = resolved_voxel_to_world.shape[0]
        index_affine = np.stack(
            [np.eye(len(present_voxel_dims) + 1, dtype=np.float64)] * npose
        )
        for pose_index in range(npose):
            index_affine[pose_index, :-1, :-1] = resolved_voxel_to_world[
                pose_index, :3, :3
            ][np.ix_(present_indices, present_indices)]
            index_affine[pose_index, :-1, -1] = resolved_voxel_to_world[
                pose_index, :3, -1
            ][present_indices]
    else:
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
    validate_voxeldata(
        result,
        require_regular_spacing=True,
        regular_spacing_dims=regular_spacing_dims,
        require_canonical_dim_order=True,
    )
    if per_pose_time is not None:
        result = result.drop_vars(TIME_DIM).assign_coords(
            {TIME_DIM: ((TIME_DIM, POSE_DIM), per_pose_time, per_pose_time_attrs)}
        )
    return result
