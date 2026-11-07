"""ConfUSIus constructor defaults layered on the standalone VoxelData model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import xarray as xr
from voxeldata import create_voxeldata as create_generic_voxeldata
from voxeldata import require_positive_finite

from confusius._dims import TIME_DIM
from confusius._utils.coordinates import get_probe_surface_origin
from confusius.timing import TIMING_REFERENCE_FACTORS, VolumeAcquisitionReference
from confusius.validation.voxeldata import validate_voxeldata

if TYPE_CHECKING:
    import dask.array as da


def _default_per_pose_time_duration(
    per_pose_time: npt.NDArray[np.float64], dt_fallback: float | None
) -> float | None:
    """Infer one pose's own acquisition duration from a `(time, pose)` array.

    `dt` (the shared spacing between successive samples of the *same* pose, i.e. the
    repetition period) is not a safe default: it describes a completely different
    quantity. Instead, default from the smallest nonzero gap between poses' own
    timestamps, per time point -- poses sharing the same timestamp (e.g. a
    stacked-linear-probe pose acquired simultaneously with another) don't count
    toward the gap.

    Parameters
    ----------
    per_pose_time : (n_time, n_pose) numpy.ndarray
        Each pose's own acquisition timestamp, per time point.
    dt_fallback : float, optional
        Value to fall back to when every pose shares the same timestamp at every time
        point, leaving no gap to infer from.

    Returns
    -------
    float or None
        Inferred duration, `dt_fallback`, or `None` when neither is available.
    """
    diffs = np.diff(np.sort(per_pose_time, axis=-1), axis=-1)
    nonzero_diffs = diffs[diffs > 0]
    if nonzero_diffs.size:
        return float(nonzero_diffs.min())
    return dt_fallback


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


def create_voxeldata(
    data: npt.ArrayLike | da.Array,
    *,
    dims: Sequence[str],
    extra_coords: Mapping[str, npt.ArrayLike | xr.DataArray] | None = None,
    time: npt.ArrayLike | xr.DataArray | None = None,
    pose: npt.ArrayLike | xr.DataArray | None = None,
    k: npt.ArrayLike | xr.DataArray | None = None,
    j: npt.ArrayLike | xr.DataArray | None = None,
    i: npt.ArrayLike | xr.DataArray | None = None,
    dt: float | None = None,
    t0: float | npt.ArrayLike = 0.0,
    volume_acquisition_reference: VolumeAcquisitionReference = "start",
    volume_acquisition_duration: float | None = None,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    direction: npt.ArrayLike | None = None,
    voxel_to_world: npt.ArrayLike | None = None,
    geometry_dims: tuple[str, ...] | None = None,
    affines: Mapping[str, npt.ArrayLike] | None = None,
    units: str = "mm",
    name: str | None = None,
    attrs: dict[str, Any] | None = None,
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
        spatial dimensions reordered to trailing `(k, j, i)`. Leading dimensions
        retain their relative order; canonicalization is optional.
    extra_coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray], optional
        Coordinates for non-core dimensions only.
    time : numpy.typing.ArrayLike or xarray.DataArray, optional
        Floating coordinates for the `time` dimension. A named DataArray may depend
        on `time` and any additional nonspatial dimensions, independently of
        geometry. Select all timestamp dependencies before using a shared-time
        analysis. A 2D `(n_time, npose)` array or
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
    k, j, i : numpy.typing.ArrayLike or xarray.DataArray, optional
        Integer native voxel coordinates for the corresponding voxel dimensions. If
        not provided, dense zero-based coordinates are generated.
    dt : float, optional
        Time spacing in seconds, used when `time` is not provided. For multi-pose
        arrays, `dt` is shared across poses.
    t0 : float or numpy.typing.ArrayLike, default: 0.0
        First time coordinate value when `dt` is used. For multi-pose arrays, a 1D
        `t0` with one value per pose generates a pose-dependent `(time, pose)` time
        coordinate using the shared `dt`.
    volume_acquisition_reference : {"start", "center", "end"}, default: "start"
        Time reference stored on generated `time` coordinates.
    volume_acquisition_duration : float, optional
        Acquisition duration stored on generated `time` coordinates. If not provided,
        it defaults to `dt` for a plain 1D `time` coordinate, or to the smallest
        nonzero gap between `t0` values for a pose-dependent `(time, pose)`
        coordinate—`dt` there is the repetition period between successive samples of
        the *same* pose, not one pose's own acquisition time. Poses sharing the same
        `t0` (e.g. a stacked-linear-probe pose acquired simultaneously with another)
        don't count toward the gap.
    spacing : sequence[float], optional
        World spacing in `z/y/x` order. Mutually exclusive with `voxel_to_world`.
    origin : sequence[float], optional
        World origin in `z/y/x` order. If not provided, ConfUSIus probe defaults are
        used.
    direction : numpy.typing.ArrayLike, optional
        3x3 direction matrix in world `z/y/x` row and voxel `k/j/i` column order.
    voxel_to_world : numpy.typing.ArrayLike, optional
        4x4 homogeneous affine in world `z/y/x` row and voxel `k/j/i` column order,
        or a stack indexed by `geometry_dims`. `(npose, 4, 4)` stacks use
        the `pose` dimension. Mutually exclusive with `spacing`, `origin`, and
        `direction`.
    geometry_dims : tuple[str, ...], optional
        Ordered nonspatial dimensions indexing the affine stack. Its shape must
        be `(*[sizes[d] for d in geometry_dims], 4, 4)`. A shared affine uses `()`.
        If not provided, 3D stacks use `("pose",)` when a `pose` dimension is present.
        Timing dependencies are independent of geometry dependencies.
    affines : mapping[str, numpy.typing.ArrayLike], optional
        Named world-frame transforms stored on the VoxelData index.
    units : str, default: "mm"
        Physical unit shared by every derived world coordinate.
    name : str, optional
        DataArray name.
    attrs : dict, optional
        DataArray attributes.

    Returns
    -------
    xarray.DataArray
        VoxelData array with native voxel dimensions and world
        coordinates.

    Raises
    ------
    ValueError
        If `dims` uses world `z`/`y`/`x` names instead of native voxel names, if a
        stacked `voxel_to_world` has missing or incompatible geometry dimensions,
        or if dimensions, coordinates, geometry, timing, or VoxelData validation
        otherwise fail.
    """
    if volume_acquisition_reference not in TIMING_REFERENCE_FACTORS:
        raise ValueError(
            "volume_acquisition_reference must be one of "
            f"{tuple(TIMING_REFERENCE_FACTORS)!r}, got {volume_acquisition_reference!r}."
        )
    if TIME_DIM not in dims and (
        time is not None or volume_acquisition_duration is not None
    ):
        raise ValueError(
            "time and volume_acquisition_duration require a 'time' dimension."
        )

    if voxel_to_world is None and spacing is not None and origin is None:
        if len(spacing) != 3:
            raise ValueError("spacing must have length 3 in z/y/x order.")
        sizes = dict(zip(dims, np.shape(data), strict=True))
        if all(dim in sizes for dim in ("k", "j", "i")):
            origin = get_probe_surface_origin(
                sizes, (spacing[0], spacing[1], spacing[2])
            )

    if dt is not None and t0 is None:
        raise ValueError("t0 must be provided when dt is provided.")

    # Probe acquisition defaults belong here; VoxelData requires named dependencies.
    coords = {} if extra_coords is None else dict(extra_coords)
    if pose is not None:
        coords["pose"] = pose
    if (
        geometry_dims is None
        and voxel_to_world is not None
        and np.ndim(voxel_to_world) == 3
        and "pose" in dims
    ):
        geometry_dims = ("pose",)
    timing_dim = "pose"
    if (
        time is not None
        and np.ndim(time) == 2
        and (not isinstance(time, xr.DataArray) or time.dims == ("dim_0", "dim_1"))
    ):
        if timing_dim not in dims:
            raise ValueError("A 2D time array requires a 'pose' dimension in dims.")
        time = xr.DataArray(
            np.asarray(time),
            dims=(TIME_DIM, timing_dim),
            attrs=dict(time.attrs) if isinstance(time, xr.DataArray) else {},
        )
    if time is None and TIME_DIM in dims and np.ndim(t0) > 0:
        if timing_dim not in dims:
            raise ValueError("A 1D t0 array requires a 'pose' dimension in dims.")
        offsets = np.asarray(t0, dtype=float)
        if offsets.ndim != 1:
            raise ValueError(f"t0 must be scalar or 1D, got shape {offsets.shape}.")
        if offsets.size != np.shape(data)[dims.index(timing_dim)]:
            raise ValueError(
                f"t0 has length {offsets.size}, but the 'pose' dimension size is {np.shape(data)[dims.index(timing_dim)]}."
            )
        if dt is None:
            raise ValueError("Spacing for dimension 'time' is required. Provide dt.")
        step = require_positive_finite(dt, "dt")
        time = xr.DataArray(
            offsets[None, :]
            + np.arange(np.shape(data)[dims.index(TIME_DIM)])[:, None] * step,
            dims=(TIME_DIM, timing_dim),
        )
    generic_t0 = float(np.asarray(t0).item()) if np.ndim(t0) == 0 else 0.0

    # VoxelData accepts Dask arrays, but its ArrayLike annotation excludes them.
    result = create_generic_voxeldata(
        data,  # ty: ignore[invalid-argument-type]
        dims=dims,
        extra_coords=coords,
        time=time,
        k=k,
        j=j,
        i=i,
        dt=dt,
        t0=generic_t0,
        spacing=spacing,
        origin=origin,
        direction=direction,
        voxel_to_world=voxel_to_world,
        geometry_dims=geometry_dims,
        affines=affines,
        units=units,
        name=name,
        attrs=attrs,
    )
    if TIME_DIM in result.dims:
        coord = result.coords[TIME_DIM]
        time_attrs = dict(coord.attrs)
        time_attrs.setdefault("units", "s")
        time_attrs.setdefault(
            "volume_acquisition_reference", volume_acquisition_reference
        )
        if time_attrs["volume_acquisition_reference"] not in TIMING_REFERENCE_FACTORS:
            raise ValueError(
                "volume_acquisition_reference must be one of "
                f"{tuple(TIMING_REFERENCE_FACTORS)!r}, got "
                f"{time_attrs['volume_acquisition_reference']!r}."
            )
        duration = time_attrs.get(
            "volume_acquisition_duration", volume_acquisition_duration
        )
        if duration is None:
            values = np.asarray(coord.values)
            if coord.ndim > 1:
                values = np.asarray(coord.transpose(TIME_DIM, ...).values)
                duration = _default_per_pose_time_duration(
                    values.reshape(coord.sizes[TIME_DIM], -1), dt
                )
            else:
                duration = _regular_step(values) or dt
                if duration is None:
                    duration = _representative_positive_step(values, TIME_DIM)
        if duration is not None:
            time_attrs["volume_acquisition_duration"] = require_positive_finite(
                duration, "volume_acquisition_duration"
            )
        result.coords[TIME_DIM].attrs = time_attrs
    validate_voxeldata(result)
    return result
