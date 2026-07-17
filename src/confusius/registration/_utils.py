"""Internal utilities shared by registration modules."""

import os
import signal
import threading
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from types import FrameType
from typing import TYPE_CHECKING, Generator, TypeGuard

import numpy as np
import xarray as xr

from confusius._utils.coordinates import get_coordinate_spacing_info
from confusius._utils.geometry import (
    get_voxel_affine_physical_coord_names,
    get_voxel_affine_spatial_dims,
    has_voxel_affine_geometry,
    restore_physical_coords_from_voxel_affine,
)

if TYPE_CHECKING:
    from threading import Event

    import SimpleITK as sitk


def _raise_undefined_spatial_spacing_error(undefined_dims: list[str]) -> None:
    """Raise a consistent registration spacing error.

    Parameters
    ----------
    undefined_dims : list[str]
        Spatial dimensions whose spacing could not be determined.

    Raises
    ------
    ValueError
        Always raised with a message explaining how to repair singleton axes.
    """
    raise ValueError(
        "Registration requires defined spatial spacing for all spatial "
        f"dimensions, but {undefined_dims!r} are undefined. For singleton spatial "
        "axes, set a `voxdim` coordinate attribute (e.g. `da.coords['z'].attrs['voxdim'] = 0.5`) "
        "before registering."
    )


def get_defined_spatial_spacing(da: xr.DataArray) -> tuple[list[str], list[float]]:
    """Return spatial dims and their defined spacings for registration.

    Parameters
    ----------
    da : xarray.DataArray
        Spatial or spatiotemporal DataArray.

    Returns
    -------
    spatial_dims : list[str]
        Spatial dimension names in DataArray order.
    spacing : list[float]
        Physical spacing for each spatial dimension.

    Raises
    ------
    ValueError
        If any spatial spacing is undefined.
    """
    spatial_dims = [str(dim) for dim in da.dims if str(dim) != "time"]

    if has_voxel_affine_geometry(da):
        spacing_dict = da.fusi.spacing
        undefined_dims = [dim for dim in spatial_dims if spacing_dict.get(dim) is None]
        if undefined_dims:
            _raise_undefined_spatial_spacing_error(undefined_dims)
        return spatial_dims, [float(spacing_dict[dim]) for dim in spatial_dims]

    spacing: list[float] = []
    undefined_dims: list[str] = []
    for dim in spatial_dims:
        spacing_info = get_coordinate_spacing_info(dim, da, uniformity_tolerance=1e-2)
        if spacing_info.value is None:
            undefined_dims.append(dim)
        else:
            spacing.append(float(spacing_info.value))
    if undefined_dims:
        _raise_undefined_spatial_spacing_error(undefined_dims)

    return spatial_dims, spacing


def _rotation_matrix_aligning_vectors(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Return the minimal rotation that maps one unit vector onto another.

    Parameters
    ----------
    source : (N,) numpy.ndarray
        Source unit vector.
    target : (N,) numpy.ndarray
        Target unit vector.

    Returns
    -------
    (N, N) numpy.ndarray
        Proper rotation matrix satisfying `R @ source == target` up to numerical
        precision.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)

    cross = np.cross(source, target)
    sin_theta = np.linalg.norm(cross)
    cos_theta = float(np.dot(source, target))

    if np.isclose(sin_theta, 0.0):
        if cos_theta > 0.0:
            return np.eye(source.size, dtype=np.float64)

        helper = np.eye(source.size, dtype=np.float64)[np.argmin(np.abs(source))]
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(source.size, dtype=np.float64) + 2.0 * (skew @ skew)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(source.size, dtype=np.float64)
        + skew
        + skew @ skew * ((1.0 - cos_theta) / (sin_theta**2))
    )


def _voxel_affine_plane_center(data: xr.DataArray) -> np.ndarray:
    """Return the physical center point of a voxel-affine slab."""
    return np.array(
        [
            float(np.asarray(data.coords[name].values).mean())
            for name in get_voxel_affine_physical_coord_names(data)
        ],
        dtype=np.float64,
    )


def _voxel_affine_slice_normal(data: xr.DataArray) -> np.ndarray:
    """Return the physical-space normal of a singleton voxel-affine slab."""
    voxel_dims = get_voxel_affine_spatial_dims(data)
    singleton_axes = [i for i, dim in enumerate(voxel_dims) if data.sizes[dim] == 1]
    if len(singleton_axes) != 1:
        raise ValueError(
            "Voxel-affine plane initialization requires exactly one singleton "
            f"spatial dimension, got sizes {[data.sizes[dim] for dim in voxel_dims]!r}."
        )
    return np.asarray(data.fusi.direction, dtype=np.float64)[:, singleton_axes[0]]


def build_voxel_affine_plane_initial_transform(
    fixed: xr.DataArray,
    moving: xr.DataArray,
) -> np.ndarray:
    """Build a rigid fixed-to-moving initializer for thin voxel-affine slabs.

    Parameters
    ----------
    fixed : xarray.DataArray
        Fixed voxel-affine slab.
    moving : xarray.DataArray
        Moving voxel-affine slab.

    Returns
    -------
    (4, 4) numpy.ndarray
        Rigid transform in physical space mapping fixed coordinates into moving
        coordinates.

    Raises
    ------
    ValueError
        If either input is not a 3D voxel-affine slab with exactly one singleton
        spatial dimension.
    """
    if not has_voxel_affine_geometry(fixed) or not has_voxel_affine_geometry(moving):
        raise ValueError(
            "Voxel-affine plane initialization requires voxel-affine geometry on "
            "both fixed and moving data."
        )

    fixed_dims = get_voxel_affine_spatial_dims(fixed)
    moving_dims = get_voxel_affine_spatial_dims(moving)
    if fixed_dims != moving_dims or len(fixed_dims) != 3:
        raise ValueError(
            "Voxel-affine plane initialization requires matching 3D voxel-affine "
            f"dimensions, got fixed={fixed_dims!r} and moving={moving_dims!r}."
        )

    fixed_normal = _voxel_affine_slice_normal(fixed)
    moving_normal = _voxel_affine_slice_normal(moving)
    rotation = _rotation_matrix_aligning_vectors(fixed_normal, moving_normal)

    fixed_center = _voxel_affine_plane_center(fixed)
    moving_center = _voxel_affine_plane_center(moving)
    translation = moving_center - rotation @ fixed_center

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


SignalHandler = Callable[[int, FrameType | None], object]
"""Python-level SIGINT handler callable."""


def _is_python_signal_handler(handler: object) -> TypeGuard[SignalHandler]:
    """Return whether `handler` is a callable Python SIGINT handler."""
    return callable(handler)


def replace_affines_attr(result: xr.DataArray, reference: xr.DataArray) -> None:
    """Replace `result.attrs["affines"]` with affines from a reference array.

    Parameters
    ----------
    result : xarray.DataArray
        DataArray whose affine metadata should be updated in place.
    reference : xarray.DataArray
        DataArray providing the physical-to-reference affines for the output grid.

    Notes
    -----
    If `reference` does not define `attrs["affines"]`, any existing affines on
    `result` are removed. This is appropriate for resampled outputs, whose affine
    metadata should match the grid they now live on rather than the source grid they
    were sampled from.
    """
    if "affines" in reference.attrs:
        result.attrs["affines"] = deepcopy(reference.attrs["affines"])
    else:
        result.attrs.pop("affines", None)


def replace_spatial_geometry_attrs(
    result: xr.DataArray,
    reference: xr.DataArray,
) -> xr.DataArray:
    """Replace spatial geometry metadata on `result` with `reference` geometry.

    Parameters
    ----------
    result : xarray.DataArray
        DataArray whose spatial geometry metadata should be updated.
    reference : xarray.DataArray
        DataArray providing the output spatial geometry.

    Returns
    -------
    xarray.DataArray
        `result` with spatial geometry metadata synchronized to `reference`.

    Notes
    -----
    This copies both `attrs["affines"]` and the canonical voxel-affine geometry
    attribute `attrs["voxel_to_physical"]` when present on `reference`. If the
    reference uses voxel-affine geometry, lazy CTI-backed physical coordinates are
    rebuilt on the returned DataArray so its coordinate index matches the copied
    metadata.
    """
    replace_affines_attr(result, reference)

    if "voxel_to_physical" in reference.attrs:
        result.attrs["voxel_to_physical"] = np.asarray(
            reference.attrs["voxel_to_physical"], dtype=np.float64
        )
        return restore_physical_coords_from_voxel_affine(result)

    result.attrs.pop("voxel_to_physical", None)
    return result


@contextmanager
def set_sitk_thread_count(n: int) -> Generator[None, None, None]:
    """Temporarily override SimpleITK's global thread count.

    Follows joblib's `n_jobs` sign convention: positive values are used
    directly; negative values are interpreted as `max(1, n_cpus + 1 + n)`,
    so `-1` means all CPUs, `-2` means all minus one, and so on.

    Saves the current value on entry and restores it on exit, even if an
    exception is raised inside the `with` block.

    Parameters
    ----------
    n : int
        Desired number of threads, following joblib's `n_jobs` convention.

    Yields
    ------
    None
         This is a context manager that does not yield any value; it only manages the
         thread count.
    """
    import SimpleITK as sitk

    if n < 0:
        n = max(1, (os.cpu_count() or 1) + 1 + n)

    prev = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(n)
    try:
        yield
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(prev)


@contextmanager
def abort_on_sigint(
    abort_event: "Event | None",
) -> Generator["Event", None, None]:
    """Return an abort event that is set cooperatively on the first Ctrl+C.

    Parameters
    ----------
    abort_event : threading.Event or None
        Existing cooperative-cancellation event to reuse. If not provided, a
        new event is created for the duration of the context.

    Yields
    ------
    threading.Event
        Event that is set when cooperative cancellation is requested, either
        explicitly by the caller or via a Ctrl+C signal handled on the main
        thread.

    Notes
    -----
    On the main thread, the first `SIGINT`/Ctrl+C is converted into
    `abort_event.set()` so long-running registrations can stop cleanly at the
    next SimpleITK iteration boundary and return their current partial result.
    A second Ctrl+C falls back to the previous signal handler so users can
    still force an immediate interrupt if graceful cancellation stalls.
    """
    shared_abort_event = abort_event or threading.Event()

    if threading.current_thread() is not threading.main_thread():
        yield shared_abort_event
        return

    previous_handler = signal.getsignal(signal.SIGINT)
    saw_sigint = False

    def _handle_sigint(signum: int, frame: FrameType | None) -> None:
        nonlocal saw_sigint
        if not saw_sigint:
            saw_sigint = True
            shared_abort_event.set()
            return

        if previous_handler in {signal.SIG_DFL, signal.default_int_handler}:
            raise KeyboardInterrupt
        if previous_handler == signal.SIG_IGN:
            return
        if _is_python_signal_handler(previous_handler):
            previous_handler(signum, frame)

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        yield shared_abort_event
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def dataarray_to_sitk_image(da: xr.DataArray) -> "sitk.Image":
    """Convert a spatial or spatiotemporal DataArray to a SimpleITK image.

    Uses the transpose convention: `da.values.T` is passed to `GetImageFromArray`,
    so that the first DataArray axis maps to SimpleITK's physical x-axis. For data
    with a time dimension, the time dimension is converted to a vector image channel
    dimension.

    Parameters
    ----------
    da : xarray.DataArray
        2D or 3D spatial DataArray, or 2D+t or 3D+t DataArray with a time dimension.
        Spacing and origin are derived from its coordinates; missing coordinates warn
        and fall back to spacing `1.0` and origin `0.0`.

    Returns
    -------
    SimpleITK.Image
        SimpleITK image with spacing and origin set from the DataArray coordinates.
        For `time`-stacked input, returns a vector image where time is the vector
        dimension.
    """
    import SimpleITK as sitk

    spatial_dims, spacing = get_defined_spatial_spacing(da)
    origin_dict = da.fusi.origin

    has_time = "time" in da.dims
    if has_voxel_affine_geometry(da):
        origin_names = get_voxel_affine_physical_coord_names(da)
        origin = tuple(origin_dict[d] for d in origin_names)
    else:
        origin = tuple(origin_dict[d] for d in spatial_dims)

    if has_time:
        data = da.values
        time_idx = da.dims.index("time")
        # SimpleITK expects the vector dimension to be the last axis, so move time
        # to the start and let the transpose place it last.
        data = np.moveaxis(data, time_idx, 0)
        image = sitk.GetImageFromArray(data.T, isVector=True)
    else:
        image = sitk.GetImageFromArray(da.values.T)

    image.SetSpacing(tuple(spacing))
    image.SetOrigin(origin)
    if has_voxel_affine_geometry(da):
        image.SetDirection(
            np.asarray(da.fusi.direction, dtype=np.float64).ravel().tolist()
        )
    return image


def expand_thin_dims(img: "sitk.Image", min_size: int = 4) -> "sitk.Image":
    """Expand any image dimension smaller than `min_size` by replication.

    SimpleITK's registration, multi-resolution pyramid, and displacement-field
    inversion fail when a spatial dimension is smaller than a handful of voxels
    (common for 2D+t fUSI recordings with a 1-voxel depth). This helper replicates
    thin dimensions so that the image is safe to process, while preserving the
    physical extent (spacing is divided by the expansion factor, keeping
    `size * spacing` constant).

    Parameters
    ----------
    img : SimpleITK.Image
        Input image. May be 2D or 3D, scalar or vector-valued.
    min_size : int, default: 4
        Minimum acceptable size along each dimension.

    Returns
    -------
    SimpleITK.Image
        Image with all dimensions >= `min_size`. Returns `img` unchanged if no
        dimension is too small.
    """
    import SimpleITK as sitk

    size = np.array(img.GetSize())
    factors = np.ones(len(size), dtype=int)
    thin = size < min_size
    if not thin.any():
        return img

    factors[thin] = np.ceil(min_size / size[thin]).astype(int)

    # sitk.Expand replicates voxels and halves spacing proportionally.
    return sitk.Expand(img, factors.tolist())
