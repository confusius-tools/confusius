"""Internal utilities shared by registration modules."""

import os
import signal
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from copy import deepcopy
from types import FrameType
from typing import TYPE_CHECKING, Literal, TypeGuard

import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation

from confusius._dims import VOXEL_DIMS, WORLD_DIMS
from confusius._utils.geometry import get_voxel_to_world_spatial_dims
from confusius.validation import ensure_voxeldata

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
        Always raised with a message explaining that the coordinates are irregular.
    """
    raise ValueError(
        "Registration requires defined spatial spacing for all spatial "
        f"dimensions, but {undefined_dims!r} are irregularly spaced (spacing is not "
        "uniform along the coordinate). Resample onto a regularly spaced grid before "
        "registering."
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
        World spacing for each spatial dimension.

    Raises
    ------
    ValueError
        If any spatial spacing is undefined.
    """
    da = ensure_voxeldata(
        da,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=False,
    )
    spatial_dims = [str(dim) for dim in da.dims if str(dim) != "time"]
    spacing_dict = da.fusi.spacing
    undefined_dims = [dim for dim in spatial_dims if spacing_dict.get(dim) is None]
    if undefined_dims:
        _raise_undefined_spatial_spacing_error(undefined_dims)
    return spatial_dims, [float(spacing_dict[dim]) for dim in spatial_dims]


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
    source = np.asarray(source, dtype=np.float64).reshape(1, -1)
    target = np.asarray(target, dtype=np.float64).reshape(1, -1)
    rotation = Rotation.align_vectors(target, source)[0]
    return rotation.as_matrix()


def _get_voxel_to_world_plane_center(data: xr.DataArray) -> np.ndarray:
    """Return the world center point of a single-slice voxel-to-world volume."""
    return np.array(
        [float(np.asarray(data.coords[name].values).mean()) for name in WORLD_DIMS],
        dtype=np.float64,
    )


def _get_voxel_to_world_slice_normal(data: xr.DataArray) -> np.ndarray:
    """Return the world-space normal of a single-slice voxel-to-world volume."""
    voxel_dims = get_voxel_to_world_spatial_dims(data)
    singleton_axes = [i for i, dim in enumerate(voxel_dims) if data.sizes[dim] == 1]
    if len(singleton_axes) != 1:
        raise ValueError(
            "Voxel-to-world plane initialization requires exactly one singleton "
            f"spatial dimension, got sizes {[data.sizes[dim] for dim in voxel_dims]!r}."
        )
    return np.asarray(data.fusi.direction, dtype=np.float64)[:, singleton_axes[0]]


def initialize_single_slice_rigid_transform(
    fixed: xr.DataArray,
    moving: xr.DataArray,
) -> np.ndarray:
    """Build a rigid fixed-to-moving initializer for single-slice voxel-to-world volumes.

    Parameters
    ----------
    fixed : xarray.DataArray
        Fixed single-slice voxel-to-world volume.
    moving : xarray.DataArray
        Moving single-slice voxel-to-world volume.

    Returns
    -------
    (4, 4) numpy.ndarray
        Rigid transform in world space mapping fixed coordinates into moving
        coordinates.

    Raises
    ------
    ValueError
        If either input is not a VoxelData array with a time-free, pose-free,
        3D voxel-to-world grid (see `confusius.validation.ensure_voxeldata`).
    """
    # ensure_voxeldata (allow_extra_dims=False) already guarantees spatial_dims ==
    # VOXEL_DIMS for both inputs, so fixed/moving spatial dims can never actually
    # differ here -- no separate dims-match check needed.
    fixed = ensure_voxeldata(
        fixed,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=False,
    )
    moving = ensure_voxeldata(
        moving,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=False,
    )

    fixed_normal = _get_voxel_to_world_slice_normal(fixed)
    moving_normal = _get_voxel_to_world_slice_normal(moving)
    rotation = _rotation_matrix_aligning_vectors(fixed_normal, moving_normal)

    fixed_center = _get_voxel_to_world_plane_center(fixed)
    moving_center = _get_voxel_to_world_plane_center(moving)
    translation = moving_center - rotation @ fixed_center

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


SignalHandler = Callable[[int, FrameType | None], object]
"""Python-level SIGINT handler callable."""


def validate_intensity_scaling(
    intensity_scaling: Literal["none", "db", "sqrt"] | float, name: str
) -> None:
    """Validate an intensity-scaling argument for the registration optimizer.

    Parameters
    ----------
    intensity_scaling : {"none", "db", "sqrt"} or float
        Scaling mode or positive power-scaling exponent to check.
    name : str
        Argument name used in the error message.

    Raises
    ------
    ValueError
        If `intensity_scaling` is neither a known mode nor a positive finite float.
    """
    valid_modes = {"none", "db", "sqrt"}
    is_valid_exponent = (
        isinstance(intensity_scaling, (int, float))
        and not isinstance(intensity_scaling, bool)
        and np.isfinite(intensity_scaling)
        and intensity_scaling > 0
    )
    if intensity_scaling not in valid_modes and not is_valid_exponent:
        raise ValueError(
            f"Invalid {name} {intensity_scaling!r}. Expected one of "
            f"{sorted(valid_modes)} or a positive finite exponent."
        )


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
        DataArray providing the world-to-reference affines for the output grid.

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


def voxeldata_to_sitk_image(da: xr.DataArray) -> "sitk.Image":
    """Convert a VoxelData array to a SimpleITK image.

    Uses the transpose convention: `da.values.T` is passed to `GetImageFromArray`,
    so that the first DataArray axis maps to SimpleITK's world x-axis. For data with
    an `extra` dimension, that dimension is converted to a vector image channel
    dimension.

    Parameters
    ----------
    da : xarray.DataArray
        Canonical VoxelData array, optionally with an `extra` dimension (produced by
        stacking `time`/other non-spatial dims together, since SimpleITK only
        supports one vector channel dimension). Spacing, origin, and direction are
        derived from its voxel-to-world index.

    Returns
    -------
    SimpleITK.Image
        SimpleITK image with spacing, origin, and direction set from the DataArray's
        voxel-to-world index. For `extra`-stacked input, returns a vector image where
        `extra` is the vector dimension.
    """
    import SimpleITK as sitk

    has_extra = "extra" in da.dims
    spacing = [float(da.fusi.spacing[dim]) for dim in VOXEL_DIMS]
    origin_dict = da.fusi.origin
    origin = tuple(origin_dict[d] for d in WORLD_DIMS)

    if has_extra:
        data = da.values
        extra_idx = da.dims.index("extra")
        # SimpleITK expects the vector dimension to be the last axis, so move extra
        # to the start and let the transpose place it last.
        data = np.moveaxis(data, extra_idx, 0)
        image = sitk.GetImageFromArray(data.T, isVector=True)
    else:
        image = sitk.GetImageFromArray(da.values.T)

    image.SetSpacing(tuple(spacing))
    image.SetOrigin(tuple(origin))
    image.SetDirection(np.asarray(da.fusi.direction, dtype=np.float64).ravel().tolist())
    return image


def expand_thin_dims(img: "sitk.Image", min_size: int = 4) -> "sitk.Image":
    """Expand any image dimension smaller than `min_size` by replication.

    SimpleITK's registration, multi-resolution pyramid, and displacement-field
    inversion fail when a spatial dimension is smaller than a handful of voxels
    (common for 2D+t fUSI recordings with a 1-voxel depth). This helper replicates
    thin dimensions so that the image is safe to process, while preserving the
    world extent (spacing is divided by the expansion factor, keeping
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
