"""Internal utilities shared by registration modules."""

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

from confusius._dims import WORLD_DIMS
from confusius._utils.geometry import get_voxel_to_world_spatial_dims
from confusius._utils.itk import resolve_itk_threads
from confusius.validation import ensure_voxeldata

if TYPE_CHECKING:
    from threading import Event

    import itk


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


def validate_fill_value(fill_value: float, dtype: np.dtype) -> None:
    """Check that `fill_value` can be stored losslessly in `dtype`.

    ITK resamples into the moving image's pixel type, so a fill value that an
    integer dtype cannot hold (NaN, inf, a fractional value, or one outside the dtype's
    range) would otherwise be silently cast to garbage such as `INT_MIN`.

    Parameters
    ----------
    fill_value : float
        Value assigned to voxels outside the moving image's field of view.
    dtype : numpy.dtype
        Dtype of the moving image.

    Raises
    ------
    ValueError
        If `dtype` is an integer dtype and `fill_value` is not a finite integer within
        its range.
    """
    # Garbage casting only occurs for integer dtypes
    if not np.issubdtype(dtype, np.integer):
        return

    info = np.iinfo(dtype)
    if (
        not np.isfinite(fill_value)
        or float(fill_value) != int(fill_value)
        or not info.min <= fill_value <= info.max
    ):
        raise ValueError(
            f"fill_value={fill_value!r} cannot be represented in the moving image's "
            f"integer dtype {dtype}. Pass an integer within [{info.min}, {info.max}], "
            "or cast the moving image to a float dtype first (e.g. "
            "`moving.astype(np.float32)`)."
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
def set_itk_thread_count(n: int) -> Generator[None, None, None]:
    """Temporarily override ITK's global thread count.

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
    import itk

    n = resolve_itk_threads(n)
    prev = itk.MultiThreaderBase.GetGlobalDefaultNumberOfThreads()
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(n)
    try:
        yield
    finally:
        itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(prev)


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
    next ITK iteration boundary and return their current partial result.
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


def expand_thin_dims(img: "itk.Image", min_size: int = 4) -> "itk.Image":
    """Expand thin image dimensions while preserving their physical support.

    Registration's recursive Gaussian filters require at least four voxels per
    dimension, including for single-slice fUSI recordings. ITK expands using
    linear interpolation, divides spacing by each expansion factor, and shifts
    the origin to preserve voxel-support boundaries (`size * spacing`), rather
    than the span between the first and last voxel centers.

    Parameters
    ----------
    img : itk.Image
        Input image.
    min_size : int, default: 4
        Minimum acceptable size along each dimension.

    Returns
    -------
    itk.Image
        Image with all dimensions >= `min_size`. Returns `img` unchanged if no
        dimension is too small.
    """
    import itk

    size = img.GetLargestPossibleRegion().GetSize()
    factors = [max(1, (min_size + int(s) - 1) // int(s)) for s in size]
    if all(factor == 1 for factor in factors):
        return img

    result = itk.expand_image_filter(img, expand_factors=factors)
    result.DisconnectPipeline()
    return result
