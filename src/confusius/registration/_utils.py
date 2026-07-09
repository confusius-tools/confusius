"""Internal utilities shared by registration modules."""

import os
import signal
import threading
from contextlib import contextmanager
from copy import deepcopy
from collections.abc import Callable
from types import FrameType
from typing import TYPE_CHECKING, Generator, TypeGuard

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from threading import Event

    import SimpleITK as sitk


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

    spacing_dict = da.fusi.spacing
    origin_dict = da.fusi.origin

    has_time = "time" in da.dims
    spacing = tuple(
        s if s is not None else 1.0 for d, s in spacing_dict.items() if str(d) != "time"
    )
    origin = tuple(o for d, o in origin_dict.items() if str(d) != "time")

    if has_time:
        data = da.values
        time_idx = da.dims.index("time")
        # SimpleITK expects the vector dimension to be the last axis, so move time
        # to the start and let the transpose place it last.
        data = np.moveaxis(data, time_idx, 0)
        image = sitk.GetImageFromArray(data.T, isVector=True)
    else:
        image = sitk.GetImageFromArray(da.values.T)

    image.SetSpacing(spacing)
    image.SetOrigin(origin)
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
