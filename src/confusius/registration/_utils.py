"""Internal utilities shared by registration modules."""

import os
from copy import deepcopy
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import SimpleITK as sitk


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


def dataarray_to_sitk_image(da: xr.DataArray) -> "sitk.Image":
    """Convert a spatial or spatiotemporal DataArray to a SimpleITK image.

    Uses the transpose convention: `da.values.T` is passed to `GetImageFromArray`,
    so that the first DataArray axis maps to SimpleITK's physical x-axis. For data
    with a time dimension, the time dimension is converted to a vector image channel
    dimension.

    Parameters
    ----------
    da : xarray.DataArray
        2D or 3D spatial DataArray, or 2D+t or 3D+t DataArray with a time
        dimension. Spacing and origin are derived from its coordinates; missing
        coordinates warn and fall back to spacing `1.0` and origin `0.0`.

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
