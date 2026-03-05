"""Volume resampling utilities for fUSI data."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import SimpleITK as sitk
import numpy.typing as npt
import xarray as xr

from confusius._utils import _compute_origin, _compute_spacing
from confusius.registration.bspline import _dataarray_to_sitk_bspline
from confusius.registration.volume import _sitk_thread_count


def _dataarray_to_sitk_image(
    da: xr.DataArray,
) -> tuple["sitk.Image", Sequence[str]]:  # noqa: F821
    """Convert a spatial or spatial+time DataArray to a SimpleITK image.

    Uses the transpose convention: `da.values.T` is passed to `GetImageFromArray`,
    so that the first DataArray axis maps to SimpleITK's physical x-axis. For 3D+t data
    with a time dimension, the time dimension is converted to a vector image channel
    dimension.

    Parameters
    ----------
    da : xarray.DataArray
        2D or 3D spatial DataArray, or 3D+t or 2D+t DataArray with a time dimension.
        Spacing and origin are derived from its coordinates; missing coordinates warn
        and fall back to spacing `1.0` and origin `0.0`.

    Returns
    -------
    SimpleITK.Image
        SimpleITK image with spacing and origin set from the DataArray coordinates. For
        3D+t input, returns a vector image where time is the vector dimension.
    spatial_dims : sequence of str
        Names of the spatial dimensions.
    """
    import SimpleITK as sitk

    spacing_dict = _compute_spacing(da)
    origin_dict = _compute_origin(da)

    spatial_dims = [str(d) for d in da.dims if str(d) != "time"]
    has_time = "time" in da.dims

    spacing = tuple(
        s if s is not None else 1.0 for d, s in spacing_dict.items() if str(d) != "time"
    )
    origin = tuple(o for d, o in origin_dict.items() if str(d) != "time")

    if has_time:
        data = da.values
        time_idx = da.dims.index("time")
        # SimpleITK expects the vector dimension to be the last axis, so we move it to
        # the start and the transpose in GetImageFromArray will put it last.
        data = np.moveaxis(data, time_idx, 0)
        image = sitk.GetImageFromArray(data.T, isVector=True)
    else:
        image = sitk.GetImageFromArray(da.values.T)

    image.SetSpacing(spacing)
    image.SetOrigin(origin)

    return image, spatial_dims


def resample_volume(
    moving: xr.DataArray,
    transform: "npt.NDArray[np.float64] | xr.DataArray",
    *,
    shape: Sequence[int],
    spacing: Sequence[float],
    origin: Sequence[float],
    dims: Sequence[str],
    interpolation: Literal["linear", "bspline"] = "linear",
    default_value: float = 0.0,
    sitk_threads: int = -1,
) -> xr.DataArray:
    """Resample a volume onto an explicit output grid using a pre-computed transform.

    Low-level resampling primitive. For the common case of resampling onto the grid of
    another DataArray, use [`resample_like`][confusius.registration.resample_like]
    instead.

    Parameters
    ----------
    moving : xarray.DataArray
        2D or 3D spatial DataArray to resample, or 3D+t or 2D+t DataArray with a time
        dimension. If a time dimension is present, the same transform is applied to
        all time points.
    transform : (N+1, N+1) numpy.ndarray or xarray.DataArray
        Registration transform, as returned by
        [`register_volume`][confusius.registration.register_volume].

        - **Affine** (`numpy.ndarray`): homogeneous matrix of shape `(N+1, N+1)`
          mapping output (fixed) physical coordinates to moving physical coordinates
          (pull/inverse convention).
        - **B-spline** (`xarray.DataArray`): control-point DataArray as returned by
          `register_volume(transform="bspline")`.
    shape : sequence of int
        Number of voxels along each output axis, in DataArray dimension order.
    spacing : sequence of float
        Voxel spacing along each output axis, in DataArray dimension order.
    origin : sequence of float
        Physical origin (first voxel centre) along each output axis, in DataArray
        dimension order.
    dims : sequence of str
        Dimension names of the output DataArray.
    interpolation : {"linear", "bspline"}, default: "linear"
        Interpolation method used during resampling.
    default_value : float, default: 0.0
        Value assigned to voxels that fall outside the moving image's field of
        view after resampling.
    sitk_threads : int, default: -1
        Number of threads SimpleITK may use internally. Negative values resolve to
        `max(1, os.cpu_count() + 1 + sitk_threads)`, so `-1` means all CPUs, `-2`
        means all minus one, and so on. You may want to set this to a lower value or
        `1` when running multiple registrations in parallel (e.g. with joblib) to
        avoid over-subscribing the CPU.

    Returns
    -------
    xarray.DataArray
        Resampled volume on the specified grid with `moving`'s attributes. If the
        input had a time dimension, the output will also have a time dimension.

    Raises
    ------
    ValueError
        If `moving` is not 2D, 3D+t, 3D, or 3D+t.
    ValueError
        If `transform` is a numpy array whose shape does not match the spatial image
        dimensionality.
    """
    import SimpleITK as sitk

    has_time = "time" in moving.dims
    spatial_dims = [str(d) for d in moving.dims if str(d) != "time"]
    ndim = len(spatial_dims)

    if ndim not in (2, 3):
        raise ValueError(
            f"'moving' must have 2 or 3 spatial dimensions; got {ndim}D "
            f"spatial array with dims {spatial_dims}."
        )

    if isinstance(transform, np.ndarray):
        expected_shape = (ndim + 1, ndim + 1)
        if transform.shape != expected_shape:
            raise ValueError(
                f"affine shape {transform.shape} does not match spatial dimensionality "
                f"{ndim}D (expected {expected_shape})."
            )

        # Reconstruct a SimpleITK AffineTransform from the homogeneous matrix.
        # Pull convention: x_moving = A @ x_fixed + t, where A is the linear part
        # and t is the translation extracted from the last column.
        tx: sitk.Transform = sitk.AffineTransform(ndim)
        tx.SetMatrix(transform[:ndim, :ndim].flatten().tolist())
        tx.SetTranslation(transform[:ndim, ndim].tolist())
    else:
        tx = _dataarray_to_sitk_bspline(transform)

    moving_sitk, _ = _dataarray_to_sitk_image(moving)

    # SimpleITK will automatically create a vector output if the input is a vector
    # image.
    ref = sitk.Image(list(shape), sitk.sitkFloat32)
    ref.SetSpacing(list(spacing))
    ref.SetOrigin(list(origin))

    interp = sitk.sitkLinear if interpolation == "linear" else sitk.sitkBSpline

    with _sitk_thread_count(sitk_threads):
        result_sitk = sitk.Resample(
            moving_sitk, ref, tx, interp, default_value, moving_sitk.GetPixelID()
        )
        # .T restores DataArray axis order, inverse of the .T applied in
        # _dataarray_to_sitk.
        registered_arr = sitk.GetArrayFromImage(result_sitk).T

    coords = {
        d: np.array(origin[i]) + np.arange(shape[i]) * np.array(spacing[i])
        for i, d in enumerate(dims)
    }

    if has_time:
        coords["time"] = moving.coords["time"].values
        dims = ["time"] + list(dims)

    result = xr.DataArray(
        registered_arr,
        coords=coords,
        dims=dims,
        attrs=moving.attrs.copy(),
    )
    result.attrs["registration"] = "volume"
    return result


def resample_like(
    moving: xr.DataArray,
    reference: xr.DataArray,
    transform: "npt.NDArray[np.float64] | xr.DataArray",
    interpolation: Literal["linear", "bspline"] = "linear",
    default_value: float = 0.0,
    sitk_threads: int = -1,
) -> xr.DataArray:
    """Resample a volume onto the grid of a reference DataArray.

    Convenience wrapper around
    [`resample_volume`][confusius.registration.resample_volume] that extracts the output
    grid (`shape`, `spacing`, `origin`) from `reference`'s coordinates.

    Parameters
    ----------
    moving : xarray.DataArray
        2D or 3D spatial DataArray to resample, or 3D+t DataArray with a time dimension.
        If a time dimension is present, the same transform is applied to all time points.
    reference : xarray.DataArray
        DataArray defining the output grid. Must be 2D or 3D spatial (no time dimension).
    transform : (N+1, N+1) numpy.ndarray or xarray.DataArray
        Registration transform, as returned by
        [`register_volume`][confusius.registration.register_volume].  Maps points from
        the reference physical space to moving physical space (pull/inverse convention).

        - **Affine** (`numpy.ndarray`): homogeneous matrix.
        - **B-spline** (`xarray.DataArray`): control-point DataArray.
    interpolation : {"linear", "bspline"}, default: "linear"
        Interpolation method used during resampling.
    default_value : float, default: 0.0
        Value assigned to voxels that fall outside the moving image's field of view
        after resampling.
    sitk_threads : int, default: os.cpu_count() or 1
        Number of threads SimpleITK may use for the `Resample` call.
        Defaults to all available CPUs.

    Returns
    -------
    xarray.DataArray
        Resampled volume on the grid of `reference`, with `reference`'s coordinates and
        dimensions and `moving`'s attributes. If `moving` had a time dimension, the
        output will also have a time dimension.

    Raises
    ------
    ValueError
        If `reference` contains a `time` dimension or is not 2D or 3D.
    """
    if "time" in reference.dims:
        raise ValueError(
            f"'reference' must not have a time dimension; got dims {reference.dims}."
        )
    if reference.ndim not in (2, 3):
        raise ValueError(
            f"'reference' must be 2D or 3D; got {reference.ndim}D array with dims {reference.dims}."
        )

    spacing_dict = _compute_spacing(reference)
    origin_dict = _compute_origin(reference)

    shape = list(reference.sizes[str(d)] for d in reference.dims)
    spacing = [s if s is not None else 1.0 for s in spacing_dict.values()]
    origin = list(origin_dict.values())
    dims = [str(d) for d in reference.dims]

    result = resample_volume(
        moving,
        transform,
        shape=shape,
        spacing=spacing,
        origin=origin,
        dims=dims,
        interpolation=interpolation,
        default_value=default_value,
        sitk_threads=sitk_threads,
    )

    # Overwrite the reconstructed arithmetic coordinates with reference's exact
    # coordinate arrays. resample_volume rebuilds coords as origin + k * spacing, which
    # can diverge from the reference coordinates by floating-point accumulation errors.
    # Those sub-epsilon differences break xarray coordinate alignment (e.g. plot_volume)
    # that uses strict tolerances to match coordinates across DataArrays.
    result = result.assign_coords(
        {d: reference.coords[d] for d in dims if d in reference.coords}
    )
    return result
