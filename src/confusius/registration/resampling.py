"""Volume resampling utilities for fUSI data."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils import _compute_origin, _compute_spacing
from confusius.registration.volume import _dataarray_to_sitk


def resample_volume(
    moving: xr.DataArray,
    affine: npt.NDArray[np.float64],
    *,
    shape: Sequence[int],
    spacing: Sequence[float],
    origin: Sequence[float],
    dims: Sequence[str],
    interpolation: Literal["linear", "bspline"] = "linear",
    default_value: float = 0.0,
) -> xr.DataArray:
    """Resample a volume onto an explicit output grid using a pre-computed affine.

    Low-level resampling primitive. For the common case of resampling onto the grid of
    another DataArray, use `resample_like` instead.

    Parameters
    ----------
    moving : xarray.DataArray
        2D or 3D spatial DataArray to resample. Must not have a ``time`` dimension.
    affine : (N+1, N+1) numpy.ndarray
        Homogeneous affine matrix in physical space, as returned by
        [`register_volume`][confusius.registration.register_volume]. Maps points from
        the output (fixed) physical space to moving physical space.
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

    Returns
    -------
    xarray.DataArray
        Resampled volume on the specified grid with ``moving``'s attributes.

    Raises
    ------
    ValueError
        If ``moving`` contains a ``time`` dimension or is not 2D or 3D.
    ValueError
        If the affine shape does not match the image dimensionality.
    """
    import SimpleITK as sitk

    if "time" in moving.dims:
        raise ValueError(
            f"'moving' must not have a time dimension; got dims {moving.dims}."
        )
    if moving.ndim not in (2, 3):
        raise ValueError(
            f"'moving' must be 2D or 3D; got {moving.ndim}D array with dims {moving.dims}."
        )

    ndim = moving.ndim
    expected_shape = (ndim + 1, ndim + 1)
    if affine.shape != expected_shape:
        raise ValueError(
            f"affine shape {affine.shape} does not match image dimensionality "
            f"{ndim}D (expected {expected_shape})."
        )

    # Reconstruct a SimpleITK AffineTransform from the homogeneous matrix.
    # Pull convention: x_moving = A @ x_fixed + t, where A is the linear part
    # and t is the translation extracted from the last column.
    tx = sitk.AffineTransform(ndim)
    tx.SetMatrix(affine[:ndim, :ndim].flatten().tolist())
    tx.SetTranslation(affine[:ndim, ndim].tolist())

    moving_sitk = _dataarray_to_sitk(moving)

    # Build the output reference image from explicit grid parameters using the same
    # convention as _dataarray_to_sitk: size, spacing, and origin are passed in
    # DataArray dimension order (e.g. z, y, x). This keeps the moving and reference
    # images in the same coordinate convention so that Resample produces correct output.
    ref = sitk.Image(list(shape), moving_sitk.GetPixelID())
    ref.SetSpacing(list(spacing))
    ref.SetOrigin(list(origin))

    interp = sitk.sitkLinear if interpolation == "linear" else sitk.sitkBSpline

    # .T restores DataArray axis order, inverse of the .T applied in _dataarray_to_sitk.
    registered_arr = sitk.GetArrayFromImage(
        sitk.Resample(
            moving_sitk, ref, tx, interp, default_value, moving_sitk.GetPixelID()
        )
    ).T

    coords = {
        d: np.array(origin[i]) + np.arange(shape[i]) * np.array(spacing[i])
        for i, d in enumerate(dims)
    }
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
    affine: npt.NDArray[np.float64],
    interpolation: Literal["linear", "bspline"] = "linear",
    default_value: float = 0.0,
) -> xr.DataArray:
    """Resample a volume onto the grid of a reference DataArray.

    Convenience wrapper around
    [`resample_volume`][confusius.registration.resample_volume] that extracts the output
    grid (``shape``, ``spacing``, ``origin``) from `reference`'s coordinates.

    Parameters
    ----------
    moving : xarray.DataArray
        2D or 3D spatial DataArray to resample. Must not have a ``time`` dimension.
    reference : xarray.DataArray
        DataArray defining the output grid. Must not have a ``time`` dimension and must
        be 2D or 3D.
    affine : (N+1, N+1) numpy.ndarray
        Homogeneous affine matrix in physical space, as returned by
        [`register_volume`][confusius.registration.register_volume]. Maps points from
        the reference physical space to moving physical space.
    interpolation : {"linear", "bspline"}, default: "linear"
        Interpolation method used during resampling.
    default_value : float, default: 0.0
        Value assigned to voxels that fall outside the moving image's field of view
        after resampling.

    Returns
    -------
    xarray.DataArray
        Resampled volume on the grid of `reference`, with `reference`'s coordinates and
        dimensions and `moving`'s attributes.

    Raises
    ------
    ValueError
        If `reference` contains a ``time`` dimension or is not 2D or 3D.
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

    return resample_volume(
        moving,
        affine,
        shape=shape,
        spacing=spacing,
        origin=origin,
        dims=dims,
        interpolation=interpolation,
        default_value=default_value,
    )
