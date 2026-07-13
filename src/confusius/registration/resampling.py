"""Volume resampling utilities for fUSI data."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils.coordinates import get_grid_kwargs_from_dataarray
from confusius.registration._utils import (
    dataarray_to_sitk_image,
    replace_affines_attr,
    set_sitk_thread_count,
)
from confusius.registration.bspline import (
    _dataarray_to_sitk_bspline,
    _dataarray_to_sitk_displacement_field,
)
from confusius.validation import (
    validate_fusi_dataarray,
    validate_matching_spatial_units,
)


def resample_volume(
    moving: xr.DataArray,
    transform: "npt.NDArray[np.floating] | xr.DataArray",
    *,
    shape: Sequence[int],
    spacing: Sequence[float],
    origin: Sequence[float],
    dims: Sequence[str],
    interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    default_value: float | None = None,
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
        - **B-spline** (`xarray.DataArray`): control-point DataArray with `attrs["type"]
          == "bspline_transform"` as returned by `register_volume(transform="bspline")`.
        - **Displacement field** (`xarray.DataArray`): dense field with
          `attrs["type"] == "displacement_field_transform"`, as returned by
          [`sample_displacement_field`][confusius.registration.sample_displacement_field]
          or
          [`invert_displacement_field`][confusius.registration.invert_displacement_field].

    shape : sequence of int
        Number of voxels along each output axis, in DataArray dimension order.
    spacing : sequence of float
        Voxel spacing along each output axis, in DataArray dimension order.
    origin : sequence of float
        Physical origin (first voxel centre) along each output axis, in DataArray
        dimension order.
    dims : sequence of str
        Dimension names of the output DataArray.
    interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used during resampling.
    default_value : float, optional
        Value assigned to voxels that fall outside the moving image's field of view
        after resampling. If not provided, defaults to `float(moving.min())`, which
        renders out-of-FOV voxels as background regardless of intensity scale (important
        for dB data where 0 is maximum intensity).
    sitk_threads : int, default: -1
        Number of threads SimpleITK may use internally. Negative values resolve to
        `max(1, os.cpu_count() + 1 + sitk_threads)`, so `-1` means all CPUs, `-2`
        means all minus one, and so on. You may want to set this to a lower value or
        `1` when running multiple registrations in parallel (e.g. with joblib) to
        avoid over-subscribing the CPU.

    Returns
    -------
    xarray.DataArray
        Resampled volume on the specified grid with `moving`'s attributes. If the input
        had a time dimension, the output will also have a time dimension. This low-level
        function does not infer world-space affines for the output grid.

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

    validate_fusi_dataarray(
        moving,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=False,
        minimum_spatial_dims=2,
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
    elif transform.attrs.get("type") == "displacement_field_transform":
        tx = sitk.DisplacementFieldTransform(
            _dataarray_to_sitk_displacement_field(transform)
        )
    else:
        tx = _dataarray_to_sitk_bspline(transform)

    moving_sitk = dataarray_to_sitk_image(moving)

    _default_value = default_value if default_value is not None else float(moving.min())

    # SimpleITK will automatically create a vector output if the input is a vector
    # image.
    ref = sitk.Image(list(shape), sitk.sitkFloat32)
    ref.SetSpacing(list(spacing))
    ref.SetOrigin(list(origin))

    if interpolation == "nearest":
        sitk_interpolation = sitk.sitkNearestNeighbor
    elif interpolation == "linear":
        sitk_interpolation = sitk.sitkLinear
    elif interpolation == "bspline":
        sitk_interpolation = sitk.sitkBSpline
    else:
        raise ValueError(f"Invalid interpolation: {interpolation}")

    with set_sitk_thread_count(sitk_threads):
        result_sitk = sitk.Resample(
            moving_sitk,
            ref,
            tx,
            sitk_interpolation,
            _default_value,
            moving_sitk.GetPixelID(),
        )
        # .T restores DataArray axis order, inverse of the .T used to build the SITK
        # image.
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
    return result


def resample_like(
    moving: xr.DataArray,
    reference: xr.DataArray,
    transform: "npt.NDArray[np.floating] | xr.DataArray",
    interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    default_value: float | None = None,
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
        When spatial coordinate `units` metadata is present on both `moving` and
        `reference`, they must match.
    transform : (N+1, N+1) numpy.ndarray or xarray.DataArray
        Registration transform, as returned by
        [`register_volume`][confusius.registration.register_volume]. Maps points from
        the reference physical space to moving physical space (pull/inverse convention).

        - **Affine** (`numpy.ndarray`): homogeneous matrix whose translation entries
          are expressed in the same physical units as `moving` and `reference`.
        - **B-spline** (`xarray.DataArray`): control-point DataArray.
        - **Displacement field** (`xarray.DataArray`): dense field with `attrs["type"]
          == "displacement_field_transform"`.

        When `transform` is a DataArray and spatial coordinate `units` metadata is
        present on both it and `reference`, those units must also match.

    interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used during resampling.
    default_value : float, optional
        Value assigned to voxels that fall outside the moving image's field of view
        after resampling. If not provided, defaults to `float(moving.min())`, which
        renders out-of-FOV voxels as background regardless of intensity scale (important
        for dB data where 0 is maximum intensity).
    sitk_threads : int, default: os.cpu_count() or 1
        Number of threads SimpleITK may use for the `Resample` call.
        Defaults to all available CPUs.

    Returns
    -------
    xarray.DataArray
        Resampled volume on the grid of `reference`, with `reference`'s coordinates and
        dimensions, `moving`'s non-spatial attributes, and physical-space affines
        inherited from `reference`. If `moving` had a time dimension, the output will
        also have a time dimension.

    Raises
    ------
    ValueError
        If `reference` contains a `time` dimension or is not 2D or 3D.
    """
    if "time" in reference.dims:
        raise ValueError(
            f"'reference' must not have a time dimension; got dims {reference.dims}."
        )

    validate_fusi_dataarray(
        moving,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=False,
        minimum_spatial_dims=2,
    )
    validate_fusi_dataarray(
        reference,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=False,
        minimum_spatial_dims=2,
    )
    validate_matching_spatial_units((("moving", moving), ("reference", reference)))
    if isinstance(transform, xr.DataArray):
        validate_matching_spatial_units(
            (("transform", transform), ("reference", reference))
        )

    grid = get_grid_kwargs_from_dataarray(reference)
    dims = grid["dims"]

    result = resample_volume(
        moving,
        transform,
        **grid,
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
    replace_affines_attr(result, reference)
    return result
