"""ITK array, geometry, and resampling boundaries.

ITK axes follow native `k/j/i` order; physical components follow world `z/y/x`.
Only NumPy's array order is reversed at the image boundary.
"""

import os
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from confusius._dims import VOXEL_DIMS, WORLD_DIMS
from confusius.validation import ensure_voxeldata

if TYPE_CHECKING:
    import itk
    import xarray as xr


def resolve_itk_threads(threads: int) -> int:
    """Resolve a worker count using joblib's sign convention.

    Parameters
    ----------
    threads : int
        Positive worker count or negative CPU-relative count.

    Returns
    -------
    int
        Positive worker count.

    Raises
    ------
    ValueError
        If `threads` is zero.
    """
    if threads == 0:
        raise ValueError("itk_threads must not be zero.")
    return threads if threads > 0 else max(1, (os.cpu_count() or 1) + 1 + threads)


def voxeldata_to_itk_image(data: "xr.DataArray") -> "itk.Image":
    """Convert spatial-only VoxelData to a real scalar ITK image.

    Parameters
    ----------
    data : xarray.DataArray
        Spatial-only VoxelData. Position-anchored origin and spacing account for
        cropping and striding, rather than assuming zero-based voxel labels.

    Returns
    -------
    itk.Image
        Scalar image with physical axes in world `z/y/x` order. Float32 data
        retains its precision; other real data uses float64 working values.

    Raises
    ------
    ValueError
        If `data` is not spatial-only VoxelData.
    TypeError
        If values are complex; conversion must not discard their phase.
    """
    import itk

    data = ensure_voxeldata(data, allow_pose=False, allow_extra_dims=False)
    if "time" in data.dims:
        raise ValueError("ITK image conversion requires spatial-only VoxelData.")
    if np.issubdtype(data.dtype, np.complexfloating):
        raise TypeError("ITK requires real-valued data, not complex values.")
    dtype = np.float32 if data.dtype == np.float32 else np.float64
    image = itk.image_from_array(np.ascontiguousarray(data.values.T, dtype=dtype))
    image.SetSpacing(tuple(float(data.fusi.spacing[dim]) for dim in VOXEL_DIMS))
    image.SetOrigin(tuple(float(data.fusi.origin[dim]) for dim in WORLD_DIMS))
    image.SetDirection(
        itk.matrix_from_array(np.asarray(data.fusi.direction, dtype=np.float64))
    )
    return image


def resample_itk_image(
    moving: "itk.Image",
    reference: "itk.Image",
    transform: "itk.Transform",
    *,
    interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    fill_value: float = 0.0,
    itk_threads: int = -1,
) -> "itk.Image":
    """Resample a scalar ITK image onto a reference grid.

    Parameters
    ----------
    moving : itk.Image
        Double-precision scalar image.
    reference : itk.Image
        Image defining output geometry; pixel allocation is unnecessary.
    transform : itk.Transform
        Output-to-input pull transform.
    interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method; B-spline interpolation is cubic.
    fill_value : float, default: 0.0
        Out-of-FOV value.
    itk_threads : int, default: -1
        Filter worker count, following joblib's sign convention.

    Returns
    -------
    itk.Image
        Detached double-precision resampled image.

    Raises
    ------
    ValueError
        If the interpolation name or worker count is invalid.
    """
    import itk

    if interpolation == "bspline":
        # Wheels tie spline coefficient precision to pixel type; use double coefficients.
        target_type = itk.Image[itk.D, moving.GetImageDimension()]
        if type(moving) is not target_type:
            moving = itk.cast_image_filter(moving, ttype=(type(moving), target_type))
    image_type = type(moving)
    if interpolation == "nearest":
        interpolator = itk.NearestNeighborInterpolateImageFunction[
            image_type, itk.D
        ].New()
    elif interpolation == "linear":
        interpolator = itk.LinearInterpolateImageFunction[image_type, itk.D].New()
    elif interpolation == "bspline":
        interpolator = itk.BSplineInterpolateImageFunction[
            image_type, itk.D, itk.D
        ].New()
    else:
        raise ValueError(f"Invalid `interpolation`: {interpolation!r}.")
    resampler = itk.ResampleImageFilter[image_type, image_type].New(
        Input=moving,
        ReferenceImage=reference,
        UseReferenceImage=True,
        Transform=transform,
        Interpolator=interpolator,
        DefaultPixelValue=fill_value,
    )
    resampler.SetNumberOfWorkUnits(resolve_itk_threads(itk_threads))
    resampler.Update()
    result = resampler.GetOutput()
    result.DisconnectPipeline()
    return result


def cast_resampled_values(values: npt.NDArray, dtype: np.dtype) -> npt.NDArray:
    """Cast interpolated values with ITK's saturating pixel-range semantics.

    Parameters
    ----------
    values : numpy.ndarray
        Interpolated working values.
    dtype : numpy.dtype
        Requested output pixel dtype.

    Returns
    -------
    numpy.ndarray
        Saturated values in the requested dtype, with nonfinite float fills kept.
    """
    if values.dtype == dtype:
        return values
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        lower = values <= limits.min
        upper = values >= limits.max
        result = np.empty(values.shape, dtype=dtype)
        result[lower] = limits.min
        result[upper] = limits.max
        # Partition before casting: int64/uint64 upper bounds round upward in float64.
        interior = ~(lower | upper)
        result[interior] = values[interior]
        return result
    if np.issubdtype(dtype, np.floating):
        limits = np.finfo(dtype)
        values = values.copy()
        finite = np.isfinite(values)
        values[finite] = np.clip(values[finite], limits.min, limits.max)
    return values.astype(dtype)


def resample_voxeldata(
    moving: "xr.DataArray",
    transform: "itk.Transform",
    *,
    sizes: list[int],
    spacing: list[float],
    origin: list[float],
    direction: npt.NDArray[np.float64],
    interpolation: Literal["linear", "nearest", "bspline"],
    fill_value: float,
    itk_threads: int,
) -> npt.NDArray:
    """Resample VoxelData with an ITK transform, preserving dtype and extra axes.

    Parameters
    ----------
    moving : xarray.DataArray
        Spatial-only VoxelData or VoxelData with one stacked `extra` dimension.
    transform : itk.Transform
        Output-to-input world transform.
    sizes : list[int]
        Output sizes in native `k/j/i` order.
    spacing : list[float]
        Output spacing in native axis order.
    origin : list[float]
        Output world origin in `z/y/x` order.
    direction : (3, 3) numpy.ndarray
        Output direction matrix, with native voxel axes as columns.
    interpolation : {"linear", "nearest", "bspline"}
        Resolved interpolation method.
    fill_value : float
        Validated output fill value, representable in the source dtype.
    itk_threads : int
        Filter worker count, following joblib's sign convention.

    Returns
    -------
    numpy.ndarray
        Values in `(extra, k, j, i)` or `(k, j, i)` order with the source dtype.

    Raises
    ------
    TypeError
        If source values are complex.

    Notes
    -----
    ITK wheels do not wrap resampling for every integer dtype. Nearest-neighbor
    resampling transforms an index image and gathers original values, preserving
    labels beyond float64's exact integer range and sharing one lookup across all
    components. Other interpolation uses double-precision working images, as ITK's
    scalar interpolators do, then casts back to the source dtype.
    """
    import itk

    if np.issubdtype(moving.dtype, np.complexfloating):
        raise TypeError("ITK requires real-valued data, not complex values.")
    has_extra = "extra" in moving.dims
    spatial = moving.isel(extra=0, drop=True) if has_extra else moving
    reference = itk.Image[itk.D, 3].New()
    reference.SetRegions(sizes)
    reference.SetSpacing(spacing)
    reference.SetOrigin(origin)
    reference.SetDirection(itk.matrix_from_array(direction))
    if interpolation == "nearest":
        indices = spatial.copy(
            data=np.arange(spatial.size, dtype=np.float64).reshape(spatial.shape)
        )
        lookup = itk.array_from_image(
            resample_itk_image(
                voxeldata_to_itk_image(indices),
                reference,
                transform,
                interpolation="nearest",
                fill_value=-1.0,
                itk_threads=itk_threads,
            )
        ).T.astype(np.intp)
        values = (
            moving.transpose("extra", *VOXEL_DIMS).values
            if has_extra
            else moving.values[None]
        )
        gathered = values.reshape(len(values), -1)[:, lookup]
        gathered[:, lookup < 0] = fill_value
        return gathered if has_extra else gathered[0]
    components = moving.sizes["extra"] if has_extra else 1
    result = np.empty((components, *sizes), dtype=moving.dtype)
    for index in range(components):
        component = moving.isel(extra=index, drop=True) if has_extra else moving
        image = resample_itk_image(
            voxeldata_to_itk_image(component),
            reference,
            transform,
            interpolation=interpolation,
            fill_value=fill_value,
            itk_threads=itk_threads,
        )
        result[index] = cast_resampled_values(
            itk.array_from_image(image).T, moving.dtype
        )
    return result if has_extra else result[0]
