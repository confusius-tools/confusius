"""Volume resampling utilities for fUSI data."""

from collections.abc import Hashable, Mapping
from typing import Literal, SupportsFloat, SupportsIndex

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._dims import SPATIAL_DIMS, VOXEL_DIMS
from confusius._utils.geometry import get_voxel_to_world_direction_matrix
from confusius.registration._utils import (
    replace_affines_attr,
    set_sitk_thread_count,
    voxeldata_to_sitk_image,
)
from confusius.registration.bspline import (
    _voxeldata_to_sitk_bspline,
    _voxeldata_to_sitk_displacement_field,
)
from confusius.validation import ensure_voxeldata, validate_matching_spatial_units
from confusius.xarray.affine import reindex_voxels_like
from confusius.xarray.create import create_voxeldata


def _require_grid_keys(
    values: Mapping[Hashable, object], keys: tuple[str, ...], name: str
) -> None:
    """Validate that a grid mapping contains all required keys.

    Parameters
    ----------
    values : mapping of hashable to object
        Mapping to validate.
    keys : tuple of str
        Required keys.
    name : str
        Parameter name used in validation errors.

    Raises
    ------
    ValueError
        If any required key is missing.
    """
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError(f"{name} is missing required key(s): {missing!r}.")


def _resolve_int_grid_mapping(
    values: Mapping[Hashable, SupportsIndex], keys: tuple[str, ...], name: str
) -> list[int]:
    """Return grid values from a keyed mapping.

    Parameters
    ----------
    values : mapping of hashable to typing.SupportsIndex
        Mapping containing integer-like entries for every key in `keys`.
    keys : tuple of str
        Keys to extract in order.
    name : str
        Parameter name used in validation errors.

    Returns
    -------
    list of int
        Values extracted in `keys` order.

    Raises
    ------
    ValueError
        If any required key is missing.
    """
    _require_grid_keys(values, keys, name)
    return [int(values[key]) for key in keys]


def _resolve_float_grid_mapping(
    values: Mapping[Hashable, SupportsFloat | SupportsIndex],
    keys: tuple[str, ...],
    name: str,
) -> list[float]:
    """Return float grid values from a keyed mapping.

    Parameters
    ----------
    values : mapping of hashable to typing.SupportsFloat or typing.SupportsIndex
        Mapping containing numeric entries for every key in `keys`.
    keys : tuple of str
        Keys to extract in order.
    name : str
        Parameter name used in validation errors.

    Returns
    -------
    list of float
        Values extracted in `keys` order.

    Raises
    ------
    ValueError
        If any required key is missing.
    """
    _require_grid_keys(values, keys, name)
    return [float(values[key]) for key in keys]


def resample_volume(
    moving: xr.DataArray,
    transform: "npt.NDArray[np.floating] | xr.DataArray",
    *,
    output_sizes: Mapping[Hashable, SupportsIndex],
    output_spacing: Mapping[Hashable, SupportsFloat | SupportsIndex],
    output_origin: Mapping[Hashable, SupportsFloat | SupportsIndex],
    output_direction: npt.ArrayLike,
    interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    fill_value: float | None = None,
    sitk_threads: int = -1,
) -> xr.DataArray:
    """Resample a volume onto an explicit output grid using a pre-computed transform.

    Low-level resampling primitive. For the common case of resampling onto the grid of
    another DataArray, use [`resample_like`][confusius.registration.resample_like]
    instead.

    The output grid is specified as a position-anchored `shape`/`spacing`/`origin`/
    `direction` (matching what a SimpleITK image expects), not as a single affine.
    A DataArray's `voxel_to_world` affine is defined in terms of its voxel
    *coordinate values*, which stay unchanged across cropping or striding (see
    [VoxelToWorldIndex][confusius._utils.geometry.VoxelToWorldIndex]) — it does not
    generally describe where the array's *position* `(0, ..., 0)` sits, or the
    world distance between consecutive *positions*, once the array has been
    cropped or strided. Use `reference.fusi.spacing`/`reference.fusi.origin` (as
    [`resample_like`][confusius.registration.resample_like] does) to derive a
    position-anchored grid from an existing DataArray.

    Parameters
    ----------
    moving : xarray.DataArray
        VoxelData array to resample. May be spatial-only or have a `time` dimension
        (single-slice recordings use a singleton `k` axis). If a time dimension is
        present, the same transform is applied to all time points.
    transform : (4, 4) numpy.ndarray or xarray.DataArray
        Registration transform, as returned by
        [`register_volume`][confusius.registration.register_volume].

        - **Affine** (`numpy.ndarray`): homogeneous `(4, 4)` matrix mapping output
          (fixed) world coordinates to moving world coordinates (pull/inverse
          convention).
        - **B-spline** (`xarray.DataArray`): control-point DataArray with `attrs["type"]
          == "bspline_transform"` as returned by `register_volume(transform="bspline")`.
        - **Displacement field** (`xarray.DataArray`): dense field with
          `attrs["type"] == "displacement_field_transform"`, as returned by
          [`sample_displacement_field`][confusius.registration.sample_displacement_field]
          or
          [`invert_displacement_field`][confusius.registration.invert_displacement_field].

    output_sizes : mapping of str to int
        Number of voxels along each output voxel axis, read by `k`/`j`/`i` keys.
        `reference.sizes` can be passed directly.
    output_spacing : mapping of str to float
        World distance between consecutive voxel positions along each output voxel
        axis, read by `k`/`j`/`i` keys. `reference.fusi.spacing` can be passed
        directly.
    output_origin : mapping of str to float
        World location of output voxel position `(0, 0, 0)`, read by `z`/`y`/`x`
        keys. `reference.fusi.origin` can be passed directly.
    output_direction : numpy.typing.ArrayLike
        `(3, 3)` matrix whose columns are the unit world-space direction of each
        output voxel axis, in native `k/j/i` column order. `reference.fusi.direction`
        can be passed directly.
    interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used during resampling.
    fill_value : float, optional
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
        VoxelData array resampled onto the requested grid, with
        `moving`'s attributes. If the input had a time dimension, the output will also
        have a time dimension.

    Raises
    ------
    ValueError
        If `transform` is a numpy array whose shape does not match the spatial
        dimensionality, or if `output_sizes`, `output_spacing`, or `output_origin`
        is missing a required key.
    """
    import SimpleITK as sitk

    moving = ensure_voxeldata(
        moving,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=False,
    )

    has_time = "time" in moving.dims
    ndim = len(VOXEL_DIMS)

    resolved_output_sizes = _resolve_int_grid_mapping(
        output_sizes, VOXEL_DIMS, "output_sizes"
    )
    resolved_output_spacing = _resolve_float_grid_mapping(
        output_spacing, VOXEL_DIMS, "output_spacing"
    )
    resolved_output_origin = _resolve_float_grid_mapping(
        output_origin, SPATIAL_DIMS, "output_origin"
    )
    direction = np.asarray(output_direction, dtype=np.float64)
    if direction.shape != (ndim, ndim):
        raise ValueError(
            f"output_direction must have shape ({ndim}, {ndim}), got {direction.shape}."
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
            _voxeldata_to_sitk_displacement_field(transform)
        )
    else:
        tx = _voxeldata_to_sitk_bspline(transform)

    moving_sitk = voxeldata_to_sitk_image(moving)

    resolved_fill_value = fill_value if fill_value is not None else float(moving.min())

    # SimpleITK will automatically create a vector output if the input is a vector
    # image.
    ref = sitk.Image(resolved_output_sizes, sitk.sitkFloat32)
    ref.SetSpacing(resolved_output_spacing)
    ref.SetOrigin(resolved_output_origin)
    ref.SetDirection(direction.ravel().tolist())

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
            resolved_fill_value,
            moving_sitk.GetPixelID(),
        )
        # .T restores DataArray axis order, inverse of the .T used to build the SITK
        # image.
        registered_arr = sitk.GetArrayFromImage(result_sitk).T

    voxel_to_world_arr = np.eye(ndim + 1, dtype=np.float64)
    voxel_to_world_arr[:ndim, :ndim] = direction @ np.diag(resolved_output_spacing)
    voxel_to_world_arr[:ndim, ndim] = resolved_output_origin

    attrs = moving.attrs.copy()
    result = create_voxeldata(
        registered_arr,
        dims=("time", *VOXEL_DIMS) if has_time else VOXEL_DIMS,
        time=moving.coords["time"] if has_time else None,
        voxel_to_world=voxel_to_world_arr,
        attrs=attrs,
        name=str(moving.name) if moving.name is not None else None,
    )
    for name in SPATIAL_DIMS:
        units = moving.coords[name].attrs.get("units")
        if units is not None:
            result.coords[name].attrs["units"] = units
    return result


def resample_like(
    moving: xr.DataArray,
    reference: xr.DataArray,
    transform: "npt.NDArray[np.floating] | xr.DataArray",
    interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    fill_value: float | None = None,
    default_value: float | None = None,
    sitk_threads: int = -1,
) -> xr.DataArray:
    """Resample a volume onto the grid of a reference DataArray.

    Convenience wrapper around
    [`resample_volume`][confusius.registration.resample_volume] that extracts the
    position-anchored output grid (`shape`, `spacing`, `origin`, `direction`) from
    `reference`'s coordinates via the `fusi` accessor, so the grid is correct even if
    `reference` has been cropped or strided from a larger DataArray.

    Parameters
    ----------
    moving : xarray.DataArray
        VoxelData array to resample. May be spatial-only or have a
        `time` dimension (single-slice recordings use a singleton `k` axis). If a time
        dimension is present, the same transform is applied to all time points.
    reference : xarray.DataArray
        VoxelData array defining the output grid. Must be spatial-only
        with dimensions `k`, `j`, `i` (no time dimension). When spatial coordinate
        `units` metadata is present on both `moving` and `reference`, they must match.
    transform : (4, 4) numpy.ndarray or xarray.DataArray
        Registration transform, as returned by
        [`register_volume`][confusius.registration.register_volume]. Maps points from
        the reference world space to moving world space (pull/inverse convention).

        - **Affine** (`numpy.ndarray`): homogeneous matrix whose translation entries
          are expressed in the same physical units as `moving` and `reference`.
        - **B-spline** (`xarray.DataArray`): control-point DataArray.
        - **Displacement field** (`xarray.DataArray`): dense field with `attrs["type"]
          == "displacement_field_transform"`.

        When `transform` is a DataArray and spatial coordinate `units` metadata is
        present on both it and `reference`, those units must also match.

    interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used during resampling.
    fill_value : float, optional
        Value assigned to voxels that fall outside the moving image's field of view
        after resampling. If not provided, defaults to `float(moving.min())`, which
        renders out-of-FOV voxels as background regardless of intensity scale (important
        for dB data where 0 is maximum intensity).
    default_value : float, optional
        Alias for `fill_value` kept for compatibility with older branch code. If both
        values are provided, `fill_value` takes precedence.
    sitk_threads : int, default: -1
        Number of threads SimpleITK may use internally. Negative values resolve to
        `max(1, os.cpu_count() + 1 + sitk_threads)`, so `-1` means all CPUs, `-2`
        means all minus one, and so on. You may want to set this to a lower value or
        `1` when running multiple registrations in parallel (e.g. with joblib) to
        avoid over-subscribing the CPU.

    Returns
    -------
    xarray.DataArray
        Resampled volume on the grid of `reference`, with `reference`'s coordinates and
        dimensions, `moving`'s non-spatial attributes, and world-space affines
        inherited from `reference`. If `moving` had a time dimension, the output will
        also have a time dimension.

    Raises
    ------
    ValueError
        If `reference` contains a `time` dimension or does not contain the spatial
        dimensions `k`, `j`, and `i`.
    """
    if "time" in reference.dims:
        raise ValueError(
            f"'reference' must not have a time dimension; got dims {reference.dims}."
        )

    moving = ensure_voxeldata(
        moving,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=False,
    )
    reference = ensure_voxeldata(
        reference,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=False,
    )
    validate_matching_spatial_units((("moving", moving), ("reference", reference)))
    if isinstance(transform, xr.DataArray):
        validate_matching_spatial_units(
            (("transform", transform), ("reference", reference))
        )

    output_direction = get_voxel_to_world_direction_matrix(reference)

    result = resample_volume(
        moving,
        transform,
        output_sizes=reference.sizes,
        output_spacing=reference.fusi.spacing,
        output_origin=reference.fusi.origin,
        output_direction=output_direction,
        interpolation=interpolation,
        fill_value=fill_value if fill_value is not None else default_value,
        sitk_threads=sitk_threads,
    )

    # resample_volume builds a fresh, position-anchored grid (dense zero-based voxel
    # labels), which physically matches reference's grid but not necessarily its
    # voxel labels (e.g. reference may itself be cropped or strided from a larger
    # array). Adopt reference's own labels and affine so the two are directly
    # alignable by voxel label as well as by world position.
    result = reindex_voxels_like(result, reference)
    replace_affines_attr(result, reference)
    return result
