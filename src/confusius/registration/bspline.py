"""B-spline and dense displacement pull transforms.

Both representations are VoxelData arrays with dims `(component, k, j, i)` and
components labeled `k/j/i`, storing world `z/y/x` displacement components.
B-splines store a sparse coefficient lattice (`type='bspline_transform'`, `order`);
dense fields store one world displacement per voxel
(`type='displacement_field_transform'`). Geometry belongs to each array's index.

An optional `attrs['affines']['bspline_initialization']` stores the pre-alignment
matrix. The full pull transform is `pre_affine(bspline(point))`: the B-spline
acts on a fixed-world point, then the affine maps it into moving world space.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.ndimage import map_coordinates

from confusius._dims import VOXEL_DIMS, WORLD_DIMS
from confusius._utils.coordinates import get_dim_keyed_origin
from confusius._utils.itk import resolve_itk_threads
from confusius.registration._utils import get_defined_spatial_spacing
from confusius.validation import (
    ensure_voxeldata,
    validate_bspline,
    validate_displacement_field,
    validate_matching_spatial_units,
)
from confusius.xarray import create_voxeldata

if TYPE_CHECKING:
    import itk


def voxeldata_to_itk_bspline(transform: xr.DataArray) -> "itk.Transform":
    """Convert a cubic B-spline coefficient VoxelData array to an ITK transform.

    Parameters
    ----------
    transform : xarray.DataArray
        B-spline coefficient array with optional
        `attrs['affines']['bspline_initialization']` metadata.

    Returns
    -------
    itk.Transform
        B-spline, optionally composed as `pre_affine(bspline(point))`.

    Raises
    ------
    ValueError
        If the coefficient array is invalid.

    Notes
    -----
    The resampling dispatcher routes non-cubic splines through sampled
    displacement fields before calling this cubic-only converter.
    """
    import itk

    from confusius.registration.affines import affine_to_itk_linear_transform

    validate_bspline(transform)
    sizes = [transform.sizes[dim] for dim in VOXEL_DIMS]
    origin = get_dim_keyed_origin(transform)
    _, spacing = get_defined_spatial_spacing(transform.isel(component=0, drop=True))
    fixed = np.concatenate(
        [
            sizes,
            [origin[dim] for dim in VOXEL_DIMS],
            spacing,
            transform.fusi.direction.ravel(),
        ]
    ).astype(np.float64)
    tx = itk.BSplineTransform[itk.D, 3, 3].New()
    tx.SetFixedParameters(itk.OptimizerParameters[itk.D](fixed))
    params = np.ascontiguousarray(
        transform.transpose("component", *VOXEL_DIMS).values.transpose(0, 3, 2, 1)
    ).ravel()
    tx.SetParameters(itk.OptimizerParameters[itk.D](params.astype(np.float64)))
    pre_affine = transform.attrs.get("affines", {}).get("bspline_initialization")
    if pre_affine is None:
        return tx
    composite = itk.CompositeTransform[itk.D, 3].New()
    composite.AddTransform(affine_to_itk_linear_transform(np.asarray(pre_affine)))
    composite.AddTransform(tx)
    return composite


def itk_bspline_to_dataarray(
    transform: "itk.Transform",
    pre_affine: npt.NDArray[np.floating] | None = None,
) -> xr.DataArray:
    """Serialize ITK B-spline coefficients and their lattice geometry.

    Parameters
    ----------
    transform : itk.Transform
        Cubic B-spline or composite with the B-spline as its innermost transform.
    pre_affine : (4, 4) numpy.ndarray, optional
        Affine applied after the B-spline.

    Returns
    -------
    xarray.DataArray
        Coefficient VoxelData in `(component, k, j, i)` order.

    Raises
    ------
    TypeError
        If `transform` does not contain a B-spline.
    """
    import itk

    if transform.GetNameOfClass() == "CompositeTransform":
        transform = itk.down_cast(transform.GetBackTransform())
    if transform.GetNameOfClass() != "BSplineTransform":
        raise TypeError("Expected an ITK BSplineTransform.")
    fixed = np.asarray(transform.GetFixedParameters())
    shape = tuple(int(v) for v in fixed[:3])
    values = (
        np.asarray(transform.GetParameters())
        .reshape((3, *shape[::-1]))
        .transpose(0, 3, 2, 1)
        .copy()
    )
    affine = np.eye(4)
    affine[:3, :3] = fixed[9:].reshape(3, 3) @ np.diag(fixed[6:9])
    affine[:3, 3] = fixed[3:6]
    attrs = {
        "type": "bspline_transform",
        "transform_type": "bspline_transform",
        "order": 3,
    }
    if pre_affine is not None:
        attrs["affines"] = {"bspline_initialization": np.asarray(pre_affine).tolist()}
    return create_voxeldata(
        values,
        voxel_to_world=affine,
        dims=["component", *VOXEL_DIMS],
        extra_coords={"component": list(VOXEL_DIMS)},
        attrs=attrs,
    )


def voxeldata_to_itk_displacement_field(field: xr.DataArray) -> "itk.Image":
    """Convert a displacement VoxelData array to an ITK vector image.

    Parameters
    ----------
    field : xarray.DataArray
        Dense world displacement vectors in `(component, k, j, i)` order.

    Returns
    -------
    itk.Image
        Owned double vector image, suitable for `DisplacementFieldTransform`.

    Raises
    ------
    ValueError
        If the displacement field is invalid.
    """
    import itk

    validate_displacement_field(field)
    image = itk.Image[itk.Vector[itk.D, 3], 3].New()
    image.SetRegions([field.sizes[dim] for dim in VOXEL_DIMS])
    image.SetSpacing([float(field.fusi.spacing[dim]) for dim in VOXEL_DIMS])
    image.SetOrigin([float(field.fusi.origin[dim]) for dim in WORLD_DIMS])
    image.SetDirection(
        itk.matrix_from_array(np.asarray(field.fusi.direction, dtype=np.float64))
    )
    # Double-vector ImageDuplicator is not wrapped; allocate owned ITK memory.
    image.Allocate()
    itk.array_view_from_image(image)[:] = field.transpose(
        "i", "j", "k", "component"
    ).values
    return image


def sample_displacement_field(
    transform: xr.DataArray,
    *,
    dims: Sequence[str],
    shape: Sequence[int],
    spacing: Sequence[float],
    origin: Sequence[float],
    direction: npt.ArrayLike | None = None,
    itk_threads: int = -1,
) -> xr.DataArray:
    """Sample a B-spline pull transform on an explicit voxel grid.

    Parameters
    ----------
    transform : xarray.DataArray
        B-spline coefficient VoxelData, optionally including affine initialization.
    dims : sequence of str
        Native voxel dimensions, `['k', 'j', 'i']`.
    shape : sequence of int
        Output grid sizes in native order.
    spacing : sequence of float
        Output voxel spacing in native dimension order.
    origin : sequence of float
        Output world origin in native dimension order.
    direction : (3, 3) numpy.ndarray, optional
        Output direction. If not provided, use the identity direction.
    itk_threads : int, default: -1
        Number of ITK workers, with joblib semantics.

    Returns
    -------
    xarray.DataArray
        Dense world displacement VoxelData, including any affine contribution.

    Raises
    ------
    ValueError
        If the transform or grid is invalid.
    """
    if direction is not None and np.asarray(direction).shape != (len(dims), len(dims)):
        raise ValueError(
            f"direction must have shape {(len(dims), len(dims))}, got {np.asarray(direction).shape}."
        )
    validate_bspline(transform)
    resolve_itk_threads(itk_threads)
    if tuple(dims) != VOXEL_DIMS or len(shape) != 3:
        raise ValueError(
            "Output grid must use native voxel dimensions ['k', 'j', 'i']."
        )
    direction_array = (
        np.eye(3) if direction is None else np.asarray(direction, dtype=np.float64)
    )
    order = int(transform.attrs["order"])
    if order not in range(4):
        raise ValueError("B-spline order must be between 0 and 3.")
    affine = np.eye(4)
    affine[:3, :3] = direction_array @ np.diag(spacing)
    affine[:3, 3] = origin
    world = (
        np.einsum("ij,j...->i...", affine[:3, :3], np.indices(shape, dtype=np.float64))
        + np.asarray(origin)[:, None, None, None]
    )
    _, coefficient_spacing = get_defined_spatial_spacing(
        transform.isel(component=0, drop=True)
    )
    coefficient_basis = transform.fusi.direction @ np.diag(coefficient_spacing)
    coefficient_origin = np.asarray([transform.fusi.origin[dim] for dim in WORLD_DIMS])
    indices = np.einsum(
        "ij,j...->i...",
        np.linalg.inv(coefficient_basis),
        world - coefficient_origin[:, None, None, None],
    )
    # The wheel's sampler outputs only float32; evaluate the same cardinal basis in double precision.
    coefficients = transform.transpose("component", *VOXEL_DIMS).values.astype(
        np.float64
    )
    displacement = np.stack(
        [
            map_coordinates(
                component, indices, order=order, prefilter=False, mode="nearest"
            )
            for component in coefficients
        ]
    )
    inside = np.ones(tuple(shape), dtype=bool)
    for axis, size in enumerate(coefficients.shape[1:]):
        inside &= (indices[axis] >= (order - 1) / 2) & (
            indices[axis] <= size - (order + 1) / 2
        )
    if order == 0:
        # ITK's unsigned (order - 1) support offset makes order-zero transforms identity.
        inside[:] = False
    displacement[:, ~inside] = 0
    pre_affine = transform.attrs.get("affines", {}).get("bspline_initialization")
    if pre_affine is not None:
        pre_affine = np.asarray(pre_affine)
        displacement = (
            np.einsum("ij,j...->i...", pre_affine[:3, :3], world + displacement)
            + pre_affine[:3, 3, None, None, None]
            - world
        )
    return create_voxeldata(
        displacement,
        voxel_to_world=affine,
        dims=["component", *VOXEL_DIMS],
        extra_coords={"component": list(VOXEL_DIMS)},
        attrs={"type": "displacement_field_transform"},
    )


def sample_displacement_field_like(
    transform: xr.DataArray,
    reference: xr.DataArray,
    *,
    itk_threads: int = -1,
) -> xr.DataArray:
    """Sample a B-spline pull transform on a reference VoxelData grid.

    Parameters
    ----------
    transform : xarray.DataArray
        B-spline coefficient VoxelData.
    reference : xarray.DataArray
        VoxelData defining the output grid.
    itk_threads : int, default: -1
        Number of ITK workers, with joblib semantics.

    Returns
    -------
    xarray.DataArray
        Dense displacement VoxelData on the reference grid.

    Raises
    ------
    ValueError
        If either VoxelData array is invalid.
    """
    from confusius._utils.coordinates import get_grid_info_from_dataarray

    if "time" in reference.dims:
        raise ValueError(
            f"'reference' must not have a time dimension; got dims {reference.dims}."
        )
    transform = ensure_voxeldata(transform, allow_pose=False)
    reference = ensure_voxeldata(reference, allow_pose=False, allow_extra_dims=False)
    validate_matching_spatial_units(
        (("transform", transform), ("reference", reference))
    )
    return sample_displacement_field(
        transform,
        **get_grid_info_from_dataarray(reference),
        direction=reference.fusi.direction,
        itk_threads=itk_threads,
    )


def invert_displacement_field(
    field: xr.DataArray,
    *,
    max_iterations: int = 20,
    max_error_tolerance: float = 0.1,
    itk_threads: int = -1,
) -> xr.DataArray:
    """Estimate the inverse of a displacement pull field by fixed-point iteration.

    Parameters
    ----------
    field : xarray.DataArray
        Dense displacement VoxelData to invert on its own grid.
    max_iterations : int, default: 20
        Maximum number of fixed-point updates.
    max_error_tolerance : float, default: 0.1
        Stopping threshold for the maximum world-space residual norm.
    itk_threads : int, default: -1
        Worker count, validated but unused by the single-threaded SciPy solver.

    Returns
    -------
    xarray.DataArray
        Approximate inverse displacement on the original grid. Boundary vectors
        are zero along axes of at least four voxels, as for ITK's iterative
        inverter after expansion and cropping of thin axes.

    Raises
    ------
    ValueError
        If the field, iteration count, tolerances, or worker count is invalid.

    Notes
    -----
    ITK wheels do not wrap its inverse-displacement filters. The same inverse
    relation is solved here: `v(x) = -u(x + v(x))`, with linear interpolation.
    As with ITK's inverter, folded or large deformations need not converge.
    """
    validate_displacement_field(field)
    resolve_itk_threads(itk_threads)
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if not np.isfinite(max_error_tolerance) or max_error_tolerance < 0:
        raise ValueError("max_error_tolerance must be finite and nonnegative.")
    values = field.transpose("component", *VOXEL_DIMS).values.astype(np.float64)
    shape = values.shape[1:]
    grid = np.indices(shape, dtype=np.float64)
    basis = field.fusi.direction @ np.diag(
        [field.fusi.spacing[dim] for dim in VOXEL_DIMS]
    )
    world_to_index = np.linalg.inv(basis)
    boundary = np.zeros(shape, dtype=bool)
    for axis, size in enumerate(shape):
        if size >= 4:
            boundary |= (grid[axis] == 0) | (grid[axis] == size - 1)
    inverse = np.zeros_like(values)
    for _ in range(max_iterations):
        indices = grid + np.einsum("ij,j...->i...", world_to_index, inverse)
        sampled = np.stack(
            [
                map_coordinates(component, indices, order=1, mode="nearest")
                for component in values
            ]
        )
        residual = sampled + inverse
        residual[:, boundary] = 0
        inverse -= residual
        norms = np.linalg.norm(residual[:, ~boundary], axis=0)
        if norms.size == 0 or norms.max() <= max_error_tolerance:
            break
    return field.copy(data=inverse)
