"""B-spline transform helpers for fUSI registration.

A B-spline deformation field is represented as a DataArray with:

- **dims**: `("component", <spatial dims>)` — e.g. `("component", "z", "y", "x")`.
- **coords**: `component` is labeled by the spatial dim names
  (e.g. `("z", "y", "x")`), and each spatial axis stores the physical mm
  positions of the control-point grid.
- **attrs**:

  ```python
  {
      "type":      "bspline_transform",
      "order":     3,                          # B-spline polynomial order
      "direction": [[...], [...], [...]],      # (ndim, ndim) direction cosine matrix
      "affines":   {
          "bspline_initialization": [[...]]   # optional (N+1, N+1) pre-affine;
                                              # only present when register_volume
                                              # was called with affine initialization.
      }
  }
  ```

When a pre-affine is stored in `attrs["affines"]["bspline_initialization"]`, the full
transform is a `CompositeTransform(pre_affine, bspline)` — i.e. the pre-affine is
applied *first* (coarse global alignment) and the B-spline is applied *second* (local
deformation refinement).  This mirrors the `inPlace=True` composite that SimpleITK
optimises during registration.

A dense displacement field is represented the same way, minus the B-spline-specific
attributes:

- **dims**: `("component", <spatial dims>)`.
- **coords**: `component` is labeled by the spatial dim names, and each
  spatial axis stores the physical mm positions of every voxel.
- **attrs**: `{"type": "displacement_field_transform", "direction": [[...], ...]}`.

Unlike the sparse B-spline coefficient lattice, a displacement field stores one
displacement vector per voxel of an explicit grid. It is produced by sampling a B-spline
(or composite) transform with
[`sample_displacement_field`][confusius.registration.bspline.sample_displacement_field]
or
[`sample_displacement_field_like`][confusius.registration.bspline.sample_displacement_field_like],
and can be inverted with
[`invert_displacement_field`][confusius.registration.bspline.invert_displacement_field].
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils.geometry import has_voxel_affine_geometry
from confusius.registration._utils import (
    expand_thin_dims,
    get_defined_spatial_spacing,
    set_sitk_thread_count,
)
from confusius.registration.affines import affine_to_sitk_linear_transform
from confusius.validation import (
    validate_fusi_dataarray,
    validate_matching_spatial_units,
)

if TYPE_CHECKING:
    import SimpleITK as sitk


def sitk_bspline_to_dataarray(
    transform: "sitk.Transform",
    pre_affine: npt.NDArray[np.floating] | None = None,
) -> xr.DataArray:
    """Convert a SimpleITK B-spline (or composite) transform to a DataArray.

    Parameters
    ----------
    transform : SimpleITK.Transform
        A `BSplineTransform` or a `CompositeTransform` whose last sub-transform
        is a `BSplineTransform`.  Any other type raises `TypeError`.
    pre_affine : (N+1, N+1) numpy.ndarray, optional
        Homogeneous affine matrix to store as
        `attrs["affines"]["bspline_initialization"]`. Pass the affine that was used as
        the pre-alignment so that `_dataarray_to_sitk_bspline` can reconstruct the full
        composite for resampling.  If not provided, no `"affines"` key is written.

    Returns
    -------
    xarray.DataArray
        B-spline control-point DataArray with `attrs["type"] == "bspline_transform"`.

    Raises
    ------
    TypeError
        If `transform` is not a `BSplineTransform` or a `CompositeTransform`
        containing a `BSplineTransform` as its last sub-transform.
    """
    import SimpleITK as sitk

    bspline = _extract_bspline(transform)
    ndim = bspline.GetDimension()
    order = bspline.GetOrder()

    coeff_images = bspline.GetCoefficientImages()

    # .T restores DataArray axis order, the same convention used throughout
    # confusius.registration (see dataarray_to_sitk_image): sitk axis i corresponds
    # directly to DataArray dim i, with no axis reversal. Stack components along a
    # new leading axis: shape (ndim, *grid_shape).
    coefficients = np.stack(
        [sitk.GetArrayFromImage(im).T for im in coeff_images], axis=0
    )

    # Grid geometry from the first coefficient image (all share the same grid). No
    # axis reversal: sitk axis i corresponds directly to DataArray dim i.
    spacing = np.array(coeff_images[0].GetSpacing())
    origin = np.array(coeff_images[0].GetOrigin())
    direction = np.array(coeff_images[0].GetDirection()).reshape(ndim, ndim)

    grid_shape = coefficients.shape[1:]  # (nz, ny, nx) or (ny, nx)
    spatial_dims = ["z", "y", "x"][-ndim:]  # ["y", "x"] or ["z", "y", "x"]
    coords: dict[str, object] = {
        "component": np.array(spatial_dims, dtype=np.str_),
    }
    for i, dim in enumerate(spatial_dims):
        coords[dim] = origin[i] + np.arange(grid_shape[i]) * spacing[i]

    attrs: dict[str, object] = {
        "type": "bspline_transform",
        "order": order,
        "direction": direction.tolist(),
    }
    if pre_affine is not None:
        attrs["affines"] = {"bspline_initialization": np.asarray(pre_affine).tolist()}

    return xr.DataArray(
        coefficients,
        dims=["component", *spatial_dims],
        coords=coords,
        attrs=attrs,
    )


def _dataarray_to_sitk_bspline(da: xr.DataArray) -> "sitk.Transform":
    """Reconstruct a SimpleITK transform from a B-spline DataArray.

    If `da.attrs["affines"]["bspline_initialization"]` is present, returns a
    `CompositeTransform(pre_affine, bspline)`; otherwise returns a plain
    `BSplineTransform`.

    Parameters
    ----------
    da : xarray.DataArray
        B-spline DataArray as produced by `_sitk_bspline_to_dataarray`.

    Returns
    -------
    SimpleITK.Transform
        A `BSplineTransform` or `CompositeTransform` ready to be passed to
        `sitk.Resample`.

    Raises
    ------
    ValueError
        If `da` does not look like a valid B-spline transform DataArray.
    """
    import SimpleITK as sitk

    _validate_bspline_dataarray(da)

    ndim = da.ndim - 1  # subtract the component axis
    order = int(da.attrs["order"])
    # No axis reversal: sitk axis i corresponds directly to DataArray dim i (see
    # sitk_bspline_to_dataarray, which stores this matrix with the same convention).
    direction_sitk = np.array(da.attrs["direction"])

    spatial_dims = list(da.dims[1:])  # e.g. ["z", "y", "x"]

    # Recover grid geometry from DataArray coordinates. The coordinates store the
    # physical position of each control-point node; spacing is the step between
    # consecutive nodes, and origin is the first node. No axis reversal needed.
    spacing_sitk = [float(da.coords[dim].diff(dim).mean()) for dim in spatial_dims]
    origin_sitk = [float(da.coords[dim][0]) for dim in spatial_dims]
    node_counts_sitk = [da.sizes[dim] for dim in spatial_dims]

    # FixedParameters layout (2D, length 10):
    #   [nodeCount_x, nodeCount_y, origin_x, origin_y, spacing_x, spacing_y,
    #    dir_00, dir_01, dir_10, dir_11]
    # For 3D, 13 entries (3 counts + 3 origin + 3 spacing + 9 direction).
    fixed_params = (
        node_counts_sitk
        + origin_sitk
        + spacing_sitk
        + direction_sitk.flatten().tolist()
    )

    bspline = sitk.BSplineTransform(ndim, order)
    bspline.SetFixedParameters(fixed_params)

    # Parameters vector: coefficient values concatenated across components in the
    # same flattened order that SimpleITK uses (C-order / row-major per component,
    # in sitk's own reversed-numpy-axis convention). da.values[d] is in DataArray
    # axis order, so .T converts back to sitk's native layout, the inverse of the .T
    # applied in sitk_bspline_to_dataarray.
    params = np.concatenate(
        [da.values[d].T.astype(np.float64).ravel() for d in range(ndim)]
    )
    bspline.SetParameters(params.tolist())

    pre_affine_list = da.attrs.get("affines", {}).get("bspline_initialization")
    if pre_affine_list is not None:
        pre_tx = affine_to_sitk_linear_transform(np.array(pre_affine_list))

        composite = sitk.CompositeTransform(ndim)
        composite.AddTransform(pre_tx)
        composite.AddTransform(bspline)
        return composite

    return bspline


def _extract_bspline(transform: "sitk.Transform") -> "sitk.BSplineTransform":
    """Return the BSplineTransform from a transform or its last composite sub-transform.

    Parameters
    ----------
    transform : SimpleITK.Transform
        A `BSplineTransform` or a `CompositeTransform` whose last sub-transform is
        a `BSplineTransform`.

    Returns
    -------
    SimpleITK.BSplineTransform

    Raises
    ------
    TypeError
        If no `BSplineTransform` can be found.
    """

    name = transform.GetName()
    if "BSpline" in name:
        return transform  # type: ignore
    if name == "CompositeTransform":
        n = transform.GetNumberOfTransforms()  # type: ignore
        # The B-spline is the last sub-transform (it was added last and is optimised).
        last = transform.GetNthTransform(n - 1)  # type: ignore
        if "BSpline" in last.GetName():
            return last
    raise TypeError(
        f"Expected a BSplineTransform or a CompositeTransform ending with a "
        f"BSplineTransform; got {transform.GetName()!r}."
    )


def _validate_bspline_dataarray(da: xr.DataArray) -> None:
    """Raise ValueError if `da` does not look like a valid B-spline transform DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to validate.

    Raises
    ------
    ValueError
        If `da.attrs["type"] != "bspline_transform"` or required attrs are missing.
    """
    if da.attrs.get("type") != "bspline_transform":
        raise ValueError(
            f"Expected a DataArray with attrs['type'] == 'bspline_transform'; "
            f"got {da.attrs.get('type')!r}."
        )
    for key in ("order", "direction"):
        if key not in da.attrs:
            raise ValueError(
                f"B-spline transform DataArray is missing required attribute {key!r}."
            )
    if da.dims[0] != "component":
        raise ValueError(
            f"B-spline transform DataArray must have 'component' as its first "
            f"dimension; got {da.dims[0]!r}."
        )


def sample_displacement_field(
    transform: xr.DataArray,
    *,
    shape: Sequence[int],
    spacing: Sequence[float],
    origin: Sequence[float],
    dims: Sequence[str],
    direction: npt.ArrayLike | None = None,
    sitk_threads: int = -1,
) -> xr.DataArray:
    """Sample a registration transform onto an explicit output grid.

    Low-level sampling primitive. For the common case of sampling onto the grid of
    another DataArray, use
    [`sample_displacement_field_like`][confusius.registration.sample_displacement_field_like]
    instead.

    Parameters
    ----------
    transform : xarray.DataArray
        Registration transform DataArray to sample. Currently this accepts B-spline
        control-point DataArrays produced by
        [`register_volume`][confusius.registration.register_volume].
    shape : sequence of int
        Number of voxels along each output axis, in DataArray dimension order.
    spacing : sequence of float
        Voxel spacing along each output axis, in DataArray dimension order.
    origin : sequence of float
        Physical origin (first voxel centre) along each output axis, in DataArray
        dimension order.
    dims : sequence of str
        Dimension names of the output displacement field.
    direction : array-like, optional
        Spatial direction matrix for the output grid, in DataArray spatial-dimension
        order. If not provided, the output grid is treated as axis-aligned.
    sitk_threads : int, default: -1
        Number of threads SimpleITK may use internally. Negative values resolve to
        `max(1, os.cpu_count() + 1 + sitk_threads)`, so `-1` means all CPUs, `-2`
        means all minus one, and so on.

    Returns
    -------
    xarray.DataArray
        Dense displacement field DataArray with `attrs["type"] ==
        "displacement_field_transform"`, one displacement vector per voxel of the
        requested grid.
    """
    import SimpleITK as sitk

    tx = _dataarray_to_sitk_bspline(transform)

    ref = sitk.Image(list(shape), sitk.sitkFloat32)
    ref.SetSpacing(list(spacing))
    ref.SetOrigin(list(origin))
    if direction is None:
        ref.SetDirection(np.eye(len(dims), dtype=np.float64).flatten().tolist())
    else:
        direction_array = np.asarray(direction, dtype=np.float64)
        if direction_array.shape != (len(dims), len(dims)):
            raise ValueError(
                f"direction must have shape {(len(dims), len(dims))}, got {direction_array.shape}."
            )
        ref.SetDirection(direction_array.flatten().tolist())

    field_filter = sitk.TransformToDisplacementFieldFilter()
    field_filter.SetReferenceImage(ref)
    with set_sitk_thread_count(sitk_threads):
        field_sitk = field_filter.Execute(tx)

    return _sitk_displacement_field_to_dataarray(
        field_sitk, shape, spacing, origin, dims, ref.GetDirection()
    )


def sample_displacement_field_like(
    transform: xr.DataArray, reference: xr.DataArray, *, sitk_threads: int = -1
) -> xr.DataArray:
    """Sample a registration transform onto the grid of a reference DataArray.

    Convenience wrapper around
    [`sample_displacement_field`][confusius.registration.sample_displacement_field] that
    extracts the output grid from `reference`'s coordinates.

    Parameters
    ----------
    transform : xarray.DataArray
        Registration transform DataArray to sample. Currently this accepts the B-spline
        control-point DataArrays produced by
        [`register_volume`][confusius.registration.register_volume].
    reference : xarray.DataArray
        DataArray defining the output grid. Must be 2D or 3D spatial (no time
        dimension). When spatial coordinate `units` metadata is present on both
        `transform` and `reference`, they must match.
    sitk_threads : int, default: -1
        Number of threads SimpleITK may use internally. Negative values resolve to
        `max(1, os.cpu_count() + 1 + sitk_threads)`, so `-1` means all CPUs, `-2` means
        all minus one, and so on.

    Returns
    -------
    xarray.DataArray
        Dense displacement field DataArray sampled on `reference`'s grid.

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
        reference,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=False,
        minimum_spatial_dims=2,
    )
    validate_matching_spatial_units(
        (("transform", transform), ("reference", reference))
    )

    dims, spacing = get_defined_spatial_spacing(reference)
    origin_dict = reference.fusi.origin
    result = sample_displacement_field(
        transform,
        shape=[int(reference.sizes[dim]) for dim in dims],
        spacing=spacing,
        origin=(
            [origin_dict[dim] for dim in dims]
            if not has_voxel_affine_geometry(reference)
            else [o for d, o in origin_dict.items() if d != "time"]
        ),
        dims=dims,
        direction=reference.fusi.direction,
        sitk_threads=sitk_threads,
    )
    return result.assign_coords(
        {
            dim: reference.coords[dim]
            for dim in reference.dims
            if dim in reference.coords
        }
    )


def invert_displacement_field(
    field: xr.DataArray,
    *,
    max_iterations: int = 20,
    max_error_tolerance: float = 0.1,
    sitk_threads: int = -1,
) -> xr.DataArray:
    """Invert a dense displacement field DataArray with SimpleITK.

    Uses `InvertDisplacementFieldImageFilter`, a fixed-point iterative solver, on the
    same grid as `field`. The inverse maps physical points in the opposite direction of
    `field`: if `field` is a fixed-to-moving pull transform, the returned field is
    (approximately) the corresponding moving-to-fixed pull transform.

    Parameters
    ----------
    field : xarray.DataArray
        Dense displacement field DataArray as produced by
        [`sample_displacement_field`][confusius.registration.sample_displacement_field]
        or
        [`sample_displacement_field_like`][confusius.registration.sample_displacement_field_like].
    max_iterations : int, default: 20
        Maximum number of fixed-point iterations.
    max_error_tolerance : float, default: 0.1
        Maximum error tolerance (mm) at which the solver is considered converged.
    sitk_threads : int, default: -1
        Number of threads SimpleITK may use internally. Negative values resolve to
        `max(1, os.cpu_count() + 1 + sitk_threads)`, so `-1` means all CPUs, `-2`
        means all minus one, and so on.

    Returns
    -------
    xarray.DataArray
        Inverted displacement field DataArray on the same grid as `field`.

    Raises
    ------
    ValueError
        If `field` does not look like a valid displacement field DataArray.
    """
    import SimpleITK as sitk

    dims = [str(dim) for dim in field.dims[1:]]
    shape = [field.sizes[d] for d in dims]

    field_sitk = _dataarray_to_sitk_displacement_field(field)

    invert_filter = sitk.InvertDisplacementFieldImageFilter()
    invert_filter.SetMaximumNumberOfIterations(max_iterations)
    invert_filter.SetMaxErrorToleranceThreshold(max_error_tolerance)

    with set_sitk_thread_count(sitk_threads):
        # InvertDisplacementFieldImageFilter needs a real spatial neighborhood along
        # every axis and silently returns an all-zero field otherwise, exactly the
        # case for fUSI data stored as a single 2D slice, e.g. (1, y, x). Reuse the
        # same expand-then-crop trick register_volume already uses for its own
        # thin-dimension images (see expand_thin_dims): a degenerate axis has no
        # spatial variation to invert against anyway, so replicating its one slice
        # and cropping back down afterward is lossless.
        expanded = expand_thin_dims(field_sitk)
        inverted_expanded = invert_filter.Execute(expanded)

    expanded_shape = list(inverted_expanded.GetSize())
    if expanded_shape != shape:
        # Crop from the interior, not the edge: InvertDisplacementFieldImageFilter
        # has no real neighbor structure beyond the boundary of the (expanded)
        # domain, so the outermost slices of the replicated axis are themselves
        # degenerate (they come back all zero). Any interior slice is equally
        # representative, since expand_thin_dims only replicates a single input
        # slice along that axis.
        index = [(s - o) // 2 for s, o in zip(expanded_shape, shape)]
        inverted_expanded = sitk.RegionOfInterest(
            inverted_expanded, size=shape, index=index
        )

    field_grid = field.isel(component=0, drop=True)
    _, spacing = get_defined_spatial_spacing(field_grid)
    origin = [field_grid.fusi.origin[d] for d in dims]
    direction = np.asarray(
        field.attrs.get("direction", np.eye(len(dims))), dtype=np.float64
    )
    return _sitk_displacement_field_to_dataarray(
        inverted_expanded, shape, spacing, origin, dims, direction.tolist()
    )


def _sitk_displacement_field_to_dataarray(
    field: "sitk.Image",
    shape: Sequence[int],
    spacing: Sequence[float],
    origin: Sequence[float],
    dims: Sequence[str],
    direction: Sequence[float],
) -> xr.DataArray:
    """Wrap a SimpleITK vector displacement field image as a DataArray.

    Parameters
    ----------
    field : SimpleITK.Image
        Vector image produced by `TransformToDisplacementFieldFilter` or
        `InvertDisplacementFieldImageFilter`.
    shape : sequence of int
        Number of voxels along each output axis, in DataArray dimension order.
    spacing : sequence of float
        Voxel spacing along each output axis, in DataArray dimension order.
    origin : sequence of float
        Physical origin (first voxel centre) along each output axis, in DataArray
        dimension order.
    dims : sequence of str
        Dimension names of the output DataArray.
    direction : sequence of float
        Spatial direction matrix in row-major order.

    Returns
    -------
    xarray.DataArray
        Displacement field DataArray with `attrs["type"] ==
        "displacement_field_transform"`.
    """
    import SimpleITK as sitk

    # .T restores DataArray axis order (component first), the inverse of the .T used
    # in _dataarray_to_sitk_displacement_field.
    array = sitk.GetArrayFromImage(field).T

    # Store spacing as a 'voxdim' coordinate attribute (the codebase-wide convention;
    # see e.g. confusius.io.load_nifti) rather than relying on consumers to recover it
    # via coords[d].diff(d).mean(). fUSI data routinely carries a singleton spatial
    # axis (a single 2D slice stored as a (1, y, x) array), for which diff() is empty
    # and .mean() silently returns NaN.
    coords: dict[str, object] = {
        "component": ("component", np.array(dims, dtype=np.str_), {})
    }
    for i, d in enumerate(dims):
        coords[d] = (
            d,
            origin[i] + np.arange(shape[i]) * spacing[i],
            {"voxdim": float(spacing[i])},
        )

    return xr.DataArray(
        array,
        dims=["component", *dims],
        coords=coords,
        attrs={
            "type": "displacement_field_transform",
            "direction": np.asarray(direction, dtype=np.float64)
            .reshape(len(dims), len(dims))
            .tolist(),
        },
    )


def _dataarray_to_sitk_displacement_field(da: xr.DataArray) -> "sitk.Image":
    """Reconstruct a SimpleITK vector displacement field image from a DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        Displacement field DataArray as produced by
        `_sitk_displacement_field_to_dataarray`.

    Returns
    -------
    SimpleITK.Image
        Vector image ready to be wrapped in a `DisplacementFieldTransform` or passed
        to `InvertDisplacementFieldImageFilter`.

    Raises
    ------
    ValueError
        If `da` does not look like a valid displacement field DataArray.
    """
    import SimpleITK as sitk

    _validate_displacement_field_dataarray(da)

    field_grid = da.isel(component=0, drop=True)
    spatial_dims, spacing = get_defined_spatial_spacing(field_grid)
    origin = [field_grid.fusi.origin[dim] for dim in spatial_dims]
    direction = np.asarray(
        da.attrs.get("direction", np.eye(len(spatial_dims))), dtype=np.float64
    )
    if direction.shape != (len(spatial_dims), len(spatial_dims)):
        direction = np.eye(len(spatial_dims), dtype=np.float64)

    # .T maps the first DataArray axis to SimpleITK's physical x-axis, matching the
    # convention used throughout confusius.registration (see dataarray_to_sitk_image).
    field = sitk.GetImageFromArray(da.values.T, isVector=True)
    field.SetSpacing(spacing)
    field.SetOrigin(origin)
    field.SetDirection(direction.flatten().tolist())
    return field


def _validate_displacement_field_dataarray(da: xr.DataArray) -> None:
    """Raise ValueError if `da` does not look like a valid displacement field DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to validate.

    Raises
    ------
    ValueError
        If `da.attrs["type"] != "displacement_field_transform"` or `da` does not have
        `"component"` as its first dimension.
    """
    if da.attrs.get("type") != "displacement_field_transform":
        raise ValueError(
            "Expected a DataArray with attrs['type'] == 'displacement_field_transform'; "
            f"got {da.attrs.get('type')!r}."
        )
    if da.dims[0] != "component":
        raise ValueError(
            f"Displacement field DataArray must have 'component' as its first "
            f"dimension; got {da.dims[0]!r}."
        )
