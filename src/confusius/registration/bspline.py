"""B-spline transform helpers for fUSI registration.

A B-spline deformation field is represented as a DataArray with:

- **dims**: `("component", <spatial dims>)` — e.g. `("component", "z", "y", "x")`.
- **coords**: physical mm positions of the control-point grid along each spatial axis.
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
- **coords**: physical mm positions of every voxel along each spatial axis.
- **attrs**: `{"type": "displacement_field_transform"}`.

Unlike the sparse B-spline coefficient lattice, a displacement field stores one
displacement vector per voxel of an explicit grid. It is produced by sampling a B-spline
(or composite) transform with
[`bspline_to_displacement_field`][confusius.registration.bspline.bspline_to_displacement_field]
and can be inverted with
[`invert_displacement_field`][confusius.registration.bspline.invert_displacement_field].
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.registration._utils import expand_thin_dims, set_sitk_thread_count
from confusius.registration.affines import affine_to_sitk_linear_transform

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
    component_coords = list(range(ndim))
    coords: dict[str, npt.NDArray[np.float64]] = {
        "component": np.array(component_coords),
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
        return transform  # type: ignore[return-value]
    if name == "CompositeTransform":
        n = transform.GetNumberOfTransforms()  # type: ignore[attr-defined]
        # The B-spline is the last sub-transform (it was added last and is optimised).
        last = transform.GetNthTransform(n - 1)  # type: ignore[attr-defined]
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


def bspline_to_displacement_field(
    transform: xr.DataArray,
    *,
    shape: Sequence[int],
    spacing: Sequence[float],
    origin: Sequence[float],
    dims: Sequence[str],
    sitk_threads: int = -1,
) -> xr.DataArray:
    """Sample a B-spline (or composite) transform into a dense displacement field.

    Parameters
    ----------
    transform : xarray.DataArray
        B-spline control-point DataArray as produced by `sitk_bspline_to_dataarray`.
    shape : sequence of int
        Number of voxels along each output axis, in DataArray dimension order.
    spacing : sequence of float
        Voxel spacing along each output axis, in DataArray dimension order.
    origin : sequence of float
        Physical origin (first voxel centre) along each output axis, in DataArray
        dimension order.
    dims : sequence of str
        Dimension names of the output displacement field.
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

    field_filter = sitk.TransformToDisplacementFieldFilter()
    field_filter.SetReferenceImage(ref)
    with set_sitk_thread_count(sitk_threads):
        field_sitk = field_filter.Execute(tx)

    return _sitk_displacement_field_to_dataarray(
        field_sitk, shape, spacing, origin, dims
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
        [`bspline_to_displacement_field`][confusius.registration.bspline_to_displacement_field].
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

    dims = list(field.dims[1:])
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

    spacing_dict = field.fusi.spacing
    origin_dict = field.fusi.origin
    spacing = [spacing_dict[d] if spacing_dict[d] is not None else 1.0 for d in dims]
    origin = [origin_dict[d] for d in dims]
    return _sitk_displacement_field_to_dataarray(
        inverted_expanded, shape, spacing, origin, dims
    )


def _sitk_displacement_field_to_dataarray(
    field: "sitk.Image",
    shape: Sequence[int],
    spacing: Sequence[float],
    origin: Sequence[float],
    dims: Sequence[str],
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
    coords: dict[str, tuple[str, npt.NDArray, dict[str, float]]] = {
        "component": ("component", np.arange(len(dims)), {})
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
        attrs={"type": "displacement_field_transform"},
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

    spatial_dims = list(da.dims[1:])
    # da.fusi.spacing falls back to the 'voxdim' coordinate attribute for singleton
    # spatial dims (e.g. a single 2D slice stored as a (1, y, x) array), where
    # coords[dim].diff(dim) is empty and .mean() would silently return NaN.
    spacing_dict = da.fusi.spacing
    origin_dict = da.fusi.origin
    spacing = [
        spacing_dict[dim] if spacing_dict[dim] is not None else 1.0
        for dim in spatial_dims
    ]
    origin = [origin_dict[dim] for dim in spatial_dims]

    # .T maps the first DataArray axis to SimpleITK's physical x-axis, matching the
    # convention used throughout confusius.registration (see dataarray_to_sitk_image).
    field = sitk.GetImageFromArray(da.values.T, isVector=True)
    field.SetSpacing(spacing)
    field.SetOrigin(origin)
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
