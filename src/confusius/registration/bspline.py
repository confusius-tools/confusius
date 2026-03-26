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
                                              # was called with initial_transform.
      }
  }
  ```

When a pre-affine is stored in `attrs["affines"]["bspline_initialization"]`, the full
transform is a `CompositeTransform(pre_affine, bspline)` — i.e. the pre-affine is
applied *first* (coarse global alignment) and the B-spline is applied *second* (local
deformation refinement).  This mirrors the `inPlace=True` composite that SimpleITK
optimises during registration.
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.registration.affines import _affine_to_sitk_linear_transform

if TYPE_CHECKING:
    import SimpleITK as sitk


def _sitk_bspline_to_dataarray(
    transform: "sitk.Transform",
    pre_affine: npt.NDArray[np.float64] | None = None,
) -> xr.DataArray:
    """Convert a SimpleITK B-spline (or composite) transform to a DataArray.

    Parameters
    ----------
    transform : SimpleITK.Transform
        A `BSplineTransform` or a `CompositeTransform` whose last sub-transform
        is a `BSplineTransform`.  Any other type raises `TypeError`.
    pre_affine : (N+1, N+1) numpy.ndarray or None, default: None
        Homogeneous affine matrix to store as
        `attrs["affines"]["bspline_initialization"]`.  Pass the affine that was
        used as the pre-alignment so that `_dataarray_to_sitk_bspline` can
        reconstruct the full composite for resampling.  When `None`, no
        `"affines"` key is written.

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

    # sitk.GetArrayFromImage returns (nz, ny, nx) for a 3-D image (numpy axis order).
    # Stack components along a new leading axis: shape (ndim, nz, ny, nx).
    coefficients = np.stack([sitk.GetArrayFromImage(im) for im in coeff_images], axis=0)

    # Grid geometry from the first coefficient image (all share the same grid).
    # SimpleITK uses (x, y, z) order; DataArray uses (z, y, x).  Reverse with [::-1].
    spacing_sitk = np.array(coeff_images[0].GetSpacing())  # (x, y, z)
    origin_sitk = np.array(coeff_images[0].GetOrigin())  # (x, y, z)
    direction_sitk = np.array(coeff_images[0].GetDirection()).reshape(ndim, ndim)

    # Permutation from (x, y, z) → (z, y, x): reverse index order.
    perm = np.arange(ndim - 1, -1, -1)

    spacing = spacing_sitk[perm]  # (z, y, x)
    origin = origin_sitk[perm]  # (z, y, x)
    # Permute both rows and columns of the direction matrix.
    direction = direction_sitk[np.ix_(perm, perm)]

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
    direction_zyx = np.array(da.attrs["direction"])  # (ndim, ndim) in (z, y, x) order
    perm = np.arange(ndim - 1, -1, -1)  # (z, y, x) → (x, y, z)
    direction_sitk = direction_zyx[np.ix_(perm, perm)]

    spatial_dims = list(da.dims[1:])  # e.g. ["z", "y", "x"]

    # Recover grid geometry from DataArray coordinates (z, y, x order).
    # The coordinates store the physical position of each control-point node;
    # spacing is the step between consecutive nodes, and origin is the first node.
    spacing_zyx = np.array(
        [float(da.coords[dim].diff(dim).mean()) for dim in spatial_dims]
    )
    origin_zyx = np.array([float(da.coords[dim][0]) for dim in spatial_dims])

    # SimpleITK needs (x, y, z) order.
    spacing_sitk = spacing_zyx[perm].tolist()
    origin_sitk = origin_zyx[perm].tolist()
    node_counts_zyx = [da.sizes[dim] for dim in spatial_dims]
    node_counts_sitk = [node_counts_zyx[i] for i in perm]  # (nx, ny[, nz])

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
    # same flattened order that SimpleITK uses (C-order / row-major per component).
    # da.values has shape (ndim, [nz,] ny, nx); each component is already in the
    # correct numpy (z, y, x) / (y, x) order that sitk.GetArrayFromImage produces.
    params = np.concatenate(
        [da.values[d].astype(np.float64).ravel() for d in range(ndim)]
    )
    bspline.SetParameters(params.tolist())

    pre_affine_list = da.attrs.get("affines", {}).get("bspline_initialization")
    if pre_affine_list is not None:
        pre_tx = _affine_to_sitk_linear_transform(np.array(pre_affine_list))

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
    """Raise ValueError if *da* does not look like a valid B-spline transform DataArray.

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
