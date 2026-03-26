"""Xarray accessor for affine transform operations."""

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from confusius._dims import SPATIAL_DIMS

if TYPE_CHECKING:
    import numpy.typing as npt


def affine_to(
    da: xr.DataArray,
    other: xr.DataArray,
    via: str,
) -> "npt.NDArray[np.float64]":
    """Return the affine mapping `da`'s physical space into `other`'s.

    Computes `inv(other.attrs["affines"][via]) @ da.attrs["affines"][via]`,
    giving the transform that takes coordinates expressed in `da`'s
    physical frame and expresses them in `other`'s physical frame.  Both
    arrays must carry an `"affines"` dict in their `attrs` with the key
    `via`.

    Parameters
    ----------
    da : xarray.DataArray
        The source scan (origin physical space).
    other : xarray.DataArray
        The scan whose physical space is the target.
    via : str
        Key into `attrs["affines"]` that names the shared intermediate
        coordinate space used to bridge the two physical frames (e.g.
        `"physical_to_lab"`).

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Homogeneous affine matrix mapping `da`'s physical coordinates
        to `other`'s physical coordinates.

    Raises
    ------
    KeyError
        If `via` is not present in `da.attrs["affines"]` or
        `other.attrs["affines"]`.
    ValueError
        If either array does not have an `"affines"` entry in its `attrs`.
    """
    if "affines" not in da.attrs:
        raise ValueError("self does not have an 'affines' entry in attrs.")
    if "affines" not in other.attrs:
        raise ValueError("other does not have an 'affines' entry in attrs.")
    self_affine: "npt.NDArray[np.float64]" = np.asarray(
        da.attrs["affines"][via], dtype=np.float64
    )
    other_affine: "npt.NDArray[np.float64]" = np.asarray(
        other.attrs["affines"][via], dtype=np.float64
    )
    return np.linalg.inv(other_affine) @ self_affine


def apply_affine(
    da: xr.DataArray,
    affine: "npt.NDArray[np.float64]",
    inplace: bool = False,
) -> xr.DataArray:
    """Apply a rigid affine to a DataArray's spatial coordinates.

    Only axis-aligned transforms are supported: the rotation block of `affine` must be
    diagonal (each output axis maps to exactly one input axis, possibly with a sign flip
    and a scale factor).  Transforms that mix axes (e.g. a 45° rotation) cannot be
    represented as independent 1-D coordinate arrays and therefore raise `ValueError`.
    For such cases, resample the data instead.

    All stored affines in `da.attrs["affines"]` are updated by composing `affine` on the
    right, keeping them valid in the new coordinate frame. Per-pose `(npose, 4, 4)`
    stacks are handled correctly: `affine` is applied to each pose's `(4, 4)` slice
    independently via broadcasting.

    Parameters
    ----------
    da : xarray.DataArray
        Input scan.  Must have at least one of `"z"`, `"y"`, `"x"` as dimensions with
        associated 1-D coordinates.
    affine : numpy.ndarray, shape (4, 4)
        Homogeneous affine matrix to apply.  The rotation block `affine[:3, :3]` must be
        diagonal.
    inplace : bool, default: False
        Whether to modify the DataArray in-place.

    Returns
    -------
    xarray.DataArray
        `da` with updated spatial coordinates and updated `attrs["affines"]`.

    Raises
    ------
    ValueError
        If `affine` is not shape `(4, 4)`.
    ValueError
        If the rotation block of `affine` is not diagonal (axis-mixing
        detected).  Use resampling for non-axis-aligned transforms.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import confusius  # noqa: F401
    >>> data = xr.DataArray(
    ...     np.zeros((3, 4)),
    ...     dims=["z", "y"],
    ...     coords={"z": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0, 3.0]},
    ... )
    >>> shift = np.eye(4)
    >>> shift[:3, 3] = [10.0, 5.0, 0.0]
    >>> result = data.fusi.affine.apply(shift)
    >>> float(result.coords["z"].values[0])
    10.0
    """
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"affine must have shape (4, 4), got {affine.shape}.")

    rotation = affine[:3, :3]
    off_diag_mask = ~np.eye(3, dtype=bool)
    if np.any(np.abs(rotation[off_diag_mask]) > 1e-9):
        raise ValueError(
            "The rotation block of affine is not diagonal: this transform "
            "mixes spatial axes and cannot be represented as independent 1-D "
            "coordinate arrays.  Use resampling for non-axis-aligned "
            "transforms.\n"
            f"Rotation block:\n{rotation}"
        )

    # Apply the affine to each spatial dimension's 1D coordinate array.
    # For a diagonal rotation the new coord for axis i is:
    #   new_coord_i = rotation[i, i] * old_coord_i + translation[i]
    spatial_axes = [0, 1, 2]  # z, y, x map to affine rows 0, 1, 2.
    dim_names = list(SPATIAL_DIMS)
    new_coords = dict(da.coords)

    for axis, dim in zip(spatial_axes, dim_names):
        if dim not in da.dims:
            continue
        old_coord = da.coords[dim].values.astype(np.float64)
        scale_factor = rotation[axis, axis]
        translation = affine[axis, 3]
        new_coord = scale_factor * old_coord + translation
        new_coords[dim] = xr.DataArray(
            new_coord,
            dims=[dim],
            attrs=da.coords[dim].attrs,
        )

    # Update all stored affines so they stay valid in the new coordinate frame.
    # Each stored affine M satisfies: lab_pos = M @ physical_pos.
    # After the coordinate change physical_pos_new = affine @ physical_pos_old,
    # the updated affine M_new must satisfy: lab_pos = M_new @ physical_pos_new.
    # Therefore: M_new = M_old @ inv(affine).
    inv_affine = np.linalg.inv(affine)
    new_affines: dict[str, "npt.NDArray[np.float64]"] = {}
    stored = da.attrs.get("affines", {})
    for key, val in stored.items():
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim == 2:
            # Shape (4, 4): single affine.
            new_affines[key] = arr @ inv_affine
        elif arr.ndim == 3:
            # Shape (npose, 4, 4): broadcast over poses.
            new_affines[key] = arr @ inv_affine
        else:
            # Unexpected shape: pass through unchanged.
            new_affines[key] = arr

    new_attrs = {**da.attrs, "affines": new_affines}
    result = da.assign_coords(new_coords).assign_attrs(new_attrs)
    if inplace:
        # xarray DataArrays are not truly mutable; update the underlying variable
        # in-place so callers holding a reference see the change.
        da.coords.update(result.coords)
        da.attrs.update(new_attrs)
        return da
    return result


class FUSIAffineAccessor:
    """Accessor for affine transform operations on fUSI DataArrays.

    Provides methods to compute relative transforms between scans and to
    apply axis-aligned affines to a scan's spatial coordinates.

    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The `DataArray` to wrap.
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def to(self, other: xr.DataArray, via: str) -> "npt.NDArray[np.float64]":
        """Return the affine mapping `self`'s physical space into `other`'s.

        Computes `inv(other.attrs["affines"][via]) @ self.attrs["affines"][via]`,
        giving the transform from `self`'s physical frame to `other`'s.

        Parameters
        ----------
        other : xarray.DataArray
            The scan whose physical space is the target.
        via : str
            Key into `attrs["affines"]` naming the shared intermediate
            coordinate space (e.g. `"physical_to_lab"`).

        Returns
        -------
        numpy.ndarray, shape (4, 4)
            Homogeneous affine matrix mapping `self`'s physical coordinates
            to `other`'s physical coordinates.

        Raises
        ------
        KeyError
            If `via` is not present in either scan's `attrs["affines"]`.
        ValueError
            If either scan has no `"affines"` entry in `attrs`.

        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> import confusius  # noqa: F401
        >>> eye = np.eye(4)
        >>> a = xr.DataArray(np.zeros((2, 2)), attrs={"affines": {"to_world": eye}})
        >>> b = xr.DataArray(np.zeros((2, 2)), attrs={"affines": {"to_world": eye}})
        >>> np.allclose(a.fusi.affine.to(b, via="to_world"), np.eye(4))
        True
        """
        return affine_to(self._obj, other, via)

    def apply(
        self, affine: "npt.NDArray[np.float64]", inplace: bool = False
    ) -> xr.DataArray:
        """Apply an axis-aligned affine to the scan's spatial coordinates.

        Updates `z`, `y`, and `x` coordinate arrays so that `plot_volume` places the
        scan in the transformed frame without resampling data values. All affines stored
        in `attrs["affines"]` are updated consistently.

        Parameters
        ----------
        affine : numpy.ndarray, shape (4, 4)
            Homogeneous affine matrix.  The rotation block must be diagonal
            (axis-aligned transforms only).
        inplace : bool, default: False
            Whether to modify the DataArray in-place.

        Returns
        -------
        xarray.DataArray
            The DataArray with updated spatial coordinates and `attrs["affines"]`.

        Raises
        ------
        ValueError
            If `affine` shape is not `(4, 4)` or if the rotation block
            mixes axes.

        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> import confusius  # noqa: F401
        >>> data = xr.DataArray(
        ...     np.zeros((3, 4)),
        ...     dims=["z", "y"],
        ...     coords={"z": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0, 3.0]},
        ... )
        >>> shift = np.eye(4)
        >>> shift[:3, 3] = [10.0, 5.0, 0.0]
        >>> result = data.fusi.affine.apply(shift)
        >>> float(result.coords["z"].values[0])
        10.0
        """
        return apply_affine(self._obj, affine, inplace=inplace)
