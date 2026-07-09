"""Xarray accessor for affine transform operations."""

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from confusius._dims import SPATIAL_DIMS
from confusius._utils.coordinates import get_affine_in_axis_aligned_space
from confusius._utils.geometry import (
    has_voxel_affine_geometry,
    restore_physical_coords_from_voxel_affine,
)
from confusius.registration.affines import decompose_affine

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
) -> "tuple[xr.DataArray, npt.NDArray[np.float64]]":
    """Apply an affine to a DataArray's spatial coordinates.

    A diagonal affine (any per-axis scale or sign flip, plus translation) maps
    each spatial axis to itself and updates the independent 1D `z`, `y`, `x`
    coordinate arrays: `new_coord = scale * old_coord + translation`. The whole
    transform is absorbed into the coordinates, so the returned orientation is
    the identity.

    An affine that mixes axes (a rotation, shear, or axis permutation) cannot be
    expressed as independent 1D coordinates. The coordinates then absorb only the
    axis-aligned zoom and translation, and the residual orientation is returned
    as a 4x4 affine mapping the new physical coordinates to the affine's target
    world frame (`orientation @ new_physical == affine @ old_physical`). The
    caller decides what to do with it (compose it with a stored affine, store it
    under a key of their choosing, or ignore it).

    All affines already in `da.attrs["affines"]` are re-expressed against the new
    coordinate frame so they remain valid. Per-pose `(npose, 4, 4)` stacks are handled
    per pose via broadcasting.

    Parameters
    ----------
    da : xarray.DataArray
        Input scan. Must have at least one of `"z"`, `"y"`, `"x"` as dimensions
        with associated 1D coordinates.
    affine : numpy.ndarray, shape (4, 4)
        Homogeneous affine matrix to apply.
    inplace : bool, default: False
        Whether to modify the DataArray in-place.

    Returns
    -------
    result : xarray.DataArray
        `da` with updated spatial coordinates and updated `attrs["affines"]`.
    orientation : (4, 4) numpy.ndarray
        The residual orientation the coordinates could not absorb, mapping the
        new physical coordinates to the affine's target world frame. The identity
        when `affine` is diagonal.

    Raises
    ------
    ValueError
        If `affine` is not shape `(4, 4)`.

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
    >>> result, orientation = data.fusi.affine.apply(shift)
    >>> float(result.coords["z"].values[0])
    10.0
    """
    affine = np.asarray(affine, dtype=np.float64)

    if has_voxel_affine_geometry(da):
        voxel_to_physical = np.asarray(da.attrs["voxel_to_physical"], dtype=np.float64)
        if affine.shape != voxel_to_physical.shape:
            raise ValueError(
                "voxel-affine data requires an affine with shape matching "
                f"voxel_to_physical {voxel_to_physical.shape}, got {affine.shape}."
            )

        stored = da.attrs.get("affines", {})
        new_affines: dict[str, "npt.NDArray[np.float64]"] = {}
        inv_affine = np.linalg.inv(affine)
        for stored_key, val in stored.items():
            arr = np.asarray(val, dtype=np.float64)
            if arr.ndim in (2, 3):
                new_affines[stored_key] = arr @ inv_affine
            else:
                new_affines[stored_key] = arr

        new_attrs = dict(da.attrs)
        new_attrs["voxel_to_physical"] = affine @ voxel_to_physical
        if "affines" in da.attrs:
            new_attrs["affines"] = new_affines

        result = restore_physical_coords_from_voxel_affine(da.assign_attrs(new_attrs))
        orientation = np.eye(affine.shape[0], dtype=np.float64)
        if inplace:
            da.coords.update(result.coords)
            da.attrs.clear()
            da.attrs.update(result.attrs)
            return da, orientation
        return result, orientation

    if affine.shape != (4, 4):
        raise ValueError(f"affine must have shape (4, 4), got {affine.shape}.")

    # Classify the transform from its decomposition A = R @ diag(Z) @ S:
    #   - sign flips sit on the DIAGONAL of R (axis-aligned, not mixing),
    #   - a true rotation or axis permutation fills R's OFF-diagonal,
    #   - any shear is non-zero in S.
    # A non-mixing affine is fully absorbed into the independent 1D coordinates;
    # a mixing one absorbs only the axis-aligned zoom and leaves a residual
    # orientation, which is returned to the caller.
    translation, rotation, zoom, shear = decompose_affine(affine)
    off_diagonal = rotation[~np.eye(3, dtype=bool)]
    mixes_axes = bool(
        np.any(np.abs(off_diagonal) > 1e-9) or np.any(np.abs(shear) > 1e-9)
    )

    if mixes_axes:
        # When axes mix, the decomposition's signed zoom is not a canonical
        # per-axis coordinate scaling: decompose_affine keeps `rotation`
        # right-handed (`det(rotation) > 0`) by relocating any needed
        # reflection into one zoom entry. Absorb only the zoom magnitudes into
        # the independent 1D coordinates and leave permutations/reflections in
        # the residual orientation.
        zoom = np.abs(zoom)
        orientation = get_affine_in_axis_aligned_space(affine, translation, zoom)
    else:
        # Diagonal affine: each axis keeps its own SIGNED scale. Use the raw
        # diagonal, not the decomposed zoom -- decompose relocates a lone sign
        # flip onto axis 0, which would otherwise flip the wrong coordinate. The
        # coordinates absorb the whole transform, leaving no residual.
        zoom = np.diag(affine[:3, :3])
        orientation = np.eye(4)

    # Apply the axis-aligned part to each present spatial dimension's 1D coords:
    #   new_coord_i = zoom[i] * old_coord_i + translation[i].
    spatial_axes = [0, 1, 2]  # z, y, x map to affine rows 0, 1, 2.
    dim_names = list(SPATIAL_DIMS)
    new_coords = dict(da.coords)
    for axis, dim in zip(spatial_axes, dim_names):
        if dim not in da.dims:
            continue
        old_coord = da.coords[dim].values.astype(np.float64)
        new_coord = zoom[axis] * old_coord + translation[axis]
        new_coords[dim] = xr.DataArray(
            new_coord,
            dims=[dim],
            attrs=da.coords[dim].attrs,
        )

    # Re-express every stored affine against the new axis-aligned physical frame
    # so it stays valid: M_new = M_old @ inv(axis_aligned(translation, zoom)).
    # Per-pose (npose, 4, 4) stacks are handled via broadcasting in
    # get_affine_in_axis_aligned_space.
    stored = da.attrs.get("affines", {})
    new_affines: dict[str, "npt.NDArray[np.float64]"] = {}
    for stored_key, val in stored.items():
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim in (2, 3):
            new_affines[stored_key] = get_affine_in_axis_aligned_space(
                arr, translation, zoom
            )
        else:
            # Unexpected shape: pass through unchanged.
            new_affines[stored_key] = arr

    new_attrs = {**da.attrs, "affines": new_affines}
    result = da.assign_coords(new_coords).assign_attrs(new_attrs)
    if inplace:
        # xarray DataArrays are not truly mutable; update the underlying variable
        # in-place so callers holding a reference see the change.
        da.coords.update(result.coords)
        da.attrs.update(new_attrs)
        return da, orientation
    return result, orientation


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
        self,
        affine: "npt.NDArray[np.float64]",
        inplace: bool = False,
    ) -> "tuple[xr.DataArray, npt.NDArray[np.float64]]":
        """Apply an affine to the scan's spatial coordinates.

        A diagonal affine (any per-axis scale or sign flip, plus translation) maps
        each spatial axis to itself and updates the independent 1D `z`, `y`, `x`
        coordinate arrays: `new_coord = scale * old_coord + translation`. The whole
        transform is absorbed into the coordinates, so the returned orientation is
        the identity.

        An affine that mixes axes (a rotation, shear, or axis permutation) cannot be
        expressed as independent 1D coordinates. The coordinates then absorb only the
        axis-aligned zoom and translation, and the residual orientation is returned
        as a 4x4 affine mapping the new physical coordinates to the affine's target
        world frame (`orientation @ new_physical == affine @ old_physical`). The
        caller decides what to do with it.

        All affines already in `attrs["affines"]` are re-expressed against the new
        coordinate frame so they remain valid. Per-pose `(npose, 4, 4)` stacks are
        handled per pose via broadcasting.

        Parameters
        ----------
        affine : numpy.ndarray, shape (4, 4)
            Homogeneous affine matrix to apply.
        inplace : bool, default: False
            Whether to modify the DataArray in-place.

        Returns
        -------
        result : xarray.DataArray
            The DataArray with updated spatial coordinates and `attrs["affines"]`.
        orientation : (4, 4) numpy.ndarray
            The residual orientation the coordinates could not absorb, mapping the
            new physical coordinates to the affine's target world frame. The
            identity when `affine` is diagonal.

        Raises
        ------
        ValueError
            If `affine` shape is not `(4, 4)`.

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
        >>> result, orientation = data.fusi.affine.apply(shift)
        >>> float(result.coords["z"].values[0])
        10.0
        """
        return apply_affine(self._obj, affine, inplace=inplace)
