"""Xarray accessor for affine transform operations."""

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from confusius._utils.geometry import (
    add_world_coords_from_voxel_affine,
    get_voxel_affine_spatial_dims,
    get_voxel_to_world_affine,
    has_voxel_world_geometry,
)

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
    self_affine: npt.NDArray[np.float64] = np.asarray(
        da.attrs["affines"][via], dtype=np.float64
    )
    other_affine: npt.NDArray[np.float64] = np.asarray(
        other.attrs["affines"][via], dtype=np.float64
    )
    return np.linalg.inv(other_affine) @ self_affine


def apply_affine(
    da: xr.DataArray,
    affine: "npt.NDArray[np.float64] | str",
    inplace: bool = False,
) -> "tuple[xr.DataArray, npt.NDArray[np.float64]]":
    """Apply a physical-space affine to voxel-affine geometry.

    The transform is composed into `attrs["voxel_to_world"]`, derived physical
    coordinates are regenerated, and existing `attrs["affines"]` entries are
    re-expressed against the new physical frame. Per-pose `(npose, N, N)` stacks
    are handled by NumPy broadcasting.

    Parameters
    ----------
    da : xarray.DataArray
        Input scan with voxel-affine geometry in `attrs["voxel_to_world"]`.
    affine : numpy.ndarray, shape (N, N), or str
        Homogeneous physical-space affine matrix to apply. If a string, it is
        looked up as a key in `da.attrs["affines"]`.
    inplace : bool, default: False
        Whether to modify the DataArray in-place.

    Returns
    -------
    result : xarray.DataArray
        `da` with updated spatial coordinates and updated `attrs["affines"]`.
    orientation : (N, N) numpy.ndarray
        Identity matrix with the same shape as `affine`.

    Raises
    ------
    ValueError
        If `da` lacks voxel-affine geometry, if `affine` shape does not match
        `attrs["voxel_to_world"]`, or if `affine` is a string and `da` has no
        `"affines"` entry in `attrs`.
    KeyError
        If `affine` is a string not present in `da.attrs["affines"]`.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import confusius  # noqa: F401
    >>> from confusius._utils.geometry import add_world_coords_from_voxel_affine
    >>> data = add_world_coords_from_voxel_affine(
    ...     xr.DataArray(np.zeros((3, 4)), dims=["j", "i"]), np.eye(3)
    ... )
    >>> shift = np.eye(3)
    >>> shift[:2, 2] = [10.0, 5.0]
    >>> result, orientation = data.fusi.affine.apply(shift)
    >>> float(result.attrs["voxel_to_world"][0, 2])
    10.0
    """
    if isinstance(affine, str):
        if "affines" not in da.attrs:
            raise ValueError("da does not have an 'affines' entry in attrs.")
        if affine not in da.attrs["affines"]:
            raise KeyError(f"'{affine}' not found in da.attrs['affines'].")
        affine = da.attrs["affines"][affine]
    affine_array = np.asarray(affine, dtype=np.float64)

    if not has_voxel_world_geometry(da):
        raise ValueError("DataArray must have voxel-affine geometry.")

    voxel_to_world = get_voxel_to_world_affine(da)
    if affine_array.shape != voxel_to_world.shape:
        raise ValueError(
            "voxel-affine data requires an affine with shape matching "
            f"voxel_to_world {voxel_to_world.shape}, got {affine_array.shape}."
        )

    stored = da.attrs.get("affines", {})
    new_affines: dict[str, npt.NDArray[np.float64]] = {}
    inv_affine = np.linalg.inv(affine_array)
    for stored_key, val in stored.items():
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim in (2, 3):
            new_affines[stored_key] = arr @ inv_affine
        else:
            new_affines[stored_key] = arr

    new_attrs = dict(da.attrs)
    new_attrs.pop("voxel_to_world", None)
    if "affines" in da.attrs:
        new_attrs["affines"] = new_affines

    voxel_dims = get_voxel_affine_spatial_dims(da)
    result = add_world_coords_from_voxel_affine(
        da.assign_attrs(new_attrs),
        affine_array @ voxel_to_world,
        voxel_dims=voxel_dims,
    )
    orientation = np.eye(affine_array.shape[0], dtype=np.float64)
    if inplace:
        da.coords.update(result.coords)
        da.attrs.clear()
        da.attrs.update(result.attrs)
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

    @property
    def voxel_to_world(self) -> "npt.NDArray[np.float64]":
        """Affine mapping native voxel coordinates to world coordinates.

        Returns
        -------
        numpy.ndarray
            Homogeneous voxel-to-world affine.
        """
        return get_voxel_to_world_affine(self._obj)

    def set_voxel_to_world(
        self, voxel_to_world: "npt.ArrayLike", *, inplace: bool = False
    ) -> xr.DataArray:
        """Replace voxel-to-world geometry.

        Parameters
        ----------
        voxel_to_world : numpy.typing.ArrayLike
            Homogeneous affine mapping native voxel coordinates to world coordinates.
        inplace : bool, default: False
            Whether to modify the wrapped DataArray in-place.

        Returns
        -------
        xarray.DataArray
            DataArray with rebuilt VoxelToWorldIndex-backed coordinates.
        """
        voxel_dims = get_voxel_affine_spatial_dims(self._obj)
        result = add_world_coords_from_voxel_affine(
            self._obj,
            voxel_to_world,
            voxel_dims=voxel_dims,
        )
        if inplace:
            self._obj.coords.update(result.coords)
            self._obj.attrs.clear()
            self._obj.attrs.update(result.attrs)
            return self._obj
        return result

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
        affine: "npt.NDArray[np.float64] | str",
        inplace: bool = False,
    ) -> "tuple[xr.DataArray, npt.NDArray[np.float64]]":
        """Apply a physical-space affine to voxel-affine geometry.

        The transform is composed into `attrs["voxel_to_world"]`, derived physical
        coordinates are regenerated, and existing `attrs["affines"]` entries are
        re-expressed against the new physical frame. Per-pose `(npose, N, N)` stacks
        are handled by NumPy broadcasting.

        Parameters
        ----------
        affine : numpy.ndarray, shape (N, N), or str
            Homogeneous physical-space affine matrix to apply. If a string, it is
            looked up as a key in `self.attrs["affines"]`.
        inplace : bool, default: False
            Whether to modify the DataArray in-place.

        Returns
        -------
        result : xarray.DataArray
            The DataArray with updated spatial coordinates and `attrs["affines"]`.
        orientation : (N, N) numpy.ndarray
            Identity matrix with the same shape as `affine`.

        Raises
        ------
        ValueError
            If `self` lacks voxel-affine geometry, if `affine` shape does not match
            `attrs["voxel_to_world"]`, or if `affine` is a string and `self` has
            no `"affines"` entry in `attrs`.
        KeyError
            If `affine` is a string not present in `self.attrs["affines"]`.

        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> import confusius  # noqa: F401
        >>> from confusius._utils.geometry import add_world_coords_from_voxel_affine
        >>> data = add_world_coords_from_voxel_affine(
        ...     xr.DataArray(np.zeros((3, 4)), dims=["j", "i"]), np.eye(3)
        ... )
        >>> shift = np.eye(3)
        >>> shift[:2, 2] = [10.0, 5.0]
        >>> result, orientation = data.fusi.affine.apply(shift)
        >>> float(result.attrs["voxel_to_world"][0, 2])
        10.0
        """
        return apply_affine(self._obj, affine, inplace=inplace)
