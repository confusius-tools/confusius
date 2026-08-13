"""Xarray accessor for affine transform operations."""

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from confusius._utils.geometry import (
    add_world_coords_from_voxel_affine,
    get_affine_orientation_matrix,
    get_voxel_affine_spatial_dims,
    get_voxel_affine_world_coord_names,
    get_voxel_to_world_affine,
    get_voxel_world_origin,
    get_voxel_world_spacing,
    has_voxel_world_geometry,
)

if TYPE_CHECKING:
    import numpy.typing as npt


def affine_to(
    da: xr.DataArray,
    other: xr.DataArray,
    via: str,
) -> "npt.NDArray[np.float64]":
    """Return the affine mapping `da`'s world space into `other`'s.

    Computes `inv(other.attrs["affines"][via]) @ da.attrs["affines"][via]`,
    giving the transform that takes coordinates expressed in `da`'s
    world frame and expresses them in `other`'s world frame.  Both
    arrays must carry an `"affines"` dict in their `attrs` with the key
    `via`.

    Parameters
    ----------
    da : xarray.DataArray
        The source scan (origin world space).
    other : xarray.DataArray
        The scan whose world space is the target.
    via : str
        Key into `attrs["affines"]` that names the shared intermediate
        coordinate space used to bridge the two world frames (e.g.
        `"world_to_lab"`).

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Homogeneous affine matrix mapping `da`'s world coordinates
        to `other`'s world coordinates.

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
) -> xr.DataArray:
    """Apply a world-space affine to voxel-affine geometry.

    The transform is composed into `attrs["voxel_to_world"]`, derived world
    coordinates are regenerated, and existing `attrs["affines"]` entries are
    re-expressed against the new world frame. Per-pose `(npose, N, N)` stacks
    are handled by NumPy broadcasting.

    Parameters
    ----------
    da : xarray.DataArray
        Input scan with voxel-affine geometry in `attrs["voxel_to_world"]`.
    affine : numpy.ndarray, shape (N, N), or str
        Homogeneous world-space affine matrix to apply. If a string, it is
        looked up as a key in `da.attrs["affines"]`.
    inplace : bool, default: False
        Whether to modify the DataArray in-place.

    Returns
    -------
    xarray.DataArray
        `da` with updated spatial coordinates and updated `attrs["affines"]`.
        When `affine` is a string, that key is dropped from the result: composing
        a stored affine with itself is deterministically identity, so the entry
        would carry no information -- the world frame now simply *is* that
        named space.

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
    >>> result = data.fusi.affine.apply(shift)
    >>> float(result.attrs["voxel_to_world"][0, 2])
    10.0
    """
    applied_key = affine if isinstance(affine, str) else None
    if applied_key is not None:
        if "affines" not in da.attrs:
            raise ValueError("da does not have an 'affines' entry in attrs.")
        if applied_key not in da.attrs["affines"]:
            raise KeyError(f"'{applied_key}' not found in da.attrs['affines'].")
        affine = da.attrs["affines"][applied_key]
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
        if stored_key == applied_key:
            # Composing this stored affine with itself is deterministically
            # identity (arr @ inv(arr) == I), regardless of what it held --
            # applying "by key" means "move the world frame to align with
            # this named affine," which by construction leaves nothing to
            # report here.
            continue
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
    if inplace:
        da.coords.update(result.coords)
        da.attrs.clear()
        da.attrs.update(result.attrs)
        return da
    return result


def reindex_voxels(da: xr.DataArray) -> xr.DataArray:
    """Rebase voxel coordinates to dense positions without moving world coordinates.

    A voxel-affine DataArray's stored `voxel_to_world` affine is defined in terms of
    voxel *coordinate values*, which stay unchanged across cropping or striding by
    design (see [VoxelToWorldIndex][confusius._utils.geometry.VoxelToWorldIndex]).
    Because of this, the affine generally does not describe where voxel *position*
    `(0, ..., 0)` sits in world space, or the world distance between
    consecutive positions, once `da` has been cropped or strided from a larger
    array. This replaces each voxel dimension's coordinate with `0, 1, ..., dim - 1`
    and rebuilds `voxel_to_world` so the resulting affine directly maps those dense
    positions to `da`'s existing world coordinates, producing a DataArray whose
    affine is directly usable by software that assumes dense, zero-based voxel
    indices (e.g. ITK, nilearn).

    Parameters
    ----------
    da : xarray.DataArray
        Input scan with voxel-affine geometry.

    Returns
    -------
    xarray.DataArray
        `da` with voxel coordinates rebased to `0, 1, ..., dim - 1` and an updated
        `voxel_to_world` affine. World coordinates are unchanged.

    Raises
    ------
    ValueError
        If `da` lacks voxel-affine geometry, or if world spacing is undefined for
        any voxel dimension.
    """
    if not has_voxel_world_geometry(da):
        raise ValueError("DataArray must have voxel-affine geometry.")

    voxel_dims = get_voxel_affine_spatial_dims(da)
    world_coord_names = get_voxel_affine_world_coord_names(da)

    spacing = get_voxel_world_spacing(da)
    missing_spacing = [dim for dim in voxel_dims if spacing[dim] is None]
    if missing_spacing:
        raise ValueError(
            f"Cannot reindex voxels because spacing is undefined for dimensions "
            f"{missing_spacing!r}."
        )
    origin = get_voxel_world_origin(da)
    direction = get_affine_orientation_matrix(get_voxel_to_world_affine(da))

    ndim = len(voxel_dims)
    new_affine = np.eye(ndim + 1, dtype=np.float64)
    new_affine[:ndim, :ndim] = direction @ np.diag([spacing[dim] for dim in voxel_dims])
    new_affine[:ndim, ndim] = [origin[name] for name in world_coord_names]

    world_coord_attrs = {
        name: dict(da.coords[name].attrs)
        for name in world_coord_names
        if name in da.coords
    }
    reindexed = da.assign_coords(
        {dim: np.arange(da.sizes[dim], dtype=np.float64) for dim in voxel_dims}
    )
    return add_world_coords_from_voxel_affine(
        reindexed,
        new_affine,
        voxel_dims=voxel_dims,
        world_coord_names=world_coord_names,
        world_coord_attrs=world_coord_attrs,
    )


def reindex_voxels_like(
    data: xr.DataArray, reference: xr.DataArray, *, atol: float = 1e-6
) -> xr.DataArray:
    """Rebase voxel coordinates onto `reference`'s voxel labels.

    `data` and `reference`'s `voxel_to_world` affines can differ even when they describe
    the exact same world grid: the affine is defined in terms of voxel *coordinate
    values*, so two arrays occupying identical world positions can still carry
    different affines if their voxel dimensions happen to be labeled differently (e.g.
    `reference` was cropped or strided from a larger array, while `data` was freshly
    built with dense labels). This verifies the two occupy the same world grid, then
    relabels `data`'s voxel coordinates and affine to match `reference`'s exactly, so
    the two become directly alignable (`.sel()`, arithmetic, `xarray.align`, ...) by
    voxel label as well as by world position.

    Parameters
    ----------
    data : xarray.DataArray
        Input scan with voxel-affine geometry, physically aligned with `reference`.
    reference : xarray.DataArray
        DataArray whose voxel labels and affine `data` should adopt.
    atol : float, default: 1e-6
        Absolute tolerance, in `reference`'s world units, for the world-coordinate
        alignment check between `data` and `reference`.

    Returns
    -------
    xarray.DataArray
        `data` with voxel coordinates and `voxel_to_world` replaced by `reference`'s.
        World coordinates are unchanged, since `data` and `reference` are verified to
        already occupy the same world grid.

    Raises
    ------
    ValueError
        If `data` or `reference` lacks voxel-affine geometry, if their voxel
        dimensions or shapes differ, or if their world coordinates do not match
        within `atol`.
    """
    if not has_voxel_world_geometry(data):
        raise ValueError("data must have voxel-affine geometry.")
    if not has_voxel_world_geometry(reference):
        raise ValueError("reference must have voxel-affine geometry.")

    voxel_dims = get_voxel_affine_spatial_dims(data)
    reference_voxel_dims = get_voxel_affine_spatial_dims(reference)
    if voxel_dims != reference_voxel_dims:
        raise ValueError(
            f"data and reference must have the same voxel dimensions; got "
            f"{voxel_dims!r} and {reference_voxel_dims!r}."
        )
    shape = tuple(data.sizes[dim] for dim in voxel_dims)
    reference_shape = tuple(reference.sizes[dim] for dim in voxel_dims)
    if shape != reference_shape:
        raise ValueError(
            f"data and reference must have the same voxel grid shape to reindex; "
            f"got {shape!r} and {reference_shape!r}."
        )

    world_coord_names = get_voxel_affine_world_coord_names(data)
    reference_world_coord_names = get_voxel_affine_world_coord_names(reference)
    for name, reference_name in zip(
        world_coord_names, reference_world_coord_names, strict=True
    ):
        data_values = data.coords[name].transpose(*voxel_dims).values
        reference_values = (
            reference.coords[reference_name].transpose(*voxel_dims).values
        )
        if not np.allclose(data_values, reference_values, atol=atol):
            raise ValueError(
                f"data and reference are not aligned in world space: coordinate "
                f"{name!r} differs by more than {atol}. reindex_voxels_like requires "
                "data to already occupy reference's exact world grid."
            )

    world_coord_attrs = {
        name: dict(reference.coords[name].attrs)
        for name in reference_world_coord_names
        if name in reference.coords
    }
    relabeled = data.assign_coords({dim: reference.coords[dim] for dim in voxel_dims})
    return add_world_coords_from_voxel_affine(
        relabeled,
        get_voxel_to_world_affine(reference),
        voxel_dims=voxel_dims,
        world_coord_names=reference_world_coord_names,
        world_coord_attrs=world_coord_attrs,
    )


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
        """Return the affine mapping `self`'s world space into `other`'s.

        Computes `inv(other.attrs["affines"][via]) @ self.attrs["affines"][via]`,
        giving the transform from `self`'s world frame to `other`'s.

        Parameters
        ----------
        other : xarray.DataArray
            The scan whose world space is the target.
        via : str
            Key into `attrs["affines"]` naming the shared intermediate
            coordinate space (e.g. `"world_to_lab"`).

        Returns
        -------
        numpy.ndarray, shape (4, 4)
            Homogeneous affine matrix mapping `self`'s world coordinates
            to `other`'s world coordinates.

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
    ) -> xr.DataArray:
        """Apply a world-space affine to voxel-affine geometry.

        The transform is composed into `attrs["voxel_to_world"]`, derived world
        coordinates are regenerated, and existing `attrs["affines"]` entries are
        re-expressed against the new world frame. Per-pose `(npose, N, N)` stacks
        are handled by NumPy broadcasting.

        Parameters
        ----------
        affine : numpy.ndarray, shape (N, N), or str
            Homogeneous world-space affine matrix to apply. If a string, it is
            looked up as a key in `self.attrs["affines"]`.
        inplace : bool, default: False
            Whether to modify the DataArray in-place.

        Returns
        -------
        xarray.DataArray
            The DataArray with updated spatial coordinates and `attrs["affines"]`.
            When `affine` is a string, that key is dropped from the result:
            composing a stored affine with itself is deterministically identity,
            so the entry would carry no information -- the world frame now
            simply *is* that named space.

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
        >>> result = data.fusi.affine.apply(shift)
        >>> float(result.attrs["voxel_to_world"][0, 2])
        10.0
        """
        return apply_affine(self._obj, affine, inplace=inplace)
