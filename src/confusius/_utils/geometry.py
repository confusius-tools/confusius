"""Geometry helpers for voxel-space to world-space transforms.

This module implements the indexing machinery underlying ConfUSIus's VoxelData
model: a DataArray keeps 1D voxel-space coordinates (for example `i`, `j`, `k`) and
stores a single affine that maps those voxel-space coordinates into world-space
coordinates (for example `x`, `y`, `z`).

The derived world coordinates are always exposed lazily via a single joint
[VoxelToWorldIndex][confusius._utils.geometry.VoxelToWorldIndex] backed by Xarray's
`CoordinateTransformIndex`, for both axis-aligned and oblique affines — see
`VoxelToWorldIndex`'s docstring for why a single shared index (rather than one per
axis) is required for compatibility with `.stack()`. For any world coordinate whose
own affine row depends on exactly one voxel dimension (every row, for an
axis-aligned affine; possibly just one row of an otherwise-oblique affine),
`VoxelToWorldIndex.sel` reimplements ordinary per-axis `.sel()` (slices, single
labels) directly against the joint transform, so it stays as ergonomic as a plain
1D coordinate despite each world coordinate technically depending on every voxel
dimension.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from xarray import Index, Variable
from xarray.core.indexing import IndexSelResult
from xarray.indexes import CoordinateTransform, CoordinateTransformIndex

from confusius._dims import POSE_DIM, VOXEL_DIMS, WORLD_DIMS
from confusius._utils.coordinates import get_representative_step


def _is_scalar_indexer(indexer: Any) -> bool:
    """Return whether `indexer` reduces its dimension to a scalar (drops it).

    Parameters
    ----------
    indexer : Any
        A single dimension's indexer, as passed to `Index.isel`.

    Returns
    -------
    bool
        Whether `indexer` is a bare integer or a 0D array/Variable. `slice`, `list`,
        `tuple`, and >=1D array/Variable indexers all return `False`, even when they
        select only one element (those keep the dimension, just with size 1).
    """
    if isinstance(indexer, slice | list | tuple):
        return False
    if isinstance(indexer, int | np.integer):
        return True
    if isinstance(indexer, Variable | np.ndarray):
        return np.ndim(indexer) == 0
    return False


def _scalar_indexer_value(indexer: Any) -> int:
    """Return the integer position selected by a scalar indexer.

    Parameters
    ----------
    indexer : Any
        A scalar indexer, as identified by
        [_is_scalar_indexer][confusius._utils.geometry._is_scalar_indexer].

    Returns
    -------
    int
        The selected dense position.
    """
    if isinstance(indexer, Variable):
        return int(indexer.values)
    return int(indexer)


def _reverse_lookup_positions(
    values: npt.NDArray[np.float64], voxel_axis: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return floating dense positions of `values` within `voxel_axis`.

    Parameters
    ----------
    values : numpy.ndarray
        Values to locate within `voxel_axis`.
    voxel_axis : numpy.ndarray
        1D array of voxel-space coordinates, in dense-position order.

    Returns
    -------
    numpy.ndarray
        Dense positions, same shape as `values`. When `voxel_axis` is monotonic
        (increasing or decreasing), positions are linearly interpolated, so
        `values` falling between two samples resolve to a fractional position.
        Otherwise, only exact matches resolve; a value with no exact match in
        `voxel_axis` resolves to `nan`.
    """
    shape = values.shape
    flat = values.reshape(-1)
    if voxel_axis.size <= 1 or np.all(np.diff(voxel_axis) > 0):
        xp, fp = voxel_axis, np.arange(voxel_axis.size, dtype=np.float64)
    elif np.all(np.diff(voxel_axis) < 0):
        xp = voxel_axis[::-1]
        fp = np.arange(voxel_axis.size - 1, -1, -1, dtype=np.float64)
    else:
        lookup = {
            float(value): float(position) for position, value in enumerate(voxel_axis)
        }
        return np.array(
            [lookup.get(float(value), np.nan) for value in flat], dtype=np.float64
        ).reshape(shape)
    return np.interp(flat, xp, fp).reshape(shape)


class VoxelToWorldIndex(Index):
    """Xarray index owning voxel-to-world geometry.

    Always wraps a single [VoxelToWorldTransform][confusius._utils.geometry.VoxelToWorldTransform]
    covering `z`/`y`/`x` jointly — for both oblique and axis-aligned affines. A single
    shared index (rather than one per axis) is what keeps `.stack()` working: xarray
    skips an index from stack-index consideration when it's associated with more than
    one coordinate name (`xarray.core.indexes.IndexVariable.is_multi`), which is
    exactly what lets `.stack(space=("k", "j", "i"))` fall through cleanly to
    `k`/`j`/`i`'s own plain indexes instead of colliding with `z`/`y`/`x`'s.

    For pose-dependent geometry, `pose` is deliberately *not* one of this index's
    owned coordinate names, even though the wrapped transform still depends on
    `pose`'s position to pick the right affine out of the stack (exactly like it
    depends on `k`/`j`/`i`'s positions). `pose` instead gets its own plain,
    independently indexed coordinate — the same treatment `k`/`j`/`i` already get.
    Coupling `pose` to this joint index (as `z`/`y`/`x`/`k`/`j`/`i` are) previously
    broke alignment after a scalar `.sel(pose=0)`: xarray's `_apply_indexes` blindly
    re-associates every *old* coordinate name with the new index object, so `pose`
    stayed registered against an index that no longer produced a `pose` variable,
    and aligning against a genuinely pose-free array (e.g. an atlas mask) raised a
    spurious `AlignmentError`. Splitting `pose` out sidesteps this: scalar `pose`
    selection is now resolved by `pose`'s own plain index, which xarray already
    knows how to correctly drop.

    One consequence: a single combined `.sel(pose=0, z=..., y=..., x=...)` call is no
    longer supported for pose-dependent geometry, because xarray resolves `pose` and
    the world coordinates against two different indexes and this index never sees the
    `pose` label. Reduce `pose` to a scalar first — `.isel(pose=0).sel(z=..., y=...,
    x=...)` or `.sel(pose=0).sel(z=..., y=..., x=...)` — see `VoxelToWorldIndex.sel`.

    For axis-aligned affines specifically, `.sel()` additionally reimplements ordinary
    per-axis selection (slices, single labels, independent per-axis queries) directly
    against the joint transform's diagonal — rather than delegating to
    `xarray.indexes.CoordinateTransformIndex.sel`, which only supports `nearest`,
    point-wise, all-axes-at-once queries. That's what a plain, non-index-backed
    coordinate would offer, and it's what most axis-aligned VoxelData (the common
    case) expects.

    Parameters
    ----------
    index : xarray.indexes.CoordinateTransformIndex
        Index wrapping a
        [VoxelToWorldTransform][confusius._utils.geometry.VoxelToWorldTransform].
    voxel_to_world : numpy.typing.ArrayLike
        Homogeneous voxel-to-world affine.
    units : str, default: "mm"
        Physical unit shared by every derived world coordinate (`z`/`y`/`x` are always
        expressed in the same unit, since they're derived jointly from one affine).
    """

    def __init__(
        self,
        index: CoordinateTransformIndex,
        voxel_to_world: npt.ArrayLike,
        units: str = "mm",
    ) -> None:
        self._index = index
        self.voxel_to_world = np.asarray(voxel_to_world, dtype=np.float64)
        self.units = units

    @property
    def voxel_dims(self) -> tuple[str, ...]:
        """Active voxel dimensions.

        Returns
        -------
        tuple[str, ...]
            Active voxel dimension names.
        """
        return tuple(dim for dim in self._index.transform.dims if dim != POSE_DIM)

    @property
    def is_pose_dependent(self) -> bool:
        """Whether this index owns a pose-dependent (stacked) affine.

        Returns
        -------
        bool
            Whether the wrapped transform still depends on a `pose` dimension (a
            stacked `voxel_to_world` affine, not yet reduced by a scalar `pose`
            selection).
        """
        transform = self._index.transform
        assert isinstance(transform, VoxelToWorldTransform)
        return transform.pose_coord is not None

    @property
    def fixed_voxel_coords(self) -> dict[str, float]:
        """Voxel-space coordinate value pinned by a prior scalar `isel` selection.

        The dimension was removed as an array dimension (e.g. by `.isel(k=0)`) but
        its contribution to the world coordinates is still tracked exactly, so it
        can later be reinstated by
        [canonicalize_voxeldata][confusius.validation.canonicalize_voxeldata].

        Returns
        -------
        dict[str, float]
            Pinned voxel-space coordinate value keyed by dimension name.
        """
        transform = self._index.transform
        assert isinstance(transform, VoxelToWorldTransform)
        return dict(transform.fixed_voxel_coords)

    @classmethod
    def from_affine(
        cls,
        active_voxel_coords: Mapping[str, npt.ArrayLike],
        voxel_to_world: npt.ArrayLike,
        *,
        units: str = "mm",
        pose_coord: npt.ArrayLike | None = None,
    ) -> Self:
        """Create a voxel-to-world index from an affine.

        Parameters
        ----------
        active_voxel_coords : mapping[str, array-like]
            Ordered mapping from voxel dimension name (`k`/`j`/`i`) to its 1D
            voxel-space coordinates.
        voxel_to_world : numpy.typing.ArrayLike
            Homogeneous affine mapping voxel-space coordinates to world-space
            coordinates, or a stack of `pose_coord`-many such affines with a leading
            pose axis.
        units : str, default: "mm"
            Physical unit shared by every derived world coordinate.
        pose_coord : numpy.typing.ArrayLike, optional
            1D pose coordinate labels. Required when `voxel_to_world` is a stack of
            affines (one per pose); must be left unset otherwise.

        Returns
        -------
        VoxelToWorldIndex
            Index wrapping the resolved joint transform, always exposing world
            coordinates `z`/`y`/`x`.
        """
        affine = np.asarray(voxel_to_world, dtype=np.float64)
        transform = VoxelToWorldTransform(
            active_voxel_coords, affine, pose_coord=pose_coord
        )
        return cls(
            CoordinateTransformIndex(transform),
            affine,
            units=units,
        )

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> dict[Hashable, Variable]:
        """Create coordinate variables from the wrapped transform index.

        Parameters
        ----------
        variables : mapping[Any, xarray.Variable], optional
            Unused compatibility argument required by Xarray's index API.

        Returns
        -------
        dict[Hashable, xarray.Variable]
            World coordinate variables, keyed by coordinate name. Never includes
            `pose`, which this index does not own -- see the class docstring.
        """
        new_variables = dict(self._index.create_variables())
        # Always regenerated from self.units, never merged with whatever the
        # variable's own attrs happened to hold before -- this is what keeps units
        # correct even after an xarray operation (e.g. a broadcasting xr.where) that
        # rebuilds the variable without properly re-deriving it from this index.
        for variable in new_variables.values():
            variable.attrs["units"] = self.units
        return new_variables

    def rename(
        self,
        name_dict: Mapping[Any, Hashable],
        dims_dict: Mapping[Any, Hashable],
    ) -> Self:
        """Rename dimensions and coordinates, keeping the wrapped transform consistent.

        The default `xarray.Index.rename` only renames the base
        `CoordinateTransform` fields (`dims`, `coord_names`, `dim_size`); it does not
        touch this module's own transform-specific fields (`active_voxel_coords`,
        `fixed_voxel_coords`, and the affine's original dimension order), which would
        otherwise desync from the renamed dimensions.

        Parameters
        ----------
        name_dict : mapping[Any, Hashable]
            Mapping of current coordinate names to desired names.
        dims_dict : mapping[Any, Hashable]
            Mapping of current dimension names to desired names.

        Returns
        -------
        VoxelToWorldIndex
            Renamed index.
        """
        transform = self._index.transform
        assert isinstance(transform, VoxelToWorldTransform)

        def _rename_dim(dim: str) -> str:
            return str(dims_dict.get(dim, dim))

        new_transform = VoxelToWorldTransform(
            {
                _rename_dim(dim): values
                for dim, values in transform.active_voxel_coords.items()
            },
            transform.voxel_to_world,
            fixed_voxel_coords={
                _rename_dim(dim): value
                for dim, value in transform.fixed_voxel_coords.items()
            },
            pose_coord=transform.pose_coord,
        )
        return type(self)(
            CoordinateTransformIndex(new_transform),
            self.voxel_to_world,
            units=self.units,
        )

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> Self | None:
        """Preserve voxel-to-world geometry through indexing.

        A scalar indexer on a voxel dimension fixes that dimension (see
        `fixed_voxel_coords`) rather than dropping the geometry outright.

        Parameters
        ----------
        indexers : mapping[Any, int or slice or numpy.ndarray or xarray.Variable]
            Indexers keyed by dimension name.

        Returns
        -------
        VoxelToWorldIndex or None
            The indexed index, or `None` when the indexer is unsupported (a
            multi-dimensional fancy index) or every active dimension would be fixed.
        """
        transform = self._index.transform
        assert isinstance(transform, VoxelToWorldTransform)
        new_transform = transform.isel(indexers)
        if new_transform is None:
            return None
        return type(self)(
            CoordinateTransformIndex(new_transform),
            new_transform.voxel_to_world,
            units=self.units,
        )

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        """Select by world coordinates.

        Each *requested* world coordinate is resolved independently per axis
        (supporting slices and single-axis queries, like an ordinary dimension
        coordinate) whenever its own affine row depends on exactly one voxel
        dimension -- true for every row on ordinary axis-aligned geometry, but also
        true for just one row of an otherwise-oblique affine (e.g. a probe swept
        along a physically world-aligned axis while the other two stay genuinely
        oblique). If any requested row depends on more than one voxel dimension,
        the whole query instead delegates to `CoordinateTransformIndex.sel`, which
        only supports point-wise `nearest` queries providing all axes at once.

        `pose` is a separate, independently indexed coordinate (see the class
        docstring), so a combined one-call query like `.sel(pose=0, z=..., y=...,
        x=...)` is not supported: xarray resolves `pose` and the world coordinates
        against two different indexes, and this index never sees the `pose` label.
        Reduce `pose` to a scalar first, e.g. `.isel(pose=0).sel(z=..., y=..., x=...)`
        or `.sel(pose=0).sel(z=..., y=..., x=...)`.

        Parameters
        ----------
        labels : dict[Any, Any]
            World coordinate labels to select, keyed by world coordinate name.
        method : str, optional
            Selection method. Unused when every requested axis resolves
            independently (per-axis selection always supports both exact and
            nearest lookup); forwarded to `CoordinateTransformIndex.sel` otherwise,
            defaulting to `"nearest"`.
        tolerance : Any, optional
            Forwarded to `CoordinateTransformIndex.sel` when the query doesn't
            resolve independently per axis.

        Returns
        -------
        xarray.core.indexing.IndexSelResult
            Resolved voxel-dimension indexers.

        Raises
        ------
        ValueError
            If `data` still carries pose-dependent (stacked) geometry and `labels`
            selects world coordinates; reduce `pose` to a scalar first.
        KeyError
            If an independently-resolved world coordinate label has no exact match
            and `method` is not `"nearest"`, or falls outside the axis's range.
        """
        transform = self._index.transform
        assert isinstance(transform, VoxelToWorldTransform)

        coord_labels: dict[Hashable, Any] = {
            name: labels[name] for name in WORLD_DIMS if name in labels
        }
        if transform.pose_coord is not None:
            if coord_labels:
                raise ValueError(
                    "Selecting world coordinates on pose-dependent geometry requires "
                    "reducing `pose` to a scalar first, e.g. `.sel(pose=0).sel(z=..., "
                    "y=..., x=...)`. A single combined "
                    "`.sel(pose=0, z=..., y=..., x=...)` call is not supported."
                )
            return IndexSelResult({})
        if not coord_labels:
            return IndexSelResult({})

        # A requested world coordinate is axis-aligned when its affine row has exactly
        # one nonzero column. Then it depends on one voxel dimension only. The test is
        # per row, not per affine: an oblique affine can still have one axis-aligned
        # row.
        linear = np.asarray(transform.voxel_to_world, dtype=np.float64)[:-1, :-1]
        single_axis: dict[Hashable, tuple[int, int]] = {}
        for name in coord_labels:
            row = WORLD_DIMS.index(name)
            nonzero_columns = np.nonzero(~np.isclose(linear[row], 0.0, atol=1e-12))[0]
            if nonzero_columns.size == 1:
                single_axis[name] = (row, int(nonzero_columns[0]))

        # If any requested coordinate is not axis-aligned, the whole query goes to
        # `CoordinateTransformIndex.sel`, which supports only nearest lookup with
        # all axes given at once.
        if len(single_axis) != len(coord_labels):
            return self._index.sel(
                coord_labels, method=method or "nearest", tolerance=tolerance
            )

        # Each axis-aligned coordinate resolves on its own voxel dimension. This
        # path supports slices, exact lookup, and nearest lookup.
        dim_indexers: dict[Hashable, Any] = {}
        for name, (row, column) in single_axis.items():
            dim = VOXEL_DIMS[column]
            if dim not in transform.active_voxel_coords:
                continue
            scale = linear[row, column]
            offset = transform.voxel_to_world[row, -1]
            voxel_axis = transform.active_voxel_coords[dim]
            label = coord_labels[name]
            if isinstance(label, slice):
                values = scale * voxel_axis + offset
                dim_indexers[dim] = pd.Index(values).slice_indexer(
                    label.start, label.stop, label.step
                )

            else:
                voxel_label = (np.asarray(label) - offset) / scale
                position = _reverse_lookup_positions(voxel_label, voxel_axis)
                # `np.interp` clamps out-of-domain queries to the boundary sample
                # instead of raising, so the domain check must run on `voxel_label`
                # itself rather than on the already-clamped `position`. `atol`
                # absorbs float roundoff from the `(label - offset) / scale`
                # division, e.g. an exact grid point landing a ULP outside its
                # own axis's bounds.
                atol = 1e-8
                valid = ~np.isnan(position) & (
                    (voxel_label >= np.min(voxel_axis) - atol)
                    & (voxel_label <= np.max(voxel_axis) + atol)
                )
                if method != "nearest":
                    valid &= np.isclose(position, np.rint(position), atol=atol)
                if not np.all(valid):
                    raise KeyError(
                        f"World coordinate {name}={label!r} not found along "
                        f"dimension {dim!r}."
                    )
                dim_indexers[dim] = np.rint(position).astype(int)
        return IndexSelResult(dim_indexers)

    def join(self, other: Index, how: str = "inner") -> Self:
        """Reject implicit alignment of different world grids with a useful error.

        Parameters
        ----------
        other : xarray.Index
            The other index Xarray is trying to align with this one.
        how : str, default: "inner"
            Join method requested by Xarray.

        Returns
        -------
        VoxelToWorldIndex
            This index, when `other` already represents the same world grid.

        Raises
        ------
        ValueError
            If `other` represents a different world grid.
        """
        if self.equals(other):
            return self
        raise ValueError(
            "Cannot automatically align VoxelToWorldIndex-backed arrays with "
            f"different world coordinates using join={how!r}. Resample one array "
            "onto the other's grid first, for example with "
            "`confusius.registration.resample_like`."
        )

    def reindex_like(self, other: Index) -> dict[Hashable, Any]:
        """Reject implicit reindexing of different world grids with a useful error.

        Parameters
        ----------
        other : xarray.Index
            The other index Xarray is trying to reindex against.

        Returns
        -------
        dict[Hashable, Any]
            Empty positional indexers when `other` already represents the same world
            grid.

        Raises
        ------
        ValueError
            If `other` represents a different world grid.
        """
        if self.equals(other):
            return {}
        raise ValueError(
            "Cannot automatically reindex VoxelToWorldIndex-backed arrays with "
            "different world coordinates. Resample one array onto the other's grid "
            "first, for example with `confusius.registration.resample_like`."
        )

    def equals(
        self, other: Index, *, exclude: frozenset[Hashable] | None = None
    ) -> bool:
        """Check equality with another voxel-to-world index.

        Parameters
        ----------
        other : xarray.Index
            Index to compare against.
        exclude : frozenset[Hashable], optional
            Dimensions to ignore, as used by Xarray's alignment machinery when
            checking whether indexes can be concatenated along one of them (e.g.
            `xr.concat(..., dim="pose")` first aligns every other dimension while
            excluding `"pose"` from the comparison). When `"pose"` is excluded and
            either side is pose-dependent, pose labels and affine values are not
            compared — only spatial geometry (voxel coordinates, fixed dims, world
            coordinate names) must match.

        Returns
        -------
        bool
            Whether the two indexes have equal affines and wrapped transforms (or,
            with `"pose"` excluded, equal spatial geometry only).
        """
        if not isinstance(other, VoxelToWorldIndex):
            return False
        self_transform = self._index.transform
        other_transform = other._index.transform
        assert isinstance(self_transform, VoxelToWorldTransform)
        assert isinstance(other_transform, VoxelToWorldTransform)
        if exclude is not None and POSE_DIM in exclude:
            return (
                self_transform.fixed_voxel_coords == other_transform.fixed_voxel_coords
                and all(
                    np.array_equal(
                        self_transform.active_voxel_coords[dim],
                        other_transform.active_voxel_coords[dim],
                    )
                    for dim in self_transform.dims
                    if dim != POSE_DIM
                )
            )
        return (
            self.voxel_to_world.shape == other.voxel_to_world.shape
            and np.allclose(self.voxel_to_world, other.voxel_to_world)
            and bool(
                cast(Index, self._index).equals(
                    cast(Index, other._index), exclude=exclude
                )
            )
        )

    @classmethod
    def concat(
        cls,
        indexes: Sequence[Index],
        dim: Hashable,
        positions: Iterable[Iterable[int]] | None = None,
    ) -> Self:
        """Concatenate pose-dependent indexes along `pose`.

        Parameters
        ----------
        indexes : sequence[xarray.Index]
            Indexes to concatenate, in `xarray.concat`'s array order. Each must
            already be pose-dependent (own a stacked affine), and all must share
            identical spatial geometry (voxel coordinates, fixed dims, world
            coordinate names) — only the pose labels and affines differ.
        dim : Hashable
            Dimension being concatenated. Only `"pose"` is supported; concatenating
            along any other dimension covered by this index (a native voxel
            dimension) is not yet implemented.
        positions : iterable[iterable[int]], optional
            Unused compatibility argument required by Xarray's index API;
            concatenation always follows `indexes`' order.

        Returns
        -------
        VoxelToWorldIndex
            Index with concatenated pose labels and affine stacks.

        Raises
        ------
        ValueError
            If `dim` is not `"pose"`, if any index is pose-independent, or if the
            indexes' spatial geometry does not match.
        """
        if dim != POSE_DIM:
            raise ValueError(
                f"VoxelToWorldIndex only supports concat along {POSE_DIM!r}, got "
                f"{dim!r}."
            )
        transforms: list[VoxelToWorldTransform] = []
        for index in indexes:
            assert isinstance(index, VoxelToWorldIndex)
            transform = index._index.transform
            assert isinstance(transform, VoxelToWorldTransform)
            if transform.pose_coord is None:
                raise ValueError(
                    "Cannot concatenate pose-independent geometry along "
                    f"{POSE_DIM!r}; every input must already be pose-dependent."
                )
            transforms.append(transform)
        first = transforms[0]
        for other in transforms[1:]:
            if (
                other.dims != first.dims
                or other.fixed_voxel_coords != first.fixed_voxel_coords
                or not all(
                    np.array_equal(
                        other.active_voxel_coords[dim], first.active_voxel_coords[dim]
                    )
                    for dim in first.dims
                    if dim != POSE_DIM
                )
            ):
                raise ValueError(
                    "Cannot concatenate VoxelToWorldIndex objects with different "
                    f"spatial geometry along {POSE_DIM!r}."
                )
        new_pose_coord = np.concatenate(
            [transform.pose_coord for transform in transforms]
        )
        new_affine = np.concatenate(
            [transform.voxel_to_world for transform in transforms], axis=0
        )
        new_transform = VoxelToWorldTransform(
            first.active_voxel_coords,
            new_affine,
            fixed_voxel_coords=first.fixed_voxel_coords,
            pose_coord=new_pose_coord,
        )
        first_index = indexes[0]
        assert isinstance(first_index, VoxelToWorldIndex)
        return cls(
            CoordinateTransformIndex(new_transform),
            new_affine,
            units=first_index.units,
        )


def _validate_voxel_to_world_affine(
    affine: npt.NDArray[np.float64], ndim: int, *, n_poses: int | None
) -> None:
    """Validate a voxel-to-world affine's shape, finiteness, and homogeneous row.

    Parameters
    ----------
    affine : (ndim+1, ndim+1) numpy.ndarray or (n_poses, ndim+1, ndim+1) numpy.ndarray
        Homogeneous affine, or a stack of one such affine per pose.
    ndim : int
        Number of spatial dimensions the affine covers.
    n_poses : int, optional
        Expected leading pose-stack length. If not provided, `affine` must be a
        single (pose-independent) affine.

    Raises
    ------
    ValueError
        If `affine`'s shape does not match, if it (or any pose affine, for a
        pose-stacked `affine`) is not finite or has an invalid homogeneous final row,
        or if pose affines do not share equal spatial scale magnitudes.
    """
    expected_shape = (ndim + 1, ndim + 1)
    label = "voxel_to_world"
    if n_poses is not None:
        expected_shape = (n_poses, *expected_shape)
        label = f"voxel_to_world for {n_poses} poses"
    if affine.shape != expected_shape:
        raise ValueError(
            f"{label} must have shape {expected_shape}, got {affine.shape}."
        )

    homogeneous_row = np.zeros(ndim + 1)
    homogeneous_row[-1] = 1.0
    valid = np.all(np.isfinite(affine)) and np.allclose(
        affine[..., -1, :], homogeneous_row
    )
    if not valid:
        subject = "Each pose affine" if n_poses is not None else "voxel_to_world"
        raise ValueError(
            f"{subject} must be finite with a valid homogeneous final row."
        )

    if n_poses is not None and n_poses > 1:
        scalings = np.stack(
            [
                np.array(list(get_affine_axis_scalings(affine[p], VOXEL_DIMS).values()))
                for p in range(n_poses)
            ]
        )
        if not np.allclose(scalings, scalings[0], rtol=1e-6):
            raise ValueError(
                "All pose affines must share equal spatial scale magnitudes."
            )


class VoxelToWorldTransform(CoordinateTransform):
    """Coordinate transform from voxel-space coordinates to world space.

    The transform combines:

    1. a dense array-position -> voxel-space lookup using the 1D coordinate arrays
       attached to each dimension, and
    2. a homogeneous affine that maps voxel-space coordinates to world-space
       coordinates.

    This lets a dense array with dimensions `(k, j, i)` carry irregular voxel-space
    coordinates such as `i = [0, 2, 3]`, while still exposing exact world coordinates
    through Xarray's `CoordinateTransformIndex`.

    Voxel dimensions are always exactly `(k, j, i)`, and world coordinates are always
    exactly `(z, y, x)` in that fixed row order. A voxel dimension can become "fixed" (see
    `fixed_voxel_coords`) when a scalar `isel` selection removes it as an array
    dimension, but it is never dropped from the affine: the affine itself is never
    reduced so a fixed dimension's contribution to the world coordinates stays exact,
    and a fixed dimension can later be reinstated (e.g. by
    [canonicalize_voxeldata][confusius.validation.canonicalize_voxeldata]) without any
    loss of precision.

    Parameters
    ----------
    active_voxel_coords : mapping[str, array-like]
        Ordered mapping from active dimension name `(k, j, i)` to its 1D voxel-space
        coordinates.
    voxel_to_world : (4, 4) or (pose, 4, 4) numpy.ndarray
        Homogeneous affine mapping voxel-space to world-space. The input column order
        must be `(k, j, i)`. The output row order must be `(z, y, x)`. When `pose_coord`
        is set, the affine must carry a leading pose axis and have shape `(pose, 4, 4)`,
        to provide one affine per pose label.
    fixed_voxel_coords : mapping[str, float], optional
        Voxel-space coordinate value pinned for each dimension that was removed as an
        array dimension by a scalar `isel` selection.
    pose_coord : numpy.typing.ArrayLike, optional
        1D pose coordinate labels. Required (and `voxel_to_world` must carry a
        leading pose axis) for pose-dependent geometry; must be left unset for
        pose-independent geometry.

    Raises
    ------
    ValueError
        If any voxel coordinate is not 1D, if `active_voxel_coords` and
        `fixed_voxel_coords` together do not exactly cover `confusius._dims.VOXEL_DIMS`,
        if the affine shape does not match (and, for pose-dependent geometry,
        `pose_coord`'s length), if the affine (or any pose affine, for pose-dependent
        geometry) is not finite or has an invalid homogeneous final row, or if pose
        affines do not share equal spatial scale magnitudes.
    """

    active_voxel_coords: dict[str, npt.NDArray[np.float64]]
    voxel_to_world: npt.NDArray[np.float64]
    fixed_voxel_coords: dict[str, float]
    pose_coord: npt.NDArray[Any] | None

    def __init__(
        self,
        active_voxel_coords: Mapping[str, npt.ArrayLike],
        voxel_to_world: npt.ArrayLike,
        *,
        fixed_voxel_coords: Mapping[str, float] | None = None,
        pose_coord: npt.ArrayLike | None = None,
    ) -> None:
        active_voxel_coords_np = {
            str(dim): np.asarray(values, dtype=np.float64)
            for dim, values in active_voxel_coords.items()
        }
        fixed = {
            str(dim): float(value) for dim, value in (fixed_voxel_coords or {}).items()
        }
        ndim = len(VOXEL_DIMS)

        if not active_voxel_coords_np:
            raise ValueError("VoxelToWorldTransform requires at least one active dim.")
        if (
            set(active_voxel_coords_np) | set(fixed) != set(VOXEL_DIMS)
            or len(active_voxel_coords_np) + len(fixed) != ndim
        ):
            raise ValueError(
                f"active_voxel_coords {tuple(active_voxel_coords_np)!r} and fixed_voxel_coords "
                f"{tuple(fixed)!r} together must exactly cover {VOXEL_DIMS!r}."
            )

        for dim, values in active_voxel_coords_np.items():
            if values.ndim != 1:
                raise ValueError(
                    f"Voxel coordinate {dim!r} must be 1D, got shape {values.shape}."
                )

        affine = np.asarray(voxel_to_world, dtype=np.float64)
        pose_coord_np: npt.NDArray[Any] | None = None
        if pose_coord is not None:
            pose_coord_np = np.asarray(pose_coord)
            if pose_coord_np.ndim != 1:
                raise ValueError(
                    f"pose_coord must be 1D, got shape {pose_coord_np.shape}."
                )

        _validate_voxel_to_world_affine(
            affine, ndim, n_poses=None if pose_coord_np is None else pose_coord_np.size
        )

        dim_size = {dim: len(values) for dim, values in active_voxel_coords_np.items()}
        if pose_coord_np is not None:
            dim_size = {POSE_DIM: pose_coord_np.size, **dim_size}

        super().__init__(coord_names=WORLD_DIMS, dim_size=dim_size)
        self.active_voxel_coords = active_voxel_coords_np
        self.voxel_to_world = affine
        self.fixed_voxel_coords = fixed
        self.pose_coord = pose_coord_np

    def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
        """Transform dense array positions into world coordinates.

        Parameters
        ----------
        dim_positions : dict[str, Any]
            Dense integer array positions keyed by the dimensions in `self.dims`.

        Returns
        -------
        dict[Hashable, Any]
            World-space coordinate values keyed by `self.coord_names`.
        """
        active_values = {
            dim: self.active_voxel_coords[dim][np.asarray(dim_positions[dim])]
            for dim in self.dims
            if dim != POSE_DIM
        }
        shape = np.asarray(next(iter(active_values.values()))).shape
        ones = np.ones(shape, dtype=np.float64)
        voxel_values = [
            active_values[dim]
            if dim in active_values
            else np.full(shape, self.fixed_voxel_coords[dim], dtype=np.float64)
            for dim in VOXEL_DIMS
        ]
        stacked = np.stack([*voxel_values, ones], axis=0).reshape(
            len(VOXEL_DIMS) + 1, -1
        )
        num_world = len(self.coord_names)

        if self.pose_coord is not None:
            pose_positions = (
                np.broadcast_to(np.asarray(dim_positions[POSE_DIM]), shape)
                .reshape(-1)
                .astype(int)
            )
            selected_affines = self.voxel_to_world[pose_positions]
            transformed_flat = np.einsum("mij,jm->mi", selected_affines, stacked)
            transformed = transformed_flat.T.reshape((num_world + 1, *shape))
            return {name: transformed[i] for i, name in enumerate(self.coord_names)}

        transformed = (self.voxel_to_world @ stacked).reshape((num_world + 1, *shape))
        return {name: transformed[i] for i, name in enumerate(self.coord_names)}

    def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
        """Transform world coordinates back into dense array positions.

        The returned positions are floating-point dense positions. Xarray rounds them
        during nearest-neighbour selection.

        Parameters
        ----------
        coord_labels : dict[Hashable, Any]
            World-space coordinate labels keyed by `self.coord_names`.

        Returns
        -------
        dict[str, Any]
            Dense array positions keyed by `self.dims`.

        Raises
        ------
        ValueError
            If this transform is pose-dependent. Pose-dependent geometry must be
            reduced to a single pose's affine before reverse world-coordinate lookup
            (see `VoxelToWorldIndex.sel`).
        """
        if self.pose_coord is not None:
            raise ValueError(
                "Reverse world-coordinate lookup requires pose-independent "
                "geometry; select a scalar pose first."
            )
        world_values = [np.asarray(coord_labels[name]) for name in self.coord_names]
        shape = np.asarray(world_values[0]).shape
        ones = np.ones(shape, dtype=np.float64)
        stacked = np.stack([*world_values, ones], axis=0).reshape(
            len(self.coord_names) + 1, -1
        )
        voxel_values = (np.linalg.inv(self.voxel_to_world) @ stacked).reshape(
            (len(VOXEL_DIMS) + 1, *shape)
        )

        dim_positions: dict[str, Any] = {}
        for i, dim in enumerate(VOXEL_DIMS):
            if dim not in self.dims:
                continue
            dim_positions[dim] = _reverse_lookup_positions(
                voxel_values[i].reshape(-1), self.active_voxel_coords[dim]
            ).reshape(shape)
        return dim_positions

    def equals(
        self,
        other: CoordinateTransform,
        *,
        exclude: frozenset[Hashable] | None = None,
    ) -> bool:
        """Check equality with another voxel-space affine transform.

        Parameters
        ----------
        other : xarray.indexes.CoordinateTransform
            Transform to compare against.
        exclude : frozenset[Hashable], optional
            Unused compatibility argument required by Xarray's transform API.

        Returns
        -------
        bool
            Whether the two transforms have identical coordinate names, dimensions,
            voxel-space coordinates, fixed voxel coordinates, and affine.
        """
        if not isinstance(other, VoxelToWorldTransform):
            return False
        self_pose, other_pose = self.pose_coord, other.pose_coord
        pose_equal = (self_pose is None) == (other_pose is None) and (
            self_pose is None
            or (other_pose is not None and np.array_equal(self_pose, other_pose))
        )
        return (
            self.dims == other.dims
            and self.fixed_voxel_coords == other.fixed_voxel_coords
            and pose_equal
            and all(
                np.array_equal(
                    self.active_voxel_coords[dim], other.active_voxel_coords[dim]
                )
                for dim in self.dims
                if dim != POSE_DIM
            )
            and self.voxel_to_world.shape == other.voxel_to_world.shape
            and np.allclose(self.voxel_to_world, other.voxel_to_world)
        )

    def isel(self, indexers: Mapping[str, Any]) -> Self | None:
        """Return the indexed transform, or `None` for unsupported indexers.

        Scalar (0D) indexers fix the corresponding dimension (see
        `fixed_voxel_coords`) rather than dropping the transform outright.

        Parameters
        ----------
        indexers : mapping[str, Any]
            Indexers keyed by dimension name, covering any subset of `self.dims`.

        Returns
        -------
        VoxelToWorldTransform or None
            The updated transform, or `None` when an indexer is unsupported (a
            multi-dimensional fancy index) or every active dimension would be fixed.
            A scalar `pose` indexer resolves to a single pose's affine and drops the
            `pose` dependency entirely, rather than fixing it like a spatial
            dimension — there is only ever one selected pose left to track, so
            nothing is gained by keeping it around as `fixed_voxel_coords`-style
            bookkeeping.
        """
        new_active_voxel_coords = dict(self.active_voxel_coords)
        new_fixed = dict(self.fixed_voxel_coords)
        new_pose_coord = self.pose_coord
        new_affine = self.voxel_to_world
        for dim, indexer in indexers.items():
            if dim == POSE_DIM:
                if self.pose_coord is None:
                    continue
                if _is_scalar_indexer(indexer):
                    new_affine = self.voxel_to_world[_scalar_indexer_value(indexer)]
                    new_pose_coord = None
                    continue
                if isinstance(indexer, Variable):
                    if indexer.ndim != 1:
                        return None
                    indexer = indexer.values
                elif isinstance(indexer, list | tuple):
                    indexer = np.asarray(indexer)
                elif not isinstance(indexer, slice | np.ndarray):
                    return None
                pose_values = self.pose_coord[indexer]
                if np.ndim(pose_values) != 1:
                    return None
                new_pose_coord = pose_values
                new_affine = self.voxel_to_world[indexer]
                continue
            if dim not in self.dims:
                continue
            if _is_scalar_indexer(indexer):
                new_fixed[dim] = float(
                    self.active_voxel_coords[dim][_scalar_indexer_value(indexer)]
                )
                del new_active_voxel_coords[dim]
                continue
            if isinstance(indexer, Variable):
                if indexer.ndim != 1:
                    return None
                indexer = indexer.values
            elif isinstance(indexer, list | tuple):
                indexer = np.asarray(indexer)
            elif not isinstance(indexer, slice | np.ndarray):
                return None
            values = self.active_voxel_coords[dim][indexer]
            if np.ndim(values) != 1:
                return None
            new_active_voxel_coords[dim] = values
        if not new_active_voxel_coords:
            return None
        return type(self)(
            new_active_voxel_coords,
            new_affine,
            fixed_voxel_coords=new_fixed,
            pose_coord=new_pose_coord,
        )

    def __repr__(self) -> str:
        """Return a compact repr."""
        return (
            f"VoxelToWorldTransform(dims={self.dims!r}, "
            f"coord_names={self.coord_names!r})"
        )


def _is_axis_aligned_affine(voxel_to_world: npt.ArrayLike) -> bool:
    """Return whether the affine has no cross-axis mixing.

    Parameters
    ----------
    voxel_to_world : (N+1, N+1) numpy.ndarray or (npose, N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel space to world space, or a stack of one
        such affine per pose.

    Returns
    -------
    bool
        Whether the affine linear part is diagonal up to floating-point noise, for
        every pose when a stack is given.
    """
    affine = np.asarray(voxel_to_world, dtype=np.float64)
    linear = affine[..., :-1, :-1]
    diagonal = np.zeros_like(linear)
    axis = np.arange(linear.shape[-1])
    diagonal[..., axis, axis] = linear[..., axis, axis]
    return bool(np.allclose(linear, diagonal, rtol=1e-10, atol=1e-12))


def attach_voxel_to_world_index(
    data: xr.DataArray,
    voxel_to_world: npt.ArrayLike,
    *,
    units: str = "mm",
) -> xr.DataArray:
    """Attach world coordinates to a DataArray, making it a VoxelData array.

    `data` must carry all three native voxel dims `k`/`j`/`i` (a lower-dimensional
    slice is represented as a singleton voxel dim, never a missing one). The
    resulting world coordinates are always `z`/`y`/`x`: a world coordinate's affine
    row is, in general (e.g. an oblique/rotated affine), a linear combination of
    every voxel dimension, so there is no meaningful per-dim `k`->`z` style
    correspondence to preserve for a partial dim set.

    Parameters
    ----------
    data : xarray.DataArray
        Input array that already carries 1D integer voxel-space coordinates on its
        native voxel dims `k`/`j`/`i`.
    voxel_to_world : (4, 4) numpy.ndarray or (npose, 4, 4) numpy.ndarray
        Homogeneous affine mapping voxel-space coordinates to world-space
        coordinates, or a stack of one such affine per pose. A stack requires `data`
        to have a matching `pose` dimension with a 1D coordinate; a single affine
        applies to every pose (or to no pose dimension at all).
    units : str, default: "mm"
        Physical unit shared by every derived world coordinate.

    Returns
    -------
    xarray.DataArray
        A new DataArray with derived world coordinates attached. Axis-aligned
        affines produce ordinary 1D coordinates; oblique affines produce lazily
        generated coordinates attached via a `CoordinateTransformIndex`.

    Raises
    ------
    ValueError
        If `data` does not have all three native voxel dims `k`/`j`/`i`, if their
        coordinates are not 1D dimension coordinates, or if a pose-stacked
        `voxel_to_world` is given without a matching `pose` dimension/coordinate on
        `data`.
    TypeError
        If a voxel dimension's coordinate does not have integer dtype.
    """
    if not all(dim in data.dims for dim in VOXEL_DIMS):
        raise ValueError(
            f"data must have all native voxel dims {VOXEL_DIMS!r}, got dims "
            f"{data.dims!r}."
        )
    active_voxel_coords: dict[str, npt.NDArray[np.int64]] = {}
    for dim in VOXEL_DIMS:
        if dim not in data.coords:
            raise ValueError(
                f"Voxel dimension {dim!r} must have a matching 1D coordinate."
            )
        coord = data.coords[dim]
        if coord.dims != (dim,):
            raise ValueError(
                f"Voxel coordinate {dim!r} must be a 1D dimension coordinate; got "
                f"dims {coord.dims!r}."
            )
        if not np.issubdtype(coord.dtype, np.integer):
            raise TypeError(
                f"Voxel coordinate {dim!r} must have integer dtype (native voxel "
                f"indices), got {coord.dtype}."
            )
        active_voxel_coords[dim] = np.asarray(coord.values, dtype=np.int64)

    voxel_to_world_array = np.asarray(voxel_to_world, dtype=np.float64)

    pose_coord = None
    if voxel_to_world_array.ndim == 3:
        if POSE_DIM not in data.dims:
            raise ValueError(
                "A stacked voxel_to_world affine (one per pose) requires data to "
                f"have a {POSE_DIM!r} dimension."
            )
        if POSE_DIM not in data.coords or data.coords[POSE_DIM].dims != (POSE_DIM,):
            raise ValueError(
                f"{POSE_DIM!r} dimension must have a matching 1D coordinate."
            )
        if voxel_to_world_array.shape[0] != data.sizes[POSE_DIM]:
            raise ValueError(
                f"voxel_to_world pose stack length {voxel_to_world_array.shape[0]} "
                f"does not match data's {POSE_DIM!r} size {data.sizes[POSE_DIM]}."
            )
        pose_coord = data.coords[POSE_DIM].values

    base = data.drop_vars(
        [name for name in WORLD_DIMS if name in data.coords], errors="ignore"
    )
    index = VoxelToWorldIndex.from_affine(
        active_voxel_coords,
        voxel_to_world_array,
        units=units,
        pose_coord=pose_coord,
    )
    return base.assign_coords(xr.Coordinates.from_xindex(index))


def _fold_fixed_dims_into_affine(
    full_affine: npt.NDArray[np.float64],
    all_dims: tuple[str, ...],
    active_dims: tuple[str, ...],
    fixed_voxel_coords: Mapping[str, float],
) -> npt.NDArray[np.float64]:
    """Reduce a voxel-to-world affine to its active dimensions.

    Parameters
    ----------
    full_affine : (N+1, N+1) numpy.ndarray or (npose, N+1, N+1) numpy.ndarray
        Homogeneous affine covering all of `all_dims`, or a stack of one such affine
        per pose.
    all_dims : tuple[str, ...]
        Voxel dimensions matching `full_affine`'s column order.
    active_dims : tuple[str, ...]
        Subset of `all_dims` to keep as columns in the reduced affine, in the desired
        output column order.
    fixed_voxel_coords : mapping[str, float]
        Voxel-space coordinate value pinned for each dimension in `all_dims` that is
        not in `active_dims`.

    Returns
    -------
    (W+1, M+1) numpy.ndarray or (npose, W+1, M+1) numpy.ndarray
        Homogeneous affine (or pose-stacked affines) covering only `active_dims` as
        its input (columns), with each fixed dimension's contribution folded into
        the translation. `W` is the number of world coordinates, `M` the number of
        `active_dims`. This is square (`W == M`) only when there are no fixed
        dimensions; with fixed dimensions present the affine is genuinely
        rectangular, since a world coordinate can depend on both active and fixed
        voxel dimensions.
    """
    if not fixed_voxel_coords:
        return full_affine.copy()
    is_pose_stacked = full_affine.ndim == 3
    linear = full_affine[..., :-1, :-1]
    translation = full_affine[..., :-1, -1].copy()
    for dim, value in fixed_voxel_coords.items():
        translation = translation + linear[..., :, all_dims.index(dim)] * value
    active_columns = [all_dims.index(dim) for dim in active_dims]
    num_world = full_affine.shape[-2] - 1
    output_shape = (
        (full_affine.shape[0], num_world + 1, len(active_dims) + 1)
        if is_pose_stacked
        else (num_world + 1, len(active_dims) + 1)
    )
    reduced = np.zeros(output_shape, dtype=np.float64)
    reduced[..., :num_world, :-1] = linear[..., :, active_columns]
    reduced[..., :num_world, -1] = translation
    reduced[..., -1, -1] = 1.0
    return reduced


def get_voxel_to_world_index(data: xr.DataArray) -> VoxelToWorldIndex | None:
    """Return the `VoxelToWorldIndex` attached to `data`, if any.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    VoxelToWorldIndex or None
        The index, or `None` if `data` carries no voxel-to-world geometry. `data`
        carries at most one: xarray only ever registers one index object per
        coordinate name, and [VoxelToWorldIndex][confusius._utils.geometry.VoxelToWorldIndex]
        always wraps `z`/`y`/`x` jointly, so the same single object is simply
        registered three times.
    """
    for index in data.xindexes.values():
        if isinstance(index, VoxelToWorldIndex):
            return index
    return None


def get_voxel_to_world_affine(data: xr.DataArray) -> npt.NDArray[np.float64]:
    """Return the voxel-to-world affine owned by the DataArray's index(es).

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array.

    Returns
    -------
    (W+1, M+1) numpy.ndarray or (npose, W+1, M+1) numpy.ndarray
        Homogeneous affine mapping the DataArray's `M` active voxel dimensions to its
        `W` world coordinates, or a stack of one such affine per pose while
        pose-dependent geometry remains (see
        [VoxelToWorldIndex.is_pose_dependent][confusius._utils.geometry.VoxelToWorldIndex.is_pose_dependent]).
        Square (`W == M`, the common case) unless a dimension was previously fixed by
        a scalar `isel` selection on oblique geometry (see
        [VoxelToWorldIndex.fixed_voxel_coords][confusius._utils.geometry.VoxelToWorldIndex.fixed_voxel_coords]):
        its contribution is folded into the translation rather than dropped, which
        can leave a world coordinate depending on both active and fixed dimensions,
        making the affine genuinely rectangular.

    Raises
    ------
    ValueError
        If `data` does not carry voxel-to-world geometry.
    """
    index = get_voxel_to_world_index(data)
    if index is None:
        raise ValueError("DataArray must have a voxel-to-world index.")
    return _fold_fixed_dims_into_affine(
        index.voxel_to_world,
        VOXEL_DIMS,
        index.voxel_dims,
        index.fixed_voxel_coords,
    )


def get_voxel_to_world_units(data: xr.DataArray) -> str:
    """Return the physical unit shared by the DataArray's world coordinates.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array.

    Returns
    -------
    str
        Physical unit (e.g. `"mm"`) shared by every world coordinate (`z`/`y`/`x` are
        always expressed in the same unit, since they're derived jointly from one
        affine).

    Raises
    ------
    ValueError
        If `data` does not carry voxel-to-world geometry.
    """
    index = get_voxel_to_world_index(data)
    if index is None:
        raise ValueError("DataArray must have a voxel-to-world index.")
    return index.units


def restore_voxel_to_world_index(data: xr.DataArray) -> xr.DataArray:
    """Rebuild voxel-to-world geometry for a dimension restored after being fixed.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray that may carry voxel-to-world geometry with a dimension fixed by a
        prior scalar `isel` selection, since restored as an array dimension (e.g. via
        `DataArray.expand_dims`).

    Returns
    -------
    xarray.DataArray
        `data` unchanged when there is nothing to restore. Otherwise a DataArray with
        geometry rebuilt for the restored dimension(s), exactly, from the untouched
        original affine (see
        [VoxelToWorldIndex.fixed_voxel_coords][confusius._utils.geometry.VoxelToWorldIndex.fixed_voxel_coords]).
    """
    index = get_voxel_to_world_index(data)
    if index is None or not index.fixed_voxel_coords:
        return data
    if any(
        dim not in data.dims or data.coords[dim].dims != (dim,) for dim in VOXEL_DIMS
    ):
        return data
    return attach_voxel_to_world_index(data, index.voxel_to_world, units=index.units)


def has_axis_aligned_voxel_to_world_index(data: xr.DataArray) -> bool:
    """Return whether `data` has voxel-to-world geometry with no cross-axis mixing.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    bool
        Whether `data` carries voxel-to-world geometry and its `voxel_to_world`
        affine is axis-aligned.
    """
    if not has_voxel_to_world_index(data):
        return False
    return _is_axis_aligned_affine(get_voxel_to_world_affine(data))


def has_voxel_to_world_index(data: xr.DataArray) -> bool:
    """Return whether a DataArray is a VoxelData array.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    bool
        Whether `data` has a `VoxelToWorldIndex` attached. Every `VoxelToWorldIndex`
        always covers all of `confusius._dims.WORLD_DIMS`
        (`z`/`y`/`x`) jointly, and a fully-fixed one (every voxel dim scalar-`isel`'d
        away) is never attached in the first place -- xarray drops it, since
        `VoxelToWorldTransform.isel` returns `None` once no active dim remains -- so
        presence alone is the only real question.
    """
    return get_voxel_to_world_index(data) is not None


def get_voxel_to_world_spatial_dims(data: xr.DataArray) -> tuple[str, ...]:
    """Return voxel-space dimensions present on a VoxelData array.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    tuple[str, ...]
        Present voxel-space dimensions in canonical affine column order.
    """
    index = get_voxel_to_world_index(data)
    if index is not None:
        return index.voxel_dims
    return tuple(dim for dim in VOXEL_DIMS if dim in data.dims)


def require_scalar_pose_affine(
    data: xr.DataArray, context: str
) -> npt.NDArray[np.float64]:
    """Return `data`'s voxel-to-world affine, requiring pose-independent geometry.

    Single-grid operations (origin/direction accessors, `reindex_voxels`,
    registration, resampling, ...) cannot operate on a per-pose affine stack; this
    gives them a clear, specific failure at entry instead of a confusing NumPy
    broadcasting error partway through.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array.
    context : str
        Short description of the calling operation, used in the error message
        (e.g. `"Computing the voxel-to-world origin"`).

    Returns
    -------
    (W+1, M+1) numpy.ndarray
        Voxel-to-world affine.

    Raises
    ------
    ValueError
        If `data` does not carry voxel-to-world geometry, or if it carries
        pose-dependent geometry (a stacked affine).
    """
    affine = get_voxel_to_world_affine(data)
    if affine.ndim == 3:
        raise ValueError(
            f"{context} requires pose-independent geometry; select a scalar pose "
            "first, e.g. `data.isel(pose=0)`."
        )
    return affine


def get_voxel_to_world_index_origin(data: xr.DataArray) -> dict[str, float]:
    """Return the world location of the first sampled voxel.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array.

    Returns
    -------
    dict[str, float]
        World origin keyed by world coordinate name.

    Raises
    ------
    ValueError
        If `data` carries pose-dependent geometry (a stacked affine); select a
        scalar pose first.

    Notes
    -----
    This returns the world location of array index `(0, ..., 0)`, i.e. the first
    sampled voxel, not necessarily the affine translation at voxel-space `(0, ..., 0)`.
    The two coincide only when the voxel coordinates themselves start at zero.
    """
    voxel_dims = get_voxel_to_world_spatial_dims(data)
    first_voxel = np.array(
        [
            np.float64(np.asarray(data.coords[dim].values)[0]).item()
            for dim in voxel_dims
        ]
        + [1.0],
        dtype=np.float64,
    )
    affine = require_scalar_pose_affine(data, "Computing the voxel-to-world origin")
    origin = affine @ first_voxel
    return {name: np.float64(origin[i]).item() for i, name in enumerate(WORLD_DIMS)}


def get_voxel_to_world_index_spacing(data: xr.DataArray) -> dict[str, float | None]:
    """Return world spacing per voxel-space axis for a VoxelData array.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array.

    Returns
    -------
    dict[str, float | None]
        World spacing keyed by voxel-space dimension.
    """
    voxel_dims = get_voxel_to_world_spatial_dims(data)
    active_voxel_coords = {dim: data.coords[dim].values for dim in voxel_dims}
    return get_voxel_to_world_spacings_from_coords(
        active_voxel_coords, get_voxel_to_world_affine(data)
    )


def get_voxel_to_world_direction_matrix(
    data: xr.DataArray,
) -> npt.NDArray[np.float64]:
    """Return the world-space direction matrix of a VoxelData array.

    The voxel-to-world affine maps voxel-space *coordinate values* to world space, so
    its own orientation says nothing about whether `data`'s voxel coordinate for a
    given dimension happens to run ascending or descending (e.g. after
    `da.isel(dim=slice(None, None, -1))`) -- dense array position always counts up
    from 0 regardless of the coordinate's own direction. This folds that sign in, so
    the returned direction matrix is expressed in dense-position terms: paired with
    [get_voxel_to_world_index_spacing][confusius._utils.geometry.get_voxel_to_world_index_spacing]
    (a magnitude) and origin, it reconstructs a position-space affine that correctly
    represents a flipped voxel dimension, as SimpleITK's own `(origin, spacing,
    direction)` grid convention expects.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array.

    Returns
    -------
    (N, N) numpy.ndarray
        Unit direction vectors in world-space row order and voxel-space column
        order.

    Raises
    ------
    ValueError
        If `data` carries pose-dependent geometry (a stacked affine); select a
        scalar pose first.
    """
    affine = require_scalar_pose_affine(
        data, "Computing the voxel-to-world direction matrix"
    )
    direction = get_affine_direction_matrix(affine)
    voxel_dims = get_voxel_to_world_spatial_dims(data)
    label_signs = [
        -1.0
        if data.sizes[dim] > 1
        and data.coords[dim].values[1] < data.coords[dim].values[0]
        else 1.0
        for dim in voxel_dims
    ]
    return direction * np.asarray(label_signs)


def get_affine_axis_scalings(
    voxel_to_world: npt.ArrayLike,
    voxel_dims: tuple[str, ...],
) -> dict[str, float]:
    """Return world distance per one voxel-space unit along each axis.

    Parameters
    ----------
    voxel_to_world : (N+1, N+1) numpy.ndarray or (npose, N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel space to world space, or a stack of one
        such affine per pose. For a stack, the first pose's scalings are returned;
        pose-dependent geometry validates equal spatial scale magnitudes across
        poses before using this helper.
    voxel_dims : tuple[str, ...]
        Voxel-space dimension names in affine column order.

    Returns
    -------
    dict[str, float]
        Euclidean norms of the affine column vectors, keyed by voxel-space dimension.
    """
    affine = np.asarray(voxel_to_world, dtype=np.float64)
    if affine.ndim == 3:
        affine = affine[0]
    linear = affine[:-1, :-1]
    return {
        dim: np.float64(np.linalg.norm(linear[:, i])).item()
        for i, dim in enumerate(voxel_dims)
    }


def get_affine_direction_matrix(
    voxel_to_world: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return unit world-space axis directions from a voxel-to-world affine.

    Parameters
    ----------
    voxel_to_world : (N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel space to world space.

    Returns
    -------
    (N, N) numpy.ndarray
        Matrix whose columns are unit world-space vectors for each voxel-space axis.
        Zero-length columns are preserved as zeros.
    """
    affine = np.asarray(voxel_to_world, dtype=np.float64)
    linear = affine[:-1, :-1].copy()
    norms = np.linalg.norm(linear, axis=0)
    nonzero = norms > 0
    linear[:, nonzero] /= norms[nonzero]
    linear[:, ~nonzero] = 0.0
    return linear


def get_voxel_to_world_spacings_from_coords(
    active_voxel_coords: Mapping[str, npt.ArrayLike],
    voxel_to_world: npt.ArrayLike,
) -> dict[str, float | None]:
    """Return world spacing for regularly sampled voxel axes.

    Parameters
    ----------
    active_voxel_coords : mapping[str, array-like]
        Ordered mapping from voxel-space dimension name to its 1D coordinates.
    voxel_to_world : (N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel space to world space.

    Returns
    -------
    dict[str, float | None]
        World spacing keyed by voxel-space dimension. Returns `None` only when the
        voxel-space coordinate is irregular. For singleton voxel axes, the spacing is
        inferred from one voxel-space unit along that affine column.
    """
    scalings = get_affine_axis_scalings(voxel_to_world, tuple(active_voxel_coords))
    spacings: dict[str, float | None] = {}
    for dim, values in active_voxel_coords.items():
        values_array = np.asarray(values, dtype=np.float64)
        step, approximate = get_representative_step(values_array)
        if approximate:
            spacings[dim] = None
        elif step is None:
            spacings[dim] = scalings[dim]
        else:
            spacings[dim] = abs(step) * scalings[dim]
    return spacings
