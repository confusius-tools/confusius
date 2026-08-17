"""Geometry helpers for voxel-space to world-space transforms.

This module implements the indexing machinery underlying ConfUSIus's VoxelData
model: a DataArray keeps 1D voxel-space coordinates (for example `i`, `j`, `k`) and
stores a single affine that maps those voxel-space coordinates into world-space
coordinates (for example `x`, `y`, `z`).

The derived world coordinates are always exposed lazily via a single joint
[VoxelToWorldIndex][confusius._utils.geometry.VoxelToWorldIndex] backed by Xarray's
`CoordinateTransformIndex`, for both axis-aligned and oblique affines — see
`VoxelToWorldIndex`'s docstring for why a single shared index (rather than one per
axis) is required for compatibility with `.stack()`. For axis-aligned affines,
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
import xarray as xr
from xarray import Index, Variable
from xarray.core.indexing import IndexSelResult
from xarray.indexes import CoordinateTransform, CoordinateTransformIndex

from confusius._dims import POSE_DIM, SPATIAL_DIMS, VOXEL_DIMS
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
    covering all of a DataArray's voxel dimensions jointly — for both oblique and
    axis-aligned affines. A single shared index (rather than one per axis) is what
    keeps `.stack()` working: xarray skips an index from stack-index consideration
    when it's associated with more than one coordinate name
    (`xarray.core.indexes.IndexVariable.is_multi`), which is exactly what lets
    `.stack(space=("k", "j", "i"))` fall through cleanly to `k`/`j`/`i`'s own plain
    indexes instead of colliding with `z`/`y`/`x`'s. Splitting per axis (so each axis
    could be dropped by a scalar `isel` independently of the others) breaks this,
    since then each axis's index is associated with only one name and is no longer
    skipped — this is not `.stack()`-compatible for *any* number of separate index
    objects greater than one, so it isn't a viable design.

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
        Index wrapping a [VoxelToWorldTransform][confusius._utils.geometry.VoxelToWorldTransform].
    voxel_to_world : numpy.typing.ArrayLike
        Homogeneous voxel-to-world affine.
    world_coord_attrs : mapping[Any, mapping[str, Any]], optional
        Attributes to attach to the derived world coordinates, keyed by world
        coordinate name.
    """

    def __init__(
        self,
        index: CoordinateTransformIndex,
        voxel_to_world: npt.ArrayLike,
        world_coord_attrs: Mapping[Any, Mapping[str, Any]] | None = None,
    ) -> None:
        self._index = index
        self.voxel_to_world = np.asarray(voxel_to_world, dtype=np.float64)
        self.world_coord_attrs = {
            name: dict(attrs) for name, attrs in (world_coord_attrs or {}).items()
        }

    @property
    def voxel_dims(self) -> tuple[str, ...]:
        """Active voxel dimensions.

        Never includes `pose` — see
        [is_pose_dependent][confusius._utils.geometry.VoxelToWorldIndex.is_pose_dependent]
        for that.

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
        [canonicalize_fusi][confusius.validation.canonicalize_fusi].

        Returns
        -------
        dict[str, float]
            Pinned voxel-space coordinate value keyed by dimension name.
        """
        transform = self._index.transform
        assert isinstance(transform, VoxelToWorldTransform)
        return dict(transform.fixed_voxel_coords)

    @property
    def all_voxel_dims(self) -> tuple[str, ...]:
        """All original voxel dimensions covered by this index (active and fixed).

        Returns
        -------
        tuple[str, ...]
            All original voxel dimension names, in affine column order.
        """
        transform = self._index.transform
        assert isinstance(transform, VoxelToWorldTransform)
        return transform._all_dims

    @property
    def world_coord_names(self) -> tuple[Hashable, ...]:
        """World coordinate names.

        Returns
        -------
        tuple[Hashable, ...]
            World coordinate names, in affine row order. Never includes `pose`.
        """
        transform = self._index.transform
        assert isinstance(transform, VoxelToWorldTransform)
        return transform.world_coord_names

    @classmethod
    def from_affine(
        cls,
        voxel_coords: Mapping[str, npt.ArrayLike],
        voxel_to_world: npt.ArrayLike,
        *,
        world_coord_names: tuple[Hashable, ...] | None = None,
        world_coord_attrs: Mapping[Any, Mapping[str, Any]] | None = None,
        pose_coord: npt.ArrayLike | None = None,
    ) -> Self:
        """Create a voxel-to-world index from an affine.

        Parameters
        ----------
        voxel_coords : mapping[str, array-like]
            Ordered mapping from voxel dimension name to its 1D voxel-space
            coordinates.
        voxel_to_world : numpy.typing.ArrayLike
            Homogeneous affine mapping voxel-space coordinates to world-space
            coordinates, or a stack of `pose_coord`-many such affines with a leading
            pose axis.
        world_coord_names : tuple[Hashable, ...], optional
            Names of the world-space coordinates exposed by the index. If not
            provided, defaults to `("z", "y", "x")` for 3D and `("y", "x")` for 2D.
        world_coord_attrs : mapping[Any, mapping[str, Any]], optional
            Attributes to attach to the derived world coordinates, keyed by world
            coordinate name.
        pose_coord : numpy.typing.ArrayLike, optional
            1D pose coordinate labels. Required when `voxel_to_world` is a stack of
            affines (one per pose); must be left unset otherwise.

        Returns
        -------
        VoxelToWorldIndex
            Index wrapping the resolved joint transform.
        """
        affine = np.asarray(voxel_to_world, dtype=np.float64)
        if world_coord_names is None:
            world_coord_names = (
                ("y", "x") if len(voxel_coords) == 2 else ("z", "y", "x")
            )
        transform = VoxelToWorldTransform(
            voxel_coords,
            affine,
            world_coord_names=world_coord_names,
            pose_coord=pose_coord,
        )
        return cls(
            CoordinateTransformIndex(transform),
            affine,
            world_coord_attrs=world_coord_attrs,
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
            World coordinate variables, keyed by coordinate name. For pose-dependent
            geometry, also includes a 1D `pose` dimension coordinate — the generic
            transform-derived variable would otherwise broadcast `pose` labels over
            every voxel dimension, since `CoordinateTransformIndex.create_variables`
            has no notion of a coordinate depending on only one of the transform's
            several dims.
        """
        transform = self._index.transform
        assert isinstance(transform, VoxelToWorldTransform)
        new_variables = dict(self._index.create_variables())
        if transform.pose_coord is not None and POSE_DIM in new_variables:
            new_variables[POSE_DIM] = Variable((POSE_DIM,), transform.pose_coord)
        for name, attrs in self.world_coord_attrs.items():
            if name in new_variables:
                new_variables[name].attrs.update(attrs)
        return new_variables

    def rename(
        self,
        name_dict: Mapping[Any, Hashable],
        dims_dict: Mapping[Any, Hashable],
    ) -> Self:
        """Rename dimensions and coordinates, keeping the wrapped transform consistent.

        The default `xarray.Index.rename` only renames the base
        `CoordinateTransform` fields (`dims`, `coord_names`, `dim_size`); it does not
        touch this module's own transform-specific fields (`voxel_coords`,
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
                for dim, values in transform.voxel_coords.items()
            },
            transform.voxel_to_world,
            tuple(name_dict.get(name, name) for name in transform.world_coord_names),
            fixed_voxel_coords={
                _rename_dim(dim): value
                for dim, value in transform.fixed_voxel_coords.items()
            },
            all_dims=tuple(_rename_dim(dim) for dim in transform._all_dims),
            pose_coord=transform.pose_coord,
        )
        new_world_coord_attrs = {
            name_dict.get(name, name): attrs
            for name, attrs in self.world_coord_attrs.items()
        }
        return type(self)(
            CoordinateTransformIndex(new_transform),
            self.voxel_to_world,
            world_coord_attrs=new_world_coord_attrs,
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
            world_coord_attrs=self.world_coord_attrs,
        )

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        """Select by world coordinates.

        For axis-aligned geometry, each world coordinate depends on exactly one
        voxel dimension, so labels are resolved independently per axis (supporting
        slices and single-axis queries, like an ordinary dimension coordinate).
        Oblique geometry delegates to `CoordinateTransformIndex.sel`, which only
        supports point-wise `nearest` queries providing all axes at once.

        Parameters
        ----------
        labels : dict[Any, Any]
            World coordinate labels to select, keyed by world coordinate name.
        method : str, optional
            Selection method. Unused for axis-aligned geometry (per-axis selection
            always supports both exact and nearest lookup); forwarded to
            `CoordinateTransformIndex.sel` for oblique geometry, defaulting to
            `"nearest"`.
        tolerance : Any, optional
            Forwarded to `CoordinateTransformIndex.sel` for oblique geometry.

        Returns
        -------
        xarray.core.indexing.IndexSelResult
            Resolved voxel-dimension indexers.

        Raises
        ------
        ValueError
            If `data` carries pose-dependent geometry and `labels` selects world
            coordinates without also giving a scalar `pose` label.
        KeyError
            If a requested `pose` label is not one of this index's pose labels.
        """
        transform = self._index.transform
        assert isinstance(transform, VoxelToWorldTransform)
        coord_labels = {
            name: labels[name] for name in transform.world_coord_names if name in labels
        }
        if transform.pose_coord is not None:
            return self._sel_pose_dependent(labels, coord_labels, method, tolerance)
        if not coord_labels:
            return IndexSelResult({})
        if not _is_axis_aligned_affine(transform.voxel_to_world):
            return self._index.sel(
                coord_labels, method=method or "nearest", tolerance=tolerance
            )
        dim_indexers: dict[Hashable, Any] = {}
        for column, dim in enumerate(transform._all_dims):
            name = transform.coord_names[column]
            if name not in coord_labels or dim not in transform.voxel_coords:
                continue
            scale = transform.voxel_to_world[column, column]
            offset = transform.voxel_to_world[column, -1]
            voxel_axis = transform.voxel_coords[dim]
            label = coord_labels[name]
            if isinstance(label, slice):
                values = scale * voxel_axis + offset
                start = np.nanmin(values) if label.start is None else label.start
                stop = np.nanmax(values) if label.stop is None else label.stop
                lower, upper = sorted((start, stop))
                hits = np.nonzero((values >= lower) & (values <= upper))[0]
                if hits.size == 0:
                    dim_indexers[dim] = slice(0, 0)
                elif label.step is None:
                    dim_indexers[dim] = slice(hits[0], hits[-1] + 1)
                else:
                    dim_indexers[dim] = hits[:: label.step]
            else:
                voxel_label = (np.asarray(label) - offset) / scale
                position = _reverse_lookup_positions(voxel_label, voxel_axis)
                dim_indexers[dim] = np.rint(position).astype(int)
        return IndexSelResult(dim_indexers)

    def _sel_pose_dependent(
        self,
        labels: dict[Any, Any],
        coord_labels: dict[Any, Any],
        method,
        tolerance,
    ) -> IndexSelResult:
        """Resolve a `.sel()` query against pose-dependent geometry.

        Requires a scalar `pose` label, which is resolved first; the matching
        affine is then selected and the remaining world-coordinate labels are
        resolved against it exactly as for pose-independent geometry.

        Parameters
        ----------
        labels : dict[Any, Any]
            Full label mapping passed to `sel`, potentially including `pose`.
        coord_labels : dict[Any, Any]
            World-coordinate labels (excluding `pose`) extracted from `labels`.
        method : Any
            Forwarded to the pose-reduced index's `sel`.
        tolerance : Any
            Forwarded to the pose-reduced index's `sel`.

        Returns
        -------
        xarray.core.indexing.IndexSelResult
            Resolved indexers, including a scalar `pose` indexer.

        Raises
        ------
        ValueError
            If world coordinates are requested without a scalar `pose` label.
        KeyError
            If the requested `pose` label has no match.
        """
        transform = self._index.transform
        assert isinstance(transform, VoxelToWorldTransform)
        pose_label = labels.get(POSE_DIM)
        if pose_label is None:
            if coord_labels:
                raise ValueError(
                    "Selecting world coordinates on pose-dependent geometry requires "
                    "a scalar `pose` label, e.g. `.sel(pose=0, z=..., y=..., "
                    "x=...)`."
                )
            return IndexSelResult({})
        if isinstance(pose_label, Variable):
            if pose_label.ndim != 0:
                raise ValueError(
                    "Selecting world coordinates on pose-dependent geometry requires "
                    "a scalar `pose` label, not a slice or array of labels."
                )
            pose_value = pose_label.values.item()
        elif isinstance(pose_label, slice | list | tuple | np.ndarray):
            raise ValueError(  # noqa: TRY004
                "Selecting world coordinates on pose-dependent geometry requires a "
                "scalar `pose` label, not a slice or array of labels."
            )
        else:
            pose_value = pose_label
        pose_coord = transform.pose_coord
        assert pose_coord is not None
        matches = np.nonzero(pose_coord == pose_value)[0]
        if matches.size == 0:
            raise KeyError(f"No pose labeled {pose_value!r}.")
        pose_position = int(matches[0])
        reduced_transform = VoxelToWorldTransform(
            transform.voxel_coords,
            transform.voxel_to_world[pose_position],
            transform.world_coord_names,
            fixed_voxel_coords=transform.fixed_voxel_coords,
            all_dims=transform._all_dims,
        )
        reduced_index = type(self)(
            CoordinateTransformIndex(reduced_transform),
            transform.voxel_to_world[pose_position],
            world_coord_attrs=self.world_coord_attrs,
        )
        result = reduced_index.sel(coord_labels, method=method, tolerance=tolerance)
        dim_indexers = dict(result.dim_indexers)
        dim_indexers[POSE_DIM] = pose_position
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
                self_transform.world_coord_names == other_transform.world_coord_names
                and self_transform._all_dims == other_transform._all_dims
                and self_transform.fixed_voxel_coords
                == other_transform.fixed_voxel_coords
                and all(
                    np.array_equal(
                        self_transform.voxel_coords[dim],
                        other_transform.voxel_coords[dim],
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
                other.world_coord_names != first.world_coord_names
                or other.dims != first.dims
                or other._all_dims != first._all_dims
                or other.fixed_voxel_coords != first.fixed_voxel_coords
                or not all(
                    np.array_equal(other.voxel_coords[dim], first.voxel_coords[dim])
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
            first.voxel_coords,
            new_affine,
            first.world_coord_names,
            fixed_voxel_coords=first.fixed_voxel_coords,
            all_dims=first._all_dims,
            pose_coord=new_pose_coord,
        )
        first_index = indexes[0]
        assert isinstance(first_index, VoxelToWorldIndex)
        return cls(
            CoordinateTransformIndex(new_transform),
            new_affine,
            world_coord_attrs=first_index.world_coord_attrs,
        )


class VoxelToWorldTransform(CoordinateTransform):
    """Coordinate transform from voxel-space coordinates to world space.

    The transform combines:

    1. a dense array-position -> voxel-space lookup using the 1D coordinate arrays
       attached to each dimension, and
    2. a homogeneous affine that maps voxel-space coordinates to world-space
       coordinates.

    This lets a dense array with dimensions like `(j, i)` or `(k, j, i)` carry
    irregular voxel-space coordinates such as `i = [0, 2, 3]`, while still exposing
    exact world coordinates through Xarray's
    `CoordinateTransformIndex`.

    Dimensions can become "fixed" (see `fixed_voxel_coords`) when a scalar `isel`
    selection removes them as array dimensions. The affine itself is never reduced —
    it always covers `all_dims`, the full original set of voxel dimensions — so a
    fixed dimension's contribution to the world coordinates stays exact, and a
    fixed dimension can later be reinstated (e.g. by
    [canonicalize_fusi][confusius.validation.canonicalize_fusi]) without any loss of
    precision.

    Parameters
    ----------
    voxel_coords : mapping[str, array-like]
        Ordered mapping from active dimension name to its 1D voxel-space coordinates.
    voxel_to_world : (N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel-space coordinates to world-space
        coordinates, covering all of `all_dims` (not just the active `voxel_coords`).
        The input column order must match `all_dims`. The output row order must match
        `world_coord_names`.
    world_coord_names : tuple[Hashable, ...], optional
        Names of the world-space coordinates exposed by the transform. If not
        provided, defaults to `("z", "y", "x")` for 3D and `("y", "x")` for 2D.
    fixed_voxel_coords : mapping[str, float], optional
        Voxel-space coordinate value pinned for each dimension that was removed as an
        array dimension by a scalar `isel` selection.
    all_dims : tuple[str, ...], optional
        Full original voxel dimension order (active and fixed), matching the affine's
        column order. If not provided, defaults to `tuple(voxel_coords) +
        tuple(fixed_voxel_coords)`, which is only correct for a from-scratch
        construction; callers reducing an existing transform must pass the original
        `all_dims` through unchanged.
    pose_coord : numpy.typing.ArrayLike, optional
        1D pose coordinate labels. Required (and `voxel_to_world` must carry a
        leading pose axis) for pose-dependent geometry; must be left unset for
        pose-independent geometry.

    Raises
    ------
    ValueError
        If any voxel coordinate is not 1D, if `all_dims` is inconsistent with
        `voxel_coords`/`fixed_voxel_coords`, if the affine shape does not match the
        number of voxel-space dimensions in `all_dims` (and, for pose-dependent
        geometry, `pose_coord`'s length), if a pose-stacked affine is not finite or
        has an invalid homogeneous final row, or if pose affines do not share equal
        spatial scale magnitudes.
    """

    voxel_coords: dict[str, npt.NDArray[np.float64]]
    voxel_to_world: npt.NDArray[np.float64]
    fixed_voxel_coords: dict[str, float]
    world_coord_names: tuple[Hashable, ...]
    pose_coord: npt.NDArray[Any] | None

    def __init__(
        self,
        voxel_coords: Mapping[str, npt.ArrayLike],
        voxel_to_world: npt.ArrayLike,
        world_coord_names: tuple[Hashable, ...] | None = None,
        *,
        fixed_voxel_coords: Mapping[str, float] | None = None,
        all_dims: tuple[str, ...] | None = None,
        pose_coord: npt.ArrayLike | None = None,
    ) -> None:
        voxel_coords_np = {
            str(dim): np.asarray(values, dtype=np.float64)
            for dim, values in voxel_coords.items()
        }
        fixed = {
            str(dim): float(value) for dim, value in (fixed_voxel_coords or {}).items()
        }
        if all_dims is None:
            all_dims = (*voxel_coords_np, *fixed)
        ndim = len(all_dims)

        if set(all_dims) != set(voxel_coords_np) | set(fixed) or len(all_dims) != len(
            voxel_coords_np
        ) + len(fixed):
            raise ValueError(
                f"all_dims {all_dims!r} must exactly cover active dims "
                f"{tuple(voxel_coords_np)!r} and fixed dims {tuple(fixed)!r}."
            )
        if not voxel_coords_np:
            raise ValueError("VoxelToWorldTransform requires at least one active dim.")
        if ndim not in {2, 3}:
            raise ValueError(
                f"VoxelToWorldTransform only supports 2D or 3D inputs, got {ndim}."
            )

        for dim, values in voxel_coords_np.items():
            if values.ndim != 1:
                raise ValueError(
                    f"Voxel coordinate {dim!r} must be 1D, got shape {values.shape}."
                )

        if world_coord_names is None:
            world_coord_names = ("y", "x") if ndim == 2 else ("z", "y", "x")

        if len(world_coord_names) != ndim:
            raise ValueError(
                "world_coord_names must have one entry per voxel dimension; got "
                f"{len(world_coord_names)} names for {ndim} dimensions."
            )

        affine = np.asarray(voxel_to_world, dtype=np.float64)
        expected_shape = (ndim + 1, ndim + 1)
        pose_coord_np: npt.NDArray[Any] | None = None
        if pose_coord is not None:
            pose_coord_np = np.asarray(pose_coord)
            if pose_coord_np.ndim != 1:
                raise ValueError(
                    f"pose_coord must be 1D, got shape {pose_coord_np.shape}."
                )
            expected_stack_shape = (pose_coord_np.size, *expected_shape)
            if affine.shape != expected_stack_shape:
                raise ValueError(
                    f"voxel_to_world must have shape {expected_stack_shape} for "
                    f"{pose_coord_np.size} poses, got {affine.shape}."
                )
            if not np.all(np.isfinite(affine)):
                raise ValueError("Pose-dependent voxel_to_world must be finite.")
            homogeneous_row = np.zeros(ndim + 1)
            homogeneous_row[-1] = 1.0
            if not np.allclose(affine[:, -1, :], homogeneous_row):
                raise ValueError(
                    "Each pose affine must have a valid homogeneous final row."
                )
            if affine.shape[0] > 1:
                scalings = np.stack(
                    [
                        np.array(
                            list(get_affine_axis_scalings(affine[p], all_dims).values())
                        )
                        for p in range(affine.shape[0])
                    ]
                )
                if not np.allclose(scalings, scalings[0], rtol=1e-6):
                    raise ValueError(
                        "All pose affines must share equal spatial scale magnitudes."
                    )
        elif affine.shape != expected_shape:
            raise ValueError(
                f"voxel_to_world must have shape {expected_shape}, got {affine.shape}."
            )

        dim_size = {dim: len(values) for dim, values in voxel_coords_np.items()}
        coord_names = tuple(world_coord_names)
        if pose_coord_np is not None:
            dim_size = {POSE_DIM: pose_coord_np.size, **dim_size}
            coord_names = (*coord_names, POSE_DIM)

        super().__init__(coord_names=coord_names, dim_size=dim_size)
        self.voxel_coords = voxel_coords_np
        self.voxel_to_world = affine
        self.fixed_voxel_coords = fixed
        self._all_dims = all_dims
        self.world_coord_names = tuple(world_coord_names)
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
            World-space coordinate values keyed by `self.world_coord_names`, plus a
            `pose` entry for pose-dependent geometry.
        """
        active_values = {
            dim: self.voxel_coords[dim][np.asarray(dim_positions[dim])]
            for dim in self.dims
            if dim != POSE_DIM
        }
        shape = np.asarray(next(iter(active_values.values()))).shape
        ones = np.ones(shape, dtype=np.float64)
        voxel_values = [
            active_values[dim]
            if dim in active_values
            else np.full(shape, self.fixed_voxel_coords[dim], dtype=np.float64)
            for dim in self._all_dims
        ]
        stacked = np.stack([*voxel_values, ones], axis=0).reshape(
            len(self._all_dims) + 1, -1
        )
        num_world = len(self.world_coord_names)

        if self.pose_coord is not None:
            pose_positions = (
                np.broadcast_to(np.asarray(dim_positions[POSE_DIM]), shape)
                .reshape(-1)
                .astype(int)
            )
            selected_affines = self.voxel_to_world[pose_positions]
            transformed_flat = np.einsum("mij,jm->mi", selected_affines, stacked)
            transformed = transformed_flat.T.reshape((num_world + 1, *shape))
            result: dict[Hashable, Any] = {
                name: transformed[i] for i, name in enumerate(self.world_coord_names)
            }
            result[POSE_DIM] = self.pose_coord[pose_positions].reshape(shape)
            return result

        transformed = (self.voxel_to_world @ stacked).reshape((num_world + 1, *shape))
        return {name: transformed[i] for i, name in enumerate(self.world_coord_names)}

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
        world_values = [
            np.asarray(coord_labels[name]) for name in self.world_coord_names
        ]
        shape = np.asarray(world_values[0]).shape
        ones = np.ones(shape, dtype=np.float64)
        stacked = np.stack([*world_values, ones], axis=0).reshape(
            len(self.world_coord_names) + 1, -1
        )
        voxel_values = (np.linalg.inv(self.voxel_to_world) @ stacked).reshape(
            (len(self._all_dims) + 1, *shape)
        )

        dim_positions: dict[str, Any] = {}
        for i, dim in enumerate(self._all_dims):
            if dim not in self.dims:
                continue
            dim_positions[dim] = _reverse_lookup_positions(
                voxel_values[i].reshape(-1), self.voxel_coords[dim]
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
            self.world_coord_names == other.world_coord_names
            and self.dims == other.dims
            and self._all_dims == other._all_dims
            and self.fixed_voxel_coords == other.fixed_voxel_coords
            and pose_equal
            and all(
                np.array_equal(self.voxel_coords[dim], other.voxel_coords[dim])
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
        new_voxel_coords = dict(self.voxel_coords)
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
                    self.voxel_coords[dim][_scalar_indexer_value(indexer)]
                )
                del new_voxel_coords[dim]
                continue
            if isinstance(indexer, Variable):
                if indexer.ndim != 1:
                    return None
                indexer = indexer.values
            elif isinstance(indexer, list | tuple):
                indexer = np.asarray(indexer)
            elif not isinstance(indexer, slice | np.ndarray):
                return None
            values = self.voxel_coords[dim][indexer]
            if np.ndim(values) != 1:
                return None
            new_voxel_coords[dim] = values
        if not new_voxel_coords:
            return None
        return type(self)(
            new_voxel_coords,
            new_affine,
            self.world_coord_names,
            fixed_voxel_coords=new_fixed,
            all_dims=self._all_dims,
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
    world_coord_attrs: Mapping[Any, Mapping[str, Any]] | None = None,
) -> xr.DataArray:
    """Attach world coordinates to a DataArray, making it VoxelData-compatible.

    Voxel dimensions are the native voxel names (`k`/`j`/`i`) present on `data`, in
    canonical affine input-space column order; each gets the matching fixed world
    coordinate name (`k`→`z`, `j`→`y`, `i`→`x`), so e.g. a `(k, i)` DataArray gets
    `z`/`x` world coordinates, not `z`/`y`. Each derived world coordinate always ends
    up with `voxdim` (spacing along that axis) and `units` (`"mm"`, the project-wide
    world-coordinate unit) attrs; `world_coord_attrs` can override either, and any
    value left unset there is filled in with these defaults.

    Parameters
    ----------
    data : xarray.DataArray
        Input array that already carries 1D integer voxel-space coordinates on its
        native voxel dims (`k`/`j`/`i`).
    voxel_to_world : (N+1, N+1) numpy.ndarray or (npose, N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel-space coordinates to world-space
        coordinates, or a stack of one such affine per pose. A stack requires `data`
        to have a matching `pose` dimension with a 1D coordinate; a single affine
        applies to every pose (or to no pose dimension at all).
    world_coord_attrs : mapping[str, mapping[str, Any]], optional
        Attributes to attach to the derived world coordinates, keyed by world
        coordinate name. `voxdim` and `units` are filled in with their defaults
        (computed spacing and `"mm"`) for any coordinate that doesn't specify them
        here.

    Returns
    -------
    xarray.DataArray
        A new DataArray with derived world coordinates attached. Axis-aligned
        affines produce ordinary 1D coordinates; oblique affines produce lazily
        generated coordinates attached via a `CoordinateTransformIndex`.

    Raises
    ------
    ValueError
        If `data` has no native voxel dims (`k`/`j`/`i`), if their coordinates are
        not 1D dimension coordinates, or if a pose-stacked `voxel_to_world` is given
        without a matching `pose` dimension/coordinate on `data`.
    TypeError
        If a voxel dimension's coordinate does not have integer dtype.
    """
    voxel_dims = tuple(dim for dim in VOXEL_DIMS if dim in data.dims)
    if not voxel_dims:
        raise ValueError(
            f"data must have at least one native voxel dim {VOXEL_DIMS!r}, got dims "
            f"{data.dims!r}."
        )
    voxel_to_world_name = dict(zip(VOXEL_DIMS, SPATIAL_DIMS, strict=True))
    world_coord_names: tuple[Hashable, ...] = tuple(
        voxel_to_world_name[dim] for dim in voxel_dims
    )
    voxel_coords: dict[str, npt.NDArray[np.int64]] = {}
    for dim in voxel_dims:
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
        voxel_coords[dim] = np.asarray(coord.values, dtype=np.int64)

    voxel_to_world_array = np.asarray(voxel_to_world, dtype=np.float64)

    pose_coord = None
    drop_names = list(world_coord_names)
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
        drop_names.append(POSE_DIM)

    base = data.drop_vars(
        [name for name in drop_names if name in data.coords], errors="ignore"
    )
    index = VoxelToWorldIndex.from_affine(
        voxel_coords,
        voxel_to_world_array,
        world_coord_names=world_coord_names,
        world_coord_attrs=world_coord_attrs,
        pose_coord=pose_coord,
    )
    result = base.assign_coords(xr.Coordinates.from_xindex(index))

    if world_coord_attrs is not None:
        for name, attrs in world_coord_attrs.items():
            if name in result.coords:
                result.coords[name].attrs.update(dict(attrs))

    representative_affine = (
        voxel_to_world_array[0] if pose_coord is not None else voxel_to_world_array
    )
    world_spacings = get_voxel_to_world_spacings_from_coords(
        voxel_coords, representative_affine
    )
    for i, (dim, name) in enumerate(zip(voxel_dims, world_coord_names, strict=True)):
        spacing = world_spacings[dim]
        if spacing is None:
            spacing = np.linalg.norm(representative_affine[:-1, i])
        if name not in result.coords:
            continue
        if "voxdim" not in result.coords[name].attrs:
            voxdim = np.float64(spacing).item()
            result.coords[name].attrs["voxdim"] = voxdim
            index.world_coord_attrs.setdefault(name, {})["voxdim"] = voxdim
        if "units" not in result.coords[name].attrs:
            result.coords[name].attrs["units"] = "mm"
            index.world_coord_attrs.setdefault(name, {})["units"] = "mm"

    return result


def update_voxel_to_world_coord_attrs(
    data: xr.DataArray, attrs_by_name: Mapping[Hashable, Mapping[str, Any]]
) -> xr.DataArray:
    """Update attrs on `data`'s index-derived world coordinates.

    World coordinates are index-derived: any operation that touches the index (e.g.
    `.assign_coords()` on an unrelated coordinate) re-derives them from the
    `VoxelToWorldIndex`'s own stored `world_coord_attrs`, silently discarding a plain
    in-place `data.coords[name].attrs[...] = ...` mutation the next time that happens.
    This updates both the coordinate's own attrs and the index's stored copy, so the
    update survives.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to update. Must carry a voxel-to-world index.
    attrs_by_name : mapping[Hashable, mapping[str, Any]]
        Attrs to merge in, keyed by world coordinate name.

    Returns
    -------
    xarray.DataArray
        `data` with the given attrs merged into each named world coordinate, on both
        the coordinate itself and the underlying index.
    """
    result = data.copy()
    for index in _collect_voxel_to_world_indexes(result):
        for name, attrs in attrs_by_name.items():
            if name not in result.coords:
                continue
            result.coords[name].attrs.update(attrs)
            index.world_coord_attrs.setdefault(name, {}).update(attrs)
    return result


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


def _collect_voxel_to_world_indexes(data: xr.DataArray) -> list[VoxelToWorldIndex]:
    """Return the unique `VoxelToWorldIndex` objects present on `data`.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    list[VoxelToWorldIndex]
        Unique index objects (the same object is registered under multiple
        coordinate names, e.g. `z`, `y`, and `x`), sorted by each index's voxel
        dimensions in canonical `VOXEL_DIMS` order. `data` carries at most one, since
        [VoxelToWorldIndex][confusius._utils.geometry.VoxelToWorldIndex] always wraps
        all voxel dimensions jointly.
    """
    seen: dict[int, VoxelToWorldIndex] = {
        id(index): index
        for index in data.xindexes.values()
        if isinstance(index, VoxelToWorldIndex)
    }

    def _sort_key(index: VoxelToWorldIndex) -> int:
        for dim in index.voxel_dims:
            if dim in VOXEL_DIMS:
                return VOXEL_DIMS.index(dim)
        return len(VOXEL_DIMS)

    return sorted(seen.values(), key=_sort_key)


def get_voxel_to_world_affine(data: xr.DataArray) -> npt.NDArray[np.float64]:
    """Return the voxel-to-world affine owned by the DataArray's index(es).

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData-compatible DataArray.

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
    indexes = _collect_voxel_to_world_indexes(data)
    active_dims = tuple(dim for index in indexes for dim in index.voxel_dims)
    if (
        len(active_dims) not in {2, 3}
        or len(set(active_dims)) != len(active_dims)
        or not set(active_dims).issubset(set(VOXEL_DIMS))
    ):
        raise ValueError("DataArray must have a voxel-to-world index.")
    index = indexes[0]
    return _fold_fixed_dims_into_affine(
        index.voxel_to_world,
        index.all_voxel_dims,
        index.voxel_dims,
        index.fixed_voxel_coords,
    )


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
    for index in _collect_voxel_to_world_indexes(data):
        if not (
            set(index.voxel_dims).issubset(set(VOXEL_DIMS)) and index.fixed_voxel_coords
        ):
            continue
        all_dims = index.all_voxel_dims
        if any(
            dim not in data.dims or data.coords[dim].dims != (dim,) for dim in all_dims
        ):
            continue
        world_coord_names = index.world_coord_names
        world_coord_attrs: dict[Hashable, Mapping[str, Any]] = {
            name: {
                key: value
                for key, value in dict(data.coords[name].attrs).items()
                if key != "voxdim"
            }
            for name in world_coord_names
            if name in data.coords
        }
        return attach_voxel_to_world_index(
            data, index.voxel_to_world, world_coord_attrs=world_coord_attrs
        )
    return data


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
    """Return whether a DataArray is VoxelData-compatible.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    bool
        Whether `data` stores a `voxel_to_world` affine and has 2D or 3D voxel-space
        dimensions drawn from `confusius._dims.VOXEL_DIMS`.
    """
    indexes = _collect_voxel_to_world_indexes(data)
    active_dims = tuple(dim for index in indexes for dim in index.voxel_dims)
    return (
        len(active_dims) in {2, 3}
        and len(set(active_dims)) == len(active_dims)
        and set(active_dims).issubset(set(VOXEL_DIMS))
    )


def get_voxel_to_world_spatial_dims(data: xr.DataArray) -> tuple[str, ...]:
    """Return voxel-space dimensions present on a VoxelData-compatible DataArray.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    tuple[str, ...]
        Present voxel-space dimensions in canonical affine column order.
    """
    indexes = _collect_voxel_to_world_indexes(data)
    active_dims = tuple(dim for index in indexes for dim in index.voxel_dims)
    if active_dims and set(active_dims).issubset(set(VOXEL_DIMS)):
        return active_dims
    return tuple(dim for dim in VOXEL_DIMS if dim in data.dims)


def get_voxel_to_world_coord_names(data: xr.DataArray) -> tuple[str, ...]:
    """Return world coordinate names exposed by voxel-to-world geometry.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    tuple[str, ...]
        World coordinate names in affine row order.
    """
    indexes = _collect_voxel_to_world_indexes(data)
    active_dims = tuple(dim for index in indexes for dim in index.voxel_dims)
    if (
        active_dims
        and len(active_dims) in {2, 3}
        and set(active_dims).issubset(set(VOXEL_DIMS))
    ):
        return tuple(str(name) for index in indexes for name in index.world_coord_names)
    voxel_dims = get_voxel_to_world_spatial_dims(data)
    default_names = tuple({"k": "z", "j": "y", "i": "x"}[dim] for dim in voxel_dims)
    world_coord_names = tuple(
        name
        for name, dim in zip(default_names, voxel_dims, strict=True)
        if name in data.coords and data.coords[name].dims in {voxel_dims, (dim,)}
    )
    if len(world_coord_names) == len(voxel_dims):
        return world_coord_names
    return default_names


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
        VoxelData-compatible DataArray.
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
        VoxelData-compatible DataArray.

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
    world_coord_names = get_voxel_to_world_coord_names(data)
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
    return {
        name: np.float64(origin[i]).item() for i, name in enumerate(world_coord_names)
    }


def get_voxel_to_world_index_spacing(data: xr.DataArray) -> dict[str, float | None]:
    """Return world spacing per voxel-space axis for a VoxelData-compatible DataArray.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData-compatible DataArray.

    Returns
    -------
    dict[str, float | None]
        World spacing keyed by voxel-space dimension.
    """
    voxel_dims = get_voxel_to_world_spatial_dims(data)
    voxel_coords = {dim: data.coords[dim].values for dim in voxel_dims}
    return get_voxel_to_world_spacings_from_coords(
        voxel_coords, get_voxel_to_world_affine(data)
    )


def get_voxel_to_world_direction_matrix(
    data: xr.DataArray,
) -> npt.NDArray[np.float64]:
    """Return the world-space direction matrix of a VoxelData-compatible DataArray.

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
        VoxelData-compatible DataArray.

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


def get_affine_axis_vectors(
    voxel_to_world: npt.ArrayLike,
    voxel_dims: tuple[str, ...],
) -> dict[str, npt.NDArray[np.float64]]:
    """Return the world step vector for one voxel-space unit along each axis.

    Parameters
    ----------
    voxel_to_world : (N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel space to world space.
    voxel_dims : tuple[str, ...]
        Voxel-space dimension names in affine column order.

    Returns
    -------
    dict[str, numpy.ndarray]
        World step vectors keyed by voxel-space dimension name.
    """
    affine = np.asarray(voxel_to_world, dtype=np.float64)
    linear = affine[:-1, :-1]
    return {dim: linear[:, i].copy() for i, dim in enumerate(voxel_dims)}


def get_affine_axis_scalings(
    voxel_to_world: npt.ArrayLike,
    voxel_dims: tuple[str, ...],
) -> dict[str, float]:
    """Return world distance per one voxel-space unit along each axis.

    Parameters
    ----------
    voxel_to_world : (N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel space to world space.
    voxel_dims : tuple[str, ...]
        Voxel-space dimension names in affine column order.

    Returns
    -------
    dict[str, float]
        Euclidean norms of the affine column vectors, keyed by voxel-space dimension.
    """
    vectors = get_affine_axis_vectors(voxel_to_world, voxel_dims)
    return {
        dim: np.float64(np.linalg.norm(vector)).item()
        for dim, vector in vectors.items()
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
    voxel_coords: Mapping[str, npt.ArrayLike],
    voxel_to_world: npt.ArrayLike,
) -> dict[str, float | None]:
    """Return world spacing for regularly sampled voxel axes.

    Parameters
    ----------
    voxel_coords : mapping[str, array-like]
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
    scalings = get_affine_axis_scalings(voxel_to_world, tuple(voxel_coords))
    spacings: dict[str, float | None] = {}
    for dim, values in voxel_coords.items():
        values_array = np.asarray(values, dtype=np.float64)
        step, approximate = get_representative_step(values_array)
        if approximate:
            spacings[dim] = None
        elif step is None:
            spacings[dim] = scalings[dim]
        else:
            spacings[dim] = abs(step) * scalings[dim]
    return spacings
