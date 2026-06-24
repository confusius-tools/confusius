"""Geometry helpers for voxel-space to physical-space transforms.

This module prototypes a geometry model where a DataArray keeps 1D voxel-space
coordinates (for example `i`, `j`, `k`) and stores a single affine that maps
those voxel-space coordinates into physical-space coordinates (for example
`x`, `y`, `z`).

The physical coordinates are exposed lazily via Xarray's
`CoordinateTransformIndex`.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
from xarray.indexes import CoordinateTransform, CoordinateTransformIndex

from confusius._utils.coordinates import get_representative_step


class VoxelSpaceAffineTransform(CoordinateTransform):
    """Coordinate transform from voxel-space coordinates to physical space.

    The transform combines:

    1. a dense array-position -> voxel-space lookup using the 1D coordinate arrays
       attached to each dimension, and
    2. a homogeneous affine that maps voxel-space coordinates to physical-space
       coordinates.

    This lets a dense array with dimensions like `(j, i)` or `(k, j, i)` carry
    irregular voxel-space coordinates such as `i = [0, 2, 3]`, while still exposing
    exact physical coordinates through Xarray's
    `CoordinateTransformIndex`.

    Parameters
    ----------
    voxel_coords : mapping[str, array-like]
        Ordered mapping from dimension name to its 1D voxel-space coordinates. The key
        order defines the column order of the affine input space.
    voxel_to_physical : (N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel-space coordinates to physical-space
        coordinates. The input column order must match `voxel_coords`. The output row
        order must match `physical_coord_names`.
    physical_coord_names : tuple[Hashable, ...], optional
        Names of the physical-space coordinates exposed by the transform. If not
        provided, defaults to `("z", "y", "x")` for 3D and `("y", "x")` for 2D.

    Raises
    ------
    ValueError
        If any voxel coordinate is not 1D, not strictly increasing, or if the affine
        shape does not match the number of voxel-space dimensions.
    """

    voxel_coords: dict[str, npt.NDArray[np.float64]]
    voxel_to_physical: npt.NDArray[np.float64]

    def __init__(
        self,
        voxel_coords: Mapping[str, npt.ArrayLike],
        voxel_to_physical: npt.ArrayLike,
        physical_coord_names: tuple[Hashable, ...] | None = None,
    ) -> None:
        voxel_coords_np = {
            str(dim): np.asarray(values, dtype=np.float64)
            for dim, values in voxel_coords.items()
        }
        ndim = len(voxel_coords_np)

        if ndim not in {2, 3}:
            raise ValueError(
                f"VoxelSpaceAffineTransform only supports 2D or 3D inputs, got {ndim}."
            )

        for dim, values in voxel_coords_np.items():
            if values.ndim != 1:
                raise ValueError(
                    f"Voxel coordinate {dim!r} must be 1D, got shape {values.shape}."
                )
            if values.size > 1 and not np.all(np.diff(values) > 0):
                raise ValueError(
                    f"Voxel coordinate {dim!r} must be strictly increasing."
                )

        if physical_coord_names is None:
            physical_coord_names = ("y", "x") if ndim == 2 else ("z", "y", "x")

        if len(physical_coord_names) != ndim:
            raise ValueError(
                "physical_coord_names must have one entry per voxel dimension; got "
                f"{len(physical_coord_names)} names for {ndim} dimensions."
            )

        affine = np.asarray(voxel_to_physical, dtype=np.float64)
        expected_shape = (ndim + 1, ndim + 1)
        if affine.shape != expected_shape:
            raise ValueError(
                f"voxel_to_physical must have shape {expected_shape}, got {affine.shape}."
            )

        super().__init__(
            coord_names=tuple(physical_coord_names),
            dim_size={dim: len(values) for dim, values in voxel_coords_np.items()},
        )
        self.voxel_coords = voxel_coords_np
        self.voxel_to_physical = affine

    def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
        """Transform dense array positions into physical coordinates.

        Parameters
        ----------
        dim_positions : dict[str, Any]
            Dense integer array positions keyed by the dimensions in `self.dims`.

        Returns
        -------
        dict[Hashable, Any]
            Physical-space coordinate values keyed by `self.coord_names`.
        """
        voxel_values = [
            self.voxel_coords[dim][np.asarray(dim_positions[dim])] for dim in self.dims
        ]
        shape = np.asarray(voxel_values[0]).shape
        ones = np.ones(shape, dtype=np.float64)
        stacked = np.stack([*voxel_values, ones], axis=0).reshape(
            len(self.dims) + 1, -1
        )
        transformed = (self.voxel_to_physical @ stacked).reshape(
            (len(self.coord_names) + 1, *shape)
        )
        return {name: transformed[i] for i, name in enumerate(self.coord_names)}

    def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
        """Transform physical coordinates back into dense array positions.

        The returned positions are floating-point dense positions. Xarray rounds them
        during nearest-neighbour selection.

        Parameters
        ----------
        coord_labels : dict[Hashable, Any]
            Physical-space coordinate labels keyed by `self.coord_names`.

        Returns
        -------
        dict[str, Any]
            Dense array positions keyed by `self.dims`.
        """
        physical_values = [np.asarray(coord_labels[name]) for name in self.coord_names]
        shape = np.asarray(physical_values[0]).shape
        ones = np.ones(shape, dtype=np.float64)
        stacked = np.stack([*physical_values, ones], axis=0).reshape(
            len(self.coord_names) + 1, -1
        )
        voxel_values = (np.linalg.inv(self.voxel_to_physical) @ stacked).reshape(
            (len(self.dims) + 1, *shape)
        )

        dim_positions: dict[str, Any] = {}
        for i, dim in enumerate(self.dims):
            voxel_axis = self.voxel_coords[dim]
            dim_positions[dim] = np.interp(
                voxel_values[i].reshape(-1),
                voxel_axis,
                np.arange(voxel_axis.size, dtype=np.float64),
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
            voxel-space coordinates, and affine.
        """
        if not isinstance(other, VoxelSpaceAffineTransform):
            return False
        return (
            self.coord_names == other.coord_names
            and self.dims == other.dims
            and all(
                np.array_equal(self.voxel_coords[dim], other.voxel_coords[dim])
                for dim in self.dims
            )
            and np.allclose(self.voxel_to_physical, other.voxel_to_physical)
        )

    def __repr__(self) -> str:
        """Return a compact repr."""
        return (
            f"VoxelSpaceAffineTransform(dims={self.dims!r}, "
            f"coord_names={self.coord_names!r})"
        )


def add_physical_coords_from_voxel_affine(
    data: xr.DataArray,
    voxel_to_physical: npt.ArrayLike,
    *,
    voxel_dims: tuple[str, ...],
    physical_coord_names: tuple[Hashable, ...] | None = None,
    physical_coord_attrs: Mapping[str, Mapping[str, Any]] | None = None,
) -> xr.DataArray:
    """Attach lazy physical coordinates to a DataArray.

    Parameters
    ----------
    data : xarray.DataArray
        Input array that already carries 1D voxel-space coordinates on `voxel_dims`.
    voxel_to_physical : (N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel-space coordinates to physical-space
        coordinates.
    voxel_dims : tuple[str, ...]
        Dimension names whose coordinates define voxel space. The order determines the
        affine input-space column order.
    physical_coord_names : tuple[Hashable, ...], optional
        Names of the physical coordinates to attach lazily. If not provided, defaults
        to `("z", "y", "x")` for 3D and `("y", "x")` for 2D.
    physical_coord_attrs : mapping[str, mapping[str, Any]], optional
        Attributes to attach to the derived physical coordinates, keyed by physical
        coordinate name.

    Returns
    -------
    xarray.DataArray
        A new DataArray with lazily generated physical coordinates attached via a
        `CoordinateTransformIndex`.

    Raises
    ------
    ValueError
        If `voxel_dims` are missing from the DataArray or if their coordinates are not
        1D dimension coordinates.
    """
    voxel_coords: dict[str, npt.NDArray[np.float64]] = {}
    for dim in voxel_dims:
        if dim not in data.dims:
            raise ValueError(
                f"Voxel dimension {dim!r} is not present in the DataArray."
            )
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
        voxel_coords[dim] = np.asarray(coord.values, dtype=np.float64)

    transform = VoxelSpaceAffineTransform(
        voxel_coords,
        voxel_to_physical,
        physical_coord_names=physical_coord_names,
    )
    physical_coords = xr.Coordinates.from_xindex(CoordinateTransformIndex(transform))
    result = data.assign_coords(physical_coords)

    if physical_coord_attrs is not None:
        for name, attrs in physical_coord_attrs.items():
            if name in result.coords:
                result.coords[name].attrs.update(dict(attrs))

    return result


def get_affine_origin(
    voxel_to_physical: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return the physical location of the voxel-space origin.

    Parameters
    ----------
    voxel_to_physical : (N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel space to physical space.

    Returns
    -------
    (N,) numpy.ndarray
        Physical coordinates corresponding to voxel-space origin `(0, ..., 0)`.
    """
    affine = np.asarray(voxel_to_physical, dtype=np.float64)
    return affine[:-1, -1].copy()


def get_affine_axis_vectors(
    voxel_to_physical: npt.ArrayLike,
    voxel_dims: tuple[str, ...],
) -> dict[str, npt.NDArray[np.float64]]:
    """Return the physical step vector for one voxel-space unit along each axis.

    Parameters
    ----------
    voxel_to_physical : (N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel space to physical space.
    voxel_dims : tuple[str, ...]
        Voxel-space dimension names in affine column order.

    Returns
    -------
    dict[str, numpy.ndarray]
        Physical step vectors keyed by voxel-space dimension name.
    """
    affine = np.asarray(voxel_to_physical, dtype=np.float64)
    linear = affine[:-1, :-1]
    return {dim: linear[:, i].copy() for i, dim in enumerate(voxel_dims)}


def get_affine_axis_scalings(
    voxel_to_physical: npt.ArrayLike,
    voxel_dims: tuple[str, ...],
) -> dict[str, float]:
    """Return physical distance per one voxel-space unit along each axis.

    Parameters
    ----------
    voxel_to_physical : (N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel space to physical space.
    voxel_dims : tuple[str, ...]
        Voxel-space dimension names in affine column order.

    Returns
    -------
    dict[str, float]
        Euclidean norms of the affine column vectors, keyed by voxel-space dimension.
    """
    vectors = get_affine_axis_vectors(voxel_to_physical, voxel_dims)
    return {dim: float(np.linalg.norm(vector)) for dim, vector in vectors.items()}


def get_affine_orientation_matrix(
    voxel_to_physical: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return unit physical-space axis directions from a voxel-to-physical affine.

    Parameters
    ----------
    voxel_to_physical : (N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel space to physical space.

    Returns
    -------
    (N, N) numpy.ndarray
        Matrix whose columns are unit physical-space vectors for each voxel-space axis.
        Zero-length columns are preserved as zeros.
    """
    affine = np.asarray(voxel_to_physical, dtype=np.float64)
    linear = affine[:-1, :-1].copy()
    norms = np.linalg.norm(linear, axis=0)
    nonzero = norms > 0
    linear[:, nonzero] /= norms[nonzero]
    linear[:, ~nonzero] = 0.0
    return linear


def get_physical_spacings(
    voxel_coords: Mapping[str, npt.ArrayLike],
    voxel_to_physical: npt.ArrayLike,
) -> dict[str, float | None]:
    """Return physical spacing for regularly sampled voxel axes.

    Parameters
    ----------
    voxel_coords : mapping[str, array-like]
        Ordered mapping from voxel-space dimension name to its 1D coordinates.
    voxel_to_physical : (N+1, N+1) numpy.ndarray
        Homogeneous affine mapping voxel space to physical space.

    Returns
    -------
    dict[str, float | None]
        Physical spacing keyed by voxel-space dimension. Returns `None` when the
        voxel-space coordinate is irregular or has fewer than two samples.
    """
    scalings = get_affine_axis_scalings(voxel_to_physical, tuple(voxel_coords))
    spacings: dict[str, float | None] = {}
    for dim, values in voxel_coords.items():
        step, approximate = get_representative_step(
            np.asarray(values, dtype=np.float64)
        )
        if step is None or approximate:
            spacings[dim] = None
        else:
            spacings[dim] = abs(step) * scalings[dim]
    return spacings
