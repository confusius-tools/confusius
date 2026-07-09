"""Mesh-vertex transforms for atlas resampling (affine and nonlinear).

An atlas mesh vertex transform is a *pull* transform, following the same convention as
`confusius.registration`: it maps a point in the atlas's current physical space back to
its base (BrainGlobe OBJ) physical space. Affine transforms are homogeneous `(4, 4)`
matrices; nonlinear transforms are B-spline or dense displacement-field DataArrays. All
points, vertices, and displacement components are in DataArray dim order `(z, y, x)`:
component `i` of a displacement field displaces along axis `dims[i]`.
"""

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.interpolate import interpn

from confusius.registration.bspline import sample_displacement_field_like

MeshVertexTransform = npt.NDArray[np.float64] | xr.DataArray
"""Transform mapping current atlas physical coordinates back to base atlas space.

Affine transforms use homogeneous `(4, 4)` matrices; nonlinear transforms use B-spline or
displacement-field DataArrays (`attrs["type"]` in `{"bspline_transform",
"displacement_field_transform"}`).
"""


def _validate_mesh_vertex_transform(transform: MeshVertexTransform) -> None:
    """Raise ValueError if `transform` is not a valid mesh vertex transform.

    Parameters
    ----------
    transform : numpy.ndarray or xarray.DataArray
        Candidate transform.

    Raises
    ------
    ValueError
        If `transform` is neither a homogeneous affine nor a supported nonlinear
        transform DataArray.
    """
    if isinstance(transform, np.ndarray):
        if transform.shape != (4, 4):
            raise ValueError(
                f"Mesh affine transform must have shape (4, 4); got {transform.shape}."
            )
        return

    transform_type = transform.attrs.get("type")
    if transform_type not in {"bspline_transform", "displacement_field_transform"}:
        raise ValueError(
            "Mesh nonlinear transform must have attrs['type'] equal to "
            "'bspline_transform' or 'displacement_field_transform'; got "
            f"{transform_type!r}."
        )


def _transform_points(
    transform: MeshVertexTransform,
    points: npt.NDArray[np.float64],
    reference: xr.DataArray,
) -> npt.NDArray[np.float64]:
    """Apply a pull transform to physical points.

    Parameters
    ----------
    transform : numpy.ndarray or xarray.DataArray
        Pull transform mapping points from `reference` space into some moving/base
        space.
    points : (N, D) numpy.ndarray
        Physical points in DataArray dim order `(z, y, x)`.
    reference : xarray.DataArray
        Reference grid on which nonlinear transforms live.

    Returns
    -------
    numpy.ndarray
        Transformed points with shape `(N, D)`.
    """
    _validate_mesh_vertex_transform(transform)

    if isinstance(transform, np.ndarray):
        n_points, ndim = points.shape
        points_h = np.hstack([points, np.ones((n_points, 1), dtype=np.float64)])
        return (transform @ points_h.T).T[:, :ndim]

    field = transform
    if transform.attrs.get("type") == "bspline_transform":
        field = sample_displacement_field_like(transform, reference)
    displacement = _interpolate_displacement_field(field, points)
    return points + displacement


def _compose_mesh_vertex_transforms(
    old_transform: MeshVertexTransform,
    new_transform: MeshVertexTransform,
    new_reference: xr.DataArray,
    old_reference: xr.DataArray,
) -> MeshVertexTransform:
    """Compose mesh pull transforms as `old_transform ∘ new_transform`.

    Parameters
    ----------
    old_transform : numpy.ndarray or xarray.DataArray
        Pull transform from the current atlas physical space to the base atlas space.
    new_transform : numpy.ndarray or xarray.DataArray
        Pull transform from the new atlas physical space to the current atlas space.
    new_reference : xarray.DataArray
        Reference grid of the new atlas space.
    old_reference : xarray.DataArray
        Reference grid of the current atlas space.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Pull transform from the new atlas physical space to the base atlas space. Affine
        when both inputs are affine, otherwise a dense displacement field on
        `new_reference`'s grid.
    """
    if isinstance(old_transform, np.ndarray) and np.allclose(old_transform, np.eye(4)):
        return (
            new_transform.copy(deep=False)
            if isinstance(new_transform, xr.DataArray)
            else new_transform.copy()
        )
    if isinstance(new_transform, np.ndarray) and np.allclose(new_transform, np.eye(4)):
        return (
            old_transform.copy(deep=False)
            if isinstance(old_transform, xr.DataArray)
            else old_transform.copy()
        )
    if isinstance(old_transform, np.ndarray) and isinstance(new_transform, np.ndarray):
        return old_transform @ new_transform

    dims = [str(dim) for dim in new_reference.dims]
    grid = np.meshgrid(
        *[
            np.asarray(new_reference.coords[dim].values, dtype=np.float64)
            for dim in dims
        ],
        indexing="ij",
    )
    reference_points = np.stack(grid, axis=-1).reshape(-1, len(dims))
    current_points = _transform_points(new_transform, reference_points, new_reference)
    base_points = _transform_points(old_transform, current_points, old_reference)
    # Points and displacement components are both in DataArray dim order (component
    # `dims[i]` displaces along axis `dims[i]`), matching `_transform_points` and
    # `_interpolate_displacement_field`, so no axis reversal is needed.
    displacement = (base_points - reference_points).T.reshape(
        len(dims), *new_reference.shape
    )

    coords: dict = {"component": np.array(dims, dtype=np.str_)}
    coords.update({dim: new_reference.coords[dim] for dim in dims})
    return xr.DataArray(
        displacement,
        dims=["component", *dims],
        coords=coords,
        attrs={"type": "displacement_field_transform"},
    )


def _interpolate_displacement_field(
    field: xr.DataArray, points: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Interpolate a dense displacement field at physical points.

    Parameters
    ----------
    field : xarray.DataArray
        Dense displacement field with dimensions `("component", *spatial_dims)`.
    points : (N, D) numpy.ndarray
        Physical points in the same axis order as `field.dims[1:]`.

    Returns
    -------
    numpy.ndarray
        Interpolated displacement vectors with shape `(N, D)`. Points beyond the field
        domain and its one-voxel edge-padded margin are returned as NaN.
    """
    spatial_dims = [str(dim) for dim in field.dims[1:]]
    spacing = field.fusi.spacing

    # Pad the field by one voxel (edge replication) at each end of every spatial dim, so
    # points within one voxel of a boundary, as barely-outside mesh vertices are, can
    # still be interpolated. Points beyond the padded margin resolve to NaN; the mesh
    # caller drops the vertices that could not be warped.
    padded_field = field.pad({dim: (1, 1) for dim in spatial_dims}, mode="edge")
    grid = []
    for dim in spatial_dims:
        padded_coord = np.asarray(padded_field.coords[dim], dtype=np.float64).copy()
        padded_coord[0] = padded_coord[1] - spacing[dim]
        padded_coord[-1] = padded_coord[-2] + spacing[dim]
        grid.append(padded_coord)

    displacements = interpn(
        grid,
        np.moveaxis(np.asarray(padded_field.values, dtype=np.float64), 0, -1),
        points,
        bounds_error=False,
        fill_value=np.nan,
    )
    return np.asarray(displacements, dtype=np.float64)


def _invert_displacement_field_at_points(
    field: xr.DataArray,
    points: npt.NDArray[np.float64],
    initial_guess_affine: npt.NDArray[np.float64] | None = None,
    *,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
) -> npt.NDArray[np.float64]:
    """Map moving-space points back to fixed space with fixed-point iteration.

    Parameters
    ----------
    field : xarray.DataArray
        Dense forward displacement field that maps fixed-space points to moving-space
        points as `moving = fixed + field(fixed)`.
    points : (N, D) numpy.ndarray
        Moving-space points to invert, in the same axis order as `field.dims[1:]`.
    initial_guess_affine : (D+1, D+1) numpy.ndarray, optional
        Affine inverse used to seed the iteration. If not provided, the moving-space
        points themselves are used as the initial guess.
    max_iterations : int, default: 20
        Maximum number of fixed-point updates.
    tolerance : float, default: 1e-6
        Convergence threshold on the maximum point update, in physical units.

    Returns
    -------
    numpy.ndarray
        Approximate fixed-space points with shape `(N, D)`. When the iteration does not
        converge within `max_iterations`, the last iterate is returned.
    """
    if points.shape[0] == 0:
        return points.copy()

    if initial_guess_affine is None:
        fixed_points = points.copy()
    else:
        n_points, ndim = points.shape
        points_h = np.hstack([points, np.ones((n_points, 1), dtype=np.float64)])
        fixed_points = (initial_guess_affine @ points_h.T).T[:, :ndim]

    for _ in range(max_iterations):
        displaced = _interpolate_displacement_field(field, fixed_points)
        updated = points - displaced
        if np.max(np.linalg.norm(updated - fixed_points, axis=1)) <= tolerance:
            return updated
        fixed_points = updated

    return fixed_points


def _apply_mesh_vertex_transform(
    transform: MeshVertexTransform,
    vertices: npt.NDArray[np.float64],
    reference: xr.DataArray,
) -> npt.NDArray[np.float64]:
    """Transform mesh vertices from base atlas space into current physical space.

    Parameters
    ----------
    transform : numpy.ndarray or xarray.DataArray
        Pull transform from the current atlas physical space back to the base atlas
        physical space.
    vertices : (N, 3) numpy.ndarray
        Mesh vertices expressed in the base atlas physical coordinates (millimetres).
    reference : xarray.DataArray
        Current atlas reference grid. Used to sample B-spline transforms into a dense
        displacement field when needed.

    Returns
    -------
    numpy.ndarray
        Mesh vertices in the current physical space.
    """
    if isinstance(transform, np.ndarray):
        n_vertices, ndim = vertices.shape
        vertices_h = np.hstack([vertices, np.ones((n_vertices, 1), dtype=np.float64)])
        return (np.linalg.inv(transform) @ vertices_h.T).T[:, :ndim]

    field = transform
    if transform.attrs.get("type") == "bspline_transform":
        field = sample_displacement_field_like(transform, reference)

    initial_guess_affine = None
    pre_affine = transform.attrs.get("affines", {}).get("bspline_initialization")
    if pre_affine is not None:
        initial_guess_affine = np.linalg.inv(np.asarray(pre_affine, dtype=np.float64))

    return _invert_displacement_field_at_points(field, vertices, initial_guess_affine)


def _drop_vertices_outside_grid(
    vertices: npt.NDArray[np.float64],
    faces: npt.NDArray[np.int32],
    reference: xr.DataArray,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Drop mesh vertices outside the reference grid (plus a margin) and reindex faces.

    Applied to warped mesh vertices: a nonlinear warp can move vertices outside the grid
    and returns NaN for vertices too far outside to interpolate. Vertices beyond the grid
    by more than one voxel (the padded interpolation margin), and NaN vertices, are
    dropped along with any face that references them (NaN fails the bounds comparison, so
    NaN vertices are removed too).

    Parameters
    ----------
    vertices : (N, 3) numpy.ndarray
        Warped mesh vertices in DataArray dim order `(z, y, x)`.
    faces : (M, 3) numpy.ndarray
        Zero-indexed triangle face indices into `vertices`.
    reference : xarray.DataArray
        Reference grid whose coordinate bounds define the valid domain.

    Returns
    -------
    vertices : numpy.ndarray
        Surviving vertices, shape `(K, 3)` with `K <= N`.
    faces : numpy.ndarray
        Faces whose three vertices all survived, reindexed into the new vertex array.
    """
    dims = [str(dim) for dim in reference.dims]
    spacing = reference.fusi.spacing
    inside = np.ones(len(vertices), dtype=bool)
    for axis, dim in enumerate(dims):
        coord = reference.coords[dim].values
        # Keep the same one-voxel margin the field interpolation is padded to, so a
        # vertex within `spacing` of a boundary (e.g. the anterior/posterior tips of the
        # Allen brain) is retained rather than clipped.
        margin = spacing[dim] if spacing[dim] is not None else 0.0
        inside &= (vertices[:, axis] >= coord.min() - margin) & (
            vertices[:, axis] <= coord.max() + margin
        )

    keep_idx = np.where(inside)[0]
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)

    new_face_idx = old_to_new[faces]  # (M, 3); -1 for dropped vertices.
    valid = np.all(new_face_idx >= 0, axis=1)
    return vertices[keep_idx], new_face_idx[valid].astype(np.int32)
