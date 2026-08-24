"""Image visualization utilities for fUSI data."""

import math
import numbers
import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._dims import POSE_DIM, SPATIAL_DIMS, VOXEL_DIMS
from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius._utils.geometry import (
    get_voxel_to_world_coord_names,
    get_voxel_to_world_index_spacing,
    get_voxel_to_world_spatial_dims,
    has_axis_aligned_voxel_to_world_index,
    has_voxel_to_world_index,
    require_scalar_pose_affine,
)
from confusius._utils.mask import select_masked_features
from confusius._utils.plotting import (
    blend_red_cyan,
    compute_oblique_axis_aligned_grid_geometry,
    qr_axis_spacing,
    scale_min_max,
)
from confusius._utils.plotting import (
    resample_to_axis_aligned_world_grid as _shared_resample_to_axis_aligned_world_grid,
)
from confusius._utils.stack import find_stack_level
from confusius.plotting._hover import (
    _HoverManager,
    _normalize_roi_labels,
)
from confusius.plotting._utils import (
    _auto_fg_color,
    _get_distinct_colors,
    _materialize_axis_aligned_world_grid_for_display,
    _resolve_font_sizes,
    _style_colorbar,
    coerce_complex_to_magnitude,
    sort_coords_for_plot,
)
from confusius.signal import clean
from confusius.validation import (
    ensure_voxeldata,
    validate_matching_coordinates,
    validate_time_series,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure, SubFigure

_BASE_SIZE = 4.0
"""Base subplot size for VolumePlotter when creating new figures.

Actual figure size is computed as `(subplot_size * ncols + 1 inch for colorbar,
subplot_size * nrows)` and then constrained to a maximum size.
"""


def _compute_grid_dims(
    n_slices: int, nrows: int | None, ncols: int | None
) -> tuple[int, int]:
    """Compute grid dimensions for a grid of `n_slices` panels."""
    if nrows is None and ncols is None:
        ncols = int(np.ceil(np.sqrt(n_slices)))
        nrows = int(np.ceil(n_slices / ncols))
    elif ncols is None:
        assert nrows is not None
        ncols = int(np.ceil(n_slices / nrows))
    elif nrows is None:
        nrows = int(np.ceil(n_slices / ncols))
    return nrows, ncols


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Convert 1D coordinate centers to cell edge positions for `pcolormesh`.

    Handles non-uniform spacing by using midpoints between adjacent centers as interior
    edges, and extrapolating half a step at each end.
    """
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    interior = (centers[:-1] + centers[1:]) / 2
    left = centers[0] - (centers[1] - centers[0]) / 2
    right = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[left], interior, [right]])


def _bilinear_interpolate_grid(
    grid: np.ndarray, row_idx: np.ndarray, col_idx: np.ndarray
) -> np.ndarray:
    """Bilinearly interpolate a 2D `(H, W)` grid at fractional pixel positions.

    Out-of-bounds indices are clamped to the grid edge (flat extrapolation),
    matching `numpy.interp`'s default behaviour for the 1D rectilinear case this
    complements -- see `_slice_edges_and_centers`'s two branches, used together in
    `VolumePlotter.add_contours` to map contour vertices (produced in fractional
    pixel-index space by `skimage.measure.find_contours`) onto the exact same
    display geometry `pcolormesh` draws, whether that's a simple per-axis lookup
    (rectilinear/materialized data) or this affine projection (oblique data).
    Since the underlying transform is affine (linear), this is an exact
    evaluation, not an approximation.
    """
    n_rows, n_cols = grid.shape
    row_idx = np.clip(row_idx, 0, n_rows - 1)
    col_idx = np.clip(col_idx, 0, n_cols - 1)
    row0 = np.floor(row_idx).astype(int)
    col0 = np.floor(col_idx).astype(int)
    row1 = np.clip(row0 + 1, 0, n_rows - 1)
    col1 = np.clip(col0 + 1, 0, n_cols - 1)
    frac_row = row_idx - row0
    frac_col = col_idx - col0
    top = grid[row0, col0] * (1 - frac_col) + grid[row0, col1] * frac_col
    bottom = grid[row1, col0] * (1 - frac_col) + grid[row1, col1] * frac_col
    return top * (1 - frac_row) + bottom * frac_row


def _validate_slice_mode(data: xr.DataArray, slice_mode: str) -> None:
    """Validate slice selection semantics for plotting.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array being sliced for plotting.
    slice_mode : str
        Requested slice dimension.

    Raises
    ------
    ValueError
        If `data` is sliced along an unsupported dimension.
    """
    valid_slice_modes = (
        tuple(str(dim) for dim in data.dims if str(dim) not in VOXEL_DIMS)
        + SPATIAL_DIMS
    )
    if slice_mode not in valid_slice_modes:
        raise ValueError(
            f"Unsupported slice_mode={slice_mode!r} for plotting. "
            f"Supported modes: {valid_slice_modes!r}."
        )


def _project_voxel_to_world_plane(
    slice_da: xr.DataArray,
    dim_row: str,
    dim_col: str,
    *,
    world_row: str | None = None,
    world_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project a voxel-to-world 2D slice into an in-plane orthonormal basis.

    Parameters
    ----------
    slice_da : xarray.DataArray
        Two-dimensional slice whose dimensions are voxel-space dims.
    dim_row : str
        Voxel-space dimension displayed on rows.
    dim_col : str
        Voxel-space dimension displayed on columns.
    world_row : str, optional
        World coordinate name (`"z"`/`"y"`/`"x"`) to project onto the display row
        axis. If provided (together with `world_col`), the basis is the *fixed
        global* unit vector for this world axis instead of one derived from
        `slice_da`'s own affine columns -- see design/world-mode-resample-scoping.md,
        Design C. This makes the projected position identical to that world
        coordinate's own value (no basis to derive at all), and -- unlike the
        default per-volume-local basis -- consistent across independently oriented
        volumes overlaid on the same axes, since every volume dots the same true
        world position onto the same fixed vector.
    world_col : str, optional
        World coordinate name for the display column axis. See `world_row`.

    Returns
    -------
    x_edges : (H+1, W+1) numpy.ndarray
        In-plane x coordinates of cell corners.
    y_edges : (H+1, W+1) numpy.ndarray
        In-plane y coordinates of cell corners.
    x_centers : (H, W) numpy.ndarray
        In-plane x coordinates of cell centers.
    y_centers : (H, W) numpy.ndarray
        In-plane y coordinates of cell centers.

    Raises
    ------
    ValueError
        If `world_row`/`world_col` are not given and `dim_row`/`dim_col` are
        collinear in world space (no well-defined in-plane basis).
    """
    affine = require_scalar_pose_affine(slice_da, "Voxel-to-world plotting")
    dim_order = get_voxel_to_world_spatial_dims(slice_da)
    linear = affine[:-1, :-1]

    row_vals = (
        slice_da.coords[dim_row].values.astype(float)
        if dim_row in slice_da.coords
        else np.arange(slice_da.sizes[dim_row], dtype=float)
    )
    col_vals = (
        slice_da.coords[dim_col].values.astype(float)
        if dim_col in slice_da.coords
        else np.arange(slice_da.sizes[dim_col], dtype=float)
    )
    row_edges = _centers_to_edges(row_vals)
    col_edges = _centers_to_edges(col_vals)

    if world_row is not None and world_col is not None:
        world_dims = get_voxel_to_world_coord_names(slice_da)
        e1 = np.zeros(3)
        e1[world_dims.index(world_col)] = 1.0
        e2 = np.zeros(3)
        e2[world_dims.index(world_row)] = 1.0
    else:
        col_vec = linear[:, dim_order.index(dim_col)]
        row_vec = linear[:, dim_order.index(dim_row)]
        e1 = col_vec / np.linalg.norm(col_vec)
        row_perp = row_vec - np.dot(row_vec, e1) * e1
        row_perp_norm = np.linalg.norm(row_perp)
        if np.isclose(row_perp_norm, 0.0):
            raise ValueError(
                f"Voxel-to-world plotting requires non-collinear plane axes, got "
                f"{dim_row!r} and {dim_col!r}."
            )
        e2 = row_perp / row_perp_norm

    def _project(
        row_axis: np.ndarray, col_axis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        col_grid, row_grid = np.meshgrid(col_axis, row_axis, indexing="xy")
        voxel_components: list[np.ndarray] = []
        for dim in dim_order:
            if dim == dim_row:
                voxel_components.append(row_grid)
            elif dim == dim_col:
                voxel_components.append(col_grid)
            else:
                voxel_components.append(
                    np.full_like(
                        row_grid, float(slice_da.coords[dim].item()), dtype=float
                    )
                )
        homogeneous = np.stack(
            [*voxel_components, np.ones_like(row_grid, dtype=float)], axis=0
        ).reshape(len(dim_order) + 1, -1)
        world = (affine @ homogeneous).reshape((affine.shape[0], *row_grid.shape))[:-1]
        # No origin subtraction: `x`/`y` are the true world position dotted onto the
        # in-plane basis, not a position relative to whichever grid happened to be
        # passed. This keeps two properties that both matter: edges and centers of
        # the *same* call share one frame (needed since `add_contours` combines
        # `pcolormesh`'s edges-based geometry with centers-based interpolation for
        # the same slice -- a per-call, grid-relative origin used to zero each grid
        # at its own first element, introducing a systematic half-pixel offset
        # between the two); and this frame matches the fixed-global-basis
        # absolute, affine-derived world coordinates for the same underlying data,
        # since dotting an unshifted world position onto a unit basis vector is
        # exactly that vector's world-space component (only subtracting a nonzero
        # origin first would shift it).
        x = np.tensordot(e1, world, axes=(0, 0))
        y = np.tensordot(e2, world, axes=(0, 0))
        return x, y

    x_edges, y_edges = _project(row_edges, col_edges)
    x_centers, y_centers = _project(row_vals, col_vals)
    return x_edges, y_edges, x_centers, y_centers


@dataclass(frozen=True)
class SliceAxisGrid:
    """One world axis's regular discretization, shared across volumes on a plotter.

    See design/world-mode-resample-scoping.md, Design B. `slice_world_dim`'s own
    axis needs a shared spacing/origin/count across every volume drawn on the same
    plotter for `match_coordinates` to line up panels by physical position -- the
    two in-plane axes don't (see `compute_slice_axis_aligned_grid_geometry`), so
    this only ever describes the one sliced axis, never a full grid.

    Attributes
    ----------
    world_dim : str
        World coordinate name (`"z"`/`"y"`/`"x"`) this spec describes.
    spacing : float
        World distance between consecutive slices.
    origin : float
        World position of the first slice.
    count : int
        Number of slices.
    """

    world_dim: str
    spacing: float
    origin: float
    count: int


def compute_slice_axis_aligned_grid_geometry(
    data: xr.DataArray,
    slice_world_dim: str,
    *,
    slice_axis_grid: SliceAxisGrid | None = None,
    resample_in_plane: bool = False,
) -> tuple[
    npt.NDArray[np.float64],
    Mapping[Hashable, int],
    Mapping[Hashable, float],
    Mapping[Hashable, float],
    SliceAxisGrid,
]:
    """Compute a `resample_volume` target grid that aligns `slice_world_dim`.

    Unlike `compute_oblique_axis_aligned_grid_geometry` (which always aligns all 3
    world axes to the global frame), this forces only `slice_world_dim` to a
    fixed global direction and regular spacing by default -- required so
    `match_coordinates` can line up panels of the same physical slice across
    independently-resampled volumes (`slice_axis_grid`, shared across those
    calls). The other two output axes keep `data`'s own native directions and
    spacing: an orthonormal basis derived from `data`'s two remaining voxel
    columns via Gram-Schmidt, fixed against the slice axis's global direction.
    For axis-aligned `data` this reduces to the identity direction with `data`'s
    own native in-plane spacing, same as today's whole-grid alignment; the
    difference only shows for oblique data (in-plane rotation preserved instead
    of forced upright) or when two volumes sharing `slice_axis_grid` have
    different native in-plane resolutions (neither gets downsampled to match the
    other). `resample_in_plane=True` instead forces all 3 axes to the global frame
    (see its own parameter doc), trading that resolution/geometry preservation
    for entirely rectangular output cells.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array, oblique or axis-aligned.
    slice_world_dim : str
        World coordinate name (`"z"`/`"y"`/`"x"`) to align.
    slice_axis_grid : SliceAxisGrid, optional
        Shared spacing/origin/count for `slice_world_dim`, established by an
        earlier call on the same plotter. If not provided, derived from `data`'s
        own extent along `slice_world_dim` and returned for the caller to reuse
        on subsequent volumes.
    resample_in_plane : bool, default: False
        Whether to also force the two in-plane axes onto the global frame
        (identity direction), like the slice axis, instead of `data`'s own
        native in-plane directions. Entirely rectangular output cells either
        way, but oblique in-plane data now requires a real interpolation for
        those two axes too (their own native resolution is still used, via the
        same QR-based per-axis spacing `compute_oblique_axis_aligned_grid_geometry`
        derives, just now along global instead of native directions) --
        genuinely different geometry from the default (non-rectangular native
        cells, no interpolation beyond the slice axis). Useful to avoid
        alpha-blended pcolormesh's per-cell seam rendering artifact on oblique
        (non-rectangular) cells (see design/world-mode-resample-scoping.md,
        Design D), at the cost of that geometry fidelity.

    Returns
    -------
    output_direction : (3, 3) numpy.ndarray
        Unit world-space direction columns for output `k`/`j`/`i`, read by
        [`resample_volume`][confusius.registration.resample_volume]. `slice_world_dim`
        is always output `k`.
    output_sizes : dict[str, int]
        Number of voxels along each output axis, keyed by `k`/`j`/`i`.
    output_spacing : dict[str, float]
        World distance between output positions, keyed by `k`/`j`/`i`.
    output_origin : dict[str, float]
        World location of output position `(0, 0, 0)`, keyed by `z`/`y`/`x`.
    slice_axis_grid : SliceAxisGrid
        `slice_axis_grid` unchanged if provided, otherwise the one just
        established from `data` -- pass this to the next call on the same
        plotter so every volume's slice axis lines up.

    Raises
    ------
    ValueError
        If `data` does not carry voxel-to-world geometry, if its geometry is
        pose-dependent, or (`resample_in_plane=False` only) if the two remaining
        voxel columns are degenerate (collinear with the slice axis or each
        other) once orthogonalized.
    """
    affine = require_scalar_pose_affine(
        data, "Slice-axis-aligned resampling for display"
    )
    world_dims = get_voxel_to_world_coord_names(data)
    voxel_dims = get_voxel_to_world_spatial_dims(data)
    voxel_spacing = get_voxel_to_world_index_spacing(data)
    slice_row = world_dims.index(slice_world_dim)
    linear = affine[:3, :3]

    if resample_in_plane:
        # Same per-world-axis spacing heuristic as
        # compute_oblique_axis_aligned_grid_geometry's Notes (nilearn's
        # reorder_img) -- but keyed by row here (not `world_dims` order) since
        # the slice row needs picking out first.
        row_to_axis, qr_spacing = qr_axis_spacing(linear)
        output_direction = np.eye(3)
        in_plane_rows = [row for row in range(3) if row != slice_row]
        in_plane_spacing = [
            float(qr_spacing[row_to_axis[row]]) for row in in_plane_rows
        ]
        slice_axis_spacing_default = float(qr_spacing[row_to_axis[slice_row]])
    else:
        dominant_idx = int(np.argmax(np.abs(linear[slice_row])))
        in_plane_indices = [i for i in range(3) if i != dominant_idx]

        e_slice = np.zeros(3)
        e_slice[slice_row] = 1.0

        basis = [e_slice]
        in_plane_spacing = []
        for idx in in_plane_indices:
            vector = linear[:, idx].astype(np.float64)
            for existing in basis:
                vector = vector - (vector @ existing) * existing
            norm = np.linalg.norm(vector)
            if np.isclose(norm, 0.0):
                raise ValueError(
                    f"Cannot align {slice_world_dim!r} for display: voxel "
                    f"dimension {voxel_dims[idx]!r} has no component left once "
                    "orthogonalized against the other in-plane axis -- the "
                    "geometry is degenerate."
                )
            basis.append(vector / norm)
            dim_spacing = voxel_spacing[voxel_dims[idx]]
            if dim_spacing is None:
                raise ValueError(
                    f"Cannot align {slice_world_dim!r} for display: voxel "
                    f"dimension {voxel_dims[idx]!r} has no well-defined spacing "
                    "(irregularly spaced coordinates)."
                )
            in_plane_spacing.append(dim_spacing)

        output_direction = np.stack(basis, axis=1)  # columns = output k, j, i
        slice_dim_spacing = voxel_spacing[voxel_dims[dominant_idx]]
        if slice_dim_spacing is None:
            raise ValueError(
                f"Cannot align {slice_world_dim!r} for display: voxel dimension "
                f"{voxel_dims[dominant_idx]!r} has no well-defined spacing "
                "(irregularly spaced coordinates)."
            )
        slice_axis_spacing_default = slice_dim_spacing

    # World positions of every sampled voxel, projected onto the new basis, to
    # find each output axis's extent -- cheap array evaluation (the lazy index's
    # own affine multiply), no interpolation.
    world_points = np.stack(
        [
            np.asarray(data.coords[dim].values, dtype=np.float64).ravel()
            for dim in world_dims
        ],
        axis=0,
    )
    projected = output_direction.T @ world_points

    if slice_axis_grid is not None:
        slice_spacing = slice_axis_grid.spacing
        slice_origin = slice_axis_grid.origin
        slice_count = slice_axis_grid.count
    else:
        slice_spacing = slice_axis_spacing_default
        slice_origin = float(projected[0].min())
        n_steps = (float(projected[0].max()) - slice_origin) / slice_spacing
        slice_count = np.int64(np.ceil(n_steps - 1e-6 * max(1.0, n_steps))).item() + 1
        slice_axis_grid = SliceAxisGrid(
            world_dim=slice_world_dim,
            spacing=slice_spacing,
            origin=slice_origin,
            count=slice_count,
        )

    origins = [slice_origin]
    sizes = [slice_count]
    for row, dim_spacing in zip((1, 2), in_plane_spacing, strict=True):
        row_min = float(projected[row].min())
        row_max = float(projected[row].max())
        n_steps = (row_max - row_min) / dim_spacing
        origins.append(row_min)
        sizes.append(np.int64(np.ceil(n_steps - 1e-6 * max(1.0, n_steps))).item() + 1)

    spacings = [slice_spacing, *in_plane_spacing]
    output_origin_world = output_direction @ np.array(origins)

    return (
        output_direction,
        dict(zip(VOXEL_DIMS, sizes, strict=True)),
        dict(zip(VOXEL_DIMS, spacings, strict=True)),
        dict(zip(SPATIAL_DIMS, output_origin_world, strict=True)),
        slice_axis_grid,
    )


def _resample_slice_axis_aligned_world_grid(
    data: xr.DataArray,
    slice_world_dim: str,
    *,
    slice_axis_grid: SliceAxisGrid | None = None,
    resample_in_plane: bool = False,
    interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    fill_value: float | None = None,
) -> tuple[xr.DataArray, SliceAxisGrid | None]:
    """Resample `data` so `slice_world_dim` is axis-aligned, for display.

    See `compute_slice_axis_aligned_grid_geometry` for the geometry this builds
    on `resample_volume` with. Unlike a full-grid resample, the two in-plane
    axes keep `data`'s own native spacing and orientation by default (see
    `SliceAxisGrid`) -- only `slice_world_dim` itself is forced onto a shared,
    regular discretization so panels from different volumes/masks line up by
    physical position. `resample_in_plane=True` also forces the in-plane axes
    onto the global frame (see `compute_slice_axis_aligned_grid_geometry`'s own
    parameter doc).

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array, oblique or axis-aligned.
    slice_world_dim : str
        World coordinate name (`"z"`/`"y"`/`"x"`) to align.
    slice_axis_grid : SliceAxisGrid, optional
        Shared spacing/origin/count for `slice_world_dim`, established by an
        earlier call on the same plotter. If not provided, derived from `data`'s
        own extent and returned for the caller to reuse on subsequent volumes.
    resample_in_plane : bool, default: False
        Whether to also force the two in-plane axes onto the global frame
        (`compute_slice_axis_aligned_grid_geometry`'s `resample_in_plane`) instead
        of `data`'s own native in-plane directions -- entirely rectangular
        output cells, at the cost of geometry/resolution fidelity for oblique
        in-plane data (see design/world-mode-resample-scoping.md, Design D).
    interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used during resampling. Use `"nearest"` for
        label/mask data, where blending distinct integer labels together is
        never meaningful.
    fill_value : float, optional
        Value assigned to voxels outside `data`'s field of view after
        resampling. If not provided, defaults to `float(data.min())` (see
        [`resample_volume`][confusius.registration.resample_volume]).

    Returns
    -------
    result : xarray.DataArray
        `data` resampled so `slice_world_dim` (and, if `resample_in_plane`, the two
        in-plane axes too) is axis-aligned and (if `slice_axis_grid` was
        provided) matches its shared spacing/origin/count. `data` unchanged if
        it's already fully axis-aligned (nothing to fix, and matplotlib only
        needs matching world coordinates to overlay correctly, not a matching
        pixel grid).
    slice_axis_grid : SliceAxisGrid
        `slice_axis_grid` unchanged if provided, or the one just established
        from `data`'s own geometry otherwise -- even when `data` was already
        axis-aligned and skipped resampling, so a later oblique volume/mask on
        the same plotter still has a grid to align its own slice axis to.
    """
    if has_axis_aligned_voxel_to_world_index(data):
        if slice_axis_grid is not None:
            return data, slice_axis_grid
        _, _, _, _, established_grid = compute_slice_axis_aligned_grid_geometry(
            data, slice_world_dim
        )
        return data, established_grid

    from confusius.registration import resample_volume

    direction, sizes, spacing, origin, established_grid = (
        compute_slice_axis_aligned_grid_geometry(
            data,
            slice_world_dim,
            slice_axis_grid=slice_axis_grid,
            resample_in_plane=resample_in_plane,
        )
    )
    result = resample_volume(
        data,
        np.eye(4, dtype=np.float64),
        output_sizes=sizes,
        output_spacing=spacing,
        output_origin=origin,
        output_direction=direction,
        interpolation=interpolation,
        fill_value=fill_value,
    )
    return result, established_grid


def _slice_edges_and_centers(
    slice_da: xr.DataArray,
    dim_row: str,
    dim_col: str,
    *,
    world_row: str | None = None,
    world_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return plotting geometry for a 2D slice.

    For standard axis-aligned data, returns 1D edge/center coordinates for the plotted
    row/column dimensions. For voxel-to-world data, returns 2D corner and center meshes
    obtained by projecting the world slice into an orthonormal in-plane basis --
    `world_row`/`world_col` forward to `_project_voxel_to_world_plane`'s
    fixed-global-basis mode (see its docstring); if not provided, the basis is
    derived from `slice_da`'s own affine instead.
    """
    if has_voxel_to_world_index(slice_da):
        return _project_voxel_to_world_plane(
            slice_da, dim_row, dim_col, world_row=world_row, world_col=world_col
        )

    if dim_col in slice_da.coords:
        x_centers = slice_da.coords[dim_col].values.astype(float)
        x_edges = _centers_to_edges(x_centers)
    else:
        x_centers = np.arange(slice_da.sizes[dim_col], dtype=float)
        x_edges = np.arange(slice_da.sizes[dim_col] + 1, dtype=float)

    if dim_row in slice_da.coords:
        y_centers = slice_da.coords[dim_row].values.astype(float)
        y_edges = _centers_to_edges(y_centers)
    else:
        y_centers = np.arange(slice_da.sizes[dim_row], dtype=float)
        y_edges = np.arange(slice_da.sizes[dim_row] + 1, dtype=float)

    return x_edges, y_edges, x_centers, y_centers


def _resolve_norm(
    slices: list,
    norm: "Normalize | None",
    data_attrs_norm: "Normalize | None",
    vmin: float | None,
    vmax: float | None,
) -> "Normalize":
    """Determine the colormap normalization.

    Precedence:
    - If `norm` is passed explicitly, it wins and vmin/vmax are ignored.
    - Otherwise, vmin/vmax (if given) override whatever is in `data_attrs`.
    - Otherwise, fall back to the norm stored in `data_attrs["norm"]`, or
      compute percentile-based limits from the data.
    """
    user_set_norm = norm is not None
    data_has_norm = data_attrs_norm is not None
    user_set_vmin = vmin is not None
    user_set_vmax = vmax is not None

    resolved_norm = norm if user_set_norm else data_attrs_norm

    if not user_set_norm:
        if data_has_norm:
            assert resolved_norm is not None
            default_vmin = resolved_norm.vmin
            default_vmax = resolved_norm.vmax
        else:
            all_vals = np.concatenate([s.values.ravel().astype(float) for s in slices])
            all_vals = all_vals[np.isfinite(all_vals)]
            default_vmin = (
                float(np.percentile(all_vals, 2)) if len(all_vals) > 0 else 0.0
            )
            default_vmax = (
                float(np.percentile(all_vals, 98)) if len(all_vals) > 0 else 1.0
            )

        vmin = vmin if user_set_vmin else default_vmin
        vmax = vmax if user_set_vmax else default_vmax

        if (not data_has_norm) or user_set_vmin or user_set_vmax:
            from matplotlib.colors import Normalize

            resolved_norm = Normalize(vmin=vmin, vmax=vmax)

    assert resolved_norm is not None

    return resolved_norm


_STAT_MAP_DIVERGING_CMAP = "coolwarm"
"""Default colormap for diverging statistical maps (see `plot_stat_map`)."""

_STAT_MAP_SEQUENTIAL_CMAP = "viridis"
"""Default colormap for non-negative statistical maps (see `plot_stat_map`)."""

_STAT_MAP_SEQUENTIAL_CMAP_NEGATIVE = "viridis_r"
"""Default colormap for non-positive statistical maps (see `plot_stat_map`).

Reversed relative to `_STAT_MAP_SEQUENTIAL_CMAP` so that, in both the
non-negative and non-positive cases, values near zero map to the same end of
the colormap (dark purple) and the most extreme magnitude maps to the other
end (yellow).
"""


def _resolve_stat_map_style(
    data: xr.DataArray,
    vmin: float | None,
    vmax: float | None,
    cmap: "str | Colormap | None",
    auto_range: bool,
) -> tuple[float, float, "str | Colormap"]:
    """Resolve `(vmin, vmax, cmap)` for a statistical map.

    `vmin`/`vmax` fall back to the actual min/max of `data` when not provided.

    When `auto_range` is `True` (default), the sign of `data` determines the
    layout:

    - Both positive and negative values: diverging, symmetric `[-m, m]` range
      where `m = max(|vmin|, |vmax|)` (using the resolved bounds above), with
      `cmap` defaulting to `_STAT_MAP_DIVERGING_CMAP`.
    - Only non-negative values: sequential `[0, vmax]` range, with `cmap`
      defaulting to `_STAT_MAP_SEQUENTIAL_CMAP`.
    - Only non-positive values: sequential `[vmin, 0]` range, with `cmap`
      defaulting to `_STAT_MAP_SEQUENTIAL_CMAP_NEGATIVE`.

    When `auto_range` is `False`, the resolved `vmin`/`vmax` are used directly with
    no zero-anchoring, and `cmap` defaults to `_STAT_MAP_DIVERGING_CMAP` regardless
    of `data`'s sign. In both cases, an explicitly provided `cmap` is always used
    as-is.
    """
    values = data.values.ravel().astype(float)
    values = values[np.isfinite(values)]
    data_min = float(values.min()) if len(values) > 0 else 0.0
    data_max = float(values.max()) if len(values) > 0 else 1.0

    resolved_vmin = vmin if vmin is not None else data_min
    resolved_vmax = vmax if vmax is not None else data_max

    if not auto_range:
        return (
            resolved_vmin,
            resolved_vmax,
            cmap if cmap is not None else _STAT_MAP_DIVERGING_CMAP,
        )

    if data_min < 0 < data_max:
        abs_max = max(abs(resolved_vmin), abs(resolved_vmax))
        return -abs_max, abs_max, cmap if cmap is not None else _STAT_MAP_DIVERGING_CMAP

    if data_max > 0:
        return (
            0.0,
            resolved_vmax,
            cmap if cmap is not None else _STAT_MAP_SEQUENTIAL_CMAP,
        )

    return (
        resolved_vmin,
        0.0,
        cmap if cmap is not None else _STAT_MAP_SEQUENTIAL_CMAP_NEGATIVE,
    )


def _threshold_slices(
    slices: list[xr.DataArray],
    threshold: float | None,
    threshold_mode: Literal["lower", "upper"],
) -> list[xr.DataArray | np.ndarray]:
    """Apply thresholding to a list of slices, returning masked arrays."""
    if threshold is None:
        return [s.values for s in slices]

    thresholded = []
    for s in slices:
        if threshold_mode == "lower":
            mask = np.abs(s) >= threshold
        else:
            mask = np.abs(s) <= threshold
        thresholded.append(s.where(mask))
    return thresholded


def _resolve_cmap(
    cmap: "str | Colormap | None",
    data_attrs_cmap: "str | Colormap | None",
    norm: "Normalize",
    threshold: float | None,
    threshold_mode: Literal["lower", "upper"],
) -> "Colormap":
    """Build colormap with gray band indicating thresholded regions.

    For `threshold_mode='lower'`: gray between `[-threshold, threshold]`.
    For `threshold_mode='upper'`: gray outside `[-threshold, threshold]`.

    Raises
    ------
    ValueError
        If `norm.vmin` or `norm.vmax` is not finite.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if (
        norm.vmin is None
        or norm.vmax is None
        or not math.isfinite(norm.vmin)
        or not math.isfinite(norm.vmax)
    ):
        raise ValueError(
            "norm.vmin and norm.vmax must be finite, got "
            f"vmin={norm.vmin!r}, vmax={norm.vmax!r}."
        )

    cmap = (
        cmap
        if cmap is not None
        else data_attrs_cmap
        if data_attrs_cmap is not None
        else "gray"
    )
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    cmap_colors = [(i / (cmap.N - 1), cmap(i / (cmap.N - 1))) for i in range(cmap.N)]

    threshold = 0.0 if threshold is None else abs(threshold)
    gray_low = norm(-threshold)
    gray_high = norm(threshold)
    if threshold_mode == "lower":
        colors_before = [c for c in cmap_colors if c[0] <= gray_low]
        colors_after = [c for c in cmap_colors if c[0] >= gray_high]

        if gray_low < gray_high:
            # We clip the gray low/high points to [0, 1] because Matplotlib expects
            # colormap values in that range.
            gray_band = [
                (max(0, gray_low), "gray"),
                (min(1.0, gray_high), "gray"),
            ]
        else:
            gray_band = []
        new_colors = colors_before + gray_band + colors_after
    else:
        colors_middle = [c for c in cmap_colors if gray_low <= c[0] <= gray_high]

        if gray_low > 0.0:
            gray_band_low = [(0.0, "gray"), (gray_low, "gray")]
        else:
            gray_band_low = []

        if gray_high < 1.0:
            gray_band_high = [(gray_high, "gray"), (1.0, "gray")]
        else:
            gray_band_high = []
        new_colors = gray_band_low + colors_middle + gray_band_high

    # Matplotlib 3.11+ requires (value, color) pairs passed to `from_list` to be
    # strictly monotonically increasing in value. Our boundary points can collide
    # with neighboring cmap entries (e.g. when `gray_low == 0` the first
    # `colors_before` entry shares its value with the start of `gray_band`).
    # Collapse duplicates by value, keeping the later entry so gray-band
    # boundaries take precedence over the underlying cmap at the same value.
    deduped: dict[float, str | tuple[float, ...] | list[float]] = {}
    for value, color in new_colors:
        deduped[value] = color
    new_colors = list(deduped.items())

    # Preserve the source colormap's resolution. The default N=256 of
    # `LinearSegmentedColormap.from_list` collapses larger discrete cmaps such as
    # the atlas `ListedColormap` (N == number of regions, often >256), aliasing
    # high indices to wrong (or out-of-range) colours. Propagate under/over/bad
    # colours so the atlas's transparent under-colour for label 0 (background)
    # survives the rebuild. Cast to tuple because the getters return numpy arrays
    # but the kwargs expect a color-like. `from_list` accepts these kwargs since
    # matplotlib 3.11, no need for a separate `with_extremes` round-trip.
    return mcolors.LinearSegmentedColormap.from_list(
        f"{cmap.name}_thresholded",
        new_colors,
        N=cmap.N,
        under=tuple(cmap.get_under()),
        over=tuple(cmap.get_over()),
        bad=tuple(cmap.get_bad()),
    )


def _build_axis_label(
    da: xr.DataArray, dim: str, *, world_dim: str | None = None
) -> str:
    """Return axis label for `dim`, including units when available.

    Parameters
    ----------
    da : xarray.DataArray
        The panel being labeled.
    dim : str
        The array dimension displayed on this axis.
    world_dim : str, optional
        World coordinate name (`"z"`/`"y"`/`"x"`) this axis is actually projected
        onto, when display uses the fixed-global-basis projection (see
        `_project_voxel_to_world_plane`'s `world_row`/`world_col`) instead of
        `dim`'s own per-volume-local in-plane direction -- labeled with this name
        directly rather than `"{dim} in-plane (mm)"`, since the axis genuinely is
        that world coordinate now, not merely something derived from `dim`.
    """
    if world_dim is not None:
        label = world_dim
        if world_dim in da.coords:
            units = da.coords[world_dim].attrs.get("units")
            if units:
                label = f"{world_dim} ({units})"
        return label

    # `isel`ing away a dim the custom VoxelToWorldIndex is attached to can drop the
    # index outright (an xarray custom-index limitation, most visible for
    # pose-stacked oblique geometry) even though the geometry is fully recoverable
    # -- restore it first so plottability reflects the panel's true geometry, not
    # an isel artifact. Not always possible: an already-materialized world-space
    # panel (z/y/x dims, k/j/i intentionally dropped by
    # `_materialize_axis_aligned_world_grid_for_display`) genuinely has no voxel
    # dim left to restore, so fall back to `da` as-is in that case.
    try:
        da = ensure_voxeldata(da)
    except (TypeError, ValueError):
        pass
    if dim != POSE_DIM and has_voxel_to_world_index(da):
        return f"{dim} in-plane (mm)"

    label = dim
    if dim in da.coords:
        units = da.coords[dim].attrs.get("units")
        if units:
            label = f"{dim} ({units})"
    return label


def _format_coord(coord: Hashable) -> str:
    """Format a slice coordinate for display: `.3g` for numeric, `str` otherwise."""
    if isinstance(coord, numbers.Real):
        return f"{coord:.3g}"
    return str(coord)


def _coords_match(
    stored_coord: Hashable, target_coord: Hashable, tolerance: float
) -> bool:
    """Compare two slice coordinates, using a tolerance for numeric values."""
    if isinstance(stored_coord, numbers.Real) and isinstance(
        target_coord, numbers.Real
    ):
        return abs(float(stored_coord) - float(target_coord)) < tolerance
    return stored_coord == target_coord


def _default_slice_coords(data: xr.DataArray, slice_mode: str) -> list[Hashable]:
    """Return every coordinate value along `slice_mode`, for the default (all-panels) case.

    Parameters
    ----------
    data : xarray.DataArray
        Data already passed through `_prepare_slice_inputs`.
    slice_mode : str
        The dimension/coordinate to enumerate values for.

    Returns
    -------
    list of collections.abc.Hashable
        One entry per distinct value along `slice_mode`. Positional indices
        (`range(data.sizes[slice_mode])`) when `slice_mode` has no coordinate at
        all.

    Notes
    -----
    A spatial `slice_mode` (`"z"`/`"y"`/`"x"`) whose data is still on native
    `k`/`j`/`i` dims (Design B/C's oblique-in-plane world display -- see
    design/world-mode-resample-scoping.md) has `slice_mode`'s own coordinate
    still derived jointly over all 3 voxel dims, even though it only genuinely
    varies along `"k"` (the slice axis, always output `k` by construction of
    `compute_slice_axis_aligned_grid_geometry`). Naively listing
    `coord.values` there would yield one 2D `(j, i)`-shaped sub-array per `k`
    position instead of one scalar -- reduce the other two voxel dims first.
    """
    if slice_mode not in data.coords:
        return list(range(data.sizes[slice_mode]))
    coord = data.coords[slice_mode]
    if coord.ndim <= 1:
        return list(coord.values)
    if slice_mode in coord.dims:
        indexers = {d: 0 for d in coord.dims if d != slice_mode}
        return list(coord.isel(indexers).values)
    indexers = {d: 0 for d in coord.dims if d != "k"}
    return list(coord.isel(indexers).values)


def _extract_slices(
    data: xr.DataArray, slice_mode: str, slice_coords: Sequence[Hashable]
) -> tuple[list[xr.DataArray], list[Hashable]]:
    """Extract 2D slices from `data` along `slice_mode`.

    Numeric coordinates are matched by nearest-neighbour lookup; non-numeric
    coordinates (e.g. region labels) require an exact match.

    Returns the slices and their actual snapped coordinate values.
    """
    slices: list[xr.DataArray] = []
    actual_coords: list[Hashable] = []
    for coord in slice_coords:
        if slice_mode in data.coords:
            is_numeric = np.issubdtype(data.coords[slice_mode].dtype, np.number)
            slice_da = data.sel(
                {slice_mode: coord}, method="nearest" if is_numeric else None
            )
            # `slice_mode`'s coordinate is usually scalar here (the dim it lived on
            # was just selected away), but for Design B's oblique-in-plane world
            # slicing (see VoxelToWorldIndex.sel's per-axis fast path) it stays
            # derived over the remaining in-plane dims -- constant-valued (that's
            # exactly what made the single-axis selection possible), just not
            # collapsed to a true 0D scalar, so `.item()` alone would raise.
            actual_coord = (
                np.asarray(slice_da.coords[slice_mode].values).reshape(-1)[0].item()
            )
        else:
            if not isinstance(coord, numbers.Real):
                raise ValueError(
                    f"slice_mode '{slice_mode}' has no coordinates, so slice_coords "
                    f"must be numeric positional indices, got {coord!r}."
                )
            idx = round(coord)
            slice_da = data.isel({slice_mode: idx})
            actual_coord = float(coord)
        slices.append(slice_da)
        actual_coords.append(actual_coord)
    return slices, actual_coords


class VolumePlotter:
    """Manager for volume slice plots with coordinate-based overlay support.

    This class maintains the state of a figure with multiple axes, each representing a
    slice through a volume at a specific coordinate. It enables overlaying multiple
    volumes on the same axes by matching coordinates.

    Parameters
    ----------
    slice_mode : str
        The dimension along which slices are taken (e.g., `"z"`).
    figure : matplotlib.figure.Figure, optional
        The figure containing the axes. If not provided, a new figure will be created
        on the first call to
        [`add_volume`][confusius.plotting.VolumePlotter.add_volume].
    axes : numpy.ndarray[matplotlib.axes.Axes] or matplotlib.axes.Axes, optional
        Existing axes to draw into: either a single
        [`matplotlib.axes.Axes`][matplotlib.axes.Axes] or an array of them. If not
        provided, axes will be created on the first call to
        [`add_volume`][confusius.plotting.VolumePlotter.add_volume].
    bg_color : str, default: "black"
        Background color for the figure and axes. Any matplotlib-compatible color
        string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula
        (white on dark backgrounds, black on light ones).
    yincrease : bool, default: False
        Whether the y-axis increases upward. When `False`, y coordinates decrease
        upward.
    xincrease : bool, default: True
        Whether the x-axis increases to the right.
    resample_interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used whenever a spatial `slice_mode` (`"z"`/`"y"`/`"x"`)
        resamples oblique voxel-to-world data to align its slice axis for display.
        A non-spatial `slice_mode`'s in-plane display is a fixed-global-basis
        *projection* instead, with no interpolation to configure (see
        `resample_in_plane` to opt into one). Applies to
        [`add_volume`][confusius.plotting.VolumePlotter.add_volume]; mask data passed
        to [`add_contours`][confusius.plotting.VolumePlotter.add_contours] always
        resamples with `"nearest"` regardless of this setting, since blending
        distinct integer labels together is never meaningful.
    resample_fill_value : float, optional
        Value assigned to voxels outside the source data's field of view after
        resampling oblique data. If not provided, defaults to the source data's own
        minimum value (see
        [`resample_volume`][confusius.registration.resample_volume]).
    transpose : bool, default: False
        Whether to swap the row/column display dims of each slice panel.

    Attributes
    ----------
    slice_mode : str
        The dimension along which slices are taken.
    figure : matplotlib.figure.Figure or None
        The figure. `None` until the first call to
        [`add_volume`][confusius.plotting.VolumePlotter.add_volume] when no figure
        is provided at construction time.
    axes : numpy.ndarray or None
        Array of [`matplotlib.axes.Axes`][matplotlib.axes.Axes]. `None` until the
        first call to [`add_volume`][confusius.plotting.VolumePlotter.add_volume]
        when no axes are provided at construction time.
    """

    axes: "npt.NDArray[Any] | None"

    def __init__(
        self,
        slice_mode: str = "z",
        figure: "Figure | None" = None,
        axes: "npt.NDArray[Any] | Axes | None" = None,
        *,
        bg_color: str = "black",
        fg_color: str | None = None,
        yincrease: bool = False,
        xincrease: bool = True,
        resample_interpolation: Literal["linear", "nearest", "bspline"] = "linear",
        resample_fill_value: float | None = None,
        transpose: bool = False,
    ):
        self.slice_mode = slice_mode
        self._transpose = transpose
        if axes is not None and not isinstance(axes, np.ndarray):
            axes = np.asarray([[axes]])
        self.axes = axes
        self._user_provided_axes = axes is not None
        if figure is None and axes is not None:
            from matplotlib.axes import Axes

            first_axis = axes.flat[0]
            if not isinstance(first_axis, Axes):
                raise TypeError("axes must contain matplotlib.axes.Axes instances.")
            self.figure = first_axis.get_figure(root=True)
        else:
            self.figure = figure
        self._bg_color = bg_color
        self._fg_color = fg_color
        self._yincrease = yincrease
        self._xincrease = xincrease
        self._coord_to_axis: dict[Hashable, int] = {}
        # Explicitly tracked axis data limits to avoid matplotlib's auto-margin.
        self._axis_xlims: dict[int, tuple[float, float]] = {}
        self._axis_ylims: dict[int, tuple[float, float]] = {}

        self._resample_interpolation = resample_interpolation
        self._resample_fill_value = resample_fill_value
        self._hover_manager = _HoverManager()
        self._slice_axis_grid: SliceAxisGrid | None = None

    def _maybe_resample_slices_to_world(
        self,
        slices: list[xr.DataArray],
        *,
        interpolation: Literal["linear", "nearest", "bspline"] | None = None,
        fill_value: float | None = None,
    ) -> list[xr.DataArray]:
        """Regularize each panel for world-space display.

        No-op when `slice_mode` is spatial (its own display geometry is resolved
        upfront in `_prepare_slice_inputs`/`add_contours`). `match_coordinates` for
        a non-spatial `slice_mode` (the only case this function ever runs for)
        matches panels by the facet's own coordinate value (e.g. a region label), never
        by grid position, so no axis needs sharing across volumes here. Contrast with
        `_prepare_slice_inputs`'s whole-array resample for spatial `slice_mode`, where
        the slice axis's discretization genuinely must be shared for `match_coordinates`
        to work.

        For already axis-aligned data, this still resamples (a no-op resample:
        `_shared_resample_to_axis_aligned_world_grid` short-circuits) purely to get
        the `_materialize_axis_aligned_world_grid_for_display` rename onto proper
        1D `z`/`y`/`x` dims. For genuinely oblique data, no resample happens at all
        -- `_extract_display_slices` instead projects the panel's own native quads
        directly onto a fixed global world basis (see
        `compute_slice_axis_aligned_grid_geometry`'s design note in
        `plotting/_utils.py` and design/world-mode-resample-scoping.md, Design C),
        so the returned panel here just keeps its native `k`/`j`/`i` geometry
        unchanged, at full native resolution and with no interpolation. Only the
        cheap collapse-to-a-2D-plane check (`compute_oblique_axis_aligned_grid_geometry`,
        bounds/spacing arithmetic, no interpolation) still runs, raising `ValueError`
        up front if the panel's geometry is oblique to the world axes and would not
        lie flat on any world plane -- global-basis projection has no meaningful
        display for a genuinely 3D-extended (non-planar) panel either.

        Parameters
        ----------
        slices : list of xarray.DataArray
            Per-panel 2D (or 2D-plus-time) slices already `isel`'d along `slice_mode`.
        interpolation : {"linear", "nearest", "bspline"}, optional
            Interpolation method for the (axis-aligned-only) resample. If not
            provided, defaults to `self._resample_interpolation`.
        fill_value : float, optional
            Value assigned to voxels outside the source panel's field of view for
            the (axis-aligned-only) resample. If not provided, defaults to
            `self._resample_fill_value`.

        Returns
        -------
        list of xarray.DataArray
            The input `slices`: renamed onto 1D `z`/`y`/`x` dims for already
            axis-aligned data, unchanged (native `k`/`j`/`i`, no interpolation) for
            oblique data, unchanged for anything else (no voxel-to-world geometry,
            ...).
        """
        if self.slice_mode in SPATIAL_DIMS:
            return slices
        interp = (
            self._resample_interpolation if interpolation is None else interpolation
        )
        fill = self._resample_fill_value if fill_value is None else fill_value
        resampled = []
        for slice_da in slices:
            if has_voxel_to_world_index(
                slice_da
            ) and not has_axis_aligned_voxel_to_world_index(slice_da):
                world_dims = get_voxel_to_world_coord_names(slice_da)
                shape, _, _ = compute_oblique_axis_aligned_grid_geometry(
                    slice_da, world_dims
                )
                if sum(size > 1 for size in shape) != 2:
                    raise ValueError(
                        f"Displaying slice_mode={self.slice_mode!r}'s panel in world "
                        "space would not collapse to a 2D plane (predicted shape "
                        f"{dict(zip(world_dims, shape, strict=True))}). This happens "
                        "when the panel's spatial geometry is oblique to the world "
                        "axes and does not lie flat on any world plane."
                    )
                resampled.append(slice_da)
                continue
            grid = _shared_resample_to_axis_aligned_world_grid(
                slice_da,
                reference=None,
                interpolation=interp,
                fill_value=fill,
            )
            grid = _materialize_axis_aligned_world_grid_for_display(grid)
            world_dims = [d for d in grid.dims if str(d) in {"z", "y", "x"}]
            squeeze_dims = [d for d in world_dims if grid.sizes[d] == 1]
            if squeeze_dims:
                grid = grid.squeeze(dim=squeeze_dims)
            assert grid.ndim == 2, (
                "predicted shape should have already rejected a non-collapsing "
                f"panel; got shape {grid.shape} with dims {list(grid.dims)}"
            )
            resampled.append(grid)
        return resampled

    def _resolve_world_display_basis(
        self, data: xr.DataArray, slices: list[xr.DataArray]
    ) -> tuple[str | None, str | None]:
        """Resolve `(world_row, world_col)` for fixed-global-basis world display.

        `None, None` whenever the panel doesn't need (or can't use) a global basis:
        it carries no oblique voxel-to-world geometry left to project
        (axis-aligned data already gets proper 1D `z`/`y`/`x` dims from
        `_maybe_resample_slices_to_world`'s materialize step, so the
        plain-coordinate path in `_slice_edges_and_centers` already
        labels/projects it correctly without this).

        See design/world-mode-resample-scoping.md, Design C: the *displayed* world
        axes are always the fixed global unit vectors for the two world dims other
        than the "out-of-plane" one -- `self.slice_mode` itself when spatial (the
        only axis being sliced across positions), or the affine's own
        near-collapsed axis when non-spatial (this panel is already reduced to a
        single fixed plane, so the collapsed axis needs cheap detection via
        `compute_oblique_axis_aligned_grid_geometry`'s shape prediction -- no
        interpolation, same check `_maybe_resample_slices_to_world` already runs).

        Parameters
        ----------
        data : xarray.DataArray
            Data already passed through `_prepare_slice_inputs`, before per-panel
            extraction (only used for the non-spatial case, since the collapsed
            axis is facet-independent -- same affine regardless of which panel).
        slices : list of xarray.DataArray
            Panels already extracted (and, for a non-spatial `slice_mode`, already
            passed through `_maybe_resample_slices_to_world`).

        Returns
        -------
        world_row : str or None
            World coordinate name for the display row axis, or `None`.
        world_col : str or None
            World coordinate name for the display column axis, or `None`.
        """
        if self.slice_mode in SPATIAL_DIMS:
            if not slices or not has_voxel_to_world_index(slices[0]):
                return None, None
            world_dims = get_voxel_to_world_coord_names(slices[0])
            remaining = [d for d in world_dims if d != self.slice_mode]
            return remaining[0], remaining[1]
        if not has_voxel_to_world_index(data) or has_axis_aligned_voxel_to_world_index(
            data
        ):
            return None, None
        world_dims = get_voxel_to_world_coord_names(data)
        shape, _, _ = compute_oblique_axis_aligned_grid_geometry(data, world_dims)
        collapsed = [d for d, size in zip(world_dims, shape, strict=True) if size == 1]
        if len(collapsed) != 1:
            # `_maybe_resample_slices_to_world` already raised a clearer error for
            # this on the actual per-panel data; reaching here regardless (e.g. no
            # slice_coords, so that loop never ran) just means no global basis.
            return None, None
        remaining = [d for d in world_dims if d != collapsed[0]]
        return remaining[0], remaining[1]

    def _extract_display_slices(
        self,
        data: xr.DataArray,
        slice_coords: Sequence[Hashable],
        *,
        interpolation: Literal["linear", "nearest", "bspline"] | None = None,
        fill_value: float | None = None,
    ) -> tuple[list[xr.DataArray], list[Hashable], str, str, str | None, str | None]:
        """Extract per-panel slices, resample to world space if requested, and resolve
        the row/column display dims (applying `transpose`).

        Parameters
        ----------
        data : xarray.DataArray
            Data already passed through `_prepare_slice_inputs`.
        slice_coords : collections.abc.Sequence of collections.abc.Hashable
            Coordinate values along `slice_mode` at which to extract slices.
        interpolation : {"linear", "nearest", "bspline"}, optional
            Interpolation method for the resample. If not provided, defaults to
            `self._resample_interpolation`.
        fill_value : float, optional
            Value assigned to voxels outside the source panel's field of view. If not
            provided, defaults to `self._resample_fill_value`.

        Returns
        -------
        slices : list of xarray.DataArray
            One 2D (or 2D-plus-unsliced-extra-dim) panel per requested slice coordinate.
        actual_coords : list of collections.abc.Hashable
            The actual coordinate value matched for each panel.
        dim_row : str
            Dim displayed on the row (y) axis.
        dim_col : str
            Dim displayed on the column (x) axis.
        world_row : str or None
            World coordinate name to project the row axis onto with a fixed global
            basis (see `_resolve_world_display_basis`), or `None` to use `dim_row`'s
            own per-volume-local basis (or a plain coordinate) instead.
        world_col : str or None
            World coordinate name for the column axis. See `world_row`.
        """
        slices, actual_coords = _extract_slices(data, self.slice_mode, slice_coords)
        slices = self._maybe_resample_slices_to_world(
            slices, interpolation=interpolation, fill_value=fill_value
        )
        world_row, world_col = self._resolve_world_display_basis(data, slices)
        display_dims = (
            [str(d) for d in slices[0].dims]
            if slices
            else [str(d) for d in data.dims if d != self.slice_mode]
        )
        dim_row, dim_col = display_dims[::-1] if self._transpose else display_dims
        if self._transpose:
            world_row, world_col = world_col, world_row
        # Reorder each panel's own array axes to (dim_row, dim_col): geometry
        # (`_slice_edges_and_centers`) only reads dim_row/dim_col by name, but the
        # pixel array handed to `pcolormesh` must match that row/col axis order.
        slices = [s.transpose(dim_row, dim_col) for s in slices]
        return slices, actual_coords, dim_row, dim_col, world_row, world_col

    def _ensure_figure(
        self,
        n_slices: int,
        nrows: int | None = None,
        ncols: int | None = None,
        dpi: int | None = None,
        x_range: float | None = None,
        y_range: float | None = None,
    ) -> None:
        """Create figure and/or axes if not already fully initialized."""
        import matplotlib.pyplot as plt

        if self.figure is not None and self.axes is not None:
            return

        _nrows, _ncols = _compute_grid_dims(n_slices, nrows, ncols)

        if self.figure is None:
            if x_range is not None and y_range is not None and y_range > 0:
                aspect = x_range / y_range
                subplot_width = _BASE_SIZE * max(1.0, aspect)
                subplot_height = _BASE_SIZE * max(1.0, 1.0 / aspect)
            else:
                subplot_width = subplot_height = _BASE_SIZE

            fig_width = max(8.0, min(20.0, _ncols * subplot_width + 1.0))
            fig_height = min(16.0, _nrows * subplot_height)

            self.figure, axes_array = plt.subplots(
                _nrows,
                _ncols,
                figsize=(fig_width, fig_height),
                dpi=dpi,
                squeeze=False,
                layout="constrained",
            )
        else:
            axes_array = self.figure.subplots(_nrows, _ncols, squeeze=False)

        self.axes = np.array(axes_array)
        self.figure.patch.set_facecolor(self._bg_color)

    def _attach_or_update_hover_manager(self, roi_labels: dict[int, str]) -> None:
        """Ensure hover manager is attached to figure and update its ROI labels.

        Parameters
        ----------
        roi_labels : dict[int, str]
            Mapping from integer label to display name during mouse hover.
        """

        if self.figure is not None:
            if not self._hover_manager.is_attached():
                self._hover_manager.attach_figure(self.figure)

            self._hover_manager.roi_labels.update(roi_labels)

    def _find_matching_axes(
        self, actual_coords: list[Hashable], tolerance: float = 1e-6
    ) -> list[tuple[int, int]]:
        """Find axis indices matching the target coordinates.

        Uses the coordinate-to-axis mapping stored when the figure was first created,
        avoiding any dependency on axis titles.

        Parameters
        ----------
        actual_coords : list[collections.abc.Hashable]
            The actual coordinate values of the slices being plotted.
        tolerance : float, default: 1e-6
            Tolerance for matching numeric coordinates, accounting for floating-point
            precision. Non-numeric coordinates (e.g. region labels) are matched by
            equality.

        Returns
        -------
        list[tuple[int, int]]
            List of `(axis_flat_idx, slice_idx)` tuples for matched coordinates.
        """
        matched = []
        for slice_idx, target_coord in enumerate(actual_coords):
            for stored_coord, axis_idx in self._coord_to_axis.items():
                if _coords_match(stored_coord, target_coord, tolerance):
                    matched.append((axis_idx, slice_idx))
                    break
        return matched

    @property
    def _text_color(self) -> str:
        """Foreground color: explicit fg_color or WCAG-derived contrast color."""
        if self._fg_color is not None:
            return self._fg_color
        return _auto_fg_color(self._bg_color)

    def _style_ax(self, ax: "Axes") -> None:
        """Apply background and spine/tick styling to an axes."""
        color = self._text_color
        ax.set_facecolor(self._bg_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
        ax.tick_params(colors=color, which="both")

    def _set_ax_lims(
        self,
        ax: "Axes",
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> None:
        """Set axis limits respecting the x/y increase direction."""
        ax.set_ylim(ylim if self._yincrease else (ylim[1], ylim[0]))
        ax.set_xlim(xlim if self._xincrease else (xlim[1], xlim[0]))

    def _update_stored_lim(
        self,
        store: dict[int, tuple[float, float]],
        axis_idx: int,
        new_lim: tuple[float, float],
    ) -> tuple[float, float]:
        """Expand the stored limit for `axis_idx` to encompass `new_lim`."""
        if axis_idx in store:
            prev = store[axis_idx]
            new_lim = (min(prev[0], new_lim[0]), max(prev[1], new_lim[1]))
        store[axis_idx] = new_lim
        return new_lim

    def _warn_unmatched(self, unmatched_slices: list[tuple[int, Hashable]]) -> None:
        """Warn about slice coordinates that could not be matched to axes."""
        unmatched_str = ", ".join(
            f"{self.slice_mode}={_format_coord(coord)}" for _, coord in unmatched_slices
        )
        available_coords = [_format_coord(c) for c in self._coord_to_axis]
        warnings.warn(
            f"Could not find matching axes for slices: {unmatched_str}. "
            f"These slices will not be plotted. "
            f"Available coordinates: {available_coords}",
            stacklevel=find_stack_level(),
        )

    def _build_slice_title(self, data: xr.DataArray, coord: Hashable) -> str:
        """Build a slice title such as `z = 0.001 mm` or `region = LGN`."""
        units = (
            data.coords[self.slice_mode].attrs.get("units")
            if self.slice_mode in data.coords
            else None
        )
        title = f"{self.slice_mode} = {_format_coord(coord)}"
        if units:
            title += f" {units}"
        return title

    def _init_sequential_layout(
        self, actual_coords: list[Hashable]
    ) -> list[tuple[int, int]]:
        """Initialise the coordinate-to-axis map and return sequential plot indices."""
        if not self._coord_to_axis:
            self._coord_to_axis = {
                coord: idx for idx, coord in enumerate(actual_coords)
            }
        return [(idx, idx) for idx in range(len(actual_coords))]

    def _prepare_slice_inputs(
        self,
        data: xr.DataArray,
        *,
        caller: str,
        interpolation: Literal["linear", "nearest", "bspline"] | None = None,
        resample_in_plane: bool = False,
    ) -> xr.DataArray:
        """Coerce complex, squeeze, validate `slice_mode`/3D, and sort display coords.

        Shared by `add_volume`/`add_composite`'s data and `add_contours`' mask --
        the two need identical geometry canonicalization (a mask can carry the same
        extra facet dim as `slice_mode`, e.g. `"region"`, and must land on the same
        native-vs-world dims as the data it overlays). `interpolation` lets a mask
        force `"nearest"` regardless of `self._resample_interpolation`: mask/label
        data is a set of distinct integer regions, and blending them together
        (linear/bspline) would fabricate boundary values that match no real label.
        `resample_in_plane` forces the two in-plane axes onto the global frame too,
        for a spatial `slice_mode` (see `add_volume`'s own parameter doc) -- a
        per-call choice, not shared/reused across other volumes on this plotter
        the way `self._slice_axis_grid` is.
        """
        data = ensure_voxeldata(data)
        data = coerce_complex_to_magnitude(data, caller=caller)
        # Data is computed here to avoid repeated computations of the same Dask graph
        # downstream (per-panel .isel, etc.).
        data = data.compute()

        _validate_slice_mode(data, self.slice_mode)

        resolved_interpolation = (
            self._resample_interpolation if interpolation is None else interpolation
        )

        if self.slice_mode in SPATIAL_DIMS:
            resampled_data, slice_axis_grid = _resample_slice_axis_aligned_world_grid(
                data,
                self.slice_mode,
                slice_axis_grid=self._slice_axis_grid,
                resample_in_plane=resample_in_plane,
                interpolation=resolved_interpolation,
                fill_value=self._resample_fill_value,
            )

            # Capture the slice axis's spec so a later volume/mask on this same plotter
            # lines up on the same physical slices. The two in-plane axes are never
            # shared (each volume keeps its own native resolution/orientation, see
            # SliceAxisGrid).
            if self._slice_axis_grid is None:
                self._slice_axis_grid = slice_axis_grid

            # A no-op unless resampled_data is fully axis-aligned (e.g. axis-aligned
            # input, or a slice axis coinciding with one of data's own native
            # axes): oblique in-plane geometry stays on k/j/i, its VoxelToWorldIndex
            # still resolving world coordinates directly for slicing/projection.
            data = _materialize_axis_aligned_world_grid_for_display(resampled_data)
        else:
            data = _materialize_axis_aligned_world_grid_for_display(data)

        # Which dim must survive squeezing even at size 1: normally self.slice_mode
        # itself, but Design B/C's oblique-in-plane spatial display (see above)
        # never renames k to self.slice_mode's world name -- the slice axis is
        # always output "k" there (compute_slice_axis_aligned_grid_geometry), and
        # it can genuinely be size 1 (e.g. a single native slice sharing another
        # volume's size-1 SliceAxisGrid). Without this, "k" != self.slice_mode
        # (e.g. "z") would wrongly mark it squeezable, silently dropping the slice
        # axis entirely instead of the intended size-1-along-slice_mode panel.
        protected_dim = (
            "k"
            if self.slice_mode not in data.dims and self.slice_mode in data.coords
            else self.slice_mode
        )
        squeeze_dims = [
            d for d in data.dims if d != protected_dim and data.sizes[d] == 1
        ]
        if squeeze_dims:
            data = data.squeeze(dim=squeeze_dims)

        # A spatial slice_mode may stay a non-dimension coordinate rather than a
        # literal dim: Design B's oblique-in-plane resample keeps data on native
        # k/j/i dims (only the slice axis is forced axis-aligned; the two in-plane
        # axes stay genuinely oblique), with the slice world coordinate still
        # selectable directly through VoxelToWorldIndex.sel's single-axis fast path
        # (see confusius._utils.geometry.VoxelToWorldIndex.sel) -- `_extract_slices`
        # below handles this via `.sel` on `data.coords` the same way it already
        # does for e.g. a "region" slice_mode coordinate.
        if self.slice_mode not in data.dims and self.slice_mode not in data.coords:
            raise ValueError(
                f"slice_mode '{self.slice_mode}' is not a dimension of data. "
                f"Available dimensions: {list(data.dims)}."
            )
        if data.ndim != 3:
            raise ValueError(
                f"Data must be 3D, but got shape {data.shape} with dims "
                f"{list(data.dims)}."
            )
        # Only the two display dims need sorting for pcolormesh/contour geometry;
        # sorting slice_mode itself would silently reorder panels (e.g. non-monotonic
        # z, or a "region" dim built from an arbitrary list of acronyms).
        display_dims = [d for d in data.dims if d != self.slice_mode]
        return sort_coords_for_plot(data, display_dims)

    def _resolve_axes_layout(
        self,
        data: xr.DataArray,
        n_slices: int,
        actual_coords: list[Hashable],
        dim_row: str,
        dim_col: str,
        *,
        match_coordinates: bool,
        nrows: int | None,
        ncols: int | None,
        dpi: int | None,
    ) -> list[tuple[int, int]]:
        """Resolve the per-slice axis assignment, creating the figure if needed."""
        if match_coordinates:
            if self.axes is None:
                raise ValueError(
                    "Cannot match coordinates: no existing axes. Either create a "
                    "VolumePlotter with axes or use match_coordinates=False."
                )
            matched_indices = self._find_matching_axes(actual_coords)
            matched_slice_indices = {idx for _, idx in matched_indices}
            unmatched_slices = [
                (idx, actual_coords[idx])
                for idx in range(n_slices)
                if idx not in matched_slice_indices
            ]
            if unmatched_slices:
                self._warn_unmatched(unmatched_slices)
            return matched_indices

        if self.axes is None:
            x_range = None
            y_range = None
            if dim_col in data.coords and dim_row in data.coords:
                x_coords = data.coords[dim_col].values.astype(float)
                y_coords = data.coords[dim_row].values.astype(float)
                x_range = float(np.max(x_coords) - np.min(x_coords))
                y_range = float(np.max(y_coords) - np.min(y_coords))
            self._ensure_figure(
                n_slices,
                nrows=nrows,
                ncols=ncols,
                dpi=dpi,
                x_range=x_range,
                y_range=y_range,
            )

        if self._user_provided_axes:
            assert self.axes is not None
            if n_slices != self.axes.size:
                raise ValueError(
                    f"Number of slices ({n_slices}) must match number of axes "
                    f"({self.axes.size}). Got {n_slices} slice_coords but axes "
                    f"has shape {self.axes.shape}."
                )

        return self._init_sequential_layout(actual_coords)

    def _style_slice_axis(
        self,
        ax: "Axes",
        axis_idx: int,
        data: xr.DataArray,
        coord: Hashable,
        dim_row: str,
        dim_col: str,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
        *,
        world_row: str | None = None,
        world_col: str | None = None,
        show_titles: bool,
        show_axis_labels: bool,
        show_axis_ticks: bool,
        show_axes: bool,
        title_fontsize: float | None,
        label_fontsize: float | None,
        tick_fontsize: float | None,
    ) -> None:
        """Apply post-draw styling (aspect, spines, title, labels, lims) to a slice axis."""
        ax.set_aspect("equal")
        self._style_ax(ax)

        text_color = self._text_color
        ax.set_title(
            self._build_slice_title(data, coord) if show_titles else "",
            color=text_color,
            fontsize=title_fontsize,
        )

        if show_axes:
            if show_axis_labels:
                ax.set_xlabel(
                    _build_axis_label(data, dim_col, world_dim=world_col),
                    color=text_color,
                    fontsize=label_fontsize,
                )
                ax.set_ylabel(
                    _build_axis_label(data, dim_row, world_dim=world_row),
                    color=text_color,
                    fontsize=label_fontsize,
                )
            if show_axis_ticks:
                ax.tick_params(labelsize=tick_fontsize)
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        else:
            ax.axis("off")

        # Expand stored limits to encompass overlaid volumes with different extents.
        current_xlim = self._update_stored_lim(
            self._axis_xlims,
            axis_idx,
            (float(x_edges.min()), float(x_edges.max())),
        )
        current_ylim = self._update_stored_lim(
            self._axis_ylims,
            axis_idx,
            (float(y_edges.min()), float(y_edges.max())),
        )
        self._set_ax_lims(ax, current_xlim, current_ylim)

    def _prepare_alpha_slices(
        self,
        alpha: "float | xr.DataArray | None",
        data: xr.DataArray,
        *,
        slice_coords: Sequence[Hashable],
        dim_row: Hashable,
        dim_col: Hashable,
    ) -> "tuple[float | None, list[np.ndarray] | None]":
        """Validate and slice a per-voxel `alpha`, or pass a scalar through.

        `data` is assumed already processed by `_prepare_slice_inputs`. Returns
        `(alpha, alpha_slices)`: exactly one is non-`None` -- `alpha_slices` for a
        per-voxel DataArray input, the (possibly `None`) scalar `alpha` otherwise.
        """
        if alpha is not None and not isinstance(alpha, (int, float, xr.DataArray)):
            raise TypeError(
                "`alpha` must be a scalar or a DataArray, not "
                f"{type(alpha).__name__}; a bare array carries no coordinates, so "
                "it cannot be validated or aligned against `data`."
            )

        if not isinstance(alpha, xr.DataArray):
            return alpha, None

        # Preprocess alpha exactly like data (squeeze unitary dims, sort coords
        # ascending). Otherwise, an alpha array created from the data could end up
        # with a shape or coordinate mismatch after slicing.
        alpha = self._prepare_slice_inputs(
            alpha, caller="VolumePlotter.add_volume (alpha)"
        )
        if set(alpha.dims) != set(data.dims):
            raise ValueError(
                f"`alpha` dims {sorted(str(d) for d in alpha.dims)} do not match "
                f"`data` dims {sorted(str(d) for d in data.dims)}."
            )
        for dim in data.dims:
            if alpha.sizes[dim] != data.sizes[dim]:
                raise ValueError(
                    f"`alpha` size along '{dim}' ({alpha.sizes[dim]}) does not "
                    f"match `data` size ({data.sizes[dim]})."
                )
        validate_matching_coordinates(data, alpha, left_name="data", right_name="alpha")
        alpha_da_slices, _ = _extract_slices(alpha, self.slice_mode, slice_coords)
        alpha_da_slices = self._maybe_resample_slices_to_world(alpha_da_slices)
        alpha_da_slices = [s.transpose(dim_row, dim_col) for s in alpha_da_slices]
        return None, [s.values for s in alpha_da_slices]

    def add_volume(
        self,
        data: xr.DataArray,
        *,
        slice_coords: Sequence[Hashable] | None = None,
        match_coordinates: bool = True,
        cmap: "str | Colormap | None" = None,
        norm: "Normalize | None" = None,
        vmin: float | None = None,
        vmax: float | None = None,
        threshold: float | None = None,
        threshold_mode: Literal["lower", "upper"] = "lower",
        alpha: "float | xr.DataArray | None" = None,
        show_colorbar: bool = True,
        cbar_label: str | None = None,
        cbar_kwargs: "dict[str, Any] | None" = None,
        roi_labels: dict[int, str] | None = None,
        resample_in_plane: bool = False,
        show_titles: bool = True,
        show_axis_labels: bool = True,
        show_axis_ticks: bool = True,
        show_axes: bool = True,
        fontsize: float | None = None,
        nrows: int | None = None,
        ncols: int | None = None,
        dpi: int | None = None,
    ) -> "VolumePlotter":
        """Plot or overlay a volume on the axes.

        Parameters
        ----------
        data : xarray.DataArray
            3D volume data. Unitary dimensions (except `slice_mode`) are squeezed
            before processing. Complex-valued inputs are converted to magnitude
            (`abs(data)`) with a warning.
        slice_coords : list[collections.abc.Hashable], optional
            Specific coordinates to plot. Numeric coordinates are matched by
            nearest-neighbour lookup; non-numeric coordinates (e.g. region labels)
            require an exact match. If not provided, uses all coordinates from data.
        match_coordinates : bool, default: True
            If True, match slice coordinates to the stored coordinate mapping (for
            overlays). If False, plot sequentially on all axes (requires exact axis
            count match).
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap. When not provided, falls back to `data.attrs["cmap"]` if
            present, otherwise `"gray"`.
        norm : matplotlib.colors.Normalize, optional
            Normalization instance (e.g. `BoundaryNorm` for integer label maps). When
            not provided, falls back to `data.attrs["norm"]` if present. When a norm
            is active, `vmin` and `vmax` are ignored.
        vmin : float, optional
            Lower bound of the colormap. Defaults to the 2nd percentile. Ignored
            when `norm` is provided explicitly (that is, not just inherited from data
            attributes).
        vmax : float, optional
            Upper bound of the colormap. Defaults to the 98th percentile. Ignored
            when `norm` is provided explicitly (that is, not just inherited from data
            attributes).
        threshold : float, optional
            Threshold value for masking.
        threshold_mode : {"lower", "upper"}, default: "lower"
            Whether to mask values below or above threshold.
        alpha : float or xarray.DataArray, optional
            Opacity of the image: a single scalar value, or a DataArray sharing `data`'s
            dims, shape, and coordinates (for independent per-slice, per-voxel opacity —
            e.g. fading out low-confidence voxels). A per-voxel opacity must be a
            DataArray, not a bare array, so it can be validated and aligned against
            `data`. If not provided, the colormap's own alpha channel is respected.
        show_colorbar : bool, default: True
            Whether to add a colorbar.
        cbar_label : str, optional
            Label for the colorbar.
        cbar_kwargs : dict, optional
            Additional keyword arguments forwarded to
            [`matplotlib.figure.Figure.colorbar`][matplotlib.figure.Figure.colorbar]
            (e.g. `shrink`, `fraction`, `pad`, `aspect`). Useful to shrink the
            colorbar when it spans a multi-panel grid, since the defaults are sized
            for a single axes.
        roi_labels : dict[int, str], optional
            Mapping from integer label to display name. When provided (or when
            `data.attrs["roi_labels"]` is populated), hovering the cursor over a
            voxel shows `<layer.name>=<id> (<name>)` in the matplotlib status bar.
        resample_in_plane : bool, default: False
            For a spatial `slice_mode` (`"z"`/`"y"`/`"x"`) with oblique in-plane
            geometry only: whether to also force the two in-plane axes onto the
            global frame (entirely rectangular output cells, via a real
            interpolation for those axes too), instead of the default -- `data`'s
            own native in-plane directions and resolution, no interpolation beyond
            the slice axis, but potentially non-rectangular cells. Useful to avoid
            alpha-blended `pcolormesh`'s per-cell seam rendering artifact on
            non-rectangular cells (see
            design/world-mode-resample-scoping.md, Design D), at the cost of
            geometry/resolution fidelity. A per-call choice: unlike the slice
            axis's own discretization, not shared with other volumes on this
            plotter. No effect for a non-spatial `slice_mode` or already
            axis-aligned `data`.
        show_titles : bool, default: True
            Whether to display subplot titles.
        show_axis_labels : bool, default: True
            Whether to display axis labels.
        show_axis_ticks : bool, default: True
            Whether to display axis tick labels.
        show_axes : bool, default: True
            Whether to show all axis decorations (spines, ticks, labels). When `False`,
            overrides `show_axis_labels` and `show_axis_ticks`.
        fontsize : float, optional
            Base font size for all text elements. Subplot titles use `fontsize`
            directly; axis labels and the colorbar label use `0.9 * fontsize`; tick
            labels use `0.85 * fontsize`. If not provided, uses the active Matplotlib
            defaults.
        nrows : int, optional
            Number of rows in the subplot grid when creating a new figure.
            If not provided, computed automatically.
        ncols : int, optional
            Number of columns in the subplot grid when creating a new figure.
            If not provided, computed automatically.
        dpi : int, optional
            Figure resolution in dots per inch. Ignored when using an existing figure.

        Returns
        -------
        VolumePlotter
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If no matching coordinates are found or axis count doesn't match.
        ValueError
            If `alpha` is a DataArray and its dims, shape, or coordinates do not
            match `data`.
        TypeError
            If `alpha` is neither a scalar nor a DataArray.
        """
        resolved_roi_labels = _normalize_roi_labels(
            roi_labels if roi_labels is not None else data.attrs.get("roi_labels")
        )

        data = self._prepare_slice_inputs(
            data, caller="VolumePlotter.add_volume", resample_in_plane=resample_in_plane
        )

        if slice_coords is None:
            slice_coords = _default_slice_coords(data, self.slice_mode)

        unthresholded_slices, actual_coords, dim_row, dim_col, world_row, world_col = (
            self._extract_display_slices(data, slice_coords)
        )
        n_slices = len(unthresholded_slices)

        alpha, alpha_slices = self._prepare_alpha_slices(
            alpha, data, slice_coords=slice_coords, dim_row=dim_row, dim_col=dim_col
        )

        norm = _resolve_norm(
            slices=unthresholded_slices,
            norm=norm,
            data_attrs_norm=data.attrs.get("norm"),
            vmin=vmin,
            vmax=vmax,
        )

        thresholded_slices = _threshold_slices(
            unthresholded_slices, threshold=threshold, threshold_mode=threshold_mode
        )

        cmap = _resolve_cmap(
            cmap=cmap,
            data_attrs_cmap=data.attrs.get("cmap"),
            norm=norm,
            threshold=threshold,
            threshold_mode=threshold_mode,
        )

        plot_indices = self._resolve_axes_layout(
            data,
            n_slices,
            actual_coords,
            dim_row,
            dim_col,
            match_coordinates=match_coordinates,
            nrows=nrows,
            ncols=ncols,
            dpi=dpi,
        )

        assert (self.axes is not None) and (self.figure is not None)

        text_color = self._text_color
        title_fontsize, label_fontsize, tick_fontsize = _resolve_font_sizes(fontsize)
        plotted_quadmesh = None

        axes_flat = self.axes.ravel()
        for axis_idx, slice_idx in plot_indices:
            ax = axes_flat[axis_idx]
            arr = thresholded_slices[slice_idx]
            slice_da = unthresholded_slices[slice_idx]
            x_edges, y_edges, hover_x, hover_y = _slice_edges_and_centers(
                slice_da, dim_row, dim_col, world_row=world_row, world_col=world_col
            )

            panel_alpha = alpha_slices[slice_idx] if alpha_slices is not None else alpha
            plotted_quadmesh = ax.pcolormesh(
                x_edges,
                y_edges,
                np.ma.masked_invalid(arr),
                cmap=cmap,
                norm=norm,
                alpha=panel_alpha,
            )
            if hover_x.ndim == 1 and hover_y.ndim == 1:
                self._attach_or_update_hover_manager(resolved_roi_labels)
                self._hover_manager.register_data_to_axis(
                    ax,
                    hover_x,
                    hover_y,
                    slice_da.values,
                    role="labels" if resolved_roi_labels else "volume",
                    name=str(data.name) if data.name is not None else "value",
                    units=data.attrs.get("units"),
                )

            self._style_slice_axis(
                ax,
                axis_idx,
                slice_da,
                actual_coords[slice_idx],
                dim_row,
                dim_col,
                x_edges,
                y_edges,
                world_row=world_row,
                world_col=world_col,
                show_titles=show_titles,
                show_axis_labels=show_axis_labels,
                show_axis_ticks=show_axis_ticks,
                show_axes=show_axes,
                title_fontsize=title_fontsize,
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
            )

        if not match_coordinates:
            for ax in axes_flat[n_slices:]:
                ax.set_visible(False)

        if show_colorbar and plotted_quadmesh is not None:
            non_cbar_axes = [
                ax for ax in self.figure.axes if not hasattr(ax, "_colorbar")
            ]
            cbar = self.figure.colorbar(
                plotted_quadmesh, ax=non_cbar_axes, **(cbar_kwargs or {})
            )
            if cbar_label is None:
                long_name = data.attrs.get("long_name")
                units = data.attrs.get("units")
                if long_name and units:
                    cbar_label = f"{long_name} ({units})"
                elif long_name:
                    cbar_label = long_name
                elif units:
                    cbar_label = f"({units})"
            _style_colorbar(
                cbar,
                text_color,
                tick_fontsize,
                label=cbar_label,
                label_fontsize=label_fontsize,
            )

        return self

    def add_composite(
        self,
        data1: xr.DataArray,
        data2: xr.DataArray,
        *,
        resample: bool = True,
        resample_kwargs: "dict[str, Any] | None" = None,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        normalize_strategy: Literal["per_volume", "per_slice", "shared"] = "per_volume",
        slice_coords: Sequence[Hashable] | None = None,
        match_coordinates: bool = False,
        alpha: "float | npt.NDArray[np.floating] | None" = None,
        show_titles: bool = True,
        show_axis_labels: bool = True,
        show_axis_ticks: bool = True,
        show_axes: bool = True,
        fontsize: float | None = None,
        nrows: int | None = None,
        ncols: int | None = None,
        dpi: int | None = None,
    ) -> "VolumePlotter":
        """Plot a red/cyan composite of two volumes on the axes.

        Each slice is rendered as an RGB image where `data1` drives the red channel
        and `data2` drives the green and blue channels (cyan), making overlap
        visible as desaturated grey. This is the same visual encoding used by the
        live registration progress preview.

        Parameters
        ----------
        data1 : xarray.DataArray
            First volume, plotted in red. 3D volume data. Unitary dimensions (except
            `slice_mode`) are squeezed before processing. Complex-valued inputs are
            converted to magnitude (`abs(data)`) with a warning.
        data2 : xarray.DataArray
            Second volume, plotted in cyan. Must have the same dimensionality as
            `data1` after squeezing; when `resample=True` it is resampled onto
            `data1`'s grid before plotting, so its native shape and coordinates may
            differ.
        resample : bool, default: True
            Whether to resample `data2` onto `data1`'s grid using an identity
            transform before blending. When `False`, the two arrays must
            already share the same dimensions and shape, and their coordinates
            must match within `rtol`/`atol`; once validated, `data2`'s
            coordinates are replaced with `data1`'s so the two volumes share
            an exact coordinate frame downstream.
        resample_kwargs : dict, optional
            Extra keyword arguments forwarded to
            [`resample_like`][confusius.registration.resample_like] when
            `resample=True`. Ignored when `resample=False`.
        rtol : float, default: 1e-5
            Relative tolerance used to validate that `data1` and `data2` share
            coordinates when `resample=False`. Widen to accept acquisitions on
            slightly offset grids known to be equivalent. Ignored when
            `resample=True`.
        atol : float, default: 1e-8
            Absolute tolerance used to validate that `data1` and `data2` share
            coordinates when `resample=False`. Ignored when `resample=True`.
        normalize_strategy : {"per_volume", "per_slice", "shared"}, default: "per_volume"
            Intensity normalisation strategy.

            - `"per_volume"`: rescale each input to `[0, 1]` independently over its
              full volume. Preserves slice-to-slice contrast within each array
              but loses the absolute-intensity relationship between `data1` and
              `data2`.
            - `"per_slice"`: rescale each 2D slice independently. Maximises
              contrast on dim slices at the cost of cross-slice comparability.
            - `"shared"`: rescale both volumes together using a single shared
              `[min(data1.min(), data2.min()), max(data1.max(), data2.max())]`
              range. Preserves the absolute-intensity relationship between the
              two inputs, useful when comparing data acquired at the same
              dynamic range.
        slice_coords : Sequence[collections.abc.Hashable], optional
            Coordinate values along `slice_mode` at which to extract slices. Numeric
            coordinates are matched by nearest-neighbour lookup; non-numeric
            coordinates (e.g. region labels) require an exact match. If not provided,
            all coordinate values from `data1` are used.
        match_coordinates : bool, default: False
            If True, match slice coordinates to the stored coordinate mapping of an
            existing figure (for use as an overlay). If False, plot sequentially on
            a fresh grid of axes — the natural mode for a standalone composite plot.
        alpha : float or numpy.ndarray, optional
            Opacity of the composite image, either a single value or a per-voxel
            array matching the shape of the displayed slices. If not provided, the
            image is fully opaque.
        show_titles : bool, default: True
            Whether to display subplot titles showing the slice coordinate.
        show_axis_labels : bool, default: True
            Whether to display axis labels (with units when available).
        show_axis_ticks : bool, default: True
            Whether to display axis tick labels.
        show_axes : bool, default: True
            Whether to show axis decorations. When `False`, overrides
            `show_axis_labels` and `show_axis_ticks`.
        fontsize : float, optional
            Base font size for all text elements. Subplot titles use `fontsize`
            directly; axis labels use `0.9 * fontsize`; tick labels use
            `0.85 * fontsize`. If not provided, uses the active Matplotlib
            defaults.
        nrows : int, optional
            Number of rows in the subplot grid when creating a new figure.
            If not provided, computed automatically.
        ncols : int, optional
            Number of columns in the subplot grid when creating a new figure.
            If not provided, computed automatically.
        dpi : int, optional
            Figure resolution in dots per inch. Ignored when using an existing
            figure.

        Returns
        -------
        VolumePlotter
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If either input has a `time` dimension, is not 2D or 3D, lacks
            `slice_mode` as a dimension, or (when `resample=False`) the two
            arrays do not share dims, shape, and coordinates within
            `rtol`/`atol`.

        Notes
        -----
        The composite is rendered with
        [`pcolormesh`][matplotlib.axes.Axes.pcolormesh] using its RGB-`C`
        codepath, so panels line up with overlays drawn by
        [`add_volume`][confusius.plotting.VolumePlotter.add_volume] /
        [`add_contours`][confusius.plotting.VolumePlotter.add_contours].
        Hover tooltips, colormaps, colorbars, and intensity thresholds are
        not supported on composite axes — use `add_volume` for those.
        """
        if normalize_strategy not in ("per_volume", "per_slice", "shared"):
            raise ValueError(
                f"Invalid normalization strategy {normalize_strategy!r}. "
                f"Expected 'per_volume', 'per_slice', or 'shared'."
            )

        if resample:
            from confusius.registration.resampling import resample_like

            data2_name = data2.name or "data2"
            _kw: dict[str, Any] = dict(resample_kwargs or {})
            # The identity transform is always spatial-only (3D): resample_like reads
            # only data1's k/j/i grid, regardless of any time dimension it carries.
            data2 = resample_like(data2, data1, np.eye(len(VOXEL_DIMS) + 1), **_kw)
            data2.name = data2_name

        data1 = self._prepare_slice_inputs(
            data1, caller="VolumePlotter.add_composite (data1)"
        )
        data2 = self._prepare_slice_inputs(
            data2, caller="VolumePlotter.add_composite (data2)"
        )

        if not resample:
            if data1.dims != data2.dims:
                raise ValueError(
                    f"With resample=False, data1 and data2 must share dimensions; "
                    f"got {data1.dims} vs {data2.dims}."
                )
            if data1.shape != data2.shape:
                raise ValueError(
                    f"With resample=False, data1 and data2 must share shape; "
                    f"got {data1.shape} vs {data2.shape}."
                )
            validate_matching_coordinates(
                data1,
                data2,
                left_name="data1",
                right_name="data2",
                rtol=rtol,
                atol=atol,
            )
            # Replace data2's coords with data1's so harmless floating-point
            # drift does not propagate into downstream slicing.
            data2 = data2.assign_coords(
                {d: data1.coords[d] for d in data1.dims if d in data1.coords}
            )

        if slice_coords is None:
            slice_coords = _default_slice_coords(data1, self.slice_mode)

        input_slices1, _ = _extract_slices(data1, self.slice_mode, slice_coords)
        input_slices2, _ = _extract_slices(data2, self.slice_mode, slice_coords)
        input_slices1 = self._maybe_resample_slices_to_world(input_slices1)
        input_slices2 = self._maybe_resample_slices_to_world(input_slices2)

        if normalize_strategy == "per_volume":
            data1 = data1.copy(data=scale_min_max(data1.values.astype(float)))
            data2 = data2.copy(data=scale_min_max(data2.values.astype(float)))
        elif normalize_strategy == "shared":
            arr1 = data1.values.astype(float)
            arr2 = data2.values.astype(float)
            finite = np.concatenate([arr1[np.isfinite(arr1)], arr2[np.isfinite(arr2)]])
            if finite.size == 0:
                raise ValueError(
                    "Cannot normalize data1/data2 with 'shared' strategy: no finite "
                    "values found in either array."
                )
            lo = float(finite.min())
            hi = float(finite.max())
            if hi == lo:
                arr1 = np.zeros_like(arr1)
                arr2 = np.zeros_like(arr2)
            else:
                arr1 = np.clip((arr1 - lo) / (hi - lo), 0.0, 1.0)
                arr2 = np.clip((arr2 - lo) / (hi - lo), 0.0, 1.0)
            data1 = data1.copy(data=arr1)
            data2 = data2.copy(data=arr2)

        slices1, actual_coords, dim_row, dim_col, world_row, world_col = (
            self._extract_display_slices(data1, slice_coords)
        )
        slices2 = self._maybe_resample_slices_to_world(
            _extract_slices(data2, self.slice_mode, slice_coords)[0]
        )
        slices2 = [s.transpose(dim_row, dim_col) for s in slices2]
        input_slices1 = [s.transpose(dim_row, dim_col) for s in input_slices1]
        input_slices2 = [s.transpose(dim_row, dim_col) for s in input_slices2]
        n_slices = len(slices1)

        plot_indices = self._resolve_axes_layout(
            data1,
            n_slices,
            actual_coords,
            dim_row,
            dim_col,
            match_coordinates=match_coordinates,
            nrows=nrows,
            ncols=ncols,
            dpi=dpi,
        )

        assert (self.axes is not None) and (self.figure is not None)

        title_fontsize, label_fontsize, tick_fontsize = _resolve_font_sizes(fontsize)
        axes_flat = self.axes.ravel()

        for axis_idx, slice_idx in plot_indices:
            ax = axes_flat[axis_idx]
            slice1 = slices1[slice_idx]
            slice2 = slices2[slice_idx]

            arr1 = slice1.values.astype(float)
            arr2 = slice2.values.astype(float)
            if normalize_strategy == "per_slice":
                arr1 = scale_min_max(arr1)
                arr2 = scale_min_max(arr2)
            rgb = blend_red_cyan(arr1, arr2)

            x_edges, y_edges, hover_x, hover_y = _slice_edges_and_centers(
                slice1, dim_row, dim_col, world_row=world_row, world_col=world_col
            )

            ax.pcolormesh(x_edges, y_edges, rgb, alpha=alpha)
            if hover_x.ndim == 1 and hover_y.ndim == 1:
                self._attach_or_update_hover_manager({})
                for i, (data, input_slices) in enumerate(
                    zip((data1, data2), (input_slices1, input_slices2))
                ):
                    self._hover_manager.register_data_to_axis(
                        ax,
                        hover_x,
                        hover_y,
                        input_slices[slice_idx].values,
                        role="volume",
                        name=str(data.name)
                        if data.name is not None
                        else f"data{i + 1}",
                        units=data.attrs.get("units"),
                    )

            self._style_slice_axis(
                ax,
                axis_idx,
                slice1,
                actual_coords[slice_idx],
                dim_row,
                dim_col,
                x_edges,
                y_edges,
                world_row=world_row,
                world_col=world_col,
                show_titles=show_titles,
                show_axis_labels=show_axis_labels,
                show_axis_ticks=show_axis_ticks,
                show_axes=show_axes,
                title_fontsize=title_fontsize,
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
            )

        if not match_coordinates:
            for ax in axes_flat[n_slices:]:
                ax.set_visible(False)

        return self

    def add_contours(
        self,
        mask: xr.DataArray,
        *,
        colors: dict[int | str, str] | str | None = None,
        linewidths: float = 1.5,
        linestyles: str = "solid",
        match_coordinates: bool = True,
        slice_coords: list[Hashable] | None = None,
        fontsize: float | None = None,
        roi_labels: dict[int, str] | None = None,
        resample_in_plane: bool = False,
        **kwargs,
    ) -> "VolumePlotter":
        """Add mask contours to existing axes.

        Parameters
        ----------
        mask : xarray.DataArray
            Integer label map in one of two formats:

            - **Flat label map**: Spatial dims only, e.g. `(k, j, i)`. Background voxels
              labeled `0`; each unique non-zero integer identifies a distinct,
              non-overlapping region. The `region` coordinate of the output holds the
              integer label values.
            - **Stacked mask format**: Has a leading `mask` dimension followed by
              spatial dims, e.g. `(mask, k, j, i)`. Each layer has values in `{0,
              region_id}` and regions may overlap. The `region` coordinate of the
              output holds the `mask` coordinate values (e.g., region label).

            Drawn on the same display geometry as the plotter's own data -- world
            space, projected or resampled as appropriate (see
            [`VolumePlotter`][confusius.plotting.VolumePlotter]).
        colors : dict[int | str, str] or str, optional
            Color specification for contour lines.

            - `dict`: maps each label (integer index) or region acronym (string)
              to a color string.
            - `str`: applies one color to all regions.
            - `None`: colors are derived from `attrs["cmap"]` and
              `attrs["norm"]` when present, otherwise from the
              `tab10`/`tab20` colormap.
        linewidths : float, default: 1.5
            Width of contour lines in points.
        linestyles : str, default: "solid"
            Line style for contour lines (e.g. `"solid"`, `"dashed"`).
        match_coordinates : bool, default: True
            If `True`, overlay contours on axes whose slice coordinate matches the
            mask. If `False`, plot sequentially on all axes.
        slice_coords : list[collections.abc.Hashable], optional
            Coordinate values along the plotter's `slice_mode` at which to draw
            contours. Numeric coordinates are matched by nearest-neighbour lookup;
            non-numeric coordinates (e.g. region labels) require an exact match. If
            not provided, all coordinate values along `slice_mode` are used.
        fontsize : float, optional
            Base font size for text elements when a standalone contour figure is created
            (`match_coordinates=False`). Subplot titles use `fontsize` directly;
            axis labels use `0.9 * fontsize`; tick labels use `0.85 * fontsize`.
            If not provided, uses the active Matplotlib defaults.
        roi_labels : dict[int, str], optional
            Mapping from integer label to display name. When provided (or when
            `mask.attrs["roi_labels"]` is populated), hovering the cursor over a
            voxel shows `<data_name>=<id> (<roi_name>)` in the matplotlib status bar. The
            cursor samples the underlying label map directly, so hovering inside
            a closed contour is sufficient — there is no need to be on the line.
        resample_in_plane : bool, default: False
            See [`add_volume`][confusius.plotting.VolumePlotter.add_volume]'s own
            parameter doc. Contour lines themselves have no alpha-blended
            `pcolormesh` seam artifact to avoid, but forcing the mask onto the
            global in-plane frame can still be useful for consistency with a
            background image plotted with `resample_in_plane=True`.
        **kwargs
            Additional keyword arguments passed to
            [`matplotlib.axes.Axes.plot`][matplotlib.axes.Axes.plot].

        Returns
        -------
        VolumePlotter
            Returns `self` for method chaining.

        Raises
        ------
        ValueError
            If the plotter's `slice_mode` is not a dimension of `mask`.
        ValueError
            If `mask` is not 3D or 4D with a leading `mask` dimension.
        """
        import matplotlib.colors as mcolors
        from skimage.measure import find_contours

        resolved_roi_labels = _normalize_roi_labels(
            roi_labels if roi_labels is not None else mask.attrs.get("roi_labels")
        )

        # Stacked mask format: (mask, z, y, x) — one layer per region.
        if "mask" in mask.dims:
            # Compute once here rather than once per layer below (`np.unique`)
            # plus again inside each layer's recursive `add_contours` call (whose
            # own `_prepare_slice_inputs` would otherwise recompute a dask-backed
            # mask's already-computed-once-per-layer data yet again).
            mask = mask.compute()
            cmap_attr = mask.attrs.get("cmap")
            norm_attr = mask.attrs.get("norm")
            # cmap/norm are dropped on Zarr save; reconstruct from rgb_lookup when
            # present so structure colors survive a serialization round-trip.
            if (cmap_attr is None or norm_attr is None) and "rgb_lookup" in mask.attrs:
                cmap_attr, norm_attr = build_atlas_cmap_and_norm(
                    mask.attrs["rgb_lookup"]
                )
            acronyms = mask.coords["mask"].values

            for i in range(mask.sizes["mask"]):
                layer = mask.isel(mask=i)
                acronym = str(acronyms[i])

                unique_nonzero = [v for v in np.unique(layer.values) if v != 0]
                if not unique_nonzero:
                    continue
                label = int(unique_nonzero[0])

                if isinstance(colors, str):
                    layer_color: str = colors
                elif isinstance(colors, dict):
                    # We accept both acronym-keyed and id-keyed dicts for flexibility.
                    layer_color = colors.get(acronym, colors.get(label, "white"))
                elif cmap_attr is not None and norm_attr is not None:
                    layer_color = mcolors.to_hex(cmap_attr(norm_attr(label)))
                else:
                    layer_color = _get_distinct_colors(mask.sizes["mask"])[i]

                per_layer_colors: dict[int | str, Any] = {label: layer_color}
                self.add_contours(
                    layer,
                    colors=per_layer_colors,
                    linewidths=linewidths,
                    linestyles=linestyles,
                    match_coordinates=match_coordinates,
                    slice_coords=slice_coords,
                    fontsize=fontsize,
                    roi_labels=resolved_roi_labels or None,
                    **kwargs,
                )
            return self

        # `_prepare_slice_inputs` handles a mask exactly like `add_volume`'s data --
        # it can carry the same extra facet dim as `self.slice_mode` (e.g.
        # "region"), and must land on the same native-vs-world dims as the data it
        # overlays. `interpolation="nearest"` regardless of
        # `self._resample_interpolation`: mask/label data is a set of distinct
        # integer regions, and blending them together (linear/bspline) would
        # fabricate boundary values that match no real label.
        mask = self._prepare_slice_inputs(
            mask,
            caller="VolumePlotter.add_contours",
            interpolation="nearest",
            resample_in_plane=resample_in_plane,
        )

        unique_labels = sorted(
            [label for label in np.unique(mask.values) if label != 0]
        )
        if not unique_labels:
            return self

        if colors is None:
            cmap_attr = mask.attrs.get("cmap")
            norm_attr = mask.attrs.get("norm")
            # cmap/norm are dropped on Zarr save; reconstruct from rgb_lookup when
            # present so structure colors survive a serialization round-trip.
            if (cmap_attr is None or norm_attr is None) and "rgb_lookup" in mask.attrs:
                cmap_attr, norm_attr = build_atlas_cmap_and_norm(
                    mask.attrs["rgb_lookup"]
                )
            if cmap_attr is not None and norm_attr is not None:
                color_map = {
                    label: mcolors.to_hex(cmap_attr(norm_attr(label)))
                    for label in unique_labels
                }
            else:
                distinct_colors = _get_distinct_colors(len(unique_labels))
                color_map = {
                    label: color for label, color in zip(unique_labels, distinct_colors)
                }
        elif isinstance(colors, str):
            color_map = {label: colors for label in unique_labels}
        else:
            color_map = colors

        if slice_coords is None:
            slice_coords = _default_slice_coords(mask, self.slice_mode)

        # Always "nearest" for the per-panel world resample too: mask/label data is a
        # set of distinct integer regions, never meaningfully blended.
        slices, actual_coords, dim_row, dim_col, world_row, world_col = (
            self._extract_display_slices(mask, slice_coords, interpolation="nearest")
        )
        n_slices = len(slices)

        if match_coordinates:
            matched_indices = self._find_matching_axes(actual_coords)

            matched_slice_indices = {idx for _, idx in matched_indices}
            unmatched_slices = [
                (idx, actual_coords[idx])
                for idx in range(n_slices)
                if idx not in matched_slice_indices
            ]
            if unmatched_slices:
                self._warn_unmatched(unmatched_slices)
            plot_indices = matched_indices
        else:
            x_range = None
            y_range = None
            if dim_col in mask.coords and dim_row in mask.coords:
                x_vals_all = mask.coords[dim_col].values.astype(float)
                y_vals_all = mask.coords[dim_row].values.astype(float)
                x_range = float(np.max(x_vals_all) - np.min(x_vals_all))
                y_range = float(np.max(y_vals_all) - np.min(y_vals_all))
            self._ensure_figure(n_slices, x_range=x_range, y_range=y_range)

            if self._user_provided_axes:
                assert self.axes is not None
                if n_slices != self.axes.size:
                    raise ValueError(
                        f"Number of slices ({n_slices}) must match number of axes "
                        f"({self.axes.size}). Got {n_slices} slice_coords but axes has "
                        f"shape {self.axes.shape}."
                    )

            plot_indices = self._init_sequential_layout(actual_coords)

        if self.axes is None:
            raise RuntimeError("No axes available")

        axes_flat = self.axes.ravel()
        title_fontsize, label_fontsize, tick_fontsize = _resolve_font_sizes(fontsize)

        for axis_idx, slice_idx in plot_indices:
            ax = axes_flat[axis_idx]
            slice_da = slices[slice_idx]
            slice_data = slice_da.values

            # Same geometry `add_volume` draws the underlying pcolormesh with (1D
            # rectilinear centers for materialized/world-space data, 2D projected
            # centers for oblique data), so contour vertices land exactly on the
            # pixels they outline instead of a separately reconstructed coordinate
            # lookup.
            x_edges, y_edges, x_centers, y_centers = _slice_edges_and_centers(
                slice_da, dim_row, dim_col, world_row=world_row, world_col=world_col
            )

            for label in unique_labels:
                binary_mask = (slice_data == label).astype(np.uint8)
                if not binary_mask.any():
                    continue

                padded = np.pad(binary_mask, 1, mode="constant")
                contours = find_contours(padded, level=0.5)
                contours = [c - 1 for c in contours]

                color = color_map.get(label, "white")

                for contour in contours:
                    if len(contour) < 2:
                        continue

                    # Map pixel indices to display coordinates. Contours are at
                    # pixel boundaries, so we interpolate between coordinate
                    # centers to get edge positions.
                    # contour[:, 0] is row (y) index, contour[:, 1] is col (x) index
                    x_idx = contour[:, 1]
                    y_idx = contour[:, 0]
                    if x_centers.ndim == 1:
                        x_world = np.interp(x_idx, np.arange(len(x_centers)), x_centers)
                        y_world = np.interp(y_idx, np.arange(len(y_centers)), y_centers)
                    else:
                        x_world = _bilinear_interpolate_grid(x_centers, y_idx, x_idx)
                        y_world = _bilinear_interpolate_grid(y_centers, y_idx, x_idx)
                    ax.plot(
                        x_world,
                        y_world,
                        color=color,
                        linewidth=linewidths,
                        linestyle=linestyles,
                        **kwargs,
                    )

            # Hover only supports rectilinear (1D) coordinates, matching add_volume's
            # own `hover_x.ndim == 1 and hover_y.ndim == 1` guard -- a 2D projected
            # grid (oblique data) has no hover support yet.
            if resolved_roi_labels and self.figure is not None and x_centers.ndim == 1:
                self._attach_or_update_hover_manager(resolved_roi_labels)
                self._hover_manager.register_data_to_axis(
                    ax,
                    x_coords=np.asarray(x_centers, dtype=float),
                    y_coords=np.asarray(y_centers, dtype=float),
                    data_2d=np.asarray(slice_data),
                    role="labels",
                    name=str(mask.name) if mask.name is not None else "label",
                )

            if not match_coordinates:
                ax.set_aspect("equal")

                # Compute limits from the same edges pcolormesh would use, not from
                # auto-scaled matplotlib limits, which may include padding.
                xlim = (float(x_edges.min()), float(x_edges.max()))
                ylim = (float(y_edges.min()), float(y_edges.max()))
                self._set_ax_lims(ax, xlim, ylim)
                self._style_ax(ax)
                ax.set_xlabel(
                    _build_axis_label(slice_da, dim_col, world_dim=world_col),
                    color=self._text_color,
                    fontsize=label_fontsize,
                )
                ax.set_ylabel(
                    _build_axis_label(slice_da, dim_row, world_dim=world_row),
                    color=self._text_color,
                    fontsize=label_fontsize,
                )
                ax.set_title(
                    self._build_slice_title(slice_da, actual_coords[slice_idx]),
                    color=self._text_color,
                    fontsize=title_fontsize,
                )
                ax.tick_params(labelsize=tick_fontsize)

        return self

    def savefig(self, fname: str, **kwargs) -> None:
        """Save the figure to a file.

        Parameters
        ----------
        fname : str
            Path to save the figure. Extension determines format (e.g., `.png`, `.pdf`).
        **kwargs
            Additional arguments passed to
            [`matplotlib.figure.Figure.savefig`][matplotlib.figure.Figure.savefig].

        Raises
        ------
        RuntimeError
            If called before any data has been plotted.
        """
        if self.figure is None:
            raise RuntimeError("No figure to save.")
        self.figure.savefig(fname, **kwargs)

    def show(self) -> None:
        """Display the figure.

        Raises
        ------
        RuntimeError
            If called before any data has been plotted.
        """
        if self.figure is None:
            raise RuntimeError("No figure to show.")
        self.figure.show()

    def close(self) -> None:
        """Close the figure and release resources."""
        import matplotlib.pyplot as plt

        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None
            self.axes = None
            self._coord_to_axis.clear()
            self._axis_xlims.clear()
            self._axis_ylims.clear()
            self._hover_manager.clear()


def plot_contours(
    mask: xr.DataArray,
    *,
    colors: dict[int | str, str] | str | None = None,
    linewidths: float = 1.5,
    linestyles: str = "solid",
    slice_mode: str = "z",
    slice_coords: list[Hashable] | None = None,
    transpose: bool = False,
    fontsize: float | None = None,
    yincrease: bool = False,
    xincrease: bool = True,
    bg_color: str = "black",
    fg_color: str | None = None,
    figure: "Figure | None" = None,
    axes: "npt.NDArray[Any] | Axes | None" = None,
    roi_labels: dict[int, str] | None = None,
    **kwargs,
) -> VolumePlotter:
    """Plot mask contours as a grid of 2D slice panels.

    Displays contour lines for each labeled region in `mask` across a grid of subplots.
    Each panel shows the contours for one slice along `slice_mode`, drawn in world
    coordinates when available.

    Parameters
    ----------
    mask : xarray.DataArray
        Integer label map in one of two formats:

        - **Flat label map**: Spatial dims only, e.g. `(k, j, i)`. Background voxels
          labeled `0`; each unique non-zero integer identifies a distinct,
          non-overlapping region. The `regions` coordinate of the output holds the
          integer label values.
        - **Stacked mask format**: Has a leading `masks` dimension followed by
          spatial dims, e.g. `(masks, k, j, i)`. Each layer has values in `{0,
          region_id}` and regions may overlap. The `regions` coordinate of the
          output holds the `masks` coordinate values (e.g., region label).

    colors : dict[int | str, str] or str, optional
        Color specification for contour lines. A `dict` maps each label (integer index
        or region acronym string) to a color; a `str` applies one color to all regions.
        If not provided, colors are derived from `attrs["cmap"]` and `attrs["norm"]`
        when present, otherwise from the `tab10`/`tab20` colormap.
    linewidths : float, default: 1.5
        Width of contour lines in points.
    linestyles : str, default: "solid"
        Line style for contour lines (e.g. `"solid"`, `"dashed"`).
    slice_mode : str, default: "z"
        Dimension along which to slice (e.g. `"x"`, `"y"`, `"z"`). After
        slicing, each panel must be 2D.
    slice_coords : list[collections.abc.Hashable], optional
        Coordinate values along `slice_mode` at which to extract slices. Numeric
        coordinates are matched by nearest-neighbour lookup; non-numeric
        coordinates (e.g. region labels) require an exact match. If not provided,
        all coordinate values along `slice_mode` are used.
    transpose : bool, default: False
        Whether to swap the row/column display dims of each slice panel.
    fontsize : float, optional
        Base font size for text elements. Subplot titles use `fontsize` directly; axis
        labels use `0.9 * fontsize`; tick labels use `0.85 * fontsize`. If not provided,
        uses the active Matplotlib defaults.
    yincrease : bool, default: False
        Whether the y-axis increases upward (`True`) or downward (`False`).
    xincrease : bool, default: True
        Whether the x-axis increases to the right (`True`) or left (`False`).
    bg_color : str, default: "black"
        Background color for the figure and axes. Any matplotlib-compatible color
        string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula
        (white on dark backgrounds, black on light ones).
    figure : matplotlib.figure.Figure, optional
        Existing figure to draw into. If not provided, a new figure is created.
    axes : numpy.ndarray or matplotlib.axes.Axes, optional
        Existing axes to draw into: either a single
        [`matplotlib.axes.Axes`][matplotlib.axes.Axes] or a 2D array of them. A single
        `Axes` is wrapped automatically. If not provided, new axes are created inside
        `figure`.
    roi_labels : dict[int, str], optional
        Mapping from integer label to display name. When provided (or when
        `mask.attrs["roi_labels"]` is populated), hovering the cursor over a
        voxel shows `<data_name>=<id> (<roi_name>)` in the matplotlib status bar.
    **kwargs
        Additional keyword arguments passed to
        [`matplotlib.axes.Axes.plot`][matplotlib.axes.Axes.plot].

    Returns
    -------
    VolumePlotter
        Object managing the figure, axes, and coordinate mapping for overlays.

    Raises
    ------
    ValueError
        If `slice_mode` is not a dimension of `mask`.
    ValueError
        If `mask` is not 3D.

    Notes
    -----
    Contours are computed with `skimage.measure.find_contours` on a binary mask for each
    label, then mapped to world coordinates via linear interpolation between
    coordinate centers. Each panel has `aspect="equal"` so that 1 unit in x matches 1
    unit in y.

    The returned [`VolumePlotter`][confusius.plotting.VolumePlotter] stores the
    coordinate-to-axis mapping, so you can overlay a volume afterwards with
    [`VolumePlotter.add_volume`][confusius.plotting.VolumePlotter.add_volume].

    Examples
    --------
    >>> import xarray as xr
    >>> from confusius.plotting import plot_contours
    >>> mask = xr.open_zarr("output.zarr")["roi_mask"]
    >>> plotter = plot_contours(mask, slice_mode="z")

    >>> # Custom colors per label.
    >>> plotter = plot_contours(mask, slice_mode="z", colors={1: "red", 2: "cyan"})

    >>> # Overlay contours on an existing volume plot.
    >>> from confusius.plotting import plot_volume
    >>> volume = xr.open_zarr("output.zarr")["power_doppler"]
    >>> plotter = plot_volume(volume, slice_mode="z")
    >>> plotter.add_contours(mask, colors="yellow")
    """
    plotter = VolumePlotter(
        slice_mode=slice_mode,
        figure=figure,
        axes=axes,
        bg_color=bg_color,
        fg_color=fg_color,
        yincrease=yincrease,
        xincrease=xincrease,
        transpose=transpose,
    )

    return plotter.add_contours(
        mask,
        colors=colors,
        linewidths=linewidths,
        linestyles=linestyles,
        match_coordinates=False,
        slice_coords=slice_coords,
        fontsize=fontsize,
        roi_labels=roi_labels,
        **kwargs,
    )


def plot_volume(
    data: xr.DataArray,
    *,
    slice_coords: list[Hashable] | None = None,
    slice_mode: str = "z",
    transpose: bool = False,
    cmap: "str | Colormap | None" = None,
    norm: "Normalize | None" = None,
    vmin: float | None = None,
    vmax: float | None = None,
    threshold: float | None = None,
    threshold_mode: Literal["lower", "upper"] = "lower",
    alpha: "float | xr.DataArray | None" = None,
    show_colorbar: bool = True,
    cbar_label: str | None = None,
    cbar_kwargs: "dict[str, Any] | None" = None,
    roi_labels: dict[int, str] | None = None,
    resample_in_plane: bool = False,
    show_titles: bool = True,
    show_axis_labels: bool = True,
    show_axis_ticks: bool = True,
    show_axes: bool = True,
    fontsize: float | None = None,
    yincrease: bool = False,
    xincrease: bool = True,
    bg_color: str = "black",
    fg_color: str | None = None,
    figure: "Figure | None" = None,
    axes: "npt.NDArray[Any] | Axes | None" = None,
    nrows: int | None = None,
    ncols: int | None = None,
    dpi: int | None = None,
    resample_interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    resample_fill_value: float | None = None,
) -> VolumePlotter:
    """Plot 2D slices of a volume using matplotlib.

    Displays a series of 2D slices extracted along `slice_mode` as a grid of subplots.
    Each slice is rendered using world coordinates for axis ticks when available.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array. Unitary dimensions are squeezed before processing. After
        squeezing, data must be 3D. Complex-valued data is converted to magnitude
        before display.
    slice_coords : list[collections.abc.Hashable], optional
        Coordinate values along `slice_mode` at which to extract slices. Numeric
        coordinates are matched by nearest-neighbour lookup; non-numeric
        coordinates (e.g. region labels) require an exact match. If not provided,
        all coordinate values along `slice_mode` are used.
    slice_mode : str, default: "z"
        Dimension along which to slice (e.g., `"x"`, `"y"`, `"z"`,
        `"time"`). After slicing, each panel must be 2D.
    transpose : bool, default: False
        Whether to swap the row/column display dims of each slice panel.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap. When not provided, falls back to `data.attrs["cmap"]` if
        present, otherwise `"gray"`.
    norm : matplotlib.colors.Normalize, optional
        Normalization instance (e.g. `BoundaryNorm` for integer label maps such
        as atlas annotations). When not provided, falls back to
        `data.attrs["norm"]` if present. When a norm is active, `vmin` and
        `vmax` are ignored.
    vmin : float, optional
        Lower bound of the colormap. Defaults to the 2nd percentile. Ignored
        when a norm is active.
    vmax : float, optional
        Upper bound of the colormap. Defaults to the 98th percentile. Ignored
        when a norm is active.
    threshold : float, optional
        Threshold applied to `|data|`. See `threshold_mode` for the masking
        direction. If not provided, no thresholding is applied.
    threshold_mode : {"lower", "upper"}, default: "lower"
        Controls how `threshold` is applied:

        - `"lower"`: set pixels where `|data| < threshold` to NaN.
        - `"upper"`: set pixels where `|data| > threshold` to NaN.

    alpha : float or xarray.DataArray, optional
        Opacity of the image: a single scalar value, or a 3D DataArray sharing
        `data`'s dims, shape, and coordinates (for independent per-slice,
        per-voxel opacity). A per-voxel opacity must be a DataArray, not a bare
        array, so it can be validated and aligned against `data`. If not provided,
        the colormap's own alpha channel is respected.
    show_colorbar : bool, default: True
        Whether to add a shared colorbar to the figure.
    cbar_label : str, optional
        Label for the colorbar.
    cbar_kwargs : dict, optional
        Additional keyword arguments forwarded to
        [`matplotlib.figure.Figure.colorbar`][matplotlib.figure.Figure.colorbar]
        (e.g. `shrink`, `fraction`, `pad`, `aspect`). Useful to shrink the colorbar
        when it spans a multi-panel grid, since the defaults are sized for a single
        axes.
    roi_labels : dict[int, str], optional
        Mapping from integer label to display name. When provided (or when
        `data.attrs["roi_labels"]` is populated), hovering the cursor over a
        voxel shows `<data_name>=<id> (<roi_name>)` in the matplotlib status bar.
    resample_in_plane : bool, default: False
        See [`VolumePlotter.add_volume`][confusius.plotting.VolumePlotter.add_volume]
        for details.
    show_titles : bool, default: True
        Whether to display subplot titles showing the slice coordinate.
    show_axis_labels : bool, default: True
        Whether to display axis labels (with units when available).
    show_axis_ticks : bool, default: True
        Whether to display axis tick labels.
    show_axes : bool, default: True
        Whether to show all axis decorations (spines, ticks, labels). When `False`,
        overrides `show_axis_labels` and `show_axis_ticks`.
    fontsize : float, optional
        Base font size for all text elements. Subplot titles use `fontsize` directly;
        axis labels and the colorbar label use `0.9 * fontsize`; tick labels use `0.85 *
        fontsize`. If not provided, uses the active Matplotlib defaults.
    yincrease : bool, default: False
        Whether the y-axis increases upward (`True`) or downward (`False`).
    xincrease : bool, default: True
        Whether the x-axis increases to the right (`True`) or left (`False`).
    bg_color : str, default: "black"
        Background color for the figure and axes. Any matplotlib-compatible color
        string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula
        (white on dark backgrounds, black on light ones).
    figure : matplotlib.figure.Figure, optional
        Existing figure to draw into. If not provided, a new figure is created.
    axes : numpy.ndarray or matplotlib.axes.Axes, optional
        Existing axes to draw into: either a single
        [`matplotlib.axes.Axes`][matplotlib.axes.Axes] or a 2D array of them. Must
        contain exactly as many elements as there are slices. A single `Axes` is
        wrapped automatically and limits the plot to one slice. If not provided, new
        axes are created inside `figure`.
    nrows : int, optional
        Number of rows in the subplot grid. If not provided, computed automatically.
    ncols : int, optional
        Number of columns in the subplot grid. If not provided, computed automatically.
    dpi : int, optional
        Figure resolution in dots per inch. Ignored when `figure` is provided.
    roi_labels : dict[int, str], optional
        Mapping from integer label to display name. When provided (or when
        `data.attrs["roi_labels"]` is populated), hovering the cursor over a
        voxel shows `<data_name>=<id> (<roi_name>)` in the matplotlib status bar.
    resample_interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used when resampling oblique (non-axis-aligned)
        voxel-to-world `data` onto an axis-aligned world grid for display.
    resample_fill_value : float, optional
        Value assigned to voxels outside `data`'s field of view after resampling
        oblique data. If not provided, defaults to `data`'s own minimum value.

    Returns
    -------
    VolumePlotter
        Object managing the figure, axes, and coordinate mapping for overlays.

    Raises
    ------
    ValueError
        If `slice_mode` is not a dimension of `data`.
    ValueError
        If `data` is not 3D after squeezing unitary dimensions.
    ValueError
        If `axes` is provided but does not contain enough elements for all slices.

    Notes
    -----
    Rendering is done with [`pcolormesh`][matplotlib.pyplot.pcolormesh], which accepts
    coordinate arrays directly and therefore handles non-uniform coordinate spacing
    correctly. Because each panel is drawn in world coordinate space, multiple calls
    with different `axes` elements will overlay correctly as long as the displayed
    dimensions are the same.

    The two dimensions that remain after slicing define the panel axes: the
    first remaining dimension maps to the vertical axis and the second to the
    horizontal axis. Coordinates are used directly as axis tick values; each
    axis has `aspect="equal"` so that 1 unit in x matches 1 unit in y.

    NaN and Inf values (including those introduced by `threshold`) are rendered
    transparently via a masked array.

    When the figure is created internally, `layout="constrained"` is used so
    that subplot titles, axis labels, tick labels, and the colorbar are spaced
    automatically without overlapping. When an external `figure` or `axes`
    is provided, layout management is left to the caller.

    Examples
    --------
    >>> import xarray as xr
    >>> from confusius.plotting import plot_volume
    >>> data = xr.open_zarr("output.zarr")["power_doppler"]
    >>> plotter = plot_volume(data, slice_mode="z")

    >>> # Select specific z slices.
    >>> plotter = plot_volume(data, slice_coords=[0.0, 1.5, 3.0], slice_mode="z")

    >>> # Threshold noise and label the colorbar.
    >>> plotter = plot_volume(
    ...     data,
    ...     slice_mode="z",
    ...     threshold=0.5,
    ...     threshold_mode="lower",
    ...     cbar_label="Power (dB)",
    ... )
    """
    plotter = VolumePlotter(
        slice_mode=slice_mode,
        figure=figure,
        axes=axes,
        bg_color=bg_color,
        fg_color=fg_color,
        yincrease=yincrease,
        xincrease=xincrease,
        resample_interpolation=resample_interpolation,
        resample_fill_value=resample_fill_value,
        transpose=transpose,
    )

    return plotter.add_volume(
        data=data,
        slice_coords=slice_coords,
        match_coordinates=False,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        threshold_mode=threshold_mode,
        alpha=alpha,
        show_colorbar=show_colorbar,
        cbar_label=cbar_label,
        cbar_kwargs=cbar_kwargs,
        show_titles=show_titles,
        show_axis_labels=show_axis_labels,
        show_axis_ticks=show_axis_ticks,
        show_axes=show_axes,
        fontsize=fontsize,
        nrows=nrows,
        ncols=ncols,
        dpi=dpi,
        roi_labels=roi_labels,
        resample_in_plane=resample_in_plane,
    )


def plot_composite(
    data1: xr.DataArray,
    data2: xr.DataArray,
    *,
    resample: bool = True,
    resample_kwargs: "dict[str, Any] | None" = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    normalize_strategy: Literal["per_volume", "per_slice", "shared"] = "per_volume",
    slice_coords: Sequence[Hashable] | None = None,
    slice_mode: str = "z",
    transpose: bool = False,
    alpha: "float | npt.NDArray[np.floating] | None" = None,
    show_titles: bool = True,
    show_axis_labels: bool = True,
    show_axis_ticks: bool = True,
    show_axes: bool = True,
    fontsize: float | None = None,
    yincrease: bool = False,
    xincrease: bool = True,
    bg_color: str = "black",
    fg_color: str | None = None,
    figure: "Figure | None" = None,
    axes: "npt.NDArray[Any] | Axes | None" = None,
    nrows: int | None = None,
    ncols: int | None = None,
    dpi: int | None = None,
    resample_interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    resample_fill_value: float | None = None,
) -> VolumePlotter:
    """Plot a red/cyan composite of two volumes as a grid of 2D slice panels.

    Each slice is rendered as an RGB image where `data1` drives the red channel
    and `data2` drives the green and blue channels (cyan), making overlap
    visible as desaturated grey. This is the same visual encoding used by the
    live registration progress preview.

    Parameters
    ----------
    data1 : xarray.DataArray
        First volume, plotted in red. 3D volume data. Unitary dimensions (except
        `slice_mode`) are squeezed before processing. Complex-valued inputs are
        converted to magnitude (`abs(data)`) with a warning.
    data2 : xarray.DataArray
        Second volume, plotted in cyan. Must have the same dimensionality as `data1`
        after squeezing; when `resample=True` it is resampled onto `data1`'s grid before
        plotting, so its native shape and coordinates may differ.
    resample : bool, default: True
        Whether to resample `data2` onto `data1`'s grid using an identity transform
        before blending. When `False`, the two arrays must already share dimensions and
        shape, and their coordinates must match within `rtol`/`atol`; once validated,
        `data2`'s coordinates are replaced with `data1`'s so the two volumes share an
        exact coordinate frame downstream.
    resample_kwargs : dict, optional
        Extra keyword arguments forwarded to
        [`resample_like`][confusius.registration.resample_like] when `resample=True`.
        Ignored when `resample=False`.
    rtol : float, default: 1e-5
        Relative tolerance used to validate that `data1` and `data2` share coordinates
        when `resample=False`. Widen to accept acquisitions on slightly offset grids
        known to be equivalent. Ignored when `resample=True`.
    atol : float, default: 1e-8
        Absolute tolerance used to validate that `data1` and `data2` share coordinates
        when `resample=False`. Ignored when `resample=True`.
    normalize_strategy : {"per_volume", "per_slice", "shared"}, default: "per_volume"
        Intensity normalisation strategy.

        - `"per_volume"`: rescale each input to `[0, 1]` independently over its full volume.
          Preserves slice-to-slice contrast within each array but loses the
          absolute-intensity relationship between `data1` and `data2`.
        - `"per_slice"`: rescale each 2D slice independently. Maximises contrast on dim
          slices at the cost of cross-slice comparability.
        - `"shared"`: rescale both volumes together using a single shared
          `[min(data1.min(), data2.min()), max(data1.max(), data2.max())]` range.
          Preserves the absolute-intensity relationship between the two inputs.
    slice_coords : Sequence[collections.abc.Hashable], optional
        Coordinate values along `slice_mode` at which to extract slices. Numeric
        coordinates are matched by nearest-neighbour lookup; non-numeric
        coordinates (e.g. region labels) require an exact match. If not provided,
        from `data1` are used.
    slice_mode : str, default: "z"
        Dimension along which to slice (e.g. `"x"`, `"y"`, `"z"`). After slicing, each
        panel must be 2D.
    transpose : bool, default: False
        Whether to swap the row/column display dims of each slice panel.
    alpha : float or numpy.ndarray, optional
        Opacity of the composite image, either a single value or a per-voxel array
        matching the shape of the displayed slices. If not provided, the image is
        fully opaque.
    show_titles : bool, default: True
        Whether to display subplot titles showing the slice coordinate.
    show_axis_labels : bool, default: True
        Whether to display axis labels (with units when available).
    show_axis_ticks : bool, default: True
        Whether to display axis tick labels.
    show_axes : bool, default: True
        Whether to show all axis decorations. When `False`, overrides `show_axis_labels`
        and `show_axis_ticks`.
    fontsize : float, optional
        Base font size for all text elements. Subplot titles use `fontsize` directly;
        axis labels use `0.9 * fontsize`; tick labels use `0.85 * fontsize`. If not
        provided, uses the active Matplotlib defaults.
    yincrease : bool, default: False
        Whether the y-axis increases upward (`True`) or downward (`False`).
    xincrease : bool, default: True
        Whether the x-axis increases to the right (`True`) or left (`False`).
    bg_color : str, default: "black"
        Background color for the figure and axes. Any matplotlib-compatible color string
        (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula (white
        on dark backgrounds, black on light ones).
    figure : matplotlib.figure.Figure, optional
        Existing figure to draw into. If not provided, a new figure is created.
    axes : numpy.ndarray or matplotlib.axes.Axes, optional
        Existing axes to draw into: either a single
        [`matplotlib.axes.Axes`][matplotlib.axes.Axes] or a 2D array of them. Must
        contain exactly as many elements as there are slices. A single `Axes` is wrapped
        automatically. If not provided, new axes are created inside `figure`.
    nrows : int, optional
        Number of rows in the subplot grid. If not provided, computed automatically.
    ncols : int, optional
        Number of columns in the subplot grid. If not provided, computed automatically.
    dpi : int, optional
        Figure resolution in dots per inch. Ignored when `figure` is provided.
    resample_interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used when resampling oblique (non-axis-aligned)
        voxel-to-world `data1`/`data2` onto an axis-aligned world grid for display.
        Distinct from `resample_kwargs`, which controls resampling `data2` onto
        `data1`'s grid for compositing.
    resample_fill_value : float, optional
        Value assigned to voxels outside `data1`/`data2`'s field of view after
        display resampling. If not provided, defaults to each array's own minimum
        value.

    Returns
    -------
    VolumePlotter
        Object managing the figure, axes, and coordinate mapping for overlays.

    Raises
    ------
    ValueError
        If either input has a `time` dimension, is not 2D or 3D, lacks `slice_mode` as a
        dimension, or (when `resample=False`) the two arrays do not share dims, shape,
        and coordinates within `rtol`/`atol`.

    Notes
    -----
    Rendering uses [`pcolormesh`][matplotlib.axes.Axes.pcolormesh] with an RGB `C`
    array, so panels share their cell geometry with
    [`plot_volume`][confusius.plotting.plot_volume] /
    [`plot_contours`][confusius.plotting.plot_contours] and overlay correctly.
    Colormaps, colorbars, intensity thresholds, and hover tooltips are not supported on
    composite axes — use `plot_volume` for those.

    The returned [`VolumePlotter`][confusius.plotting.VolumePlotter] stores the
    coordinate-to-axis mapping, so you can overlay further volumes or contours with
    [`VolumePlotter.add_volume`][confusius.plotting.VolumePlotter.add_volume] or
    [`VolumePlotter.add_contours`][confusius.plotting.VolumePlotter.add_contours].

    Examples
    --------
    >>> import xarray as xr
    >>> from confusius.plotting import plot_composite
    >>> fixed = xr.open_zarr("fixed.zarr")["power_doppler"]
    >>> moving = xr.open_zarr("moving.zarr")["power_doppler"]
    >>> plotter = plot_composite(fixed, moving, slice_mode="z")

    >>> # Skip resampling when the two volumes are already aligned.
    >>> plotter = plot_composite(fixed, registered_moving, resample=False)

    >>> # Maximise contrast on dim slices.
    >>> plotter = plot_composite(fixed, moving, normalize_strategy="per_slice")
    """
    plotter = VolumePlotter(
        slice_mode=slice_mode,
        figure=figure,
        axes=axes,
        bg_color=bg_color,
        fg_color=fg_color,
        yincrease=yincrease,
        xincrease=xincrease,
        resample_interpolation=resample_interpolation,
        resample_fill_value=resample_fill_value,
        transpose=transpose,
    )

    return plotter.add_composite(
        data1,
        data2,
        resample=resample,
        resample_kwargs=resample_kwargs,
        rtol=rtol,
        atol=atol,
        normalize_strategy=normalize_strategy,
        slice_coords=slice_coords,
        match_coordinates=False,
        alpha=alpha,
        show_titles=show_titles,
        show_axis_labels=show_axis_labels,
        show_axis_ticks=show_axis_ticks,
        show_axes=show_axes,
        fontsize=fontsize,
        nrows=nrows,
        ncols=ncols,
        dpi=dpi,
    )


def plot_stat_map(
    stat_map: xr.DataArray,
    *,
    bg_volume: xr.DataArray | None = None,
    slice_coords: list[Hashable] | None = None,
    slice_mode: str = "z",
    transpose: bool = False,
    bg_kwargs: "dict[str, Any] | None" = None,
    cmap: "str | Colormap | None" = None,
    norm: "Normalize | None" = None,
    vmin: float | None = None,
    vmax: float | None = None,
    auto_range: bool = True,
    alpha: "float | xr.DataArray | None" = None,
    threshold: float | None = None,
    threshold_mode: Literal["lower", "upper"] = "lower",
    show_colorbar: bool = True,
    cbar_label: str | None = None,
    cbar_kwargs: "dict[str, Any] | None" = None,
    show_titles: bool = True,
    show_axis_labels: bool = True,
    show_axis_ticks: bool = True,
    show_axes: bool = True,
    fontsize: float | None = None,
    yincrease: bool = False,
    xincrease: bool = True,
    bg_color: str = "black",
    fg_color: str | None = None,
    figure: "Figure | None" = None,
    axes: "npt.NDArray[Any] | Axes | None" = None,
    nrows: int | None = None,
    ncols: int | None = None,
    dpi: int | None = None,
    resample_interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    resample_fill_value: float | None = None,
    resample_in_plane: bool = False,
) -> VolumePlotter:
    """Plot a statistical map, optionally over a background volume.

    Performs the recurring pattern of [`plot_volume`][confusius.plotting.plot_volume] to
    show a background anatomical volume +
    [`VolumePlotter.add_volume`][confusius.plotting.VolumePlotter.add_volume] to overlay
    a statistical map, with the colormap and range picked automatically based on
    whether the statistic is diverging (has both positive and negative values) or
    one-signed.

    Parameters
    ----------
    stat_map : xarray.DataArray
        Statistical map to plot. 3D volume data. Unitary dimensions (except
        `slice_mode`) are squeezed before processing.
    bg_volume : xarray.DataArray, optional
        Background anatomical volume, plotted underneath `stat_map`. When `alpha` is
        not provided, `stat_map` fully covers `bg_volume` wherever it has a value;
        `bg_volume` only shows through where `stat_map` is masked out by `threshold`.
        Lower `alpha` to blend the two layers instead. Must share `slice_mode` and,
        after squeezing, the same display dimensions as `stat_map`. If not provided,
        `stat_map` is plotted on its own.
    slice_coords : list[collections.abc.Hashable], optional
        Coordinate values along `slice_mode` at which to extract slices. Numeric
        coordinates are matched by nearest-neighbour lookup; non-numeric
        coordinates (e.g. region labels) require an exact match. If not provided,
        all coordinate values from `bg_volume` (or `stat_map` when `bg_volume` is not
        provided) along `slice_mode` are used.
    slice_mode : str, default: "z"
        Dimension along which to slice (e.g., `"x"`, `"y"`, `"z"`, `"time"`). After
        slicing, each panel must be 2D.
    transpose : bool, default: False
        Whether to swap the row/column display dims of each slice panel.
    bg_kwargs : dict, optional
        Additional keyword arguments forwarded to
        [`plot_volume`][confusius.plotting.plot_volume] for the background layer
        (e.g. `cmap`, `vmin`, `vmax`, `norm`, `alpha`, `roi_labels`). Ignored when
        `bg_volume` is not provided. Layout and text styling (`slice_coords`,
        `slice_mode`, `show_titles`, `fontsize`, etc.) are controlled by this
        function's own parameters instead, so that both layers share consistent
        styling.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for `stat_map`. If not provided, the default depends on
        `auto_range` and the sign of `stat_map` (see below); an explicit `cmap` is
        always used as-is regardless of `auto_range`.
    norm : matplotlib.colors.Normalize, optional
        Normalization instance (e.g. `TwoSlopeNorm`, `BoundaryNorm`, `LogNorm`) for
        cases `vmin`/`vmax`/`auto_range` can't express. When provided, `vmin`,
        `vmax`, and `auto_range`'s range computation are bypassed entirely; `cmap`
        still follows the usual rules above.
    vmin : float, optional
        Lower bound of the colormap. If not provided, defaults to the minimum value
        of `stat_map`, computed over the full array rather than just the displayed
        slices. Ignored when `norm` is provided, or when `auto_range` resolves to a
        range anchored at zero (see below).
    vmax : float, optional
        Upper bound of the colormap. If not provided, defaults to the maximum value
        of `stat_map`, computed over the full array rather than just the displayed
        slices. Ignored when `norm` is provided, or when `auto_range=True` and
        `stat_map` has only non-positive values.
    auto_range : bool, default: True
        Whether to pick the colormap range and default colormap automatically based
        on the sign of `stat_map`:

        - Both positive and negative values: diverging, symmetric `[-m, m]` range
          where `m = max(|vmin|, |vmax|)` (using the resolved bounds above),
          with `cmap` defaulting to `"coolwarm"` — the right choice for diverging
          statistics where the sign is meaningful (e.g. t-statistics, correlation
          coefficients, PCA/ICA component maps).
        - Only non-negative values: sequential `[0, vmax]` range, with `cmap`
          defaulting to `"viridis"` — the right choice for non-diverging
          statistics where only magnitude matters (e.g. R², F-statistics).
        - Only non-positive values: sequential `[vmin, 0]` range, with `cmap`
          defaulting to `"viridis_r"` (reversed, so that values near zero map to
          the same end of the colormap in both the non-negative and
          non-positive cases).

        Set to `False` to use the resolved `vmin`/`vmax` directly with no
        zero-anchoring (`cmap` then defaults to `"coolwarm"` regardless of sign).
    alpha : float or xarray.DataArray, optional
        Opacity of the `stat_map` overlay: a single scalar value, or a 3D DataArray
        sharing `stat_map`'s dims, shape, and coordinates (for independent per-slice,
        per-voxel opacity, e.g. to fade out low-magnitude voxels instead of masking them
        out with `threshold`). A per-voxel opacity must be a DataArray, not a bare
        array, so it can be validated and aligned against `stat_map`; note it is
        validated against `stat_map`, not `bg_volume`. If not provided, the colormap's
        own alpha channel is respected.
    threshold : float, optional
        Threshold applied to `|stat_map|`. See `threshold_mode` for the masking
        direction. If not provided, no thresholding is applied.
    threshold_mode : {"lower", "upper"}, default: "lower"
        Controls how `threshold` is applied:

        - `"lower"`: set pixels where `|stat_map| < threshold` to NaN.
        - `"upper"`: set pixels where `|stat_map| > threshold` to NaN.

    show_colorbar : bool, default: True
        Whether to add a shared colorbar for `stat_map` to the figure.
    cbar_label : str, optional
        Label for the colorbar.
    cbar_kwargs : dict, optional
        Additional keyword arguments forwarded to
        [`matplotlib.figure.Figure.colorbar`][matplotlib.figure.Figure.colorbar]
        (e.g. `shrink`, `fraction`, `pad`, `aspect`). Useful to shrink the colorbar
        when it spans a multi-panel grid, since the defaults are sized for a single
        axes.
    show_titles : bool, default: True
        Whether to display subplot titles showing the slice coordinate.
    show_axis_labels : bool, default: True
        Whether to display axis labels (with units when available).
    show_axis_ticks : bool, default: True
        Whether to display axis tick labels.
    show_axes : bool, default: True
        Whether to show all axis decorations (spines, ticks, labels). When `False`,
        overrides `show_axis_labels` and `show_axis_ticks`.
    fontsize : float, optional
        Base font size for all text elements. Subplot titles use `fontsize` directly;
        axis labels and the colorbar label use `0.9 * fontsize`; tick labels use `0.85 *
        fontsize`. If not provided, uses the active Matplotlib defaults.
    yincrease : bool, default: False
        Whether the y-axis increases upward (`True`) or downward (`False`).
    xincrease : bool, default: True
        Whether the x-axis increases to the right (`True`) or left (`False`).
    bg_color : str, default: "black"
        Background color for the figure and axes. Any matplotlib-compatible color
        string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula
        (white on dark backgrounds, black on light ones).
    figure : matplotlib.figure.Figure, optional
        Existing figure to draw into. If not provided, a new figure is created.
    axes : numpy.ndarray or matplotlib.axes.Axes, optional
        Existing axes to draw into: either a single
        [`matplotlib.axes.Axes`][matplotlib.axes.Axes] or a 2D array of them. Must
        contain exactly as many elements as there are slices. A single `Axes` is
        wrapped automatically and limits the plot to one slice. If not provided, new
        axes are created inside `figure`.
    nrows : int, optional
        Number of rows in the subplot grid. If not provided, computed automatically.
    ncols : int, optional
        Number of columns in the subplot grid. If not provided, computed automatically.
    dpi : int, optional
        Figure resolution in dots per inch. Ignored when `figure` is provided.
    resample_interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used when resampling oblique (non-axis-aligned)
        voxel-to-world `stat_map`/`bg_volume` onto an axis-aligned world grid for
        display. Applied to both, since they share one `VolumePlotter`.
    resample_fill_value : float, optional
        Value assigned to voxels outside `stat_map`/`bg_volume`'s field of view
        after display resampling. If not provided, defaults to each array's own
        minimum value.
    resample_in_plane : bool, default: False
        See [`VolumePlotter.add_volume`][confusius.plotting.VolumePlotter.add_volume]
        for details. Applies to `stat_map`'s own overlay only -- pass it to
        `bg_volume` via `bg_kwargs={"resample_in_plane": True}` instead.

    Returns
    -------
    VolumePlotter
        Object managing the figure, axes, and coordinate mapping for overlays.

    Raises
    ------
    ValueError
        If `slice_mode` is not a dimension of `stat_map` or `bg_volume`.
    ValueError
        If `stat_map` or `bg_volume` is not 3D after squeezing unitary dimensions.

    Notes
    -----
    When `bg_volume` is provided, this is equivalent to calling
    `plot_volume(bg_volume, ...)` followed by `plotter.add_volume(stat_map,
    alpha=alpha, cmap=resolved_cmap, vmin=resolved_vmin, vmax=resolved_vmax, ...)`.
    Use those functions directly for finer control, e.g. a custom, non-zero-anchored
    asymmetric range.

    Examples
    --------
    >>> import xarray as xr
    >>> from confusius.plotting import plot_stat_map
    >>> anatomical = xr.open_zarr("output.zarr")["power_doppler"]
    >>> t_map = xr.open_zarr("output.zarr")["t_stat"]
    >>> plotter = plot_stat_map(t_map, bg_volume=anatomical, slice_mode="z")

    >>> # Suppress subthreshold voxels and cap the colormap range explicitly.
    >>> plotter = plot_stat_map(
    ...     t_map,
    ...     bg_volume=anatomical,
    ...     threshold=3.0,
    ...     vmax=6.0,
    ...     cbar_label="t-statistic",
    ... )

    >>> # Blend the overlay with the background instead of fully covering it.
    >>> plotter = plot_stat_map(t_map, bg_volume=anatomical, alpha=0.6)

    >>> # Non-diverging statistic (e.g. R²): sequential range and colormap picked
    >>> # automatically since r2_map has only non-negative values.
    >>> r2_map = xr.open_zarr("output.zarr")["r2"]
    >>> plotter = plot_stat_map(r2_map, bg_volume=anatomical)

    >>> # No background: plot the statistical map on its own.
    >>> plotter = plot_stat_map(t_map, slice_mode="z")
    """
    # `_resolve_stat_map_style` below reads `stat_map.values` directly, ahead of
    # `add_volume`/`_prepare_slice_inputs`'s own `.compute()` -- compute once here
    # so a dask-backed `stat_map` isn't recomputed twice (once for style
    # resolution, once inside `add_volume`).
    stat_map = stat_map.compute()
    if bg_volume is not None:
        plotter = plot_volume(
            bg_volume,
            slice_coords=slice_coords,
            slice_mode=slice_mode,
            show_colorbar=False,
            show_titles=show_titles,
            show_axis_labels=show_axis_labels,
            show_axis_ticks=show_axis_ticks,
            show_axes=show_axes,
            fontsize=fontsize,
            yincrease=yincrease,
            xincrease=xincrease,
            bg_color=bg_color,
            fg_color=fg_color,
            figure=figure,
            axes=axes,
            nrows=nrows,
            ncols=ncols,
            dpi=dpi,
            resample_interpolation=resample_interpolation,
            resample_fill_value=resample_fill_value,
            transpose=transpose,
            **(bg_kwargs or {}),
        )
    else:
        plotter = VolumePlotter(
            slice_mode=slice_mode,
            figure=figure,
            axes=axes,
            bg_color=bg_color,
            fg_color=fg_color,
            yincrease=yincrease,
            xincrease=xincrease,
            resample_interpolation=resample_interpolation,
            resample_fill_value=resample_fill_value,
            transpose=transpose,
        )

    resolved_vmin, resolved_vmax, resolved_cmap = _resolve_stat_map_style(
        stat_map, vmin, vmax, cmap, auto_range
    )

    return plotter.add_volume(
        stat_map,
        slice_coords=slice_coords,
        match_coordinates=bg_volume is not None,
        cmap=resolved_cmap,
        norm=norm,
        vmin=resolved_vmin,
        vmax=resolved_vmax,
        threshold=threshold,
        threshold_mode=threshold_mode,
        alpha=alpha,
        show_colorbar=show_colorbar,
        cbar_label=cbar_label,
        cbar_kwargs=cbar_kwargs,
        show_titles=show_titles,
        show_axis_labels=show_axis_labels,
        show_axis_ticks=show_axis_ticks,
        show_axes=show_axes,
        fontsize=fontsize,
        nrows=nrows,
        ncols=ncols,
        dpi=dpi,
        resample_in_plane=resample_in_plane,
    )


def _prepare_carpet_data(
    data: xr.DataArray,
    mask: xr.DataArray | None = None,
    detrend_order: int | None = None,
    standardize: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    decimation_threshold: int | None = 800,
) -> dict:
    """Prepare carpet plot data, separating expensive computation from drawing.

    Intended to run in a background thread; the result is passed to `plot_carpet` via
    its `_precomputed` keyword.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array with a `"time"` dimension and coordinate: a VoxelData
        array, or an already-extracted signals array.
    mask : xarray.DataArray, optional
        Boolean mask to select elements. Defaults to all non-zero elements.
    detrend_order : int, optional
        Polynomial order for detrending. See `plot_carpet`.
    standardize : bool, default: True
        Whether to z-score each voxel signals.
    vmin : float, optional
        Lower colormap bound. Computed from data when `None`.
    vmax : float, optional
        Upper colormap bound. Computed from data when `None`.
    decimation_threshold : int or None, default: 800
        Downsample time axis when the number of frames exceeds this value.

    Returns
    -------
    dict
        Keys: `signals` (DataArray with shape `(time, space)`), `vmin` (float),
        `vmax` (float), `xlabel` (str), `time_coord` (DataArray | None).
    """
    if np.iscomplexobj(data):
        data = xr.ufuncs.abs(data)

    validate_time_series(data, "plot_carpet", require_unchunked_time=False)

    n_timepoints = data.sizes["time"]

    non_zero = (data != 0).any(dim="time")
    if mask is None:
        mask = non_zero
    else:
        mask = mask & non_zero

    signals = select_masked_features(data, mask)

    # Carpet plots don't need spatial coordinates, and multi-index coordinates will make
    # plotting fail.
    space_coords = [c for c in signals.coords if "space" in signals.coords[c].dims]
    signals = signals.drop_vars(space_coords).assign_coords(
        space=np.arange(signals.sizes["space"])
    )

    signals = clean(
        signals,
        detrend_order=detrend_order,
        standardize_method="zscore" if standardize else None,
    )

    if vmin is None or vmax is None:
        std_val = float(signals.std(axis=0).mean().values)
        default_vmin = float(signals.mean().values - (2 * std_val))
        default_vmax = float(signals.mean().values + (2 * std_val))
        vmin = vmin or default_vmin
        vmax = vmax or default_vmax

    if decimation_threshold is not None and n_timepoints > decimation_threshold:
        n_decimations = int(
            np.ceil(np.log2(np.ceil(n_timepoints / decimation_threshold)))
        )
        decimation_factor = 2**n_decimations
        signals = signals[::decimation_factor, :]

    return {
        "signals": signals,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "xlabel": _build_axis_label(data, "time").capitalize(),
        "time_coord": data.coords.get("time"),
    }


def _draw_carpet(
    prep: dict,
    cmap: "str | Colormap" = "gray",
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
    fontsize: float | None = None,
    bg_color: str = "white",
    fg_color: str | None = None,
    ax: "Axes | None" = None,
) -> tuple["Figure | SubFigure", "Axes"]:
    """Draw a carpet plot from pre-computed data.

    Low-level drawing counterpart of `_prepare_carpet_data`. Intended to run on the main
    thread after the expensive data preparation has been done in a background thread.

    Parameters
    ----------
    prep : dict
        Pre-computed dict returned by `_prepare_carpet_data`.
    cmap : str, default: `"gray"`
        Matplotlib colormap name.
    figsize : tuple[float, float], default: (10, 5)
        Figure size in inches, used only when *ax* is `None`.
    title : str, optional
        Plot title.
    fontsize : float, optional
        Base font size for text elements. Title uses `fontsize` directly; axis labels
        and colorbar label use `0.9 * fontsize`; tick labels use `0.85 * fontsize`. If
        not provided, uses the active Matplotlib defaults.
    bg_color : str, default: "white"
        Background color for the figure and axes. Any matplotlib-compatible color
        string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when `None`.

    Returns
    -------
    figure : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        Figure containing the carpet plot.
    axes : matplotlib.axes.Axes
        Axes with the carpet plot.
    """
    import matplotlib.pyplot as plt

    signals = prep["signals"]
    vmin = prep["vmin"]
    vmax = prep["vmax"]
    xlabel = prep["xlabel"]

    text_color = fg_color if fg_color is not None else _auto_fg_color(bg_color)
    title_fontsize, label_fontsize, tick_fontsize = _resolve_font_sizes(fontsize)

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
        figure.patch.set_facecolor(bg_color)
    else:
        figure = ax.figure

    ax.set_facecolor(bg_color)

    plotted_quadmesh = signals.T.plot(
        cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, yincrease=False
    )

    if plotted_quadmesh.colorbar is not None:
        cbar = plotted_quadmesh.colorbar
        _style_colorbar(
            cbar,
            text_color,
            tick_fontsize,
            bg_color=bg_color,
            label_fontsize=label_fontsize,
        )

    ax.grid(False)
    ax.set_yticks([])
    ax.set_ylabel("Voxels", color=text_color, fontsize=label_fontsize)
    ax.set_xlabel(xlabel, color=text_color, fontsize=label_fontsize)
    ax.tick_params(colors=text_color, labelsize=tick_fontsize)

    if title:
        ax.set_title(title, color=text_color, fontsize=title_fontsize)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_edgecolor(text_color)
    ax.spines["left"].set_edgecolor(text_color)

    return figure, ax


def plot_carpet(
    data: xr.DataArray,
    mask: xr.DataArray | None = None,
    detrend_order: int | None = None,
    standardize: bool = True,
    cmap: "str | Colormap" = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    decimation_threshold: int | None = 800,
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
    fontsize: float | None = None,
    bg_color: str = "white",
    fg_color: str | None = None,
    ax: "Axes | None" = None,
) -> tuple["Figure | SubFigure", "Axes"]:
    """Plot voxel intensities across time as a raster image.

    A carpet plot (also known as "grayplot" or "Power plot") displays voxel
    intensities as a 2D raster image with time on the x-axis and voxels on
    the y-axis. Each row represents one voxel's signals, typically
    standardized to z-scores.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array with a `time` dimension: a VoxelData array, or
        an already-extracted signals array (e.g.
        [`extract_with_labels`][confusius.extract.extract_with_labels] output).
    mask : xarray.DataArray, optional
        Boolean mask with the same non-`time` dimensions and coordinates as `data`.
        True values indicate elements to include. If not provided, all non-zero
        elements from the data are included.
    detrend_order : int, optional
        Polynomial order for detrending:

        - `0`: Remove mean (constant detrending).
        - `1`: Remove linear trend using least squares regression.
        - `2+`: Remove polynomial trend of specified order.

        If not provided, no detrending is applied.
    standardize : bool, default: True
        Whether to standardize each voxel's signals to z-scores.
    cmap : str, default: "gray"
        Matplotlib colormap name.
    vmin : float, optional
        Minimum value for colormap. If not provided, uses `mean - 2*std`.
    vmax : float, optional
        Maximum value for colormap. If not provided, uses `mean + 2*std`.
    decimation_threshold : int or None, default: 800
        If the number of timepoints exceeds this value, data is downsampled
        along the time axis to improve plotting performance. Set to `None` to
        disable downsampling.
    figsize : tuple[float, float], default: (10, 5)
        Figure size in inches `(width, height)`.
    title : str, optional
        Plot title.
    fontsize : float, optional
        Base font size for text elements. Title uses `fontsize` directly; axis labels
        and colorbar label use `0.9 * fontsize`; tick labels use `0.85 * fontsize`. If
        not provided, uses the active Matplotlib defaults.
    bg_color : str, default: "white"
        Background color for the figure and axes. Any matplotlib-compatible color
        string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula
        (white on dark backgrounds, black on light ones).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If not provided, creates new figure and axes.

    Returns
    -------
    figure : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        Figure object containing the carpet plot.
    axes : matplotlib.axes.Axes
        Axes object with the carpet plot.

    Notes
    -----
    Complex-valued data is converted to magnitude before processing.

    This function was inspired by Nilearn's `nilearn.plotting.plot_carpet`.

    References
    ----------
    [^1]:
        Power, Jonathan D. “A Simple but Useful Way to Assess fMRI Scan Qualities.”
        NeuroImage, vol. 154, July 2017, pp. 150–58. DOI.org (Crossref),
        <https://doi.org/10.1016/j.neuroimage.2016.08.009>.

    Examples
    --------
    >>> import xarray as xr
    >>> from confusius.plotting import plot_carpet
    >>> data = xr.open_zarr("output.zarr")["iq"]
    >>> fig, ax = plot_carpet(data)

    >>> # With linear detrending
    >>> fig, ax = plot_carpet(data, detrend_order=1)

    >>> # With mask
    >>> import numpy as np
    >>> mask = xr.DataArray(
    ...     np.abs(data.isel(time=0)) > threshold,
    ...     dims=["z", "y", "x"],
    ... )
    >>> fig, ax = plot_carpet(data, mask=mask)
    """
    prep = _prepare_carpet_data(
        data, mask, detrend_order, standardize, vmin, vmax, decimation_threshold
    )
    return _draw_carpet(
        prep,
        cmap=cmap,
        figsize=figsize,
        title=title,
        fontsize=fontsize,
        bg_color=bg_color,
        fg_color=fg_color,
        ax=ax,
    )
