"""Helpers shared between matplotlib- and napari-based plotting code."""

import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._dims import SPATIAL_DIMS, VOXEL_DIMS
from confusius._utils.geometry import (
    get_voxel_to_world_affine,
    get_voxel_to_world_coord_names,
    get_voxel_to_world_index_spacing,
    get_voxel_to_world_spatial_dims,
    has_axis_aligned_voxel_to_world_index,
    has_voxel_to_world_index,
    require_scalar_pose_affine,
)
from confusius._utils.stack import find_stack_level
from confusius.validation import ensure_voxeldata

if TYPE_CHECKING:
    from matplotlib.colorbar import Colorbar


def _relative_luminance(color: str) -> float:
    """Compute WCAG 2.1 relative luminance for any matplotlib color string.

    Parameters
    ----------
    color : str
        Any matplotlib-compatible color string (e.g. `"black"`, `"#1a1a2e"`).

    Returns
    -------
    float
        Relative luminance in [0, 1], where 0 is darkest and 1 is lightest.

    Notes
    -----
    Implements the WCAG 2.1 relative luminance definition:
    https://www.w3.org/TR/WCAG21/#dfn-relative-luminance
    """
    import matplotlib.colors as mcolors

    def _linearize(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = mcolors.to_rgb(color)
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def _auto_fg_color(bg_color: str) -> str:
    """Return white or black for maximum WCAG contrast against `bg_color`.

    Parameters
    ----------
    bg_color : str
        Any matplotlib-compatible background color string.

    Returns
    -------
    str
        `"white"` when the background is dark (relative luminance < 0.179),
        `"black"` otherwise.
    """
    return "white" if _relative_luminance(bg_color) < 0.179 else "black"


def _resolve_font_sizes(
    fontsize: float | None,
) -> tuple[float | None, float | None, float | None]:
    """Resolve title, label, and tick font sizes from a base size.

    Parameters
    ----------
    fontsize : float, optional
        Base font size for plot text elements.

    Returns
    -------
    title_fontsize : float, optional
        Font size for subplot titles.
    label_fontsize : float, optional
        Font size for axis and colorbar labels.
    tick_fontsize : float, optional
        Font size for tick labels.
    """
    if fontsize is None:
        return None, None, None
    return fontsize, fontsize * 0.9, fontsize * 0.85


def _get_distinct_colors(n_colors: int) -> list[tuple[float, float, float]]:
    """Generate `n_colors` visually distinct colors.

    Parameters
    ----------
    n_colors : int
        Number of colors to generate.

    Returns
    -------
    list[tuple[float, float, float]]
        RGB triplets drawn from a qualitative colormap (`tab10` for up to 10
        colors, `tab20` beyond that). Colors repeat cyclically once `n_colors`
        exceeds the colormap size.
    """
    import matplotlib

    cmap = matplotlib.colormaps["tab10" if n_colors <= 10 else "tab20"]
    return [tuple(cmap(i % cmap.N)[:3]) for i in range(n_colors)]


def _style_colorbar(
    cbar: "Colorbar",
    text_color: str,
    tick_fontsize: float | None,
    *,
    bg_color: str | None = None,
    label: str | None = None,
    label_fontsize: float | None = None,
) -> None:
    """Apply foreground and background colors to a colorbar's ticks, outline, and label.

    Parameters
    ----------
    cbar : matplotlib.colorbar.Colorbar
        Colorbar to style.
    text_color : str
        Color for the tick marks, tick labels, outline edge, and label.
    tick_fontsize : float, optional
        Font size for the tick labels. If not provided, the active Matplotlib
        default is kept.
    bg_color : str, optional
        Background color for the colorbar axes. If not provided, the axes
        background is left unchanged.
    label : str, optional
        Text for the colorbar label. If not provided, any label already set on the
        colorbar (e.g. by an xarray plot call) is kept and only recolored/resized.
    label_fontsize : float, optional
        Font size for the label. If not provided, the active Matplotlib default is
        kept.
    """
    import matplotlib.pyplot as plt

    cbar.ax.yaxis.set_tick_params(color=text_color, labelsize=tick_fontsize)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=text_color, fontsize=tick_fontsize)
    cbar.outline.set_edgecolor(text_color)  # type: ignore
    if bg_color is not None:
        cbar.ax.set_facecolor(bg_color)
    if label is not None:
        cbar.set_label(label)
    cbar.ax.yaxis.label.set_color(text_color)
    if label_fontsize is not None:
        cbar.ax.yaxis.label.set_fontsize(label_fontsize)


def coerce_complex_to_magnitude(data: xr.DataArray, caller: str) -> xr.DataArray:
    """Convert complex-valued arrays to magnitude for plotting.

    Parameters
    ----------
    data : xarray.DataArray
        Input data to display.
    caller : str
        Name of the plotting entry point used in the warning message.

    Returns
    -------
    xarray.DataArray
        `data` unchanged for non-complex inputs, otherwise `abs(data)`.

    Warns
    -----
    UserWarning
        Raised when `data` is complex-valued to make the implicit magnitude
        conversion explicit to users.
    """
    if np.iscomplexobj(data):
        warnings.warn(
            f"Complex-valued data passed to {caller}; plotting magnitude "
            "(`abs(data)`).",
            UserWarning,
            stacklevel=find_stack_level(),
        )
        return xr.ufuncs.abs(data)
    return data


def _materialize_axis_aligned_world_grid_for_display(
    data: xr.DataArray,
) -> xr.DataArray:
    """Expose axis-aligned voxel-to-world data on plain world `z/y/x` dims.

    Parameters
    ----------
    data : xarray.DataArray
        Canonical VoxelData array (every caller runs it through `ensure_voxeldata`
        first, so `has_voxel_to_world_index(data)` is always true here).

    Returns
    -------
    xarray.DataArray
        DataArray whose spatial dimensions are renamed from voxel `k/j/i` to world
        `z/y/x`, with the linked world coordinates promoted to dimension
        coordinates and `voxel_to_world` removed from attrs.
    """
    if not has_axis_aligned_voxel_to_world_index(data):
        return data
    world_dims = get_voxel_to_world_coord_names(data)

    voxel_dims = tuple(dim for dim in VOXEL_DIMS if dim in data.dims)
    dim_map = dict(zip(voxel_dims, world_dims, strict=True))
    result_dims = tuple(dim_map.get(str(dim), str(dim)) for dim in data.dims)

    coords = {}
    for dim in data.dims:
        result_dim = dim_map.get(str(dim), str(dim))
        if str(dim) in dim_map:
            source_coord = data.coords[result_dim]
            indexers = {d: 0 for d in source_coord.dims if d != dim}
            coords[result_dim] = (result_dim, source_coord.isel(indexers).values)
        elif str(dim) in data.coords:
            coords[result_dim] = (result_dim, data.coords[str(dim)].values)

    result = xr.DataArray(
        data=data.data,
        dims=result_dims,
        coords=coords,
        name=data.name,
        attrs=data.attrs.copy(),
    )
    for dim in data.dims:
        result_dim = dim_map.get(str(dim), str(dim))
        source_coord = (
            data.coords[result_dim] if str(dim) in dim_map else data.coords[str(dim)]
        )
        result.coords[result_dim].attrs = dict(source_coord.attrs)
    return result


def compute_oblique_axis_aligned_grid_geometry(
    data: xr.DataArray, world_dims: Sequence[str]
) -> tuple[list[int], list[float], list[float]]:
    """Compute the axis-aligned grid shape/spacing/origin `data` would resample onto.

    Pure geometry from `data`'s world-coordinate bounds and voxel spacing -- no
    interpolation. Callers that only need to know the resampled shape (e.g. to
    reject a panel that won't collapse to a 2D plane) can use this without paying
    for `resample_volume`'s actual interpolation.

    Parameters
    ----------
    data : xarray.DataArray
        Oblique (non-axis-aligned) VoxelData array. May have a spatial dim already
        squeezed to a scalar coordinate (e.g. a single-slice panel with `k` isel'd
        away) -- `ensure_voxeldata` restores it, affine and all, before computing
        geometry, so the result doesn't depend on whether the caller's dim was
        literal or scalar (see Notes).
    world_dims : collections.abc.Sequence of str
        World coordinate names (`"z"`/`"y"`/`"x"`) to compute grid geometry for, in
        order.

    Returns
    -------
    shape : list of int
        Number of voxels along each output world axis.
    spacing : list of float
        World distance between consecutive voxels along each output world axis.
    origin : list of float
        World location of output voxel position 0 along each output world axis.

    Notes
    -----
    Per-world-axis spacing comes from a QR decomposition of the full 3x3
    voxel-to-world linear block, following nilearn's `reorder_img`
    (`nilearn.image.resampling`): `Q, R = qr(linear)`, then each world row's
    spacing is `abs(diag(R))[argmax(abs(Q[row]))]` -- the scale of whichever
    orthonormal basis vector that row is most aligned to. This replaces a simpler,
    per-row "look at the raw affine row and pick the largest-magnitude voxel
    column" heuristic, which had a real bug: a spatial dim already squeezed to a
    scalar coordinate was silently dropped from consideration (its column removed
    from the linear block, per `get_voxel_to_world_affine`'s docstring, rather than
    the whole 3-column matrix always being used regardless of array shape), so a
    single-slice panel's predicted resample shape depended on whether the caller
    happened to squeeze it before or after handing it here. `ensure_voxeldata`
    below removes that dependency by always restoring the full, un-folded affine
    first, then QR decomposes it as a whole -- so the result is the same either
    way.
    """
    data = ensure_voxeldata(data)
    # QR only needs the affine, not the data's own per-voxel-dim coordinate
    # spacing -- but `resample_volume` (the eventual consumer of this geometry)
    # builds a regular-grid SimpleITK image from *all 3* native voxel dims
    # regardless of which one QR picks as dominant for a given world axis, so an
    # irregularly spaced dim anywhere still breaks it downstream with an opaque
    # error. Check all 3 up front for a clear message instead.
    for voxel_dim, voxel_spacing in get_voxel_to_world_index_spacing(data).items():
        if voxel_spacing is None:
            raise ValueError(
                f"Cannot resample onto an axis-aligned world grid: voxel "
                f"dimension {voxel_dim!r} has no well-defined spacing "
                "(irregularly spaced coordinates)."
            )
    linear = get_voxel_to_world_affine(data)[:3, :3]
    q_basis, r_scale = np.linalg.qr(linear)
    row_to_axis = np.abs(q_basis).argmax(axis=1)
    axis_spacing = np.abs(np.diag(r_scale))

    spacing: list[float] = []
    origin: list[float] = []
    shape: list[int] = []
    for row, dim in enumerate(world_dims):
        values = np.asarray(data.coords[dim].values, dtype=np.float64)
        lower = np.float64(np.min(values)).item()
        upper = np.float64(np.max(values)).item()
        dim_spacing = float(axis_spacing[row_to_axis[row]])
        if dim_spacing == 0.0:
            raise ValueError(
                f"Cannot resample {dim!r} onto an axis-aligned world grid: the "
                "voxel-to-world affine is singular along this axis."
            )
        origin.append(lower)
        spacing.append(dim_spacing)
        # A relative tolerance absorbs floating-point noise in `upper`/`lower`
        # (e.g. from composing several affines) that would otherwise push an
        # exact multiple of `dim_spacing` just over an integer boundary and
        # `ceil` an extra, out-of-bounds slice onto the grid.
        n_steps = (upper - lower) / dim_spacing
        shape.append(np.int64(np.ceil(n_steps - 1e-6 * max(1.0, n_steps))).item() + 1)
    return shape, spacing, origin


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
        # reorder_img): the scale of whichever orthonormal QR basis vector each
        # world row is most aligned to -- but keyed by row here (not `world_dims`
        # order) since the slice row needs picking out first.
        q_basis, r_scale = np.linalg.qr(linear)
        row_to_axis = np.abs(q_basis).argmax(axis=1)
        qr_axis_spacing = np.abs(np.diag(r_scale))
        output_direction = np.eye(3)
        in_plane_rows = [row for row in range(3) if row != slice_row]
        in_plane_spacing = [
            float(qr_axis_spacing[row_to_axis[row]]) for row in in_plane_rows
        ]
        slice_axis_spacing_default = float(qr_axis_spacing[row_to_axis[slice_row]])
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


def resample_slice_axis_aligned_world_grid(
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
    on `resample_volume` with. A no-op (returns `data` unchanged) when `data`
    doesn't carry voxel-to-world geometry at all.

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
        it carries no voxel-to-world geometry, or if it's already fully
        axis-aligned (nothing to fix, and matplotlib only needs matching world
        coordinates to overlay correctly, not a matching pixel grid -- see
        `resample_to_axis_aligned_world_grid`'s identical short-circuit).
    slice_axis_grid : SliceAxisGrid or None
        `slice_axis_grid` unchanged if provided, or the one just established
        from `data`'s own geometry otherwise -- even when `data` was already
        axis-aligned and skipped resampling, so a later oblique volume/mask on
        the same plotter still has a grid to align its own slice axis to.
        `None` only when `data` carries no voxel-to-world geometry at all.
    """
    if not has_voxel_to_world_index(data):
        return data, slice_axis_grid
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


def resample_to_axis_aligned_world_grid(
    data: xr.DataArray,
    *,
    reference: xr.DataArray | None = None,
    interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    fill_value: float | None = None,
) -> xr.DataArray:
    """Resample voxel-to-world data onto an axis-aligned world grid for display.

    Parameters
    ----------
    data : xarray.DataArray
        Three-dimensional or three-dimensional-plus-time DataArray. VoxelData
        arrays are resampled using their voxel-to-world index.
    reference : xarray.DataArray, optional
        Axis-aligned world-grid DataArray to reuse as the resampling target.
        If not provided, a new plotting grid is synthesized from `data`'s world
        bounds and per-axis world spacing.
    interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used during resampling. Use `"nearest"` for label/mask
        data, where blending distinct integer labels together is never meaningful.
    fill_value : float, optional
        Value assigned to voxels outside `data`'s field of view after resampling.
        If not provided, defaults to `float(data.min())` (see
        [`resample_volume`][confusius.registration.resample_volume]).

    Returns
    -------
    xarray.DataArray
        Axis-aligned world-grid VoxelData array when `data` has
        voxel-to-world geometry; otherwise the original input.
    """
    if not has_voxel_to_world_index(data) or has_axis_aligned_voxel_to_world_index(
        data
    ):
        return data

    from confusius.registration import resample_like, resample_volume

    world_dims = get_voxel_to_world_coord_names(data)
    if reference is not None:
        if not has_voxel_to_world_index(reference):
            from confusius.xarray import create_voxeldata

            # `reference` is a plain, caller-supplied DataArray here (not one of
            # ConfUSIus's own VoxelData arrays), so its spatial dims may be named
            # z/y/x rather than the native voxel names create_voxeldata now
            # requires; remap them, leaving any other dim name (time, pose, extras)
            # unchanged.
            spatial_to_voxel = dict(zip(SPATIAL_DIMS, VOXEL_DIMS, strict=True))
            reference_dims = tuple(
                spatial_to_voxel.get(str(dim), str(dim)) for dim in reference.dims
            )
            spacing = []
            origin = []
            for dim in ("z", "y", "x"):
                if dim in reference.coords:
                    values = np.asarray(reference.coords[dim].values, dtype=np.float64)
                    origin.append(float(values[0]))
                    spacing.append(
                        float(np.median(np.diff(values))) if values.size > 1 else 1.0
                    )
                else:
                    origin.append(0.0)
                    spacing.append(1.0)
            reference = create_voxeldata(
                reference.data,
                dims=reference_dims,
                time=reference.coords.get("time"),
                pose=reference.coords.get("pose"),
                spacing=tuple(spacing),
                origin=tuple(origin),
                attrs=reference.attrs,
                name=str(reference.name) if reference.name is not None else None,
            )
        result = resample_like(
            data,
            reference,
            np.eye(len(world_dims) + 1, dtype=np.float64),
            interpolation=interpolation,
            fill_value=fill_value,
        )
    else:
        # `data` is oblique here (axis-aligned data already returned above), so
        # `data.coords[dim]` for a world dim is (k, j, i)-shaped, not 1D -- a
        # representative per-axis spacing therefore can't come from
        # `np.diff(values)` (which would default to differencing along the last
        # voxel axis, regardless of which world axis `dim` actually corresponds
        # to). require_scalar_pose_affine is still called for its validation side
        # effect: rejecting pose-stacked data before attempting to resample it onto
        # one axis-aligned grid.
        require_scalar_pose_affine(
            data, "Resampling to an axis-aligned world grid for display"
        )

        # A voxel dim's own spacing is only a valid proxy for a world axis's
        # resolution when the affine is close to diagonal. For a rotated/permuted
        # affine (see design/multipose-voxel-to-world-index.md's "monomial
        # affines" deferred item -- e.g. a probe swept along what is nominally the
        # `k` voxel axis but physically lands mostly along world `x`), the world
        # axis that needs fine spacing is fed by whichever voxel dim actually
        # dominates that row of the affine, not the positionally-matching one.
        # Pick that dominant voxel dim's spacing for each output world axis.
        shape, spacing, origin = compute_oblique_axis_aligned_grid_geometry(
            data, world_dims
        )

        result = resample_volume(
            data,
            np.eye(len(world_dims) + 1, dtype=np.float64),
            output_sizes=dict(zip(VOXEL_DIMS, shape, strict=True)),
            output_spacing=dict(zip(VOXEL_DIMS, spacing, strict=True)),
            output_origin=dict(zip(SPATIAL_DIMS, origin, strict=True)),
            output_direction=np.eye(len(world_dims), dtype=np.float64),
            interpolation=interpolation,
            fill_value=fill_value,
        )

    return result


def sort_coords_for_plot(
    data: xr.DataArray,
    dims: Sequence[Hashable],
) -> xr.DataArray:
    """Sort coordinate axes into increasing order before plotting.

    Any plotted coordinate axis that is not already monotonic increasing,
    including monotonic-decreasing axes, is sorted to avoid ambiguous
    geometry in plotting backends that assume ordered coordinates (e.g.
    `pcolormesh` edge construction, contour interpolation, and napari array
    indexing with scale/translate).

    Parameters
    ----------
    data : xarray.DataArray
        Input DataArray whose plotted coordinate axes should be sorted.
    dims : sequence of hashable
        Dimensions whose coordinates to consider for sorting.

    Returns
    -------
    xarray.DataArray
        The input with every non-monotonic-increasing coordinate among `dims`
        sorted into ascending order.
    """
    sorted_data = data
    for dim in dims:
        if dim not in sorted_data.coords:
            continue
        if not sorted_data.get_index(dim).is_monotonic_increasing:
            sorted_data = sorted_data.sortby(dim)
    return sorted_data
