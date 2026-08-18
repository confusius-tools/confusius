"""Helpers shared between matplotlib- and napari-based plotting code."""

import warnings
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from confusius._dims import SPATIAL_DIMS, VOXEL_DIMS
from confusius._utils.geometry import (
    VoxelToWorldIndex,
    get_affine_axis_scalings,
    get_voxel_to_world_coord_names,
    get_voxel_to_world_index_spacing,
    get_voxel_to_world_spatial_dims,
    has_axis_aligned_voxel_to_world_index,
    has_voxel_to_world_index,
    require_scalar_pose_affine,
    update_voxel_to_world_coord_attrs,
)
from confusius._utils.stack import find_stack_level

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
        Axis-aligned VoxelData-compatible DataArray.

    Returns
    -------
    xarray.DataArray
        DataArray whose spatial dimensions are renamed from voxel `k/j/i` to world
        `z/y/x`, with the linked world coordinates promoted to dimension
        coordinates and `voxel_to_world` removed from attrs.
    """
    if not has_axis_aligned_voxel_to_world_index(data):
        if has_voxel_to_world_index(data):
            return data
        if any(
            isinstance(index, VoxelToWorldIndex) for index in data.xindexes.values()
        ):
            coords = {
                name: (coord.dims, coord.values, coord.attrs)
                for name, coord in data.coords.items()
            }
            result = xr.DataArray(
                data=data.data,
                dims=data.dims,
                coords=coords,
                name=data.name,
                attrs=data.attrs.copy(),
            )
            return result
        else:
            return data
    else:
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


def resample_to_axis_aligned_world_grid(
    data: xr.DataArray,
    *,
    reference: xr.DataArray | None = None,
) -> xr.DataArray:
    """Resample voxel-to-world data onto an axis-aligned world grid for display.

    Parameters
    ----------
    data : xarray.DataArray
        Three-dimensional or three-dimensional-plus-time DataArray. VoxelData-compatible
        inputs are resampled using their voxel-to-world index.
    reference : xarray.DataArray, optional
        Axis-aligned world-grid DataArray to reuse as the resampling target.
        If not provided, a new plotting grid is synthesized from `data`'s world
        bounds and per-axis world spacing.

    Returns
    -------
    xarray.DataArray
        Axis-aligned world-grid VoxelData-compatible DataArray when `data` has
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
            from confusius.xarray import create_fusi_dataarray

            # `reference` is a plain, caller-supplied DataArray here (not one of
            # ConfUSIus's own VoxelData arrays), so its spatial dims may be named
            # z/y/x rather than the native voxel names create_fusi_dataarray now
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
                        float(np.median(np.diff(values)))
                        if values.size > 1
                        else float(reference.coords[dim].attrs.get("voxdim", 1.0))
                    )
                else:
                    origin.append(0.0)
                    spacing.append(1.0)
            reference = create_fusi_dataarray(
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
        )
    else:
        # `data` is oblique here (axis-aligned data already returned above), so
        # `data.coords[dim]` for a world dim is (k, j, i)-shaped, not 1D. A
        # representative per-axis spacing therefore can't come from
        # `np.diff(values)` (which would default to differencing along the last
        # voxel axis, regardless of which world axis `dim` actually corresponds
        # to) -- derive it from the affine instead, the same way `.fusi.spacing`
        # does: world distance between consecutive sampled voxels along the
        # matching voxel axis, falling back to the affine's per-one-voxel-unit
        # scale when the voxel-space sampling itself is irregular.
        voxel_dims = get_voxel_to_world_spatial_dims(data)
        voxel_to_world_affine = require_scalar_pose_affine(
            data, "Resampling to an axis-aligned world grid for display"
        )
        index_spacing = get_voxel_to_world_index_spacing(data)
        axis_scalings = get_affine_axis_scalings(voxel_to_world_affine, voxel_dims)
        world_dim_spacing = {
            world_dim: index_spacing[voxel_dim]
            if index_spacing[voxel_dim] is not None
            else axis_scalings[voxel_dim]
            for world_dim, voxel_dim in zip(world_dims, voxel_dims, strict=True)
        }

        spacing: list[float] = []
        origin: list[float] = []
        shape: list[int] = []
        for dim in world_dims:
            values = np.asarray(data.coords[dim].values, dtype=np.float64)
            lower = np.float64(np.min(values)).item()
            upper = np.float64(np.max(values)).item()
            dim_spacing = data.coords[dim].attrs.get("voxdim")
            if dim_spacing is None:
                dim_spacing = world_dim_spacing[dim]
            dim_spacing = np.float64(dim_spacing).item()
            origin.append(lower)
            spacing.append(dim_spacing)
            # A relative tolerance absorbs floating-point noise in `upper`/`lower`
            # (e.g. from composing several affines) that would otherwise push an
            # exact multiple of `dim_spacing` just over an integer boundary and
            # `ceil` an extra, out-of-bounds slice onto the grid.
            n_steps = (upper - lower) / dim_spacing
            shape.append(
                np.int64(np.ceil(n_steps - 1e-6 * max(1.0, n_steps))).item() + 1
            )

        result = resample_volume(
            data,
            np.eye(len(world_dims) + 1, dtype=np.float64),
            output_shape=shape,
            output_spacing=spacing,
            output_origin=origin,
        )

    # `resample_volume`/`resample_like` always build VoxelData-compatible (voxel-dim,
    # indexed) output; carry over the source world coordinates' attrs (units, etc.)
    # via the index-aware helper rather than a direct `.attrs =` assignment, since
    # the world coordinates here are index-derived and a plain mutation would be
    # silently discarded by the next operation that touches the index.
    result = update_voxel_to_world_coord_attrs(
        result, {dim: dict(data.coords[dim].attrs) for dim in world_dims}
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
