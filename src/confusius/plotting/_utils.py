"""Helpers shared between matplotlib- and napari-based plotting code."""

import warnings
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._dims import VOXEL_DIMS, WORLD_DIMS
from confusius._utils.geometry import has_axis_aligned_voxel_to_world_index
from confusius._utils.stack import find_stack_level

if TYPE_CHECKING:
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import Colormap


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


_DIVERGING_CMAP = "coolwarm"
"""Default colormap for data spanning both signs (see `_resolve_colormap_style`)."""

_SEQUENTIAL_CMAP = "viridis"
"""Default colormap for non-negative data (see `_resolve_colormap_style`)."""

_SEQUENTIAL_CMAP_NEGATIVE = "viridis_r"
"""Default colormap for non-positive data (see `_resolve_colormap_style`).

Reversed relative to `_SEQUENTIAL_CMAP` so that, in both the non-negative and
non-positive cases, values near zero map to the same end of the colormap (dark
purple) and the most extreme magnitude maps to the other end (yellow).
"""


def _resolve_colormap_style(
    data: "npt.NDArray[Any] | xr.DataArray",
    vmin: float | None,
    vmax: float | None,
    cmap: "str | Colormap | None",
    auto_range: bool,
) -> tuple[float, float, "str | Colormap"]:
    """Resolve `(vmin, vmax, cmap)` from the sign of `data` and the bounds given.

    Shared by `plot_stat_map` and `plot_matrix` so both expose the same
    `auto_range` semantics.

    When `auto_range` is `True`, the sign of `data` determines the layout:

    - Both positive and negative values: diverging, symmetric `[-m, m]` range
      where `m = max(|vmin|, |vmax|)` over the bounds actually provided, falling
      back to the largest magnitude in `data` when neither is given, with `cmap`
      defaulting to `_DIVERGING_CMAP`.
    - Only non-negative values: sequential `[0, vmax]` range, with `cmap`
      defaulting to `_SEQUENTIAL_CMAP`.
    - Only non-positive values: sequential `[vmin, 0]` range, with `cmap`
      defaulting to `_SEQUENTIAL_CMAP_NEGATIVE`.

    When `auto_range` is `False`, `vmin`/`vmax` are used directly with no
    zero-anchoring, and `cmap` defaults to `_DIVERGING_CMAP` regardless of
    `data`'s sign. In both cases, an explicitly provided `cmap` is always used
    as-is.

    Parameters
    ----------
    data : numpy.ndarray or xarray.DataArray
        Values to be colormapped. Non-finite entries are ignored.
    vmin : float, optional
        Lower bound of the colormap. If not provided, falls back to the smallest
        finite value in `data`, except in the diverging branch described above.
    vmax : float, optional
        Upper bound of the colormap. If not provided, falls back to the largest
        finite value in `data`, except in the diverging branch described above.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use. If not provided, one is picked from the sign of `data`.
    auto_range : bool
        Whether to anchor the range on zero based on the sign of `data`.

    Returns
    -------
    vmin : float
        Resolved lower bound of the colormap.
    vmax : float
        Resolved upper bound of the colormap.
    cmap : str or matplotlib.colors.Colormap
        Resolved colormap.
    """
    values = np.asarray(data).ravel().astype(float)
    values = values[np.isfinite(values)]
    data_min = float(values.min()) if values.size > 0 else 0.0
    data_max = float(values.max()) if values.size > 0 else 1.0

    resolved_vmin = vmin if vmin is not None else data_min
    resolved_vmax = vmax if vmax is not None else data_max

    if not auto_range:
        return (
            resolved_vmin,
            resolved_vmax,
            cmap if cmap is not None else _DIVERGING_CMAP,
        )

    if data_min < 0 < data_max:
        # A bound given on its own caps the symmetric range by itself: falling back
        # to the data's own min/max for the missing one would drop it silently.
        explicit = [abs(bound) for bound in (vmin, vmax) if bound is not None]
        abs_max = max(explicit) if explicit else max(abs(data_min), abs(data_max))
        return -abs_max, abs_max, cmap if cmap is not None else _DIVERGING_CMAP

    if data_max > 0:
        return 0.0, resolved_vmax, cmap if cmap is not None else _SEQUENTIAL_CMAP

    return resolved_vmin, 0.0, cmap if cmap is not None else _SEQUENTIAL_CMAP_NEGATIVE


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


def materialize_axis_aligned_world_grid_for_display(
    data: xr.DataArray,
) -> xr.DataArray:
    """Expose axis-aligned voxel-to-world data on plain world `z/y/x` dims.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array.

    Returns
    -------
    xarray.DataArray
        DataArray whose spatial dimensions are renamed from voxel `k/j/i` to world
        `z/y/x`, with the linked world coordinates promoted to dimension coordinates and
        `VoxelToWorldIndex` removed.
    """
    if not has_axis_aligned_voxel_to_world_index(data):
        return data

    voxel_dims = tuple(dim for dim in VOXEL_DIMS if dim in data.dims)
    dim_map = dict(zip(voxel_dims, WORLD_DIMS, strict=True))
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
