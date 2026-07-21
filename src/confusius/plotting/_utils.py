"""Helpers shared between matplotlib- and napari-based plotting code."""

import warnings
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

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
    *,
    figure: Any | None = None,
    axes: Any | None = None,
) -> tuple[float, float, float]:
    """Resolve title, label, and tick font sizes from a base size.

    Parameters
    ----------
    fontsize : float, optional
        Base font size for plot text elements. If not provided, a base size is
        estimated from the figure size and the number of plotted axes.
    figure : object, optional
        Matplotlib figure used for the size heuristic.
    axes : object, optional
        Matplotlib axes (or array of axes) used to estimate how many panels share
        the available figure area.

    Returns
    -------
    title_fontsize : float
        Font size for subplot titles.
    label_fontsize : float
        Font size for axis and colorbar labels.
    tick_fontsize : float
        Font size for tick labels.
    """
    if fontsize is None:
        base_fontsize = 12.0
        if figure is not None and hasattr(figure, "get_size_inches"):
            width, height = figure.get_size_inches()
            n_axes = max(1, int(np.size(axes)) if axes is not None else 1)
            panel_area = max(float(width) * float(height) / n_axes, 1.0)
            base_fontsize = float(np.clip(6.5 + 1.1 * np.sqrt(panel_area), 8.0, 18.0))
        fontsize = base_fontsize
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
