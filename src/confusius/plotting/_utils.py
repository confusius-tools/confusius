"""Helpers shared between matplotlib- and napari-based plotting code."""

import warnings
from collections.abc import Hashable, Sequence

import numpy as np
import xarray as xr

from confusius._utils.stack import find_stack_level


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
    """Generate `n_colors` visually distinct colors."""
    import matplotlib

    cmap = matplotlib.colormaps["tab10" if n_colors <= 10 else "tab20"]
    return [tuple(cmap(i % cmap.N)[:3]) for i in range(n_colors)]


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
