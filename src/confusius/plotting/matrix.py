"""2D matrix visualization utilities.

`plot_matrix` is inspired by `nilearn.plotting.plot_matrix` (BSD-3-Clause License; see
`NOTICE` for details). The `groups` colored-rectangle annotation has no Nilearn
equivalent.
"""

from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.plotting._utils import (
    _auto_fg_color,
    _get_distinct_colors,
    _resolve_font_sizes,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure, SubFigure


def _sanitize_matrix(matrix: "npt.NDArray[Any] | xr.DataArray") -> npt.NDArray[Any]:
    """Validate `matrix` is 2D and square, returning its plain `numpy.ndarray` values.

    Parameters
    ----------
    matrix : numpy.ndarray or xarray.DataArray
        Candidate matrix to plot.

    Returns
    -------
    numpy.ndarray
        `matrix` as a plain array.

    Raises
    ------
    ValueError
        If `matrix` is not a 2D square array.
    """
    values = matrix.values if isinstance(matrix, xr.DataArray) else np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(
            f"Expected a square 2D matrix, got array of shape {values.shape}."
        )
    return values


def _default_labels(matrix: "npt.NDArray[Any] | xr.DataArray") -> list[str] | None:
    """Derive default tick labels from a `DataArray`'s first-dimension coordinate.

    Parameters
    ----------
    matrix : numpy.ndarray or xarray.DataArray
        Matrix passed to `plot_matrix`.

    Returns
    -------
    list[str] or None
        Stringified coordinate values for `matrix.dims[0]`, or `None` when `matrix`
        is not a `xarray.DataArray` or carries no such coordinate.
    """
    if not isinstance(matrix, xr.DataArray):
        return None
    dim = matrix.dims[0]
    if dim not in matrix.coords:
        return None
    return [str(v) for v in matrix.coords[dim].values]


def _sanitize_labels(
    labels: "Sequence[str] | Literal[False] | None", size: int
) -> list[str] | None:
    """Validate and normalize `labels` against the matrix `size`.

    Parameters
    ----------
    labels : sequence of str, False, or None
        Candidate row/column labels.
    size : int
        Number of rows (== number of columns) in the matrix being plotted.

    Returns
    -------
    list[str] or None
        `None` when `labels` is `False`, `None`, or empty. Otherwise `labels` as a
        list.

    Raises
    ------
    ValueError
        If `labels` is provided but its length does not match `size`.
    """
    if labels is False or labels is None:
        return None
    labels = list(labels)
    if not labels:
        return None
    if len(labels) != size:
        raise ValueError(
            f"Length of labels ({len(labels)}) does not match matrix size ({size})."
        )
    return labels


def _mask_triangle(
    matrix: npt.NDArray[Any], tri: Literal["full", "lower", "diag"]
) -> np.ma.MaskedArray:
    """Mask the upper triangle of `matrix` according to `tri`.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square matrix to mask.
    tri : {"full", "lower", "diag"}
        Which part of the matrix to keep visible: `"lower"` excludes the diagonal,
        `"diag"` includes it, `"full"` keeps everything.

    Returns
    -------
    numpy.ma.MaskedArray
        `matrix` with the masked entries hidden from display.
    """
    if tri == "full":
        return np.ma.masked_array(matrix, mask=np.zeros_like(matrix, dtype=bool))
    k = -1 if tri == "lower" else 0
    mask = ~np.tri(matrix.shape[0], k=k, dtype=bool)
    return np.ma.masked_array(matrix, mask=mask)


def _draw_grid(
    ax: "Axes", tri: Literal["full", "lower", "diag"], size: int, color: str
) -> None:
    """Draw grid lines separating matrix cells.

    Adapted from Nilearn's `_configure_grid` helper (see module docstring for
    attribution); the small offsets correct the same off-by-one line/cell alignment
    issue.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing the matrix image.
    tri : {"full", "lower", "diag"}
        Matrix triangle being displayed; determines which grid lines are drawn.
    size : int
        Number of rows (== number of columns) in the matrix.
    color : str
        Line color.
    """
    for i in range(size):
        offset = 1.001 * i
        if tri == "lower":
            ax.plot(
                [offset + 0.5, offset + 0.5], [size - 0.5, offset + 0.5], color=color
            )
            ax.plot([offset + 0.5, -0.5], [offset + 0.5, offset + 0.5], color=color)
        elif tri == "diag":
            ax.plot(
                [offset + 0.5, offset + 0.5], [size - 0.5, offset - 0.5], color=color
            )
            ax.plot([offset + 0.5, -0.5], [offset - 0.5, offset - 0.5], color=color)
        else:
            ax.plot([offset + 0.5, offset + 0.5], [size - 0.5, -0.5], color=color)
            ax.plot([size - 0.5, -0.5], [offset + 0.5, offset + 0.5], color=color)


def _group_spans(groups: Sequence[Any]) -> list[tuple[Any, int, int]]:
    """Split `groups` into contiguous runs of identical values.

    Parameters
    ----------
    groups : sequence
        Group value for each row/column, e.g. `["cortex", "cortex", "thalamus"]`.

    Returns
    -------
    list[tuple[Any, int, int]]
        `(value, start, stop)` for each contiguous run, `stop` exclusive.
    """
    spans = []
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            spans.append((groups[start], start, i))
            start = i
    return spans


def _draw_group_bar(
    ax: "Axes",
    orientation: Literal["vertical", "horizontal"],
    spans: list[tuple[Any, int, int]],
    colors: Mapping[Any, str],
    show_labels: bool,
    fontsize: float | None,
    text_color: str,
) -> None:
    """Draw a strip of colored rectangles marking contiguous label groups.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Thin axes appended alongside the matrix (left for rows, top for columns).
    orientation : {"vertical", "horizontal"}
        `"vertical"` draws a column of rectangles (for row groups), `"horizontal"`
        draws a row of rectangles (for column groups).
    spans : list[tuple[Any, int, int]]
        Contiguous group runs from `_group_spans`.
    colors : Mapping
        Color for each group value.
    show_labels : bool
        Whether to annotate each rectangle with its group value.
    fontsize : float, optional
        Font size for group labels.
    text_color : str
        Color for group labels.
    """
    import matplotlib.patches as mpatches

    for value, start, stop in spans:
        color = colors[value]
        if orientation == "vertical":
            ax.add_patch(
                mpatches.Rectangle((0, start - 0.5), 1, stop - start, color=color)
            )
        else:
            ax.add_patch(
                mpatches.Rectangle((start - 0.5, 0), stop - start, 1, color=color)
            )

    # Only fix the limit along the axis *not* shared with the main matrix axes
    # (sharex/sharey ties the other one to the main plot's row/column indices).
    if orientation == "vertical":
        ax.set_xlim(0, 1)
    else:
        ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if not show_labels:
        return
    for value, start, stop in spans:
        midpoint = (start + stop - 1) / 2
        if orientation == "vertical":
            ax.annotate(
                str(value),
                xy=(0.5, midpoint),
                xycoords=("axes fraction", "data"),
                ha="center",
                va="center",
                rotation=90,
                fontsize=fontsize,
                color=text_color,
            )
        else:
            ax.annotate(
                str(value),
                xy=(midpoint, 0.5),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
            )


def plot_matrix(
    matrix: "npt.NDArray[Any] | xr.DataArray",
    labels: "Sequence[str] | Literal[False] | None" = None,
    groups: "Sequence[Any] | None" = None,
    group_colors: "Mapping[Any, str] | None" = None,
    show_group_labels: bool = True,
    tri: Literal["full", "lower", "diag"] = "full",
    grid: "str | Literal[False]" = False,
    cmap: "str | Colormap" = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    show_colorbar: bool = True,
    cbar_label: str | None = None,
    title: str | None = None,
    fontsize: float | None = None,
    bg_color: str = "white",
    fg_color: str | None = None,
    figsize: tuple[float, float] = (7, 6),
    ax: "Axes | None" = None,
) -> tuple["Figure | SubFigure", "Axes"]:
    """Plot a 2D matrix, e.g. a connectivity or correlation matrix.

    Parameters
    ----------
    matrix : (n, n) numpy.ndarray or xarray.DataArray
        Square matrix to plot.
    labels : list[str], False, or None, default: None
        Label for each row/column, in matrix order. If not provided and `matrix` is a
        `xarray.DataArray` with a coordinate on its first dimension, that coordinate's
        values are used. Pass `False` to force no labels even when such a coordinate
        exists.
    groups : sequence, optional
        Group value for each row/column, in matrix order, e.g.
        `["cortex"] * 5 + ["thalamus"] * 3`. Rows/columns are assumed to already be
        sorted by group: contiguous runs of equal values are drawn as colored
        rectangles alongside the matrix. If not provided, no group annotation is
        drawn.
    group_colors : Mapping, optional
        Color for each unique value in `groups`. If not provided, colors are assigned
        automatically from a qualitative colormap.
    show_group_labels : bool, default: True
        Whether to annotate each group's colored rectangle with its value.
    tri : {"full", "lower", "diag"}, default: "full"
        Which part of the matrix to display:

        - `"lower"`: only the part strictly below the diagonal.
        - `"diag"`: the part below the diagonal, including it.
        - `"full"`: the entire matrix.

    grid : str or False, default: False
        Color of grid lines separating cells. `False` disables the grid.
    cmap : str or matplotlib.colors.Colormap, default: "RdBu_r"
        Colormap for `matrix`.
    vmin : float, optional
        Minimum value for the colormap. If not provided, defaults to `-vmax`.
    vmax : float, optional
        Maximum value for the colormap. If not provided, defaults to the maximum
        absolute value in `matrix`, so that a diverging `cmap` centers on zero.
    show_colorbar : bool, default: True
        Whether to add a colorbar for `matrix` to the figure.
    cbar_label : str, optional
        Label for the colorbar.
    title : str, optional
        Plot title.
    fontsize : float, optional
        Base font size for text elements. Title uses `fontsize` directly; the
        colorbar label and group labels use `0.9 * fontsize`; tick labels use
        `0.85 * fontsize`. If not provided, uses the active Matplotlib defaults.
    bg_color : str, default: "white"
        Background color for the figure and axes. Any matplotlib-compatible color
        string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula
        (white on dark backgrounds, black on light ones).
    figsize : tuple[float, float], default: (7, 6)
        Figure size in inches `(width, height)`, used only when `ax` is not provided.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If not provided, creates new figure and axes.

    Returns
    -------
    figure : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        Figure object containing the matrix plot.
    axes : matplotlib.axes.Axes
        Axes object with the matrix plot.

    Notes
    -----
    This function was inspired by Nilearn's `nilearn.plotting.plot_matrix`. Unlike
    Nilearn, it does not support reordering by hierarchical clustering.

    Examples
    --------
    >>> import numpy as np
    >>> from confusius.plotting import plot_matrix
    >>> rng = np.random.default_rng(0)
    >>> corr = np.corrcoef(rng.standard_normal((10, 20)))
    >>> fig, ax = plot_matrix(corr, labels=[f"region_{i}" for i in range(10)])

    >>> # Annotate groups of regions with colored rectangles.
    >>> groups = ["cortex"] * 4 + ["thalamus"] * 3 + ["hippocampus"] * 3
    >>> fig, ax = plot_matrix(corr, groups=groups)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    values = _sanitize_matrix(matrix)
    size = values.shape[0]
    labels = _sanitize_labels(
        labels if labels is not None else _default_labels(matrix), size
    )
    if groups is not None and len(groups) != size:
        raise ValueError(
            f"Length of groups ({len(groups)}) does not match matrix size ({size})."
        )

    masked = _mask_triangle(values, tri)
    if vmax is None:
        vmax = float(np.nanmax(np.abs(values)))
    if vmin is None:
        vmin = -vmax

    text_color = fg_color if fg_color is not None else _auto_fg_color(bg_color)
    title_fontsize, label_fontsize, tick_fontsize = _resolve_font_sizes(fontsize)

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize, layout="constrained")
        figure.patch.set_facecolor(bg_color)
    else:
        figure = ax.figure
    ax.set_facecolor(bg_color)

    divider = make_axes_locatable(ax)

    image = ax.imshow(
        masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest"
    )
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)

    if labels:
        ax.set_xticks(np.arange(size))
        ax.set_xticklabels(
            labels, rotation=50, ha="right", fontsize=tick_fontsize, color=text_color
        )
        ax.set_yticks(np.arange(size))
        ax.set_yticklabels(
            labels,
            rotation=10,
            ha="right",
            va="top",
            fontsize=tick_fontsize,
            color=text_color,
        )
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(text_color)

    if grid is not False:
        _draw_grid(ax, tri, size, grid)

    if groups is not None:
        spans = _group_spans(list(groups))
        unique_values = list(dict.fromkeys(g for g, _, _ in spans))
        colors = group_colors or dict(
            zip(unique_values, _get_distinct_colors(len(unique_values)))
        )
        missing = [v for v in unique_values if v not in colors]
        if missing:
            raise ValueError(f"group_colors is missing colors for group(s): {missing}.")

        row_ax = divider.append_axes("left", size="4%", pad=0.05, sharey=ax)
        _draw_group_bar(
            row_ax,
            "vertical",
            spans,
            colors,
            show_group_labels,
            label_fontsize,
            text_color,
        )
        col_ax = divider.append_axes("top", size="4%", pad=0.05, sharex=ax)
        _draw_group_bar(
            col_ax,
            "horizontal",
            spans,
            colors,
            show_group_labels,
            label_fontsize,
            text_color,
        )

    if show_colorbar:
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = figure.colorbar(image, cax=cax)
        cbar.ax.yaxis.set_tick_params(color=text_color, labelsize=tick_fontsize)
        plt.setp(
            cbar.ax.yaxis.get_ticklabels(), color=text_color, fontsize=tick_fontsize
        )
        cbar.outline.set_edgecolor(text_color)  # type: ignore
        cbar.ax.set_facecolor(bg_color)
        if cbar_label:
            cbar.set_label(cbar_label, color=text_color, fontsize=label_fontsize)

    if title:
        ax.set_title(title, color=text_color, fontsize=title_fontsize)

    return figure, ax
