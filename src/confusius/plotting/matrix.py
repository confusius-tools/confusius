"""2D matrix visualization utilities.

`plot_matrix` is inspired by `nilearn.plotting.plot_matrix` (BSD-3-Clause License; see
`NOTICE` for details). The `groups` colored-rectangle annotation has no Nilearn
equivalent.
"""

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.plotting._utils import (
    _auto_fg_color,
    _get_distinct_colors,
    _resolve_font_sizes,
    _style_colorbar,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure, SubFigure

_MATRIX_DIVERGING_CMAP = "coolwarm"
"""Default colormap for diverging matrices (see `plot_matrix`'s `auto_range`)."""

_MATRIX_SEQUENTIAL_CMAP = "viridis"
"""Default colormap for non-negative matrices (see `plot_matrix`'s `auto_range`)."""

_MATRIX_SEQUENTIAL_CMAP_NEGATIVE = "viridis_r"
"""Default colormap for non-positive matrices (see `plot_matrix`'s `auto_range`).

Reversed relative to `_MATRIX_SEQUENTIAL_CMAP` so that, in both the non-negative and
non-positive cases, values near zero map to the same end of the colormap.
"""


def _resolve_matrix_style(
    values: npt.NDArray[Any],
    vmin: float | None,
    vmax: float | None,
    cmap: "str | Colormap | None",
    auto_range: bool,
) -> tuple[float, float, "str | Colormap"]:
    """Resolve `(vmin, vmax, cmap)` for a matrix, mirroring `plot_stat_map`'s auto_range.

    `vmin`/`vmax` fall back to the actual min/max of `values` when not provided.

    When `auto_range` is `True` (default), the sign of `values` determines the layout:

    - Both positive and negative values: diverging, symmetric `[-m, m]` range where
      `m = max(|vmin|, |vmax|)` (using the resolved bounds above), with `cmap`
      defaulting to `_MATRIX_DIVERGING_CMAP`.
    - Only non-negative values: sequential `[0, vmax]` range, with `cmap` defaulting
      to `_MATRIX_SEQUENTIAL_CMAP`.
    - Only non-positive values: sequential `[vmin, 0]` range, with `cmap` defaulting
      to `_MATRIX_SEQUENTIAL_CMAP_NEGATIVE`.

    When `auto_range` is `False`, the resolved `vmin`/`vmax` are used directly with no
    zero-anchoring, and `cmap` defaults to `_MATRIX_DIVERGING_CMAP` regardless of
    `values`'s sign. In both cases, an explicitly provided `cmap` is always used as-is.
    """
    finite = values[np.isfinite(values)]
    data_min = float(finite.min()) if finite.size > 0 else 0.0
    data_max = float(finite.max()) if finite.size > 0 else 1.0

    resolved_vmin = vmin if vmin is not None else data_min
    resolved_vmax = vmax if vmax is not None else data_max

    if not auto_range:
        return (
            resolved_vmin,
            resolved_vmax,
            cmap if cmap is not None else _MATRIX_DIVERGING_CMAP,
        )

    if data_min < 0 < data_max:
        abs_max = max(abs(resolved_vmin), abs(resolved_vmax))
        return -abs_max, abs_max, cmap if cmap is not None else _MATRIX_DIVERGING_CMAP

    if data_max > 0:
        return 0.0, resolved_vmax, cmap if cmap is not None else _MATRIX_SEQUENTIAL_CMAP

    return (
        resolved_vmin,
        0.0,
        cmap if cmap is not None else _MATRIX_SEQUENTIAL_CMAP_NEGATIVE,
    )


def _validate_matrix(matrix: "npt.NDArray[Any] | xr.DataArray") -> npt.NDArray[Any]:
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
        Stringified coordinate values for `matrix.dims[0]`, or `None` when `matrix` is
        not a `xarray.DataArray` or carries no such coordinate.
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
    matrix: npt.NDArray[Any],
    triangle: Literal["full", "lower", "diag_lower", "diag_upper", "upper"],
) -> np.ma.MaskedArray:
    """Mask part of `matrix` according to `triangle`.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square matrix to mask.
    triangle : {"full", "lower", "diag_lower", "diag_upper", "upper"}
        Which part of the matrix to keep visible: `"lower"` and `"upper"` exclude the
        diagonal, `"diag_lower"` and `"diag_upper"` include it (on the lower and upper
        side respectively), `"full"` keeps everything.

    Returns
    -------
    numpy.ma.MaskedArray
        `matrix` with the masked entries hidden from display.

    Raises
    ------
    ValueError
        If `triangle` is not one of the supported modes.

    Notes
    -----
    This is `plot_matrix`'s first consumer of `triangle`, so it validates the value
    once here; downstream helpers (e.g. `_draw_grid`) trust it thereafter.
    """
    size = matrix.shape[0]
    if triangle == "full":
        mask = np.zeros((size, size), dtype=bool)
    elif triangle == "lower":
        mask = ~np.tri(size, k=-1, dtype=bool)
    elif triangle == "diag_lower":
        mask = ~np.tri(size, k=0, dtype=bool)
    elif triangle == "diag_upper":
        mask = np.tri(size, k=-1, dtype=bool)
    elif triangle == "upper":
        mask = np.tri(size, k=0, dtype=bool)
    else:
        raise ValueError(f"Unknown triangle mode: {triangle!r}.")
    return np.ma.masked_array(matrix, mask=mask)


def _draw_grid(
    ax: "Axes",
    triangle: Literal["full", "lower", "diag_lower", "diag_upper", "upper"],
    size: int,
    color: str,
    linewidth: float | None,
) -> None:
    """Draw grid lines separating matrix cells.

    Adapted from Nilearn's `_configure_grid` helper (see module docstring for
    attribution). For each row/column `i`, one line runs along the cell's far edge and
    another along its near edge, each clipped to the visible triangle; the
    `"upper"`/`"diag_upper"` segments mirror `"lower"`/`"diag_lower"` across the
    diagonal.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing the matrix image.
    triangle : {"full", "lower", "diag_lower", "diag_upper", "upper"}
        Matrix triangle being displayed; determines which grid lines are drawn.
    size : int
        Number of rows (== number of columns) in the matrix.
    color : str
        Line color.
    linewidth : float, optional
        Line width. If not provided, uses the active Matplotlib default.

    Notes
    -----
    `triangle` is validated once by `_mask_triangle`, which `plot_matrix` always calls
    before reaching here, so any other value falls back to `"upper"` rather than
    re-validating.
    """
    lo, hi = -0.5, size - 0.5
    for i in range(size):
        near, far = i - 0.5, i + 0.5  # cell edges on either side of index i
        if triangle == "full":
            segments = [([far, far], [hi, lo]), ([hi, lo], [far, far])]
        elif triangle == "lower":
            segments = [([far, far], [hi, far]), ([far, lo], [far, far])]
        elif triangle == "diag_lower":
            segments = [([far, far], [hi, near]), ([far, lo], [near, near])]
        elif triangle == "diag_upper":
            segments = [([hi, near], [far, far]), ([near, near], [far, lo])]
        else:  # "upper"
            segments = [([far, far], [far, lo]), ([hi, far], [far, far])]
        for xs, ys in segments:
            ax.plot(xs, ys, color=color, linewidth=linewidth)


def _group_spans(groups: Sequence[Hashable]) -> list[tuple[Hashable, int, int]]:
    """Split `groups` into contiguous runs of identical values.

    Parameters
    ----------
    groups : sequence of hashable
        Group value for each row/column, e.g. `["cortex", "cortex", "thalamus"]`.

    Returns
    -------
    list[tuple[Hashable, int, int]]
        `(value, start, stop)` for each contiguous run, `stop` exclusive.
    """
    spans = []
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            spans.append((groups[start], start, i))
            start = i
    return spans


def _y_tick_label_width_inches(ax: "Axes") -> float:
    """Return the rendered width of `ax`'s y tick labels, in inches.

    Forces a draw so the tick labels' text metrics are up to date, matching Nilearn's
    `_fit_axes` helper (see module docstring for attribution).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose y tick labels to measure.

    Returns
    -------
    float
        Rendered width of the y tick labels in inches, or `0.0` if the labels have no
        extent (e.g. no labels are set).
    """
    figure = ax.figure
    # Draw so text metrics are current, then measure via get_tightbbox() with no
    # explicit renderer: it falls back to the figure's layout renderer, which every
    # backend provides. Passing canvas.get_renderer() instead would crash on non-Agg
    # backends (e.g. pdf/svg), which do not define that method.
    figure.canvas.draw()
    bbox = ax.yaxis.get_tightbbox()
    if bbox is None:
        return 0.0
    return bbox.width / figure.dpi


def _draw_group_bar(
    ax: "Axes",
    orientation: Literal["vertical", "horizontal"],
    spans: list[tuple[Hashable, int, int]],
    colors: Mapping[Hashable, str],
    show_labels: bool,
    fontsize: float | None,
) -> None:
    """Draw a strip of colored rectangles marking contiguous label groups.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Thin axes appended alongside the matrix (left for rows, top for columns).
    orientation : {"vertical", "horizontal"}
        `"vertical"` draws a column of rectangles (for row groups), `"horizontal"`
        draws a row of rectangles (for column groups).
    spans : list[tuple[Hashable, int, int]]
        Contiguous group runs from `_group_spans`.
    colors : Mapping
        Color for each group value.
    show_labels : bool
        Whether to annotate each rectangle with its group value.
    fontsize : float, optional
        Font size for group labels.
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

    # Only fix the limit/ticks along the axis *not* shared with the main matrix axes.
    # sharex/sharey ties the other axis's locator to the main plot's row/column
    # indices, so clearing it with set_xticks([])/set_yticks([]) would wipe out the
    # main plot's own tick labels too; hide it with tick_params instead.
    if orientation == "vertical":
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    else:
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.tick_params(
            axis="x",
            which="both",
            bottom=False,
            labelbottom=False,
            top=False,
            labeltop=False,
        )
    for spine in ax.spines.values():
        spine.set_visible(False)

    if not show_labels:
        return
    for value, start, stop in spans:
        midpoint = (start + stop - 1) / 2
        # Each strip's own color can be much lighter or darker than the figure
        # background, so contrast the label against that color rather than fg_color.
        label_color = _auto_fg_color(colors[value])
        if orientation == "vertical":
            ax.annotate(
                str(value),
                xy=(0.5, midpoint),
                xycoords=("axes fraction", "data"),
                ha="center",
                va="center",
                rotation=90,
                fontsize=fontsize,
                color=label_color,
            )
        else:
            ax.annotate(
                str(value),
                xy=(midpoint, 0.5),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=label_color,
            )


def plot_matrix(
    matrix: "npt.NDArray[Any] | xr.DataArray",
    labels: "Sequence[str] | Literal[False] | None" = None,
    groups: "Sequence[Hashable] | None" = None,
    group_colors: "Mapping[Hashable, str] | None" = None,
    show_group_labels: bool = True,
    triangle: Literal["full", "lower", "diag_lower", "diag_upper", "upper"] = "full",
    grid: "str | bool" = False,
    grid_linewidth: float | None = None,
    cmap: "str | Colormap | None" = None,
    vmin: float | None = None,
    vmax: float | None = None,
    auto_range: bool = True,
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
    labels : sequence[str] or False, optional
        Label for each row/column, in matrix order. If not provided and `matrix` is a
        `xarray.DataArray` with a coordinate on its first dimension, that coordinate's
        values are used. Pass `False` to force no labels even when such a coordinate
        exists.
    groups : sequence of hashable, optional
        Group value for each row/column, in matrix order, e.g. `["cortex"] * 5 +
        ["thalamus"] * 3`. Rows/columns are assumed to already be sorted by group:
        contiguous runs of equal values are drawn as colored rectangles alongside the
        matrix. If not provided, no group annotation is drawn.
    group_colors : Mapping[Hashable, str], optional
        Color for each unique value in `groups`. If not provided, colors are assigned
        automatically from a qualitative colormap.
    show_group_labels : bool, default: True
        Whether to annotate each group's colored rectangle with its value.
    triangle : {"full", "lower", "diag_lower", "diag_upper", "upper"}, default: "full"
        Which part of the matrix to display:

        - `"lower"`: only the part strictly below the diagonal.
        - `"diag_lower"`: the part below the diagonal, including it.
        - `"diag_upper"`: the part above the diagonal, including it.
        - `"upper"`: only the part strictly above the diagonal.
        - `"full"`: the entire matrix.

        `"lower"`/`"diag_lower"` and `"upper"`/`"diag_upper"` are useful for overlaying
        two matrices on the same axes (e.g. an estimate on one side and its significance
        mask on the other): call `plot_matrix` twice, passing the first call's `ax` to
        the second along with `show_colorbar=False`. Pass the same `labels` to both
        calls: since the second call reuses `ax` directly (rather than a new axes
        sharing it), it otherwise clears the labels the first call set.

    grid : str or bool, default: False
        Grid lines separating cells. Pass a color string to draw them in that color,
        `True` to use the resolved foreground color (see `fg_color`), or `False` to
        disable the grid.
    grid_linewidth : float, optional
        Width of the grid lines. If not provided, uses the active Matplotlib default.
        Ignored when `grid` is `False`.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for `matrix`. If not provided, the default depends on `auto_range`
        and the sign of `matrix` (see below); an explicitly provided `cmap` is
        always used as-is regardless of `auto_range`.
    vmin : float, optional
        Lower bound of the colormap. If not provided, defaults to the minimum value
        in `matrix`. Ignored when `auto_range` resolves to a range anchored at zero
        (see below).
    vmax : float, optional
        Upper bound of the colormap. If not provided, defaults to the maximum value
        in `matrix`. Ignored when `auto_range=True` and `matrix` has only
        non-positive values.
    auto_range : bool, default: True
        Whether to pick the colormap range and default colormap automatically based
        on the sign of `matrix`:

        - Both positive and negative values: diverging, symmetric `[-m, m]` range
          where `m = max(|vmin|, |vmax|)` (using the resolved bounds above), with
          `cmap` defaulting to `"coolwarm"`.
        - Only non-negative values: sequential `[0, vmax]` range, with `cmap`
          defaulting to `"viridis"`.
        - Only non-positive values: sequential `[vmin, 0]` range, with `cmap`
          defaulting to `"viridis_r"` (reversed, so that values near zero map to
          the same end of the colormap in both the non-negative and non-positive
          cases).

        Set to `False` to use the resolved `vmin`/`vmax` directly with no zero-anchoring
        (`cmap` then defaults to `"coolwarm"` regardless of sign).
    show_colorbar : bool, default: True
        Whether to add a colorbar for `matrix` to the figure.
    cbar_label : str, optional
        Label for the colorbar.
    title : str, optional
        Plot title.
    fontsize : float, optional
        Base font size for text elements. Title uses `fontsize` directly; the colorbar
        label and group labels use `0.9 * fontsize`; tick labels use `0.85 * fontsize`.
        If not provided, uses the active Matplotlib defaults.
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

    >>> # Overlay two matrices, e.g. an estimate and its significance mask.
    >>> pvalues = rng.uniform(size=(10, 10))
    >>> labels = [f"region_{i}" for i in range(10)]
    >>> fig, ax = plot_matrix(corr, labels=labels, triangle="diag_lower")
    >>> fig, ax = plot_matrix(
    ...     pvalues, labels=labels, triangle="upper", ax=ax, show_colorbar=False
    ... )
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    values = _validate_matrix(matrix)
    size = values.shape[0]
    labels = _sanitize_labels(
        labels if labels is not None else _default_labels(matrix), size
    )
    if groups is not None and len(groups) != size:
        raise ValueError(
            f"Length of groups ({len(groups)}) does not match matrix size ({size})."
        )

    masked = _mask_triangle(values, triangle)
    vmin, vmax, cmap = _resolve_matrix_style(values, vmin, vmax, cmap, auto_range)

    text_color = fg_color if fg_color is not None else _auto_fg_color(bg_color)
    title_fontsize, label_fontsize, tick_fontsize = _resolve_font_sizes(fontsize)

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize, layout="constrained")
        figure.patch.set_facecolor(bg_color)
    else:
        figure = ax.figure
    ax.set_facecolor(bg_color)

    divider = make_axes_locatable(ax)

    # Draw the cells as vector quads (pcolormesh) rather than a raster (imshow): imshow
    # snaps cell edges to screen pixels, so at interactive zoom they drift from the
    # grid lines (which are vector). pcolormesh keeps cells and grid aligned at any zoom.
    edges = np.arange(size + 1) - 0.5
    image = ax.pcolormesh(edges, edges, masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
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
        grid_color = text_color if grid is True else grid
        _draw_grid(ax, triangle, size, grid_color, grid_linewidth)

    if groups is not None:
        spans = _group_spans(list(groups))
        unique_values = list(dict.fromkeys(g for g, _, _ in spans))
        colors = group_colors or dict(
            zip(unique_values, _get_distinct_colors(len(unique_values)))
        )
        missing = [v for v in unique_values if v not in colors]
        if missing:
            raise ValueError(f"group_colors is missing colors for group(s): {missing}.")

        # Reserve enough room for the y tick labels so the opaque row strip does not
        # cover them: a fixed pad only fits labels up to some length (see _y_tick_
        # label_width_inches docstring for the underlying Nilearn-inspired technique).
        row_pad = _y_tick_label_width_inches(ax) + 0.1 if labels else 0.1
        row_ax = divider.append_axes("left", size="4%", pad=row_pad, sharey=ax)
        _draw_group_bar(
            row_ax,
            "vertical",
            spans,
            colors,  # type: ignore
            show_group_labels,
            label_fontsize,
        )
        col_ax = divider.append_axes("top", size="4%", pad=0.05, sharex=ax)
        _draw_group_bar(
            col_ax,
            "horizontal",
            spans,
            colors,  # type: ignore
            show_group_labels,
            label_fontsize,
        )

    if show_colorbar:
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = figure.colorbar(image, cax=cax)
        _style_colorbar(
            cbar,
            text_color,
            tick_fontsize,
            bg_color=bg_color,
            label=cbar_label,
            label_fontsize=label_fontsize,
        )

    if title:
        ax.set_title(title, color=text_color, fontsize=title_fontsize)

    return figure, ax
