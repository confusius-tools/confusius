"""Image visualization utilities for fUSI data."""

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

import napari
import numpy as np
import xarray as xr
from napari.utils.colormaps import DirectLabelColormap

from confusius._utils import find_stack_level
from confusius.atlas._structures import _build_atlas_cmap_and_norm
from confusius.extract import extract_with_mask
from confusius.signal import clean

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure, SubFigure
    from napari import Viewer
    from napari.layers import Image, Labels

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
    """Convert 1-D coordinate centers to cell edge positions for `pcolormesh`.

    Handles non-uniform spacing by using midpoints between adjacent centers as interior
    edges, and extrapolating half a step at each end.
    """
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    interior = (centers[:-1] + centers[1:]) / 2
    left = centers[0] - (centers[1] - centers[0]) / 2
    right = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[left], interior, [right]])


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


def _threshold_slices(
    slices: list[xr.DataArray],
    threshold: float | None,
    threshold_mode: Literal["lower", "upper"],
) -> list[np.ndarray]:
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
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

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

    return mcolors.LinearSegmentedColormap.from_list(
        f"{cmap.name}_thresholded", new_colors
    )


def _build_axis_label(da: xr.DataArray, dim: str) -> str:
    """Return axis label for `dim`, including units when available."""
    label = dim
    if dim in da.coords:
        units = da.coords[dim].attrs.get("units")
        if units:
            label = f"{dim} ({units})"
    return label


def _get_distinct_colors(n_colors: int) -> list[tuple[float, float, float]]:
    """Generate `n_colors` visually distinct colors."""
    import matplotlib

    cmap = matplotlib.colormaps["tab10" if n_colors <= 10 else "tab20"]
    return [tuple(cmap(i % cmap.N)[:3]) for i in range(n_colors)]


def _extract_slices(
    data: xr.DataArray,
    slice_mode: str,
    slice_coords: Sequence[float],
) -> tuple[list[xr.DataArray], list[float]]:
    """Extract 2D slices from `data` along `slice_mode`.

    Returns the slices and their actual snapped coordinate values.
    """
    slices: list[xr.DataArray] = []
    actual_coords: list[float] = []
    for coord in slice_coords:
        if slice_mode in data.coords:
            slice_da = data.sel({slice_mode: coord}, method="nearest")
            actual_coord = float(slice_da.coords[slice_mode].values)
        else:
            idx = int(round(coord))
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
    axes : numpy.ndarray, optional
        2D array of [`matplotlib.axes.Axes`][matplotlib.axes.Axes]. If not provided,
        axes will be created on the first call to
        [`add_volume`][confusius.plotting.VolumePlotter.add_volume].
    black_bg : bool, default: True
        Whether to set the figure background to black.
    yincrease : bool, default: False
        Whether the y-axis increases upward. When `False`, y coordinates decrease
        upward.
    xincrease : bool, default: True
        Whether the x-axis increases to the right.
    """

    def __init__(
        self,
        slice_mode: str = "z",
        figure: "Figure | None" = None,
        axes: "npt.NDArray[Any] | None" = None,
        *,
        black_bg: bool = True,
        yincrease: bool = False,
        xincrease: bool = True,
    ):
        self.slice_mode = slice_mode
        self.axes = axes
        self._user_provided_axes = axes is not None
        if figure is None and axes is not None:
            self.figure = axes.flat[0].figure
        else:
            self.figure = figure
        self._black_bg = black_bg
        self._yincrease = yincrease
        self._xincrease = xincrease
        self._coord_to_axis: dict[float, int] = {}
        # Explicitly tracked axis data limits to avoid matplotlib's auto-margin.
        self._axis_xlims: dict[int, tuple[float, float]] = {}
        self._axis_ylims: dict[int, tuple[float, float]] = {}

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
        self.figure.patch.set_facecolor("black" if self._black_bg else "white")

    def _find_matching_axes(
        self, actual_coords: list[float], tolerance: float = 1e-6
    ) -> list[tuple[int, int]]:
        """Find axis indices matching the target coordinates.

        Uses the coordinate-to-axis mapping stored when the figure was first created,
        avoiding any dependency on axis titles.

        Parameters
        ----------
        actual_coords : list[float]
            The actual coordinate values of the slices being plotted.
        tolerance : float, default: 1e-6
            Tolerance for matching coordinates, accounting for floating-point precision.

        Returns
        -------
        list[tuple[int, int]]
            List of `(axis_flat_idx, slice_idx)` tuples for matched coordinates.
        """
        matched = []
        for slice_idx, target_coord in enumerate(actual_coords):
            for stored_coord, axis_idx in self._coord_to_axis.items():
                if abs(stored_coord - target_coord) < tolerance:
                    matched.append((axis_idx, slice_idx))
                    break
        return matched

    @property
    def _text_color(self) -> str:
        """White for dark backgrounds, black for light ones."""
        return "white" if self._black_bg else "black"

    def _style_ax(self, ax: "Axes") -> None:
        """Apply background and spine/tick styling to an axes."""
        color = self._text_color
        ax.set_facecolor("black" if self._black_bg else "white")
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

    def _warn_unmatched(self, unmatched_slices: list[tuple[int, float]]) -> None:
        """Warn about slice coordinates that could not be matched to axes."""
        unmatched_str = ", ".join(
            f"{self.slice_mode}={coord:.3g}" for _, coord in unmatched_slices
        )
        available_coords = [f"{c:.3g}" for c in self._coord_to_axis]
        warnings.warn(
            f"Could not find matching axes for slices: {unmatched_str}. "
            f"These slices will not be plotted. "
            f"Available coordinates: {available_coords}",
            stacklevel=find_stack_level(),
        )

    def _build_slice_title(self, data: xr.DataArray, coord: float) -> str:
        """Build a slice title such as `z = 0.001 mm`."""
        units = (
            data.coords[self.slice_mode].attrs.get("units")
            if self.slice_mode in data.coords
            else None
        )
        title = f"{self.slice_mode} = {coord:.3g}"
        if units:
            title += f" {units}"
        return title

    def _init_sequential_layout(
        self, actual_coords: list[float]
    ) -> list[tuple[int, int]]:
        """Initialise the coordinate-to-axis map and return sequential plot indices."""
        if not self._coord_to_axis:
            self._coord_to_axis = {
                coord: idx for idx, coord in enumerate(actual_coords)
            }
        return [(idx, idx) for idx in range(len(actual_coords))]

    def add_volume(
        self,
        data: xr.DataArray,
        *,
        slice_coords: Sequence[float] | None = None,
        match_coordinates: bool = True,
        cmap: "str | Colormap | None" = None,
        norm: "Normalize | None" = None,
        vmin: float | None = None,
        vmax: float | None = None,
        threshold: float | None = None,
        threshold_mode: Literal["lower", "upper"] = "lower",
        alpha: float = 1.0,
        show_colorbar: bool = True,
        cbar_label: str | None = None,
        show_titles: bool = True,
        show_axis_labels: bool = True,
        show_axis_ticks: bool = True,
        show_axes: bool = True,
        nrows: int | None = None,
        ncols: int | None = None,
        dpi: int | None = None,
    ) -> "VolumePlotter":
        """Plot or overlay a volume on the axes.

        Parameters
        ----------
        data : xarray.DataArray
            3D volume data. Unitary dimensions (except `slice_mode`) are squeezed
            before processing.
        slice_coords : list[float], optional
            Specific coordinates to plot. If None, uses all coordinates from data.
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
        alpha : float, default: 1.0
            Opacity of the image.
        show_colorbar : bool, default: True
            Whether to add a colorbar.
        cbar_label : str, optional
            Label for the colorbar.
        show_titles : bool, default: True
            Whether to display subplot titles.
        show_axis_labels : bool, default: True
            Whether to display axis labels.
        show_axis_ticks : bool, default: True
            Whether to display axis tick labels.
        show_axes : bool, default: True
            Whether to show all axis decorations (spines, ticks, labels). When `False`,
            overrides `show_axis_labels` and `show_axis_ticks`.
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
        """
        import matplotlib.pyplot as plt

        if np.iscomplexobj(data):
            data = xr.ufuncs.abs(data)

        squeeze_dims = [
            d for d in data.dims if d != self.slice_mode and data.sizes[d] == 1
        ]
        if squeeze_dims:
            data = data.squeeze(dim=squeeze_dims)

        if self.slice_mode not in data.dims:
            raise ValueError(
                f"slice_mode '{self.slice_mode}' is not a dimension of data. "
                f"Available dimensions: {list(data.dims)}."
            )

        if data.ndim != 3:
            raise ValueError(
                f"Data must be 3D, but got shape {data.shape} with dims "
                f"{list(data.dims)}."
            )

        display_dims = [str(d) for d in data.dims if d != self.slice_mode]
        dim_row, dim_col = display_dims[0], display_dims[1]

        has_coords = self.slice_mode in data.coords

        if slice_coords is None:
            if has_coords:
                slice_coords = list(data.coords[self.slice_mode].values)
            else:
                slice_coords = list(range(data.sizes[self.slice_mode]))

        unthresholded_slices, actual_coords = _extract_slices(
            data, self.slice_mode, slice_coords
        )
        n_slices = len(unthresholded_slices)

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

            plot_indices = matched_indices
        else:
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
                        f"({self.axes.size}). Got {n_slices} slice_coords but axes has "
                        f"shape {self.axes.shape}."
                    )

            plot_indices = self._init_sequential_layout(actual_coords)

        assert (self.axes is not None) and (self.figure is not None)

        text_color = self._text_color
        im = None

        axes_flat = self.axes.ravel()

        for axis_idx, slice_idx in plot_indices:
            ax = axes_flat[axis_idx]
            arr = thresholded_slices[slice_idx]
            coord = actual_coords[slice_idx]
            slice_da = unthresholded_slices[slice_idx]

            if dim_col in slice_da.coords:
                x_vals = _centers_to_edges(
                    slice_da.coords[dim_col].values.astype(float)
                )
            else:
                x_vals = np.arange(slice_da.sizes[dim_col] + 1, dtype=float)

            if dim_row in slice_da.coords:
                y_vals = _centers_to_edges(
                    slice_da.coords[dim_row].values.astype(float)
                )
            else:
                y_vals = np.arange(slice_da.sizes[dim_row] + 1, dtype=float)

            im = ax.pcolormesh(
                x_vals,
                y_vals,
                np.ma.masked_invalid(arr),
                cmap=cmap,
                norm=norm,
                alpha=alpha,
            )
            ax.set_aspect("equal")
            self._style_ax(ax)
            ax.set_title(
                self._build_slice_title(data, coord) if show_titles else "",
                color=text_color,
            )

            if show_axes:
                if show_axis_labels:
                    ax.set_xlabel(_build_axis_label(data, dim_col), color=text_color)
                    ax.set_ylabel(_build_axis_label(data, dim_row), color=text_color)
                if not show_axis_ticks:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
            else:
                ax.axis("off")

            # Expand stored limits to encompass overlaid volumes with different extents.
            current_xlim = self._update_stored_lim(
                self._axis_xlims, axis_idx, (float(x_vals.min()), float(x_vals.max()))
            )
            current_ylim = self._update_stored_lim(
                self._axis_ylims, axis_idx, (float(y_vals.min()), float(y_vals.max()))
            )
            self._set_ax_lims(ax, current_xlim, current_ylim)

        if not match_coordinates:
            for ax in axes_flat[n_slices:]:
                ax.set_visible(False)

        if show_colorbar and im is not None:
            non_cbar_axes = [
                ax for ax in self.figure.axes if not hasattr(ax, "_colorbar")
            ]
            cbar = self.figure.colorbar(im, ax=non_cbar_axes)
            if cbar_label is None:
                long_name = data.attrs.get("long_name")
                units = data.attrs.get("units")
                if long_name and units:
                    cbar_label = f"{long_name} ({units})"
                elif long_name:
                    cbar_label = long_name
                elif units:
                    cbar_label = f"({units})"
            if cbar_label is not None:
                cbar.set_label(cbar_label, color=text_color)

            cbar.ax.yaxis.set_tick_params(color=text_color)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=text_color)
            cbar.outline.set_edgecolor(text_color)  # type: ignore[union-attr]

        return self

    def add_contours(
        self,
        mask: xr.DataArray,
        *,
        colors: dict[int | str, str] | str | None = None,
        linewidths: float = 1.5,
        linestyles: str = "solid",
        match_coordinates: bool = True,
        slice_coords: list[float] | None = None,
        **kwargs,
    ) -> "VolumePlotter":
        """Add mask contours to existing axes.

        Parameters
        ----------
        mask : xarray.DataArray
            Integer label map in one of two formats:

            - **Flat label map**: Spatial dims only, e.g. `(z, y, x)`. Background voxels
              labeled `0`; each unique non-zero integer identifies a distinct,
              non-overlapping region. The `region` coordinate of the output holds the
              integer label values.
            - **Stacked mask format**: Has a leading `mask` dimension followed by
              spatial dims, e.g. `(mask, z, y, x)`. Each layer has values in `{0,
              region_id}` and regions may overlap. The `region` coordinate of the
              output holds the `mask` coordinate values (e.g., region label).

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
        slice_coords : list[float], optional
            Coordinate values along the plotter's `slice_mode` at which to draw
            contours. Slices are selected by nearest-neighbour lookup. If not
            provided, all coordinate values along `slice_mode` are used.
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

        # Stacked mask format: (mask, z, y, x) — one layer per region.
        if "mask" in mask.dims:
            cmap_attr = mask.attrs.get("cmap")
            norm_attr = mask.attrs.get("norm")
            # cmap/norm are dropped on Zarr save; reconstruct from rgb_lookup when
            # present so structure colors survive a serialization round-trip.
            if (cmap_attr is None or norm_attr is None) and "rgb_lookup" in mask.attrs:
                cmap_attr, norm_attr = _build_atlas_cmap_and_norm(
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
                    **kwargs,
                )
            return self

        squeeze_dims = [
            d for d in mask.dims if d != self.slice_mode and mask.sizes[d] == 1
        ]
        if squeeze_dims:
            mask = mask.squeeze(dim=squeeze_dims)

        if self.slice_mode not in mask.dims:
            raise ValueError(f"slice_mode '{self.slice_mode}' not in mask dimensions")

        if mask.ndim != 3:
            raise ValueError(f"mask must be 3D, got shape {mask.shape}")

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
                cmap_attr, norm_attr = _build_atlas_cmap_and_norm(
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

        display_dims = [str(d) for d in mask.dims if d != self.slice_mode]
        dim_row, dim_col = display_dims[0], display_dims[1]

        if slice_coords is None:
            if self.slice_mode in mask.coords:
                slice_coords = list(mask.coords[self.slice_mode].values)
            else:
                slice_coords = list(range(mask.sizes[self.slice_mode]))

        slices, actual_coords = _extract_slices(mask, self.slice_mode, slice_coords)
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
            plot_indices = self._init_sequential_layout(actual_coords)

        if self.axes is None:
            raise RuntimeError("No axes available")

        axes_flat = self.axes.ravel()

        for axis_idx, slice_idx in plot_indices:
            ax = axes_flat[axis_idx]
            slice_da = slices[slice_idx]
            slice_data = slice_da.values

            if dim_col in slice_da.coords:
                x_coords = slice_da.coords[dim_col].values.astype(float)
            else:
                x_coords = np.arange(slice_da.sizes[dim_col])
            if dim_row in slice_da.coords:
                y_coords = slice_da.coords[dim_row].values.astype(float)
            else:
                y_coords = np.arange(slice_da.sizes[dim_row])

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

                    # Map pixel indices to physical coordinates.
                    # Contours are at pixel boundaries, so we need to interpolate
                    # between coordinate centers to get edge positions.
                    # contour[:, 0] is row (y) index, contour[:, 1] is col (x) index
                    x_idx = contour[:, 1]
                    y_idx = contour[:, 0]
                    x_physical = np.interp(
                        x_idx,
                        np.arange(len(x_coords)),
                        x_coords,
                    )
                    y_physical = np.interp(
                        y_idx,
                        np.arange(len(y_coords)),
                        y_coords,
                    )
                    ax.plot(
                        x_physical,
                        y_physical,
                        color=color,
                        linewidth=linewidths,
                        linestyle=linestyles,
                        **kwargs,
                    )

            if not match_coordinates:
                ax.set_aspect("equal")

                # Compute limits from physical coordinates, not from auto-scaled
                # matplotlib limits, which may include padding.
                x_edges = _centers_to_edges(x_coords)
                y_edges = _centers_to_edges(y_coords)
                xlim = (float(x_edges.min()), float(x_edges.max()))
                ylim = (float(y_edges.min()), float(y_edges.max()))
                self._set_ax_lims(ax, xlim, ylim)
                self._style_ax(ax)
                ax.set_xlabel(_build_axis_label(mask, dim_col), color=self._text_color)
                ax.set_ylabel(_build_axis_label(mask, dim_row), color=self._text_color)
                ax.set_title(
                    self._build_slice_title(mask, actual_coords[slice_idx]),
                    color=self._text_color,
                )

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


def plot_contours(
    mask: xr.DataArray,
    *,
    colors: dict[int | str, str] | str | None = None,
    linewidths: float = 1.5,
    linestyles: str = "solid",
    slice_mode: str = "z",
    slice_coords: list[float] | None = None,
    yincrease: bool = False,
    xincrease: bool = True,
    black_bg: bool = True,
    figure: "Figure | None" = None,
    axes: "npt.NDArray[Any] | None" = None,
    **kwargs,
) -> VolumePlotter:
    """Plot mask contours as a grid of 2D slice panels.

    Displays contour lines for each labeled region in `mask` across a grid of subplots.
    Each panel shows the contours for one slice along `slice_mode`, drawn in physical
    coordinates when available.

    Parameters
    ----------
    mask : xarray.DataArray
        Integer label map in one of two formats:

        - **Flat label map**: Spatial dims only, e.g. `(z, y, x)`. Background voxels
          labeled `0`; each unique non-zero integer identifies a distinct,
          non-overlapping region. The `regions` coordinate of the output holds the
          integer label values.
        - **Stacked mask format**: Has a leading `masks` dimension followed by
          spatial dims, e.g. `(masks, z, y, x)`. Each layer has values in `{0,
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
    slice_coords : list[float], optional
        Coordinate values along `slice_mode` at which to extract slices. Slices are
        selected by nearest-neighbour lookup. If not provided, all coordinate values
        along `slice_mode` are used.
    yincrease : bool, default: False
        Whether the y-axis increases upward (`True`) or downward (`False`).
    xincrease : bool, default: True
        Whether the x-axis increases to the right (`True`) or left (`False`).
    black_bg : bool, default: True
        Whether to set the figure background to black.
    figure : matplotlib.figure.Figure, optional
        Existing figure to draw into. If not provided, a new figure is created.
    axes : numpy.ndarray, optional
        Existing 2D array of [`matplotlib.axes.Axes`][matplotlib.axes.Axes] to draw
        into. If not provided, new axes are created inside `figure`.
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
    label, then mapped to physical coordinates via linear interpolation between
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
        black_bg=black_bg,
        yincrease=yincrease,
        xincrease=xincrease,
    )

    return plotter.add_contours(
        mask,
        colors=colors,
        linewidths=linewidths,
        linestyles=linestyles,
        match_coordinates=False,
        slice_coords=slice_coords,
        **kwargs,
    )


def plot_volume(
    data: xr.DataArray,
    *,
    slice_coords: list[float] | None = None,
    slice_mode: str = "z",
    cmap: "str | Colormap | None" = None,
    norm: "Normalize | None" = None,
    vmin: float | None = None,
    vmax: float | None = None,
    threshold: float | None = None,
    threshold_mode: Literal["lower", "upper"] = "lower",
    alpha: float = 1.0,
    show_colorbar: bool = True,
    cbar_label: str | None = None,
    show_titles: bool = True,
    show_axis_labels: bool = True,
    show_axis_ticks: bool = True,
    show_axes: bool = True,
    yincrease: bool = False,
    xincrease: bool = True,
    black_bg: bool = True,
    figure: "Figure | None" = None,
    axes: "npt.NDArray[Any] | None" = None,
    nrows: int | None = None,
    ncols: int | None = None,
    dpi: int | None = None,
) -> VolumePlotter:
    """Plot 2D slices of a volume using matplotlib.

    Displays a series of 2D slices extracted along `slice_mode` as a grid of subplots.
    Each slice is rendered using physical coordinates for axis ticks when available.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array. Unitary dimensions are squeezed before processing. After
        squeezing, data must be 3D. Complex-valued data is converted to magnitude
        before display.
    slice_coords : list[float], optional
        Coordinate values along `slice_mode` at which to extract slices. Slices are
        selected by nearest-neighbour lookup. If not provided, all coordinate values
        along `slice_mode` are used.
    slice_mode : str, default: "z"
        Dimension along which to slice (e.g., `"x"`, `"y"`, `"z"`,
        `"time"`). After slicing, each panel must be 2D.
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

    alpha : float, default: 1.0
        Opacity of the image.
    show_colorbar : bool, default: True
        Whether to add a shared colorbar to the figure.
    cbar_label : str, optional
        Label for the colorbar.
    show_titles : bool, default: True
        Whether to display subplot titles showing the slice coordinate.
    show_axis_labels : bool, default: True
        Whether to display axis labels (with units when available).
    show_axis_ticks : bool, default: True
        Whether to display axis tick labels.
    show_axes : bool, default: True
        Whether to show all axis decorations (spines, ticks, labels). When `False`,
        overrides `show_axis_labels` and `show_axis_ticks`.
    yincrease : bool, default: False
        Whether the y-axis increases upward (`True`) or downward (`False`).
    xincrease : bool, default: True
        Whether the x-axis increases to the right (`True`) or left (`False`).
    black_bg : bool, default: True
        Whether to set the figure background to black.
    figure : matplotlib.figure.Figure, optional
        Existing figure to draw into. If not provided, a new figure is created.
    axes : numpy.ndarray, optional
        Existing 2D array of [`matplotlib.axes.Axes`][matplotlib.axes.Axes] to draw
        into. Must contain at least as many elements as there are slices. If not
        provided, new axes are created inside `figure`.
    nrows : int, optional
        Number of rows in the subplot grid. If not provided, computed automatically.
    ncols : int, optional
        Number of columns in the subplot grid. If not provided, computed automatically.
    dpi : int, optional
        Figure resolution in dots per inch. Ignored when `figure` is provided.

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
    correctly. Because each panel is drawn in physical coordinate space, multiple calls
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
        black_bg=black_bg,
        yincrease=yincrease,
        xincrease=xincrease,
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
        show_titles=show_titles,
        show_axis_labels=show_axis_labels,
        show_axis_ticks=show_axis_ticks,
        show_axes=show_axes,
        nrows=nrows,
        ncols=ncols,
        dpi=dpi,
    )


def plot_napari(
    data: xr.DataArray,
    show_colorbar: bool = True,
    show_scale_bar: bool = True,
    dim_order: tuple[str, ...] | None = None,
    viewer: "Viewer | None" = None,
    layer_type: Literal["image", "labels"] = "image",
    **layer_kwargs,
) -> "tuple[Viewer, Image | Labels]":
    """Display fUSI data using the Napari viewer.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array to visualize. Expected dimensions are (time, z, y, x) where
        z is the elevation/stacking axis, y is depth, and x is lateral. Use
        `dim_order` to specify a different dimension ordering. Can be image data
        or label/mask data (e.g., ROIs, segmentations).
    show_colorbar : bool, default: True
        Whether to show the colorbar. Only applies to image layers.
    show_scale_bar : bool, default: True
        Whether to show the scale bar.
    dim_order : tuple[str, ...], optional
        Dimension ordering for the spatial axes (last three dimensions). If not
        provided, the ordering of the last three dimensions in `data` is used.
    viewer : napari.Viewer, optional
        Existing Napari viewer to add the layer to. If not provided, a new viewer
        is created.
    layer_type : {"image", "labels"}, default: "image"
        Type of layer to create. Use "image" for fUSI data and "labels" for
        ROI masks, segmentations, or other label data.
    **layer_kwargs
        Additional keyword arguments passed to the layer creation method
        (`napari.imshow` for images or `viewer.add_labels` for labels).
        For image layers, if `data.attrs` contains `"cmap"` and `"colormap"`
        is not in `layer_kwargs`, the attribute is used as the colormap.
        For labels layers, if `data.attrs` contains `"cmap"` and `"norm"`
        (as set by atlas functions) and `"colormap"` is not in `layer_kwargs`,
        a per-label color dict is built automatically from those attributes.

    Returns
    -------
    viewer : napari.Viewer
        The Napari viewer instance with the layer added.
    layer : napari.layers.Image or napari.layers.Labels
        The layer added to the viewer.

    Notes
    -----
    If all spatial dimensions have coordinates, their spacing is used as the scale
    parameter for Napari to ensure correct physical scaling. If any spatial dimension is
    missing coordinates, no scaling is applied. The spacing is computed as the median
    difference between consecutive coordinate values.

    When spatial coordinates carry a `units` attribute (e.g. `"m"`), the unit list
    is forwarded to Napari as the `units` layer parameter, which populates the status
    bar with physical coordinates. The scale bar is also updated to reflect the first
    found unit; it falls back to `"mm"` when no units are present on the coordinates.

    For unitary dimensions (e.g., a single-slice elevation axis in 2D+t data), the
    spacing cannot be inferred from coordinates. In that case, the function looks for a
    `voxdim` attribute on the coordinate variable
    (`data.coords[dim].attrs["voxdim"]`) and uses it as the spacing. If no such
    attribute is found, unit spacing is assumed and a warning is emitted.

    The first coordinate value of each spatial dimension is used as the `translate`
    parameter so that the image is positioned at its correct physical origin. For
    dimensions without coordinates, a translate of `0.0` is used. This ensures that
    multiple datasets with different fields of view overlay correctly when added to the
    same viewer.

    Examples
    --------
    >>> import xarray as xr
    >>> from confusius.plotting import plot_napari
    >>> data = xr.open_zarr("output.zarr")["iq"]
    >>> viewer, layer = plot_napari(data)

    >>> # Custom contrast limits
    >>> viewer, layer = plot_napari(data, contrast_limits=(0, 100))

    >>> # Different dimension ordering (e.g., depth, elevation, lateral)
    >>> viewer, layer = plot_napari(data, dim_order=("y", "z", "x"))

    >>> # Add a second dataset as a new layer in an existing viewer
    >>> viewer, layer = plot_napari(data1)
    >>> viewer, layer = plot_napari(data2, viewer=viewer)

    >>> # Display ROI labels (e.g., segmentation mask)
    >>> roi_mask = xr.open_zarr("output.zarr")["roi_mask"]
    >>> viewer, layer = plot_napari(roi_mask, layer_type="labels")

    >>> # Overlay labels on existing image
    >>> viewer, layer = plot_napari(data)
    >>> viewer, layer = plot_napari(roi_mask, viewer=viewer, layer_type="labels")
    """
    all_dims = list(data.dims)
    time_dim = "time" if "time" in all_dims else None
    spatial_dims = [d for d in all_dims if d != time_dim]

    if dim_order is not None and set(dim_order) != set(spatial_dims):
        raise ValueError(
            f"dim_order {dim_order} does not match spatial dimensions {spatial_dims}. "
            "Ensure 'dim_order' contains all spatial dimension names."
        )

    # Warnings for undefined dims are emitted by .fusi.spacing; fall back to 1.0 so the
    # scale bar still works.
    spacing = data.fusi.spacing
    scale = [s if (s := spacing[dim]) is not None else 1.0 for dim in spatial_dims]

    # .origin falls back to 0.0 for dimensions without coordinates.
    coord_translates = [data.fusi.origin[dim] for dim in spatial_dims]

    coord_units = [
        data.coords[dim].attrs.get("units") if dim in data.coords else None
        for dim in spatial_dims
    ]

    # Napari requires units to cover ALL dims (including time). Prepend time unit so
    # that the list length matches ndim; passing None is accepted for unlabelled axes.
    if time_dim is not None:
        time_unit = (
            data.coords[time_dim].attrs.get("units")
            if time_dim in data.coords
            else None
        )
        all_units: list[str | None] = [time_unit, *coord_units]
    else:
        all_units = coord_units

    layer_kwargs.setdefault("name", data.name)
    if any(u is not None for u in all_units):
        layer_kwargs.setdefault("units", all_units)

    if layer_type == "image":
        # The last 2 (2D) or 3 (3D) dimensions are the displayed spatial axes.
        if dim_order is not None:
            order = []
            if time_dim:
                order.append(all_dims.index(time_dim))
            for dim in dim_order:
                if dim in all_dims:
                    order.append(all_dims.index(dim))
            layer_kwargs["order"] = tuple(order)

        layer_kwargs.setdefault("axis_labels", all_dims)

        if "colormap" not in layer_kwargs:
            cmap_attr = data.attrs.get("cmap")
            if cmap_attr is not None:
                layer_kwargs["colormap"] = cmap_attr

        layer_kwargs.setdefault("translate", coord_translates)
        # napari.imshow stubs declare list[Image] but at runtime returns Image
        # directly — cast to silence the type checker.
        viewer, layer = napari.imshow(
            data,
            scale=scale,
            viewer=viewer,
            **layer_kwargs,
        )
        layer = cast("Image", layer)

        if show_colorbar:
            viewer.layers[0].colorbar.visible = True  # type: ignore[attr-defined]

    elif layer_type == "labels":
        layer_kwargs.setdefault("translate", coord_translates)
        if viewer is None:
            viewer = napari.Viewer()
        values = data.values
        if not np.issubdtype(values.dtype, np.integer):
            values = values.astype(np.int32)

        # Build a DirectLabelColormap from attrs when the caller has not already
        # supplied one.  This lets atlas annotations and masks carry their colormap
        # automatically into the viewer.  cmap/norm are not serializable, so they
        # may be absent after a Zarr round-trip; fall back to reconstructing them
        # from rgb_lookup when that is present.
        cmap_attr = data.attrs.get("cmap")
        norm_attr = data.attrs.get("norm")
        if (cmap_attr is None or norm_attr is None) and "rgb_lookup" in data.attrs:
            cmap_attr, norm_attr = _build_atlas_cmap_and_norm(data.attrs["rgb_lookup"])
        if (
            cmap_attr is not None
            and norm_attr is not None
            and "colormap" not in layer_kwargs
        ):
            color_dict: defaultdict[int | None, np.ndarray] = defaultdict(
                lambda: np.zeros(4, dtype=np.float32)  # unknown labels → transparent.
            )
            for label in np.unique(values):
                if label == 0:
                    continue  # background_value=0 is always transparent.
                color_dict[int(label)] = np.array(
                    cmap_attr(norm_attr(int(label))), dtype=np.float32
                )
            layer_kwargs["colormap"] = DirectLabelColormap(
                color_dict=color_dict, background_value=0
            )

        layer = viewer.add_labels(
            values,
            scale=scale,
            **layer_kwargs,
        )

    else:
        raise ValueError(
            f"Unknown layer_type: {layer_type!r}. Expected 'image' or 'labels'."
        )

    if show_scale_bar:
        viewer.scale_bar.visible = True
        scale_bar_unit = next((u for u in coord_units if u is not None), "mm")
        viewer.scale_bar.unit = scale_bar_unit

    return viewer, layer


def draw_napari_labels(
    data: xr.DataArray,
    labels_layer_name: str = "labels",
    viewer: "Viewer | None" = None,
    **kwargs,
) -> "tuple[Viewer, Labels]":
    """Open a napari viewer to interactively paint integer labels over fUSI data.

    Displays the data as an image layer and adds an empty Labels layer on top. The user
    can paint integer labels directly on the image using napari's brush tool. After
    painting, call [`labels_from_layer`][confusius.plotting.labels_from_layer] with the
    returned Labels layer and the original data to obtain an integer label map as a
    DataArray with the same spatial coordinates.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array to display as the background image. Typically a time-averaged
        power Doppler frame, e.g. `data.mean("time")`.
    labels_layer_name : str, default: "labels"
        Name assigned to the Labels layer added to the viewer.
    viewer : napari.Viewer, optional
        Existing Napari viewer to add layers to. If not provided, a new viewer
        is created via [`plot_napari`][confusius.plotting.plot_napari].
    **kwargs
        Additional keyword arguments forwarded to
        [`plot_napari`][confusius.plotting.plot_napari] for the image layer
        (e.g. `colormap`, `contrast_limits`).

    Returns
    -------
    viewer : napari.Viewer
        The Napari viewer instance with the image and Labels layers.
    labels_layer : napari.layers.Labels
        The empty Labels layer initialised to zeros. After the user paints
        labels in the viewer, pass this layer to
        [`labels_from_layer`][confusius.plotting.labels_from_layer] to obtain
        an integer label map.

    Notes
    -----
    The Labels layer is initialised with the same `scale` and `translate`
    parameters as the image layer so that the napari canvas shows a consistent
    physical coordinate frame regardless of voxel spacing or data origin.

    Examples
    --------
    >>> import xarray as xr
    >>> import confusius  # Register accessor.
    >>> pwd = xr.open_zarr("output.zarr")["power_doppler"].compute()
    >>> # Display the time-averaged image and add an interactive Labels layer.
    >>> viewer, labels_layer = draw_napari_labels(pwd.mean("time"))
    >>> # … paint labels in the viewer …
    >>> # Convert painted labels to an integer label map DataArray.
    >>> label_map = labels_from_layer(labels_layer, pwd.mean("time"))
    """
    viewer, image_layer = plot_napari(data, viewer=viewer, **kwargs)

    # Reuse the same spatial scale and translate that plot_napari computed for
    # the image layer so the Labels layer overlays correctly.
    all_dims = list(data.dims)
    time_dim = "time" if "time" in all_dims else None
    spatial_dims = [d for d in all_dims if d != time_dim]

    spacing = data.fusi.spacing
    spatial_scale = [
        s if (s := spacing[dim]) is not None else 1.0 for dim in spatial_dims
    ]
    spatial_translate = [
        float(data.coords[dim].values[0]) if dim in data.coords else 0.0
        for dim in spatial_dims
    ]
    spatial_shape = tuple(data.sizes[d] for d in spatial_dims)

    labels_array = np.zeros(spatial_shape, dtype=np.int32)
    labels_layer = viewer.add_labels(
        labels_array,
        scale=spatial_scale,
        translate=spatial_translate,
        name=labels_layer_name,
    )

    return viewer, labels_layer


def labels_from_layer(
    labels_layer: "Labels",
    data: xr.DataArray,
) -> xr.DataArray:
    """Convert a napari Labels layer to an integer label map DataArray.

    Reads the integer array painted in `labels_layer` and wraps it in a DataArray whose
    spatial dimensions and coordinates match those of `data`. The result is compatible
    with [`extract_with_labels`][confusius.extract.extract_with_labels],
    [`plot_contours`][confusius.plotting.plot_contours], and
    [`VolumePlotter.add_contours`][confusius.plotting.VolumePlotter.add_contours].

    Parameters
    ----------
    labels_layer : napari.layers.Labels
        A Labels layer populated by the user (e.g. via
        [`draw_napari_labels`][confusius.plotting.draw_napari_labels]). Integer values
        identify distinct regions; zero is the background and is excluded from
        downstream analyses.
    data : xarray.DataArray
        Reference data array. Its spatial dimensions and coordinates define the shape
        and labelling of the output. A time dimension, if present, is ignored: the
        label map is purely spatial.

    Returns
    -------
    xarray.DataArray
        Stacked integer DataArray with dims `["mask", *spatial_dims]`, where the
        `mask` coordinate holds each unique non-zero label integer. Each layer
        `mask=k` has values `k` where the user painted label `k` and `0` elsewhere.
        This format is directly compatible with
        [`extract_with_labels`][confusius.extract.extract_with_labels],
        [`plot_contours`][confusius.plotting.plot_contours], and
        [`VolumePlotter.add_contours`][confusius.plotting.VolumePlotter.add_contours],
        and can be sliced by label (e.g. `label_map.sel(mask=2)`) for per-label
        display. The `attrs` dict carries:

        - `"long_name"` — human-readable name.
        - `"labels_layer_name"` — name of the source napari layer.
        - `"rgb_lookup"` — `dict[int, list[int]]` mapping each non-zero
          label to its `[r, g, b]` color (0–255) as painted in napari.

    Notes
    -----
    The label array is taken directly from `labels_layer.data`. No
    rasterisation is performed: this is a direct read of the painted values.

    Per-label colors are read from `labels_layer.get_color(label)`, which works for both
    the default cyclic colormap and any `DirectLabelColormap` set on the layer.

    Examples
    --------
    >>> import xarray as xr
    >>> import confusius  # Register accessor.
    >>> pwd = xr.open_zarr("output.zarr")["power_doppler"].compute()
    >>> viewer, labels_layer = draw_napari_labels(pwd.mean("time"))
    >>> # … paint labels in the viewer …
    >>> label_map = labels_from_layer(labels_layer, pwd.mean("time"))
    >>> label_map.dims
    ('mask', 'z', 'y', 'x')
    >>> # Slice a single label for display alongside a seed map.
    >>> label_map.sel(mask=2)
    >>> # Use the label map for region-based analysis.
    >>> from confusius.extract import extract_with_labels
    >>> signals = extract_with_labels(pwd, label_map)
    """
    all_dims = list(data.dims)
    time_dim = "time" if "time" in all_dims else None
    spatial_dims = [d for d in all_dims if d != time_dim]

    coords = {dim: data.coords[dim] for dim in spatial_dims if dim in data.coords}

    # Build a color lookup from the napari layer so downstream consumers
    # (plot_napari, VolumePlotter.add_volume, add_contours) can render each
    # label with exactly the color the user painted in the viewer.
    # get_color() returns RGBA in [0, 1] for any non-zero label and works for
    # both the default CyclicLabelColormap and any DirectLabelColormap.
    label_array = np.asarray(labels_layer.data)
    unique_labels = np.unique(label_array)
    unique_labels = unique_labels[unique_labels != 0]
    rgb_lookup: dict[int, list[int]] = {}
    for label in unique_labels:
        rgba = labels_layer.get_color(int(label))
        if rgba is not None:
            # Store 0-255 RGB (drop alpha) to match the Atlas convention.
            rgb_lookup[int(label)] = [int(round(c * 255)) for c in rgba[:3]]

    # Build one layer per label so the output matches the stacked mask format
    # returned by Atlas.get_masks: dims=["mask", *spatial_dims] with the
    # mask coordinate holding integer label IDs. This allows per-label slicing
    # (e.g. label_map.sel(mask=2)) and is directly accepted by
    # extract_with_labels, plot_contours, and add_contours.
    layers = [np.where(label_array == k, k, 0).astype(np.int32) for k in unique_labels]
    stacked = (
        np.stack(layers, axis=0)
        if len(layers) > 0
        else np.empty((0, *label_array.shape), dtype=np.int32)
    )

    return xr.DataArray(
        stacked,
        dims=["mask", *spatial_dims],
        coords={"mask": unique_labels.astype(np.int32), **coords},
        attrs={
            "long_name": "Drawn label map",
            "labels_layer_name": labels_layer.name,
            "rgb_lookup": rgb_lookup,
        },
    )


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
    black_bg: bool = False,
    ax: "Axes | None" = None,
) -> tuple["Figure | SubFigure", "Axes"]:
    """Plot voxel intensities across time as a raster image.

    A carpet plot (also known as "grayplot" or "Power plot") displays voxel
    intensities as a 2D raster image with time on the x-axis and voxels on
    the y-axis. Each row represents one voxel's time series, typically
    standardized to z-scores.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array with a 'time' dimension.
    mask : xarray.DataArray, optional
        Boolean mask with same spatial dimensions and coordinates as `data`. True
        values indicate voxels to include. If not provided, all non-zero voxels
        from the data are included.
    detrend_order : int, optional
        Polynomial order for detrending:

        - `0`: Remove mean (constant detrending).
        - `1`: Remove linear trend using least squares regression.
        - `2+`: Remove polynomial trend of specified order.

        If not provided, no detrending is applied.
    standardize : bool, default: True
        Whether to standardize each voxel's time series to z-scores.
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
    black_bg : bool, default: False
        Whether to use a black figure background with white foreground elements
        (spines, ticks, labels). Use `True` for dark-themed figures.
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
    import matplotlib.pyplot as plt

    if np.iscomplexobj(data):
        data = xr.ufuncs.abs(data)

    if "time" not in data.dims or "time" not in data.coords:
        raise ValueError("Data must have 'time' dimension and coordinates.")

    n_timepoints = data.sizes["time"]

    non_zero_voxels = (data != 0).any(dim="time")
    if mask is None:
        mask = non_zero_voxels
    else:
        mask = mask & non_zero_voxels

    # extract_with_mask creates a MultiIndex on voxels that makes xarray plotting
    # fail; we need to drop the spatial coordinates.
    spatial_dims = [d for d in data.dims if d != "time"]
    signals = extract_with_mask(data, mask)
    signals = signals.drop_vars(spatial_dims + ["space"])

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
        # For decimation, we get the smallest power of 2 greater than the number of
        # volumes divided by the threshold.
        n_decimations = int(
            np.ceil(np.log2(np.ceil(n_timepoints / decimation_threshold)))
        )
        decimation_factor = 2**n_decimations
        signals = signals[::decimation_factor, :]

    text_color = "white" if black_bg else "black"
    bg_color = "black" if black_bg else "white"

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
        figure.patch.set_facecolor(bg_color)
    else:
        figure = ax.figure

    ax.set_facecolor(bg_color)

    quad = signals.T.plot(cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, yincrease=False)  # type: ignore[call-arg]

    if quad.colorbar is not None:
        cbar = quad.colorbar
        cbar.ax.yaxis.set_tick_params(color=text_color)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=text_color)
        cbar.ax.yaxis.label.set_color(text_color)
        cbar.outline.set_edgecolor(text_color)
        cbar.ax.set_facecolor(bg_color)

    ax.grid(False)
    ax.set_yticks([])
    ax.set_ylabel("Voxels", color=text_color)
    ax.set_xlabel(_build_axis_label(data, "time").capitalize(), color=text_color)
    ax.tick_params(colors=text_color)

    if title:
        ax.set_title(title, color=text_color)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_edgecolor(text_color)
    ax.spines["left"].set_edgecolor(text_color)

    return figure, ax
