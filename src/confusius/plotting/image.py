"""Image visualization utilities for fUSI data."""

import warnings
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import xarray as xr

from confusius._utils import find_stack_level
from confusius.extract import extract_with_mask
from confusius.signal import clean

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure, SubFigure
    from napari import Viewer
    from napari.layers import Image

# Import napari at module level for backward compatibility and test mocking.
# This is a lightweight import (~111ms) compared to matplotlib/scipy.
import napari  # noqa: F401

_BASE_SIZE = 4.0
"""Base subplot size for VolumePlotter when creating new figures.

Actual figure size is computed as ``(subplot_size * ncols + 1 inch for colorbar,
subplot_size * nrows)`` and then constrained to a maximum size.
"""


def _compute_grid_dims(
    n_slices: int, nrows: int | None, ncols: int | None
) -> tuple[int, int]:
    """Return (nrows, ncols) for a grid of ``n_slices`` panels."""
    if nrows is None and ncols is None:
        _ncols = int(np.ceil(np.sqrt(n_slices)))
        _nrows = int(np.ceil(n_slices / _ncols))
    elif ncols is None:
        assert nrows is not None
        _nrows = nrows
        _ncols = int(np.ceil(n_slices / _nrows))
    elif nrows is None:
        _ncols = ncols
        _nrows = int(np.ceil(n_slices / _ncols))
    else:
        _nrows, _ncols = nrows, ncols
    return _nrows, _ncols


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Convert 1-D coordinate centers to cell edge positions for ``pcolormesh``.

    Handles non-uniform spacing by using midpoints between adjacent centers as interior
    edges, and extrapolating half a step at each end.
    """
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    interior = (centers[:-1] + centers[1:]) / 2
    left = centers[0] - (centers[1] - centers[0]) / 2
    right = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[left], interior, [right]])


def _build_threshold_cmap(
    base_cmap: "Colormap",
    vmin: float,
    vmax: float,
    threshold: float,
    threshold_mode: Literal["lower", "upper"],
) -> "Colormap":
    """Build colormap with grey band indicating thresholded regions.

    For ``threshold_mode='lower'``: grey between ``[-threshold, threshold]``.
    For ``threshold_mode='upper'``: grey outside ``[-threshold, threshold]``.
    """
    import matplotlib.colors as mcolors

    grey_rgb = (0.5, 0.5, 0.5, 1.0)
    value_range = vmax - vmin

    if value_range <= 0:
        return base_cmap

    grey_start = 0.0
    grey_end = 0.0
    grey_start_low = 0.0
    grey_end_low = 0.0
    grey_start_high = 1.0
    grey_end_high = 1.0

    if threshold_mode == "lower":
        grey_start = max(0.0, (max(vmin, -threshold) - vmin) / value_range)
        grey_end = min(1.0, (min(vmax, threshold) - vmin) / value_range)
    else:
        grey_end_low = min(1.0, (min(vmax, -threshold) - vmin) / value_range)
        grey_start_high = max(0.0, (max(vmin, threshold) - vmin) / value_range)

    n_colors = 256
    cmap_colors = [
        (i / (n_colors - 1), base_cmap(i / (n_colors - 1))) for i in range(n_colors)
    ]

    new_colors: list[tuple[float, tuple[float, float, float, float]]]
    if threshold_mode == "lower":
        colors_before = [c for c in cmap_colors if c[0] <= grey_start]
        colors_after = [c for c in cmap_colors if c[0] >= grey_end]
        if colors_before and colors_after and grey_start < grey_end:
            grey_colors = [
                (grey_start, grey_rgb),
                (grey_end, colors_after[0][1]),
            ]
        else:
            grey_colors = []
        new_colors = colors_before + grey_colors + colors_after
    else:
        colors_middle = [
            c for c in cmap_colors if grey_end_low <= c[0] <= grey_start_high
        ]
        if colors_middle:
            grey_low: list[tuple[float, tuple[float, float, float, float]]] = []
            grey_high: list[tuple[float, tuple[float, float, float, float]]] = []
            if grey_start_low < grey_end_low:
                grey_low = [
                    (grey_start_low, cmap_colors[0][1]),
                    (grey_end_low, grey_rgb),
                ]
            if grey_start_high < grey_end_high:
                grey_high = [
                    (grey_start_high, grey_rgb),
                    (grey_end_high, cmap_colors[-1][1]),
                ]
            new_colors = grey_low + colors_middle + grey_high
        else:
            new_colors = cmap_colors

    return mcolors.LinearSegmentedColormap.from_list(
        f"{base_cmap.name}_thresholded", new_colors
    )


class VolumePlotter:
    """Manager for volume slice plots with coordinate-based overlay support.

    This class maintains the state of a figure with multiple axes, each representing a
    slice through a volume at a specific coordinate. It enables overlaying multiple
    volumes on the same axes by matching coordinates.

    Parameters
    ----------
    slice_mode : str
        The dimension along which slices are taken (e.g., ``"z"``).
    figure : matplotlib.figure.Figure, optional
        The figure containing the axes. If not provided, a new figure will be created
        on the first call to add_volume.
    axes : numpy.ndarray, optional
        2D array of matplotlib.axes.Axes objects. If not provided, axes will be created
        on the first call to add_volume.
    """

    def __init__(
        self,
        slice_mode: str = "z",
        figure: "Figure | None" = None,
        axes: "npt.NDArray[Any] | None" = None,
    ):
        self.slice_mode = slice_mode
        self.figure = figure
        self.axes = axes
        self._coord_to_axis: dict[float, int] = {}
        self._black_bg: bool = True
        # Explicitly tracked axis data limits to avoid matplotlib's auto-margin.
        self._axis_xlims: dict[int, tuple[float, float]] = {}
        self._axis_ylims: dict[int, tuple[float, float]] = {}

    def _ensure_figure(
        self,
        n_slices: int,
        nrows: int | None = None,
        ncols: int | None = None,
        dpi: int | None = None,
        black_bg: bool = True,
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
        self.figure.patch.set_facecolor("black" if black_bg else "white")

    def _find_matching_axes(
        self, actual_coords: list[float], tolerance: float = 1e-6
    ) -> list[tuple[int, int]]:
        """Find axis indices matching the target coordinates.

        Uses the coordinate-to-axis mapping stored when the figure was first created,
        avoiding any dependency on axis titles.

        Returns a list of (axis_flat_idx, slice_idx) tuples for matched coordinates.
        """
        matched = []
        for slice_idx, target_coord in enumerate(actual_coords):
            for stored_coord, axis_idx in self._coord_to_axis.items():
                if abs(stored_coord - target_coord) < tolerance:
                    matched.append((axis_idx, slice_idx))
                    break
        return matched

    def add_volume(
        self,
        data: xr.DataArray,
        slice_coords: Sequence[float] | None = None,
        match_coordinates: bool = True,
        cmap: "str | Colormap" = "gray",
        vmin: float | None = None,
        vmax: float | None = None,
        threshold: float | None = None,
        threshold_mode: Literal["lower", "upper"] = "lower",
        alpha: float = 1.0,
        show_colorbar: bool = True,
        cbar_label: str | None = None,
        display_titles: bool = True,
        display_axis_labels: bool = True,
        display_axis_ticks: bool = True,
        axis_off: bool = False,
        origin: Literal["upper", "lower"] = "upper",
        black_bg: bool = True,
        nrows: int | None = None,
        ncols: int | None = None,
        dpi: int | None = None,
    ) -> "VolumePlotter":
        """Plot or overlay a volume on the axes.

        Parameters
        ----------
        data : xarray.DataArray
            3D volume data. Unitary dimensions are squeezed before processing.
        slice_coords : list[float], optional
            Specific coordinates to plot. If None, uses all coordinates from data.
        match_coordinates : bool, default: True
            If True, match slice coordinates to the stored coordinate mapping (for
            overlays). If False, plot sequentially on all axes (requires exact axis
            count match).
        cmap : str, default: "gray"
            Matplotlib colormap name.
        vmin, vmax : float, optional
            Color scale limits. Auto-computed from data if not provided.
        threshold : float, optional
            Threshold value for masking.
        threshold_mode : {"lower", "upper"}, default: "lower"
            Whether to mask values below or above threshold.
        alpha : float, default: 1.0
            Opacity of the overlay (0-1).
        show_colorbar : bool, default: True
            Whether to add a colorbar.
        cbar_label : str, optional
            Label for the colorbar.
        display_titles : bool, default: True
            Whether to display subplot titles.
        display_axis_labels : bool, default: True
            Whether to display axis labels.
        display_axis_ticks : bool, default: True
            Whether to display axis tick labels.
        axis_off : bool, default: False
            Whether to turn off all axis decorations.
        origin : {"upper", "lower"}, default: "upper"
            Position of the y-axis origin.
        black_bg : bool, default: True
            Whether to set the figure background to black. Only applies when
            creating a new figure (match_coordinates=False).
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

        data = data.squeeze()

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

        slices = []
        actual_coords = []
        for coord in slice_coords:
            if has_coords:
                s = data.sel({self.slice_mode: coord}, method="nearest")
                actual_coords.append(float(s.coords[self.slice_mode].values))
            else:
                idx = int(round(float(coord)))
                idx = max(0, min(idx, data.sizes[self.slice_mode] - 1))
                s = data.isel({self.slice_mode: idx})
                actual_coords.append(float(idx))
            slices.append(s)

        n_slices = len(slices)

        if vmin is None or vmax is None:
            all_vals = np.concatenate([s.values.ravel() for s in slices])
            if len(all_vals) > 0:
                if vmin is None:
                    vmin = float(np.percentile(all_vals, 2))
                if vmax is None:
                    vmax = float(np.percentile(all_vals, 98))
            else:
                vmin = vmin if vmin is not None else 0.0
                vmax = vmax if vmax is not None else 1.0

        if threshold is not None:
            slice_arrays = []
            for s in slices:
                arr = s.values.astype(float)
                mask = (
                    np.abs(arr) < threshold
                    if threshold_mode == "lower"
                    else np.abs(arr) > threshold
                )
                arr[mask] = np.nan
                slice_arrays.append(arr)
        else:
            slice_arrays = [s.values for s in slices]

        if threshold is not None:
            base_cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
            # Build colormap with grey inserted where values are thresholded. This
            # visually indicates which regions were masked.
            cmap = _build_threshold_cmap(
                base_cmap, vmin, vmax, threshold, threshold_mode
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

            if not matched_indices:
                return self

            sorted_matched = sorted(matched_indices, key=lambda x: x[1])
            slices = [slices[orig_idx] for _, orig_idx in sorted_matched]
            slice_arrays = [slice_arrays[orig_idx] for _, orig_idx in sorted_matched]
            actual_coords = [actual_coords[orig_idx] for _, orig_idx in sorted_matched]
            plot_indices = [
                (axis_idx, new_idx)
                for new_idx, (axis_idx, _) in enumerate(sorted_matched)
            ]
            n_slices = len(slices)
        else:
            if self.axes is not None:
                if self.figure is None:
                    self.figure = self.axes.ravel()[0].figure
                if n_slices != self.axes.size:
                    raise ValueError(
                        f"Number of slices ({n_slices}) must match number of axes "
                        f"({self.axes.size}) when match_coordinates=False."
                    )
            else:
                x_range, y_range = None, None
                if (
                    slices
                    and dim_col in slices[0].coords
                    and dim_row in slices[0].coords
                ):
                    x_range = float(
                        np.max(slices[0].coords[dim_col].values)
                        - np.min(slices[0].coords[dim_col].values)
                    )
                    y_range = float(
                        np.max(slices[0].coords[dim_row].values)
                        - np.min(slices[0].coords[dim_row].values)
                    )
                self._ensure_figure(
                    n_slices,
                    nrows=nrows,
                    ncols=ncols,
                    dpi=dpi,
                    black_bg=black_bg,
                    x_range=x_range,
                    y_range=y_range,
                )

            self._black_bg = black_bg
            self.figure.patch.set_facecolor("black" if black_bg else "white")  # type: ignore[union-attr]

            if not self._coord_to_axis:
                self._coord_to_axis = {
                    coord: idx for idx, coord in enumerate(actual_coords)
                }
            plot_indices = [(idx, idx) for idx in range(n_slices)]

        assert (self.axes is not None) and (self.figure is not None)

        text_color = "white" if self._black_bg else "black"
        im = None

        axes_flat = self.axes.ravel()

        def _build_axis_label(dim: str) -> str:
            label = dim
            if dim in data.coords:
                units = data.coords[dim].attrs.get("units")
                if units:
                    label = f"{dim} ({units})"
            return label

        for axis_idx, slice_idx in plot_indices:
            ax = axes_flat[axis_idx]
            arr = slice_arrays[slice_idx]
            coord = actual_coords[slice_idx]
            slice_da = slices[slice_idx]

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
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
            )
            ax.set_aspect("equal")

            if self._black_bg:
                ax.set_facecolor("black")
                for spine in ax.spines.values():
                    spine.set_edgecolor("white")
                ax.tick_params(colors="white", which="both")
            else:
                ax.set_facecolor("white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("black")
                ax.tick_params(colors="black", which="both")

            slice_units = (
                data.coords[self.slice_mode].attrs.get("units")
                if self.slice_mode in data.coords
                else None
            )
            title = f"{self.slice_mode} = {coord:.3g}"
            if slice_units:
                title += f" {slice_units}"
            ax.set_title(title if display_titles else "", color=text_color)

            if display_axis_labels:
                ax.set_xlabel(_build_axis_label(dim_col), color=text_color)
                ax.set_ylabel(_build_axis_label(dim_row), color=text_color)

            if not display_axis_ticks:
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            if axis_off:
                ax.axis("off")

            # When overlaying volumes with different spatial extents, we need to expand
            # the axis limits to encompass all data, not just the first volume.
            current_xlim = (float(x_vals.min()), float(x_vals.max()))
            current_ylim = (float(y_vals.min()), float(y_vals.max()))

            if axis_idx in self._axis_xlims:
                prev_xlim = self._axis_xlims[axis_idx]
                current_xlim = (
                    min(prev_xlim[0], current_xlim[0]),
                    max(prev_xlim[1], current_xlim[1]),
                )
            self._axis_xlims[axis_idx] = current_xlim
            ax.set_xlim(current_xlim)

            if axis_idx in self._axis_ylims:
                prev_ylim = self._axis_ylims[axis_idx]
                current_ylim = (
                    min(prev_ylim[0], current_ylim[0]),
                    max(prev_ylim[1], current_ylim[1]),
                )
            self._axis_ylims[axis_idx] = current_ylim

            if origin == "upper":
                ax.set_ylim(current_ylim[1], current_ylim[0])
            else:
                ax.set_ylim(current_ylim)

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

            if self._black_bg:
                cbar.ax.yaxis.set_tick_params(color="white")
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
                cbar.outline.set_edgecolor("white")  # type: ignore[union-attr]
            else:
                cbar.ax.yaxis.set_tick_params(color="black")
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")
                cbar.outline.set_edgecolor("black")  # type: ignore[union-attr]

        return self

    def savefig(self, fname: str, **kwargs) -> None:
        """Save the figure to a file.

        Parameters
        ----------
        fname : str
            Path to save the figure. Extension determines format (e.g., ``.png``,
            ``.pdf```).
        **kwargs
            Additional keyword arguments passed to
            [`matplotlib.figure.Figure.savefig`][matplotlib.figure.Figure.savefig].
        """
        if self.figure is None:
            raise RuntimeError("Figure not initialized.")
        self.figure.savefig(fname, **kwargs)

    def show(self) -> None:
        """Display the figure.

        This calls [`matplotlib.pyplot.show`][matplotlib.pyplot.show] to display the
        figure in an interactive window.
        """
        import matplotlib.pyplot as plt

        plt.show()

    def close(self) -> None:
        """Close the figure window.

        This closes the matplotlib figure window associated with this plotter. After
        closing, the figure cannot be displayed or saved.
        """
        import matplotlib.pyplot as plt

        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None
            self.axes = None
            self._coord_to_axis = {}
            self._black_bg = True
            self._axis_xlims = {}
            self._axis_ylims = {}


def plot_napari(
    data: xr.DataArray,
    scale_method: Literal["db", "log", "power"] | None = None,
    scale_kwargs: dict[str, Any] | None = None,
    show_colorbar: bool = True,
    show_scale_bar: bool = True,
    dim_order: tuple[str, ...] | None = None,
    viewer: "Viewer | None" = None,
    **imshow_kwargs,
) -> tuple["Viewer", list["Image"]]:
    """Display fUSI data using the Napari viewer.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array to visualize. Expected dimensions are (time, z, y, x) where
        z is the elevation/stacking axis, y is depth, and x is lateral. Use
        ``dim_order`` to specify a different dimension ordering.
    scale_method : {"db", "log", "power"}, optional
        Scaling method to apply before display. Use ``"db"`` for decibel scaling,
        ``"log"`` for natural log, ``"power"`` for power scaling. If not provided,
        no scaling is applied.
    scale_kwargs : dict, optional
        Keyword arguments to pass to the scaling method. For example, ``{"factor": 20}``
        for db scaling or ``{"exponent": 0.5}`` for power scaling.
    show_colorbar : bool, default: True
        Whether to show the colorbar.
    show_scale_bar : bool, default: True
        Whether to show the scale bar.
    dim_order : tuple[str, ...], optional
        Dimension ordering for the spatial axes (last three dimensions). If not
        provided, the ordering of the last three dimensions in `data` is used.
    viewer : napari.Viewer, optional
        Existing Napari viewer to add the image layer to. If not provided, a new viewer
        is created.
    **imshow_kwargs
        Additional keyword arguments passed to `napari.imshow`, such as
        ``contrast_limits``, ``colormap``, etc.

    Returns
    -------
    viewer : napari.Viewer
        The Napari viewer instance.
    layer : napari.layers.Image
        The image layer added to the viewer.

    Notes
    -----
    If all spatial dimensions have coordinates, their spacing is used as the scale
    parameter for Napari to ensure correct physical scaling. If any spatial dimension is
    missing coordinates, no scaling is applied. The spacing is computed as the median
    difference between consecutive coordinate values.

    For unitary dimensions (e.g., a single-slice elevation axis in 2D+t data), the
    spacing cannot be inferred from coordinates. In that case, the function looks for a
    ``voxdim`` attribute on the coordinate variable
    (``data.coords[dim].attrs["voxdim"]``) and uses it as the spacing. If no such
    attribute is found, unit spacing is assumed and a warning is emitted.

    The first coordinate value of each spatial dimension is used as the ``translate``
    parameter so that the image is positioned at its correct physical origin. For
    dimensions without coordinates, a translate of ``0.0`` is used. This ensures that
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

    >>> # Amplitude scaling (factor=20)
    >>> viewer, layer = plot_napari(
    ...     data, scale_method="db", scale_kwargs={"factor": 20}
    ... )

    >>> # Decibel scaling
    >>> viewer, layer = plot_napari(data, scale_method="db")

    >>> # Different dimension ordering (e.g., depth, elevation, lateral)
    >>> viewer, layer = plot_napari(data, dim_order=("y", "z", "x"))

    >>> # Add a second dataset as a new layer in an existing viewer
    >>> viewer, layer1 = plot_napari(data1)
    >>> viewer, layer2 = plot_napari(data2, viewer=viewer)
    """
    if scale_method is not None:
        if scale_kwargs is None:
            scale_kwargs = {}

        if scale_method == "db":
            data = data.fusi.scale.db(**scale_kwargs)
        elif scale_method == "log":
            data = data.fusi.scale.log(**scale_kwargs)
        elif scale_method == "power":
            data = data.fusi.scale.power(**scale_kwargs)
        else:
            raise ValueError(
                f"Unknown scale_method: {scale_method}. "
                "Use 'db', 'log', 'power', or None."
            )

    all_dims = list(data.dims)
    time_dim = "time" if "time" in all_dims else None
    spatial_dims = [d for d in all_dims if d != time_dim]

    if dim_order is not None and set(dim_order) != set(spatial_dims):
        raise ValueError(
            f"dim_order {dim_order} does not match spatial dimensions {spatial_dims}. "
            "Ensure 'dim_order' contains all spatial dimension names."
        )

    # Build the Napari scale from coordinate spacing. Warnings for undefined dims are
    # emitted by .fusi.spacing; fall back to 1.0 so the scale bar still works.
    scale = None
    spacing = data.fusi.spacing
    coord_scales = [
        s if (s := spacing[dim]) is not None else 1.0 for dim in spatial_dims
    ]
    if coord_scales:
        scale = coord_scales

    # Build translate from the first coordinate value per spatial dim (physical origin).
    # Falls back to 0.0 for dimensions without coordinates.
    coord_translates = [
        float(data.coords[dim].values[0]) if dim in data.coords else 0.0
        for dim in spatial_dims
    ]

    # The last 2 (2D) or 3 (3D) dimensions are the displayed spatial axes.
    if dim_order is not None:
        order = []
        if time_dim:
            order.append(all_dims.index(time_dim))
        for dim in dim_order:
            if dim in all_dims:
                order.append(all_dims.index(dim))
        imshow_kwargs["order"] = tuple(order)

    imshow_kwargs.setdefault("axis_labels", all_dims)
    imshow_kwargs.setdefault("name", data.name)
    imshow_kwargs.setdefault("translate", coord_translates)
    viewer, image_layer = napari.imshow(
        data,
        scale=scale,
        viewer=viewer,
        **imshow_kwargs,
    )

    if show_colorbar:
        viewer.layers[0].colorbar.visible = True  # type: ignore[attr-defined]

    if show_scale_bar:
        viewer.scale_bar.visible = True
        viewer.scale_bar.gridded = True
        viewer.scale_bar.unit = "mm"

    return viewer, image_layer


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

        - ``0``: Remove mean (constant detrending).
        - ``1``: Remove linear trend using least squares regression.
        - ``2+``: Remove polynomial trend of specified order.

        If not provided, no detrending is applied.
    standardize : bool, default: True
        Whether to standardize each voxel's time series to z-scores.
    cmap : str, default: "gray"
        Matplotlib colormap name.
    vmin : float, optional
        Minimum value for colormap. If not provided, uses ``mean - 2*std``.
    vmax : float, optional
        Maximum value for colormap. If not provided, uses ``mean + 2*std``.
    decimation_threshold : int or None, default: 800
        If the number of timepoints exceeds this value, data is downsampled
        along the time axis to improve plotting performance. Set to ``None`` to
        disable downsampling.
    figsize : tuple[float, float], default: (10, 5)
        Figure size in inches ``(width, height)``.
    title : str, optional
        Plot title.
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
    signals = signals.drop_vars(spatial_dims + ["voxels"])

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

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    signals.T.plot(cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, yincrease=False)  # type: ignore[call-arg]

    ax.grid(False)
    ax.set_yticks([])
    ax.set_ylabel("Voxels")
    # TODO: we could get time units from attrs, or maybe use pint?
    ax.set_xlabel("Time (s)")

    if title:
        ax.set_title(title)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_position(("outward", 10))

    return figure, ax


def plot_volume(
    data: xr.DataArray,
    slice_coords: Sequence[float] | None = None,
    slice_mode: str = "z",
    nrows: int | None = None,
    ncols: int | None = None,
    threshold: float | None = None,
    threshold_mode: Literal["lower", "upper"] = "lower",
    cmap: "str | Colormap" = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 1.0,
    black_bg: bool = True,
    figure: "Figure | None" = None,
    axes: "npt.NDArray[Any] | None" = None,
    show_colorbar: bool = True,
    cbar_label: str | None = None,
    dpi: int | None = None,
    origin: Literal["upper", "lower"] = "upper",
    display_titles: bool = True,
    display_axis_labels: bool = True,
    display_axis_ticks: bool = True,
    axis_off: bool = False,
) -> "VolumePlotter":
    """Plot 2D slices of a volume using matplotlib.

    Displays a series of 2D slices extracted along ``slice_mode`` as a grid of subplots.
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
        Dimension along which to slice (e.g. ``"x"``, ``"y"``, ``"z"``, ``"time"``).
        After slicing, each panel must be 2D.
    nrows : int, optional
        Number of rows in the subplot grid. If not provided, computed automatically
        together with `ncols` to produce a near-square layout.
    ncols : int, optional
        Number of columns in the subplot grid. If not provided, computed automatically
        together with `nrows` to produce a near-square layout.
    threshold : float, optional
        Threshold applied to ``|data|``. See `threshold_mode` for the masking
        direction. If not provided, no thresholding is applied.
    threshold_mode : {"lower", "upper"}, default: "lower"
        Controls how `threshold` is applied:

        - ``"lower"``: set pixels where ``|data| < threshold`` to NaN
          (suppresses values below a minimum signal level).
        - ``"upper"``: set pixels where ``|data| > threshold`` to NaN
          (suppresses large absolute values, e.g. e.g. useful when thresholding
          decibel-scaled data).

    cmap : str or matplotlib.colors.Colormap, default: "gray"
        Matplotlib colormap name or colormap object.
    vmin : float, optional
        Lower bound of the colormap. Defaults to the 2nd percentile of the displayed
        data.
    vmax : float, optional
        Upper bound of the colormap. Defaults to the 98th percentile of the displayed
        data.
    alpha : float, default: 1.0
        Opacity of the image (0 transparent, 1 opaque).
    black_bg : bool, default: True
        Whether to set the figure and axes backgrounds to black and render all text and
        spines in white.
    figure : matplotlib.figure.Figure, optional
        Existing figure to draw into. If not provided, a new figure is created.
    axes : numpy.ndarray, optional
        Existing 2D array of [`matplotlib.axes.Axes`][matplotlib.axes.Axes] to draw
        into. Must contain at least as many elements as there are slices. If not
        provided, new axes are created inside `figure`.
    show_colorbar : bool, default: True
        Whether to add a shared colorbar to the figure.
    cbar_label : str, optional
        Label for the colorbar.
    dpi : int, optional
        Figure resolution in dots per inch. Ignored when `figure` is provided.
    origin : {"upper", "lower"}, default: "upper"
        Position of the y-axis origin. ``"upper"`` places the origin at the top
        (decreasing y coordinates downward), which is standard for medical imaging.
        ``"lower"`` places the origin at the bottom.
    display_titles : bool, default: True
        Whether to display subplot titles showing the slice coordinate.
    display_axis_labels : bool, default: True
        Whether to display axis labels (with units when available).
    display_axis_ticks : bool, default: True
        Whether to display axis tick labels.
    axis_off : bool, default: False
        Whether to turn off all axis decorations (spines, ticks, labels).
        When True, overrides display_axis_labels and display_axis_ticks.

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
    axis has ``aspect="equal"`` so that 1 unit in x matches 1 unit in y.

    NaN and Inf values (including those introduced by `threshold`) are rendered
    transparently via a masked array.

    When the figure is created internally, ``layout="constrained"`` is used so
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
    plotter = VolumePlotter(slice_mode=slice_mode, figure=figure, axes=axes)

    return plotter.add_volume(
        data=data,
        slice_coords=slice_coords,
        match_coordinates=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        threshold_mode=threshold_mode,
        alpha=alpha,
        show_colorbar=show_colorbar,
        cbar_label=cbar_label,
        display_titles=display_titles,
        display_axis_labels=display_axis_labels,
        display_axis_ticks=display_axis_ticks,
        axis_off=axis_off,
        origin=origin,
        black_bg=black_bg,
        nrows=nrows,
        ncols=ncols,
        dpi=dpi,
    )
