"""Time series plotter widget for the bottom dock."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import napari
import numpy as np
from matplotlib import colormaps as mpl_colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import QSize, QTimer
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from confusius._napari._export import (
    ExportSeries,
    PlotSeries,
    prepare_export_series,
    prompt_delimited_export_path,
    write_delimited_series,
)
from confusius._napari._theme import (
    create_export_button,
    get_napari_colors,
    style_export_button,
    style_plot_toolbar,
)
from confusius._napari._time_series._store import ImportedSeries, TimeSeriesStore

# Line styles cycled across reference layers so multi-layer plots are distinguishable
# even when point/label colors are the same.
_LAYER_LINESTYLES = ["-", "--", "-.", ":"]


class TimeSeriesPlotter(QWidget):
    """Bottom dock widget for live time series plotting.

    Supports three source modes: mouse hover (Shift + move), a Points layer, or a
    Labels layer. Imported series from a `TimeSeriesStore` are overlaid on every plot.
    The time dimension is resolved from xarray metadata when available, falling back to
    the first array dimension.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.
    store : TimeSeriesStore | None, optional
        Shared store containing imported time series to overlay on the live plot.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        store: TimeSeriesStore | None = None,
    ) -> None:
        super().__init__()
        self._viewer = viewer
        self._series_store = store
        self._cursor_pos: np.ndarray | None = None
        self._current_layer = None

        # Plot settings.
        self._ylim_min: float | None = None
        self._ylim_max: float | None = None
        self._autoscale: bool = True
        self._show_grid: bool = True
        self._zscore: bool = False

        # Whether at least one time series has been plotted (used to decide whether to
        # restore the zoom state on the next _update_plot call).
        self._has_plot: bool = False
        # Whether the previous _update_plot call had valid data. Used to avoid
        # restoring the default [0, 1] xlim that matplotlib sets when axes are cleared
        # with no data (e.g. when the cursor leaves the image).
        self._prev_ts_valid: bool = False

        # Time coordinate values for the current layer, used to map frame indices to
        # real time values on the x-axis.
        self._time_coords: np.ndarray | None = None

        # Source mode: "mouse" | "points" | "labels".
        self._source_mode: Literal["mouse", "points", "labels"] = "mouse"
        self._points_layer = None
        self._labels_layer = None
        self._ref_layers: list | None = None  # None = all image layers with time dim

        # Debounce timer for labels data changes (painting is fast, replot is slow).
        self._labels_debounce = QTimer(self)
        self._labels_debounce.setSingleShot(True)
        self._labels_debounce.setInterval(500)
        self._labels_debounce.timeout.connect(self._update_plot_from_labels)

        # Time cursor blitting state.
        self._show_cursor: bool = False
        self._vline = None
        self._bg = None  # Saved pixel buffer (without vline).
        self._cursor_frame: float = 0.0
        self._export_series: list[ExportSeries] = []
        self._mouse_plot_dirty: bool = True
        self._mouse_live_line = None
        self._mouse_imported_signature: tuple[str, ...] = ()
        self._mouse_legend_signature: tuple[str, ...] = ()

        # Throttle blit calls to ~60 fps so the napari time slider stays responsive when
        # the canvas is docked (docked canvases must synchronise repaints with the
        # QMainWindow, unlike floating ones).
        self._cursor_timer = QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.setInterval(16)  # ms → ~60 fps
        self._cursor_timer.timeout.connect(self._flush_cursor)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumHeight(200)
        self._setup_ui()
        self._setup_callbacks()
        if self._series_store is not None:
            self._series_store.plot_data_changed.connect(self._on_plot_data_changed)
            self._series_store.changed.connect(self._refresh_plot)
            # If there are already imported series, plot them instead of showing
            # instructions.
            imported = self._series_store.imported_series()
            if imported and self._render_imported_only():
                pass  # Successfully plotted imported series
            else:
                self._show_instructions()
        else:
            self._show_instructions()
        self._apply_theme()

    def sizeHint(self) -> QSize:
        """Return the preferred initial size of the widget.

        Returns
        -------
        QSize
            Preferred size of 800 x 320 pixels.
        """
        return QSize(800, 320)

    def _setup_ui(self) -> None:
        """Set up the widget UI with matplotlib canvas."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        self._figure = Figure(tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self._toolbar = NavigationToolbar(self._canvas, self)
        self._export_button = create_export_button(
            self._toolbar, "timeSeriesExportButton", self._save_current_plot
        )
        self._export_button.setEnabled(False)
        self._toolbar.addWidget(self._export_button)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._axes = self._figure.add_subplot(111)
        # Save background after each full redraw for blitting the time cursor.
        self._canvas.mpl_connect("draw_event", self._on_draw)

    def _setup_callbacks(self) -> None:
        """Set up napari event callbacks."""
        self._viewer.mouse_move_callbacks.append(self._on_mouse_move)
        self._viewer.layers.events.inserted.connect(self._on_layer_change)
        self._viewer.layers.events.removed.connect(self._on_layer_change)
        self._viewer.layers.selection.events.active.connect(self._on_layer_change)

    def _get_colors(self) -> dict:
        """Get current theme colors."""
        return get_napari_colors(self._viewer.theme)

    def _apply_theme(self) -> None:
        """Apply napari theme to matplotlib figure."""
        self._mouse_plot_dirty = True
        colors = self._get_colors()

        self._figure.patch.set_facecolor(colors["bg"])
        self._axes.set_facecolor(colors["bg"])

        style_plot_toolbar(self._toolbar, colors)
        style_export_button(self._export_button, colors)

        for spine in self._axes.spines.values():
            spine.set_edgecolor(colors["fg"])
        self._axes.tick_params(colors=colors["fg"])
        self._axes.xaxis.label.set_color(colors["fg"])
        self._axes.yaxis.label.set_color(colors["fg"])
        self._axes.title.set_color(colors["fg"])

        self._canvas.draw_idle()

    def set_ylim(self, min_val: float | None, max_val: float | None) -> None:
        """Set Y-axis limits.

        Parameters
        ----------
        min_val : float | None
            Minimum Y value, or None for auto.
        max_val : float | None
            Maximum Y value, or None for auto.
        """
        self._ylim_min = min_val
        self._ylim_max = max_val
        self._refresh_plot()

    def get_ylim(self) -> tuple[float, float] | None:
        """Return current Y-axis limits, or None if no plot exists.

        Returns
        -------
        tuple[float, float] | None
            Current `(min, max)` Y-axis limits, or None if axes are empty.
        """
        if not self._axes.get_lines():
            return None
        return self._axes.get_ylim()

    def set_autoscale(self, enabled: bool) -> None:
        """Enable or disable Y-axis autoscaling.

        Parameters
        ----------
        enabled : bool
            Whether to autoscale the Y-axis.
        """
        self._autoscale = enabled
        self._refresh_plot()

    def set_show_grid(self, enabled: bool) -> None:
        """Enable or disable grid display.

        Parameters
        ----------
        enabled : bool
            Whether to show the grid.
        """
        self._show_grid = enabled
        self._refresh_plot()

    def set_zscore(self, enabled: bool) -> None:
        """Enable or disable Z-scoring of time series.

        Parameters
        ----------
        enabled : bool
            Whether to Z-score the time series.
        """
        self._zscore = enabled
        self._refresh_plot()

    def set_show_cursor(self, enabled: bool) -> None:
        """Enable or disable the time cursor.

        Parameters
        ----------
        enabled : bool
            Whether to show the time cursor.
        """
        self._show_cursor = enabled
        self._vline = None
        self._bg = None
        self._refresh_plot()

    def set_time_cursor(self, frame_idx: float) -> None:
        """Schedule a blitted time-cursor update.

        The blit is deferred to a ~60 fps timer so that rapid step events
        from the napari time slider do not block the main thread waiting for
        the docked canvas to repaint.

        Parameters
        ----------
        frame_idx : float
            Frame index to position the cursor at.
        """
        self._cursor_frame = frame_idx
        if not self._cursor_timer.isActive():
            self._cursor_timer.start()

    def _frame_to_x(self, frame: float) -> float:
        """Convert a napari frame index to the x-axis value used for plotting.

        When time coordinates are available the frame index is mapped to the
        corresponding time value; otherwise the frame index is returned as-is.
        """
        if self._time_coords is not None:
            idx = int(round(frame))
            if 0 <= idx < len(self._time_coords):
                return float(self._time_coords[idx])
        return frame

    def _flush_cursor(self) -> None:
        """Perform the actual blit for the current cursor position."""
        if self._vline is not None and self._bg is not None:
            try:
                x_cursor = self._frame_to_x(self._cursor_frame)
                self._canvas.restore_region(self._bg)
                self._vline.set_xdata([x_cursor, x_cursor])
                self._axes.draw_artist(self._vline)
                self._canvas.blit(self._figure.bbox)
            except Exception:  # noqa: BLE001
                self._bg = None  # Force a full redraw next time.

    def _on_draw(self, event) -> None:
        """After each full redraw, save the background and blit the vline."""
        if self._vline is None:
            return
        try:
            self._bg = self._canvas.copy_from_bbox(self._figure.bbox)
            self._axes.draw_artist(self._vline)
            self._canvas.blit(self._figure.bbox)
        except Exception:  # noqa: BLE001
            self._bg = None

    def on_theme_changed(self) -> None:
        """Handle napari theme change (called by parent panel)."""
        self._apply_theme()
        if self._cursor_pos is not None and self._current_layer is not None:
            self._update_plot()

    def _on_mouse_move(self, viewer, event) -> None:
        """Handle mouse move events."""
        if self._source_mode != "mouse":
            return
        if "Shift" not in event.modifiers:
            return

        layer = self._active_layer()
        if layer is None:
            return

        self._current_layer = layer
        # Store raw cursor position, rounding happens during data extraction.
        self._cursor_pos = np.array(viewer.cursor.position)
        self._update_plot()

    def _on_layer_change(self, event) -> None:
        """Handle layer insertion/removal or active-layer change events."""
        if self._source_mode != "mouse":
            return
        self._current_layer = self._active_layer()
        if self._cursor_pos is not None and self._current_layer is not None:
            self._update_plot()

    def _active_layer(self):
        """Return the active layer, or None if it is not a valid image.

        A layer is considered valid when it is a 3-D+ non-RGB image.
        """
        layer = self._viewer.layers.selection.active
        if (
            layer is not None
            and layer._type_string == "image"
            and layer.data.ndim >= 3
            and not getattr(layer, "rgb", False)
        ):
            return layer
        return None

    def _invalidate_mouse_cache(self) -> None:
        """Reset all mouse-plot cache state, forcing a full redraw on the next update."""
        self._mouse_live_line = None
        self._mouse_imported_signature = ()
        self._mouse_legend_signature = ()
        self._mouse_plot_dirty = True

    def _show_message(self, text: str) -> None:
        """Show a centered message in the plot area with no axes or data."""
        self._clear_export_series()
        self._invalidate_mouse_cache()
        self._axes.clear()
        colors = self._get_colors()
        self._axes.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            transform=self._axes.transAxes,
            fontsize=11,
            color=colors["fg"],
            alpha=0.7,
        )
        self._axes.set_xticks([])
        self._axes.set_yticks([])
        for spine in self._axes.spines.values():
            spine.set_visible(False)
        self._vline = None
        self._canvas.draw_idle()

    def _show_instructions(self) -> None:
        """Show initial instructions appropriate for the current source mode."""
        if self._source_mode == "points":
            msg = "Select a Points layer and a reference\nimage layer in the panel to the right."
        elif self._source_mode == "labels":
            msg = "Select a Labels layer and a reference\nimage layer in the panel to the right."
        else:
            msg = (
                'Hold "Shift" while moving the cursor\n'
                "over an image layer to plot\n"
                "voxel time series."
            )
        self._show_message(msg)

    def _refresh_plot(self) -> None:
        """Re-render the plot for the current source mode."""
        self._mouse_plot_dirty = True
        if self._source_mode == "points":
            self._update_plot_from_points()
        elif self._source_mode == "labels":
            self._update_plot_from_labels()
        else:
            self._update_plot()

    def _on_plot_data_changed(self) -> None:
        """Reset auto x-range when plotted imported series membership changes."""
        self._has_plot = False
        self._prev_ts_valid = False
        self._mouse_plot_dirty = True

    def _update_plot(self) -> None:
        """Update the time series plot with current data."""
        if self._current_layer is None or self._cursor_pos is None:
            self._render_imported_only()
            return

        # Preserve zoom state across voxel changes. X limits (time range) are always
        # restored so the user can zoom in and keep exploring. Y limits are only
        # restored when autoscale is off, so the axis still adjusts to the new voxel's
        # amplitude when autoscale is on.
        #
        # Only save limits when the previous call had valid data. If the cursor left
        # the image, the axes were cleared with no data and matplotlib reset xlim to
        # [0, 1]; restoring that would permanently constrain the axis.
        save_zoom = self._has_plot and self._prev_ts_valid
        saved_xlim = (
            self._axes.get_xlim()
            if save_zoom and self._xlim_is_user_modified()
            else None
        )
        # Use user-specified limits if set, otherwise fall back to current axis limits
        if save_zoom and not self._autoscale:
            ymin = (
                self._ylim_min
                if self._ylim_min is not None
                else self._axes.get_ylim()[0]
            )
            ymax = (
                self._ylim_max
                if self._ylim_max is not None
                else self._axes.get_ylim()[1]
            )
            saved_ylim = (ymin, ymax)
        else:
            saved_ylim = None

        colors = self._get_colors()

        ts = self._extract_time_series(self._current_layer, self._cursor_pos)
        # Update stored time coordinates for cursor mapping.
        self._time_coords = self._get_time_coords(self._current_layer)
        if ts is not None:
            if self._zscore:
                ts = self._apply_zscore(ts)

            if self._time_coords is not None and len(self._time_coords) == len(ts):
                x_values = self._time_coords
            else:
                x_values = np.arange(len(ts))
            xlabel = self._get_time_xlabel(self._current_layer)
            live_label = self._current_layer.name

            imported_series = self._imported_plot_series()
            coord_str = self._mouse_coord_string()
            title = f"{self._current_layer.name} — ({coord_str})"

            if self._try_fast_mouse_update(
                live_label=live_label,
                x_values=np.asarray(x_values),
                ts=np.asarray(ts),
                xlabel=xlabel,
                title=title,
                saved_xlim=saved_xlim,
                saved_ylim=saved_ylim,
                imported_series=imported_series,
                colors=colors,
            ):
                return

            self._axes.clear()

            self._axes.plot(
                x_values,
                ts,
                linewidth=1.5,
                color=colors["accent"],
                label=live_label,
            )

            self._mouse_live_line = self._axes.lines[-1]
            self._plot_series_lines(imported_series, linewidth=1.4, alpha=0.95)
            self._mouse_imported_signature = self._imported_signature(imported_series)
            export_imported = self._plot_series_to_export(imported_series)
            self._set_export_series(
                [ExportSeries(live_label, np.asarray(x_values), np.asarray(ts))]
                + export_imported
            )

        if ts is not None:
            self._style_valid_axes(
                colors,
                xlabel,
                "Z-score" if self._zscore else "Intensity",
                title,
                with_legend=len(self._export_series) > 1,
            )
            self._restore_view(saved_xlim, saved_ylim)

            self._has_plot = True
            self._prev_ts_valid = True
            self._mouse_plot_dirty = False
            self._mouse_legend_signature = self._legend_signature(
                live_label, imported_series
            )
        else:
            if self._render_imported_only(saved_xlim=saved_xlim, saved_ylim=saved_ylim):
                return
            self._clear_export_series()
            self._prev_ts_valid = False
            self._mouse_plot_dirty = True
            self._mouse_live_line = None
            self._axes.clear()
            self._vline = None
            self._axes.text(
                0.5,
                0.5,
                "No valid data at cursor position",
                ha="center",
                va="center",
                transform=self._axes.transAxes,
                fontsize=11,
                color=colors["fg"],
                alpha=0.7,
            )
            self._axes.set_xticks([])
            self._axes.set_yticks([])
            for spine in self._axes.spines.values():
                spine.set_visible(False)

        self._canvas.draw_idle()

    def _time_dim_index(self, layer) -> int:
        """Return the data dimension index that corresponds to time.

        Reads `layer.metadata["xarray"]` when available; falls back to 0.
        """
        da = layer.metadata.get("xarray")
        if da is not None and "time" in da.dims:
            return list(da.dims).index("time")
        return 0

    def _get_time_coords(self, layer) -> np.ndarray | None:
        """Return the time coordinate array from the layer's xarray metadata.

        Returns None when no xarray metadata is present or the DataArray has no
        "time" coordinate.
        """
        da = layer.metadata.get("xarray")
        if da is not None and "time" in da.coords:
            return np.asarray(da.coords["time"])
        return None

    def _get_time_xlabel(self, layer) -> str:
        """Return the x-axis label for the time axis.

        Reads `long_name` and `units` from the time coordinate attributes
        when available. Falls back to `"Time"` and no units.
        """
        da = layer.metadata.get("xarray")
        if da is not None and "time" in da.coords:
            attrs = da.coords["time"].attrs
            name = attrs.get("long_name", "Time")
            units = attrs.get("units", "")
            return f"{name} ({units})" if units else name
        return "Time Frame"

    def _extract_time_series(self, layer, cursor_pos: np.ndarray) -> np.ndarray | None:
        """Extract time series at cursor position from layer.

        Always uses the nearest voxel to the cursor position.
        """
        data = layer.data
        ind = list(int(round(x)) for x in layer.world_to_data(cursor_pos))

        t_idx = self._time_dim_index(layer)
        # Replace the time index before bounds-checking: the injected time world
        # coordinate (typically 0) may fall outside the image's time range (e.g. when
        # time starts at a non-zero offset), which would cause the check to reject valid
        # spatial positions.
        ind[t_idx] = slice(None)  # type: ignore[call-overload]

        if not all(
            0 <= i < max_i for i, max_i in zip(ind, data.shape) if isinstance(i, int)
        ):
            return None

        return data[tuple(ind)]

    # ------------------------------------------------------------------
    # Source mode — public setters
    # ------------------------------------------------------------------

    def set_source_mode(self, mode: Literal["mouse", "points", "labels"]) -> None:
        """Switch the active plotting source.

        Parameters
        ----------
        mode : str
            One of `"mouse"`, `"points"`, or `"labels"`.
        """
        self._source_mode = mode
        self._invalidate_mouse_cache()
        if mode == "mouse":
            # Invalidate the saved zoom state so the first mouse-hover plot does not
            # restore the [0, 1] xlim left by the cleared axes.
            self._has_plot = False
            self._prev_ts_valid = False
            # Only show instructions if there are no imported series to display.
            if not self._render_imported_only():
                self._show_instructions()
        elif mode == "points":
            self._update_plot_from_points()
        else:
            self._update_plot_from_labels()

    def set_points_layer(self, layer) -> None:
        """Set the Points layer used in points mode.

        Parameters
        ----------
        layer : napari.layers.Points or None
            The Points layer to use, or None to clear.
        """
        if self._points_layer is not None:
            try:
                self._points_layer.events.data.disconnect(self._on_points_data_changed)
            except RuntimeError:
                pass
        self._points_layer = layer
        if layer is not None:
            layer.events.data.connect(self._on_points_data_changed)
        if self._source_mode == "points":
            self._update_plot_from_points()

    def set_labels_layer(self, layer) -> None:
        """Set the Labels layer used in labels mode.

        Parameters
        ----------
        layer : napari.layers.Labels or None
            The Labels layer to use, or None to clear.
        """
        if self._labels_layer is not None:
            try:
                self._labels_layer.events.data.disconnect(self._on_labels_data_changed)
            except RuntimeError:
                pass
            try:
                # events.paint fires when individual voxels are painted with the
                # brush tool; events.data only fires on full array replacement.
                self._labels_layer.events.paint.disconnect(self._on_labels_data_changed)
            except RuntimeError:
                pass
        self._labels_layer = layer
        if layer is not None:
            layer.events.data.connect(self._on_labels_data_changed)
            layer.events.paint.connect(self._on_labels_data_changed)
        if self._source_mode == "labels":
            self._update_plot_from_labels()

    def set_ref_layers(self, layers: list | None) -> None:
        """Set the reference image layers to extract time series from.

        Parameters
        ----------
        layers : list or None
            Specific Image layers to use, or None to use all image layers
            that have a time dimension.
        """
        self._ref_layers = layers
        if self._source_mode == "points":
            self._update_plot_from_points()
        elif self._source_mode == "labels":
            self._update_plot_from_labels()

    # ------------------------------------------------------------------
    # Source mode — private helpers
    # ------------------------------------------------------------------

    def _get_ref_image_layers(self) -> list:
        """Return the list of reference image layers to extract data from.

        When `_ref_layers` is None (all-layers mode), every Image layer in
        the viewer with at least 3 dimensions and not RGB is returned.
        Otherwise returns the stored list, filtered to layers still present.
        """
        current_names = {layer.name for layer in self._viewer.layers}
        if self._ref_layers is None:
            result = []
            for layer in self._viewer.layers:
                if layer._type_string != "image" or getattr(layer, "rgb", False):
                    continue
                # Only include layers that have a time dimension; spatial-only
                # maps (tSNR, CV, …) have ndim < 4 and no "time" dim.
                da = layer.metadata.get("xarray")
                if da is not None:
                    if "time" in da.dims:
                        result.append(layer)
                elif layer.data.ndim >= 4:
                    result.append(layer)
            return result
        return [layer for layer in self._ref_layers if layer.name in current_names]

    def _save_view(self) -> tuple:
        """Return (xlim, ylim) from the current axes if a plot is present.

        Used to preserve user zoom across replots.  Both values are `None` when the axes
        contain no data lines.
        """
        lines = self._axes.get_lines()
        xlim = self._axes.get_xlim() if lines else None
        if lines and not self._autoscale:
            # Use user-specified limits if set, otherwise fall back to current axis
            # limits.
            ymin = (
                self._ylim_min
                if self._ylim_min is not None
                else self._axes.get_ylim()[0]
            )
            ymax = (
                self._ylim_max
                if self._ylim_max is not None
                else self._axes.get_ylim()[1]
            )
            ylim = (ymin, ymax)
        else:
            ylim = None
        return xlim, ylim

    def _restore_view(self, xlim, ylim) -> None:
        """Restore a previously saved view or apply a tight x range.

        When `xlim` is `None` (first plot in a mode), the x range is set to the full
        data extent across all plotted lines so matplotlib's default 5 % margin is
        removed.
        """
        if xlim is not None:
            self._axes.set_xlim(xlim)
        else:
            x_extent = self._plotted_x_extent()
            if x_extent is not None:
                self._axes.set_xlim(x_extent)
        if ylim is not None:
            self._axes.set_ylim(ylim)

    def _xlim_is_user_modified(self) -> bool:
        """Return whether the current x view differs from the automatic data extent."""
        auto_xlim = self._plotted_x_extent()
        if auto_xlim is None:
            return False
        return not np.allclose(self._axes.get_xlim(), auto_xlim, rtol=0.0, atol=1e-9)

    def _plotted_x_extent(self) -> tuple[float, float] | None:
        """Return the overall x extent across all currently plotted lines."""
        if not self._axes.get_lines():
            return None

        self._axes.relim()
        x0 = float(self._axes.dataLim.x0)
        x1 = float(self._axes.dataLim.x1)
        if not (np.isfinite(x0) and np.isfinite(x1)):
            return None
        return min(x0, x1), max(x0, x1)

    def _set_export_series(self, series: list[ExportSeries]) -> None:
        """Store the currently plotted series for export."""
        self._export_series = prepare_export_series(series)
        self._export_button.setEnabled(bool(self._export_series))

    def _clear_export_series(self) -> None:
        """Clear the currently exported series."""
        self._set_export_series([])

    def _visible_imported_series(self) -> list[ImportedSeries]:
        """Return visible imported series from the shared store."""
        if self._series_store is None:
            return []
        return self._series_store.visible_imported_series()

    def _mouse_coord_string(self) -> str:
        """Return the current mouse coordinates formatted for the plot title."""
        if self._current_layer is None or self._cursor_pos is None:
            return ""
        t_idx = self._time_dim_index(self._current_layer)
        spatial_coords = [c for i, c in enumerate(self._cursor_pos) if i != t_idx]
        return ", ".join(f"{c:.1f}" for c in spatial_coords)

    def _imported_plot_series(self) -> list[PlotSeries]:
        """Return imported series prepared for plotting and export."""
        plotted = []
        for series in self._visible_imported_series():
            x_values = np.asarray(series.x).copy()
            y_values = np.asarray(series.y, dtype=float).copy()
            if self._zscore:
                y_values = self._apply_zscore(y_values)
            plotted.append(PlotSeries(series.name, x_values, y_values, series.color))
        return plotted

    def _plot_series_lines(
        self,
        series: list[PlotSeries],
        *,
        linewidth: float = 1.5,
        alpha: float = 1.0,
    ) -> None:
        """Plot a list of series onto the current axes."""
        for s in series:
            kwargs = {"linewidth": linewidth, "alpha": alpha, "label": s.label}
            if s.color is not None:
                kwargs["color"] = s.color
            self._axes.plot(s.x_values, s.y_values, **kwargs)

    def _imported_signature(self, series: list[PlotSeries]) -> tuple[str, ...]:
        """Return a lightweight signature for imported series already on the axes."""
        return tuple(s.label for s in series)

    def _legend_signature(
        self, live_label: str, imported_series: list[PlotSeries]
    ) -> tuple[str, ...]:
        """Return the legend content signature for the current mouse plot."""
        return (live_label, *self._imported_signature(imported_series))

    @staticmethod
    def _plot_series_to_export(series: list[PlotSeries]) -> list[ExportSeries]:
        """Convert PlotSeries to ExportSeries for CSV/TSV export.

        Parameters
        ----------
        series : list[PlotSeries]
            Series with color information.

        Returns
        -------
        list[ExportSeries]
            Series without color for export.
        """
        return [ExportSeries(s.label, s.x_values, s.y_values) for s in series]

    def _try_fast_mouse_update(
        self,
        *,
        live_label: str,
        x_values: np.ndarray,
        ts: np.ndarray,
        xlabel: str,
        title: str,
        saved_xlim,
        saved_ylim,
        imported_series: list[PlotSeries],
        colors: dict,
    ) -> bool:
        """Update the mouse plot without rebuilding imported overlay lines."""
        if self._mouse_plot_dirty or self._mouse_live_line is None:
            return False
        if self._mouse_imported_signature != self._imported_signature(imported_series):
            return False

        self._mouse_live_line.set_data(x_values, ts)
        self._mouse_live_line.set_label(live_label)
        self._mouse_live_line.set_color(colors["accent"])
        self._set_export_series(
            [ExportSeries(live_label, x_values, ts)]
            + self._plot_series_to_export(imported_series)
        )

        self._axes.set_xlabel(xlabel, color=colors["fg"], fontsize=9)
        self._axes.set_ylabel(
            "Z-score" if self._zscore else "Intensity",
            color=colors["fg"],
            fontsize=9,
        )
        self._axes.set_title(title, color=colors["fg"], fontsize=10)
        if self._show_grid:
            self._axes.grid(True, alpha=0.3, color=colors["fg"])

        self._update_legend_if_needed(colors, live_label, imported_series)

        self._axes.relim()
        self._restore_view(saved_xlim, saved_ylim)
        if self._autoscale:
            self._axes.autoscale_view(scalex=False, scaley=True)

        self._has_plot = True
        self._prev_ts_valid = True
        self._canvas.draw_idle()
        return True

    def _update_legend_if_needed(
        self, colors: dict, live_label: str, imported_series: list[PlotSeries]
    ) -> None:
        """Update the legend only when its entries actually changed."""
        signature = self._legend_signature(live_label, imported_series)
        if signature == self._mouse_legend_signature:
            return

        legend = self._axes.get_legend()
        if legend is not None:
            legend.remove()

        if len(signature) > 1:
            self._axes.legend(
                loc="upper right",
                fontsize=8,
                labelcolor=colors["fg"],
                facecolor=colors["bg"],
                edgecolor=colors["fg"],
            )
        self._mouse_legend_signature = signature

    def _render_imported_only(self, saved_xlim=None, saved_ylim=None) -> bool:
        """Render only imported series when no live data can be drawn."""
        imported_series = self._imported_plot_series()
        if not imported_series:
            return False

        colors = self._get_colors()
        self._axes.clear()
        self._plot_series_lines(imported_series, linewidth=1.4, alpha=0.95)
        self._set_export_series(self._plot_series_to_export(imported_series))
        self._mouse_live_line = None
        self._mouse_imported_signature = self._imported_signature(imported_series)
        self._mouse_legend_signature = self._imported_signature(imported_series)
        self._mouse_plot_dirty = True
        self._time_coords = None
        self._style_valid_axes(
            colors,
            "Time",
            "Z-score" if self._zscore else "Value",
            "Imported Time Series",
            with_legend=len(imported_series) > 1,
            show_cursor=False,
        )
        self._restore_view(saved_xlim, saved_ylim)
        self._has_plot = True
        self._prev_ts_valid = True
        self._canvas.draw_idle()
        return True

    def _show_message_or_imported(
        self,
        text: str,
        *,
        saved_xlim=None,
        saved_ylim=None,
    ) -> None:
        """Show a message when no imported series are visible, else plot imports only."""
        if self._render_imported_only(saved_xlim=saved_xlim, saved_ylim=saved_ylim):
            return
        self._show_message(text)

    def _write_current_plot_delimited(self, path: Path, delimiter: str) -> None:
        """Write the currently plotted time series to a delimited text file."""
        write_delimited_series(path, self._export_series, delimiter=delimiter)

    def _save_current_plot(self) -> None:
        """Open a save dialog and export the current plot as CSV or TSV."""
        if not self._export_series:
            show_error("No time series plot available to save.")
            return

        export_selection = prompt_delimited_export_path(
            self, "Export Time Series", str(Path.home() / "time_series.tsv")
        )
        if export_selection is None:
            return

        path, delimiter = export_selection

        try:
            self._write_current_plot_delimited(path, delimiter)
        except Exception as exc:
            show_error(str(exc))
            return

        show_info(f"Exported time series to {path}")

    def _apply_zscore(self, ts: np.ndarray) -> np.ndarray:
        """Normalise a time series to zero mean and unit variance.

        NaN values are ignored when computing the mean and standard deviation so that
        series with missing values (imported files with partial columns) are z-scored
        correctly rather than becoming all-NaN.
        """
        mean = np.nanmean(ts)
        std = np.nanstd(ts)
        return (ts - mean) / std if std > 0 else ts - mean

    def _get_label_color(self, label_id: int) -> tuple:
        """Return a matplotlib-compatible RGBA color for a label ID.

        Tries to read the color from the Labels layer's colormap; falls back
        to matplotlib's tab20 palette.
        """
        if self._labels_layer is not None:
            try:
                color = self._labels_layer.get_color(label_id)
                if color is not None:
                    return tuple(float(c) for c in color)
            except (AttributeError, KeyError, IndexError):
                pass
        return mpl_colormaps["tab20"](int(label_id - 1) % 20)

    def _style_valid_axes(
        self,
        colors: dict,
        xlabel: str,
        ylabel: str,
        title: str,
        with_legend: bool = False,
        show_cursor: bool = True,
    ) -> None:
        """Apply common axis styling after a valid plot has been drawn."""
        self._axes.set_xlabel(xlabel, color=colors["fg"], fontsize=9)
        self._axes.set_ylabel(ylabel, color=colors["fg"], fontsize=9)
        self._axes.set_title(title, color=colors["fg"], fontsize=10)
        if not self._autoscale:
            if self._ylim_min is not None:
                self._axes.set_ylim(bottom=self._ylim_min)
            if self._ylim_max is not None:
                self._axes.set_ylim(top=self._ylim_max)
        if self._show_grid:
            self._axes.grid(True, alpha=0.3, color=colors["fg"])
        self._axes.tick_params(colors=colors["fg"], labelsize=8)
        for spine in self._axes.spines.values():
            spine.set_edgecolor(colors["fg"])
            spine.set_visible(True)
        if with_legend:
            self._axes.legend(
                loc="upper right",
                fontsize=8,
                labelcolor=colors["fg"],
                facecolor=colors["bg"],
                edgecolor=colors["fg"],
            )
        if self._show_cursor and show_cursor:
            self._vline = self._axes.axvline(
                self._frame_to_x(self._cursor_frame),
                color=colors["cursor"],
                linewidth=1.2,
                alpha=0.85,
                zorder=10,
                animated=True,
            )
        else:
            self._vline = None

    def _on_points_data_changed(self, event) -> None:
        """Re-plot when the Points layer data changes."""
        if self._source_mode == "points":
            self._update_plot_from_points()

    def _on_labels_data_changed(self, event) -> None:
        """Schedule a debounced re-plot when the Labels layer is painted."""
        if self._source_mode == "labels":
            if not self._labels_debounce.isActive():
                self._labels_debounce.start()

    # ------------------------------------------------------------------
    # Source mode — plot methods
    # ------------------------------------------------------------------

    def _validate_source_layers(
        self,
        source_layer,
        source_name: str,
        saved_xlim,
        saved_ylim,
    ) -> list | None:
        """Validate that a source layer and at least one reference image layer are set.

        Shows an appropriate message and returns `None` when validation fails;
        returns the list of reference image layers on success.

        Parameters
        ----------
        source_layer : napari layer or None
            The currently configured source layer (Points or Labels).
        source_name : str
            Human-readable layer type used in the error message (e.g. `"Points"`).
        saved_xlim : tuple or None
            Saved x-axis limits to restore after showing the message.
        saved_ylim : tuple or None
            Saved y-axis limits to restore after showing the message.

        Returns
        -------
        list or None
            The reference image layers, or `None` if validation failed.
        """
        if source_layer is None:
            self._show_message_or_imported(
                f"No {source_name} layer selected.\n"
                "Choose one in the Source section of the panel.",
                saved_xlim=saved_xlim,
                saved_ylim=saved_ylim,
            )
            return None
        ref_layers = self._get_ref_image_layers()
        if not ref_layers:
            self._show_message_or_imported(
                "No reference image layer found.\nLoad a 4-D image layer first.",
                saved_xlim=saved_xlim,
                saved_ylim=saved_ylim,
            )
            return None
        return ref_layers

    @staticmethod
    def _series_label(base: str, img_layer, ref_layers: list) -> str:
        """Build a series label, prefixing the layer name when multiple layers are shown.

        Parameters
        ----------
        base : str
            Base label (e.g. `"Point 0"` or `"Label 3"`).
        img_layer : napari image layer
            Reference image layer contributing this series.
        ref_layers : list
            All active reference image layers.

        Returns
        -------
        str
            `"<layer name> | <base>"` when more than one reference layer is active,
            otherwise just `base`.
        """
        return f"{img_layer.name} | {base}" if len(ref_layers) > 1 else base

    def _finalize_multi_series_plot(
        self,
        export_series: list,
        source_layer_name: str,
        ref_layers: list,
        colors: dict,
        saved_xlim,
        saved_ylim,
    ) -> None:
        """Overlay imported series, style the axes, and redraw after a multi-series plot.

        Shared by `_update_plot_from_points` and `_update_plot_from_labels` once their
        per-series loops have completed successfully.

        Parameters
        ----------
        export_series : list[ExportSeries]
            Series already plotted by the caller.
        source_layer_name : str
            Name of the active Points or Labels layer, used in the plot title.
        ref_layers : list
            Active reference image layers.
        colors : dict
            Theme color dictionary from `_get_colors`.
        saved_xlim : tuple or None
            Saved x-axis limits to restore.
        saved_ylim : tuple or None
            Saved y-axis limits to restore.
        """
        imported_series = self._imported_plot_series()
        self._plot_series_lines(imported_series, linewidth=1.4, alpha=0.95)
        self._set_export_series(
            export_series + self._plot_series_to_export(imported_series)
        )
        xlabel = self._get_time_xlabel(ref_layers[0])
        ylabel = "Z-score" if self._zscore else "Intensity"
        title_ref = ref_layers[0].name if len(ref_layers) == 1 else "All layers"
        self._style_valid_axes(
            colors,
            xlabel,
            ylabel,
            f"{title_ref} — {source_layer_name}",
            with_legend=len(self._export_series) > 1,
        )
        self._restore_view(saved_xlim, saved_ylim)
        self._canvas.draw_idle()

    def _update_plot_from_points(self) -> None:
        """Plot time series for every point in the Points layer."""
        colors = self._get_colors()
        saved_xlim, saved_ylim = self._save_view()

        ref_layers = self._validate_source_layers(
            self._points_layer, "Points", saved_xlim, saved_ylim
        )
        if ref_layers is None:
            return

        n_points = len(self._points_layer.data)
        if n_points == 0:
            self._show_message_or_imported(
                "The selected Points layer is empty.",
                saved_xlim=saved_xlim,
                saved_ylim=saved_ylim,
            )
            return

        # Pre-compute time coordinates per layer: they don't depend on individual
        # points.
        layer_time_coords = [self._get_time_coords(layer) for layer in ref_layers]

        self._axes.clear()
        has_any = False
        export_series: list[ExportSeries] = []

        for pt_idx in range(n_points):
            pt_data = np.asarray(self._points_layer.data[pt_idx], dtype=float)
            n_pt = len(pt_data)
            # napari may pad layer.scale/translate to the viewer's ndim, so use the last
            # n_pt elements to match the point's dimensionality.
            pts_scale = np.asarray(self._points_layer.scale, dtype=float)[-n_pt:]
            pts_translate = np.asarray(self._points_layer.translate, dtype=float)[
                -n_pt:
            ]
            # data → world (spatial, n_pt dims)
            pt_world = pt_data * pts_scale + pts_translate
            pt_color = tuple(float(c) for c in self._points_layer.face_color[pt_idx])

            for layer_idx, img_layer in enumerate(ref_layers):
                img_ndim = img_layer.data.ndim
                if n_pt < img_ndim:
                    # 3-D point in a 4-D image: pad the world coord with 0 at the front
                    # so world_to_data receives the correct ndim. The time value (0) is
                    # irrelevant — _extract_time_series replaces it with slice(None).
                    padded = np.zeros(img_ndim)
                    padded[-n_pt:] = pt_world
                    pt_world_img = padded
                else:
                    pt_world_img = pt_world
                try:
                    ts = self._extract_time_series(img_layer, pt_world_img)
                except Exception:  # noqa: BLE001
                    ts = None
                if ts is None:
                    continue

                ts = np.asarray(ts, dtype=float)
                if self._zscore:
                    ts = self._apply_zscore(ts)

                time_coords = layer_time_coords[layer_idx]
                x = (
                    time_coords
                    if (time_coords is not None and len(time_coords) == len(ts))
                    else np.arange(len(ts))
                )
                label = self._series_label(f"Point {pt_idx}", img_layer, ref_layers)
                linestyle = _LAYER_LINESTYLES[layer_idx % len(_LAYER_LINESTYLES)]
                self._axes.plot(
                    x,
                    ts,
                    color=pt_color,
                    linewidth=1.5,
                    linestyle=linestyle,
                    label=label,
                )
                export_series.append(ExportSeries(label, np.asarray(x), np.asarray(ts)))
                has_any = True
                # Keep time coords from the last successful layer for cursor mapping.
                self._time_coords = time_coords
                self._current_layer = img_layer

        if has_any:
            self._finalize_multi_series_plot(
                export_series,
                self._points_layer.name,
                ref_layers,
                colors,
                saved_xlim,
                saved_ylim,
            )
        else:
            self._show_message_or_imported(
                "No valid data at the point positions.\n",
                saved_xlim=saved_xlim,
                saved_ylim=saved_ylim,
            )

    def _update_plot_from_labels(self) -> None:
        """Plot mean time series for each unique label in the Labels layer."""
        colors = self._get_colors()
        saved_xlim, saved_ylim = self._save_view()

        ref_layers = self._validate_source_layers(
            self._labels_layer, "Labels", saved_xlim, saved_ylim
        )
        if ref_layers is None:
            return

        # Read the labels data. We intentionally avoid an early "all-zeros" check here
        # because newer napari versions may return a lazy view from .data that doesn't
        # reflect painted regions until it is iterated inside the loop.
        labels_data = np.asarray(self._labels_layer.data)

        self._axes.clear()
        has_any = False
        shape_mismatch = False
        export_series: list[ExportSeries] = []

        for layer_idx, img_layer in enumerate(ref_layers):
            t_idx = self._time_dim_index(img_layer)
            img_data = img_layer.data
            img_spatial = tuple(s for i, s in enumerate(img_data.shape) if i != t_idx)

            # Labels can have the full image shape (T, Z, Y, X) (napari creates labels
            # with the same shape as the reference image) or the spatial shape only (Z,
            # Y, X). Collapse the time axis in the former case.
            if labels_data.shape == img_data.shape:
                labels_spatial = np.max(labels_data, axis=t_idx)
            elif labels_data.shape == img_spatial:
                labels_spatial = labels_data
            else:
                shape_mismatch = True
                continue

            unique_labels = np.unique(labels_spatial)
            unique_labels = unique_labels[unique_labels != 0]  # exclude background
            if len(unique_labels) == 0:
                continue

            # Move time to last axis so spatial boolean indexing works cleanly:
            # img_arr[mask] → (N_voxels, T).
            img_arr = np.moveaxis(np.asarray(img_data), t_idx, -1)

            time_coords = self._get_time_coords(img_layer)
            x = time_coords if time_coords is not None else np.arange(img_arr.shape[-1])

            for lid in unique_labels:
                mask = labels_spatial == lid
                ts = np.asarray(img_arr[mask].mean(axis=0), dtype=float)  # (T,)

                if self._zscore:
                    ts = self._apply_zscore(ts)

                label = self._series_label(f"Label {lid}", img_layer, ref_layers)
                linestyle = _LAYER_LINESTYLES[layer_idx % len(_LAYER_LINESTYLES)]
                self._axes.plot(
                    x,
                    ts,
                    color=self._get_label_color(int(lid)),
                    linewidth=1.5,
                    linestyle=linestyle,
                    label=label,
                )
                export_series.append(ExportSeries(label, np.asarray(x), np.asarray(ts)))
                has_any = True

            # Keep time coords from the last successful layer for cursor mapping.
            self._time_coords = time_coords
            self._current_layer = img_layer

        if has_any:
            self._finalize_multi_series_plot(
                export_series,
                self._labels_layer.name,
                ref_layers,
                colors,
                saved_xlim,
                saved_ylim,
            )
        elif shape_mismatch:
            self._show_message_or_imported(
                "Spatial shape of the Labels layer does not match\n"
                "the reference image. Make sure they share the same grid.",
                saved_xlim=saved_xlim,
                saved_ylim=saved_ylim,
            )
        else:
            self._show_message_or_imported(
                "No valid data extracted from the Labels layer.",
                saved_xlim=saved_xlim,
                saved_ylim=saved_ylim,
            )

    def closeEvent(self, a0) -> None:
        """Clean up when widget is closed.

        Parameters
        ----------
        a0 : QCloseEvent
            The close event (unused; accepted unconditionally).
        """
        if self._on_mouse_move in self._viewer.mouse_move_callbacks:
            self._viewer.mouse_move_callbacks.remove(self._on_mouse_move)
        if self._series_store is not None:
            try:
                self._series_store.plot_data_changed.disconnect(
                    self._on_plot_data_changed
                )
            except (RuntimeError, TypeError):
                pass
            try:
                self._series_store.changed.disconnect(self._refresh_plot)
            except (RuntimeError, TypeError):
                pass
        self.set_points_layer(None)
        self.set_labels_layer(None)
        self._labels_debounce.stop()
        a0.accept()
