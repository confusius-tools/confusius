"""Signals plotter widget for the bottom dock."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import napari
import numpy as np
from matplotlib import colormaps as mpl_colormaps
from matplotlib import colors as mpl_colors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import QSize, QTimer, Signal
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from confusius._dims import TIME_DIM
from confusius._napari._export import (
    ExportSignal,
    PlotSignal,
    prepare_export_signal,
    prompt_delimited_export_path,
    write_delimited_signals,
)
from confusius._napari._signals._store import (
    ImportedSignal,
    LiveSignal,
    SignalStore,
)
from confusius._napari._theme import (
    create_export_button,
    get_napari_colors,
    style_export_button,
    style_plot_toolbar,
)

# Line styles cycled across reference layers so multi-layer plots are distinguishable
# even when point/label colors are the same.
_LAYER_LINESTYLES = ["-", "--", "-.", ":"]


class SignalPlotter(QWidget):
    """Bottom dock widget for live signals plotting.

    Supports three source modes: mouse hover (Shift + move), a Points layer, or a Labels
    layer. Imported signals from a `SignalStore` are overlaid on every plot. The x-axis
    dimension is resolved from xarray metadata when available, falling back to the first
    array dimension.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.
    store : SignalStore | None, optional
        Shared store containing imported signals to overlay on the live plot.
    """

    frame_clicked = Signal(float)
    """Emitted when the user left-clicks on the plot axes.

    The payload is the frame index corresponding to the clicked x position.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        store: SignalStore | None = None,
    ) -> None:
        super().__init__()
        self._viewer = viewer
        self._signals_store = store
        self._cursor_pos: np.ndarray | None = None
        self._current_layer = None

        # Plot settings.
        self._ylim_min: float | None = None
        self._ylim_max: float | None = None
        self._autoscale: bool = True
        self._show_grid: bool = True
        self._zscore: bool = False

        # Whether at least one signals has been plotted (used to decide whether to
        # restore the zoom state on the next _update_plot call).
        self._has_plot: bool = False
        # Whether the previous _update_plot call had valid data. Used to avoid
        # restoring the default [0, 1] xlim that matplotlib sets when axes are cleared
        # with no data (e.g. when the cursor leaves the image).
        self._prev_ts_valid: bool = False

        # X-axis coordinate values for the current layer, used to map frame
        # indices to real coordinate values on the x-axis.
        self._xaxis_coords: np.ndarray | None = None

        # X-axis dimension name (e.g., "time", "lag", "feature"). When None, falls back
        # to looking for a "time" dimension in the xarray metadata, then to index 0.
        self._xaxis_dim: str | None = None

        # Source mode: "mouse" | "points" | "labels".
        self._source_mode: Literal["mouse", "points", "labels"] = "mouse"
        self._points_layer = None
        self._labels_layer = None
        self._ref_layers: list | None = None  # None = all image layers with x-axis dim

        # Debounce timer for labels data changes (painting is fast, replot is slow).
        self._labels_debounce = QTimer(self)
        self._labels_debounce.setSingleShot(True)
        self._labels_debounce.setInterval(500)
        self._labels_debounce.timeout.connect(self._update_plot_from_labels)

        # Time cursor blitting state.
        self._show_cursor: bool = False
        self._vline = None
        self._bg = None  # Saved pixel buffer (without vline).
        self._cursor_world: float = 0.0
        self._export_signals: list[ExportSignal] = []
        self._mouse_plot_dirty: bool = True
        self._mouse_live_line = None
        self._mouse_imported_signature: tuple[str, ...] = ()
        self._mouse_legend_signature: tuple[str, ...] = ()
        # Guard flag to prevent infinite color sync loops.
        self._syncing_color: bool = False
        # Guard flag to prevent reentrant plot updates.
        self._updating_plot: bool = False

        # Throttle blit calls to ~60 fps so the napari slider stays responsive when
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
        if self._signals_store is not None:
            self._signals_store.plot_data_changed.connect(self._on_plot_data_changed)
            self._signals_store.changed.connect(self._refresh_plot)
            self._signals_store.changed.connect(self._sync_live_colors_to_layers)
            # If there are already imported signals, plot them instead of showing
            # instructions.
            imported = self._signals_store.imported_signals()
            if imported and self._render_imported_only():
                pass  # Successfully plotted imported signal.
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
            self._toolbar, "signalExportButton", self._save_current_plot
        )
        self._export_button.setEnabled(False)
        self._toolbar.addWidget(self._export_button)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._axes = self._figure.add_subplot(111)
        # Save background after each full redraw for blitting the x-axis cursor.
        self._canvas.mpl_connect("draw_event", self._on_draw)
        # Left-click navigates the viewer to the clicked frame.
        self._canvas.mpl_connect("button_press_event", self._on_click)

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
        """Enable or disable Z-scoring of signals.

        Parameters
        ----------
        enabled : bool
            Whether to Z-score the signals.
        """
        self._zscore = enabled
        self._refresh_plot()

    def set_show_cursor(self, enabled: bool) -> None:
        """Enable or disable the x-axis cursor.

        Parameters
        ----------
        enabled : bool
            Whether to show the x-axis cursor.
        """
        self._show_cursor = enabled
        self._vline = None
        self._bg = None
        self._refresh_plot()

    def set_xaxis_dim(self, dim: str | None) -> None:
        """Set the dimension to use for the plot's x-axis.

        When set to a dimension name (e.g., "time", "lag", "feature"), that
        dimension will be used for the x-axis. When None, the plotter will look
        for a "time" dimension in the xarray metadata, falling back to index 0.

        Parameters
        ----------
        dim : str | None
            Name of the dimension to use for the x-axis, or None to use
            automatic detection (looks for "time" first, then defaults to 0).
        """
        if self._xaxis_dim != dim:
            self._xaxis_dim = dim
            # The previous zoom state belongs to a different coordinate space,
            # so discard it to avoid applying stale x-limits to the new axis.
            self._has_plot = False
            self._prev_ts_valid = False
            # The blitting background was captured for the old axis layout;
            # clear it so _flush_cursor does not restore a stale image.
            self._bg = None
            self._invalidate_mouse_cache()
            self._refresh_plot()

    def set_xaxis_cursor(self, world_value: float) -> None:
        """Schedule a blitted x-axis cursor update.

        The blit is deferred to a ~60 fps timer so that rapid step events
        from the napari slider do not block the main thread waiting for
        the docked canvas to repaint.

        Parameters
        ----------
        world_value : float
            World coordinate along the x-axis dimension.
        """
        self._cursor_world = world_value
        if not self._cursor_timer.isActive():
            self._cursor_timer.start()

    def _world_to_xaxis(self, world_value: float) -> float | str:
        """Convert a world coordinate to the x-axis value used for plotting.

        Uses the current layer's `world_to_data` transform to map the
        world coordinate to a data index, then looks up the actual xarray
        coordinate value.  This keeps the cursor in sync with the time
        overlay for non-uniform spacing and multi-recording setups.

        For non-numeric coordinates (e.g., feature names on a categorical
        axis) the coordinate string itself is returned so that matplotlib
        places the cursor at the correct existing category.
        """
        if self._xaxis_coords is not None and self._current_layer is not None:
            index = self._world_to_data_index(world_value)
            if 0 <= index < len(self._xaxis_coords):
                coord = self._xaxis_coords[index]
                try:
                    return float(coord)
                except (ValueError, TypeError):
                    return str(coord)
        return world_value

    def _world_to_data_index(self, world_value: float) -> int:
        """Map a scalar world coordinate to a data index on the x-axis.

        Builds a full world-space point from the current viewer position,
        overrides the x-axis component with *world_value*, and transforms
        through the current layer's `world_to_data`.
        """
        layer = self._current_layer
        world_point = np.array(self._viewer.dims.point)
        xaxis_idx = self._xaxis_dim_index(layer)
        offset = self._viewer.dims.ndim - layer.ndim
        layer_world = world_point[offset:]
        layer_world[xaxis_idx] = world_value
        data_point = layer.world_to_data(layer_world)
        return int(np.round(data_point[xaxis_idx]))

    def _x_to_frame(self, x_val: float) -> float:
        """Convert an x-axis value back to a frame index (inverse of `_frame_to_x`)."""
        if self._xaxis_coords is not None:
            try:
                coords = np.asarray(self._xaxis_coords, dtype=float)
                return float(np.argmin(np.abs(coords - x_val)))
            except (ValueError, TypeError):
                pass
        return x_val

    def _on_click(self, event) -> None:
        """Handle left-click on the axes: emit `frame_clicked` for viewer navigation."""
        if event.inaxes is not self._axes:
            return
        if event.button != 1:
            return
        if self._toolbar.mode:
            return
        self.frame_clicked.emit(self._x_to_frame(event.xdata))

    def _flush_cursor(self) -> None:
        """Perform the actual blit for the current cursor position."""
        if self._vline is not None and self._bg is not None:
            try:
                x_cursor = self._world_to_xaxis(self._cursor_world)
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
        self._clear_export_signals()
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
                "voxel signals."
            )
        self._show_message(msg)

    def _refresh_plot(self) -> None:
        """Re-render the plot for the current source mode."""
        if self._updating_plot:
            return
        self._mouse_plot_dirty = True
        if self._source_mode == "points":
            self._update_plot_from_points()
        elif self._source_mode == "labels":
            self._update_plot_from_labels()
        else:
            self._update_plot()

    def _on_plot_data_changed(self) -> None:
        """Reset auto x-range when plotted imported signal membership changes."""
        self._has_plot = False
        self._prev_ts_valid = False
        self._mouse_plot_dirty = True

    def _update_plot(self) -> None:
        """Update the signals plot with current data."""
        if self._current_layer is None or self._cursor_pos is None:
            self._render_imported_only()
            return

        # Preserve zoom state across voxel changes. X limits are always
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

        ts = self._extract_signals(self._current_layer, self._cursor_pos)
        # Update stored x-axis coordinates for cursor mapping.
        self._xaxis_coords = self._get_xaxis_coords(self._current_layer)
        if ts is not None:
            if self._zscore:
                ts = self._apply_zscore(ts)

            if self._xaxis_coords is not None and len(self._xaxis_coords) == len(ts):
                x_values = self._xaxis_coords
            else:
                x_values = np.arange(len(ts))
            xlabel = self._get_xaxis_label(self._current_layer)

            # Read name/color/visibility from the store if available.
            mouse_signal = (
                self._signals_store.get_live_signal("mouse-0")
                if self._signals_store is not None
                else None
            )
            if mouse_signal is not None and not mouse_signal.visible:
                # Mouse signal hidden: only show imported.
                self._render_imported_only(saved_xlim=saved_xlim, saved_ylim=saved_ylim)
                return

            live_label = (
                mouse_signal.name
                if mouse_signal is not None
                else self._current_layer.name
            )
            live_color = (
                mouse_signal.color if mouse_signal is not None else colors["accent"]
            )

            imported_signals = self._imported_plot_signals()
            coord_str = self._mouse_coord_string()
            title = f"{self._current_layer.name} — ({coord_str})"

            if self._try_fast_mouse_update(
                live_label=live_label,
                live_color=live_color,
                x_values=np.asarray(x_values),
                ts=np.asarray(ts),
                xlabel=xlabel,
                title=title,
                saved_xlim=saved_xlim,
                saved_ylim=saved_ylim,
                imported_signals=imported_signals,
                colors=colors,
            ):
                return

            self._axes.clear()

            self._axes.plot(
                x_values,
                ts,
                linewidth=1.5,
                color=live_color,
                label=live_label,
            )

            self._mouse_live_line = self._axes.lines[-1]
            self._plot_signal_lines(imported_signals, linewidth=1.4, alpha=0.95)
            self._mouse_imported_signature = self._imported_signature(imported_signals)
            export_imported = self._plot_signal_to_export(imported_signals)
            self._set_export_signals(
                [ExportSignal(live_label, np.asarray(x_values), np.asarray(ts))]
                + export_imported
            )

        if ts is not None:
            self._style_valid_axes(
                colors,
                xlabel,
                "Z-score" if self._zscore else "Intensity",
                title,
                with_legend=len(self._export_signals) > 1,
            )
            self._restore_view(saved_xlim, saved_ylim)

            self._has_plot = True
            self._prev_ts_valid = True
            self._mouse_plot_dirty = False
            self._mouse_legend_signature = self._legend_signature(
                live_label, imported_signals
            )
        else:
            if self._render_imported_only(saved_xlim=saved_xlim, saved_ylim=saved_ylim):
                return
            self._clear_export_signals()
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

    def _xaxis_dim_index(self, layer) -> int:
        """Return the data dimension index that corresponds to the x-axis.

        Uses the configured `_xaxis_dim` when available, otherwise looks for
        the default signal dimension in xarray dims, falling back to 0.
        """
        da = layer.metadata.get("xarray")
        if da is not None:
            if self._xaxis_dim is not None and self._xaxis_dim in da.dims:
                return list(da.dims).index(self._xaxis_dim)
            if TIME_DIM in da.dims:
                return list(da.dims).index(TIME_DIM)
        return 0

    def _get_xaxis_coords(self, layer) -> np.ndarray | None:
        """Return the x-axis coordinate array from the layer's xarray metadata.

        Returns None when no xarray metadata is present or the DataArray has no
        coordinate for the configured x-axis dimension.
        """
        da = layer.metadata.get("xarray")
        if da is None:
            return None
        dim = self._xaxis_dim if self._xaxis_dim is not None else TIME_DIM
        if dim in da.coords:
            return np.asarray(da.coords[dim])
        return None

    def _get_xaxis_label(self, layer) -> str:
        """Return the x-axis label for the selected axis.

        Reads `long_name` and `units` from the coordinate attributes
        when available. Falls back to the dimension name with capitalized first letter.
        """
        da = layer.metadata.get("xarray")
        if da is None:
            return "Index"

        dim = self._xaxis_dim if self._xaxis_dim is not None else TIME_DIM
        if dim in da.coords:
            attrs = da.coords[dim].attrs
            name = attrs.get("long_name", dim.capitalize())
            units = attrs.get("units", "")
            return f"{name} ({units})" if units else name

        # No coordinate found for this dimension, use dimension name.
        return dim.capitalize()

    def _extract_signals(self, layer, cursor_pos: np.ndarray) -> np.ndarray | None:
        """Extract signals at cursor position from layer.

        Always uses the nearest voxel to the cursor position.
        """
        data = layer.data
        ind = list(int(round(x)) for x in layer.world_to_data(cursor_pos))

        xaxis_index = self._xaxis_dim_index(layer)
        # Replace the x-axis index before bounds-checking: the injected x-axis world
        # coordinate (typically 0) may fall outside the data range (e.g. when the
        # coordinate starts at a non-zero offset), which would cause the check to
        # reject valid spatial positions.
        ind[xaxis_index] = slice(None)  # type: ignore[call-overload]

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
            self._register_mouse_live_signal()
            # Invalidate the saved zoom state so the first mouse-hover plot does not
            # restore the [0, 1] xlim left by the cleared axes.
            self._has_plot = False
            self._prev_ts_valid = False
            # Only show instructions if there are no imported signal to display.
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
            try:
                self._points_layer.events.face_color.disconnect(
                    self._on_points_color_changed
                )
            except RuntimeError:
                pass
        self._points_layer = layer
        if layer is not None:
            layer.events.data.connect(self._on_points_data_changed)
            layer.events.face_color.connect(self._on_points_color_changed)
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
            try:
                self._labels_layer.events.colormap.disconnect(
                    self._on_labels_color_changed
                )
            except (RuntimeError, AttributeError):
                pass
        self._labels_layer = layer
        if layer is not None:
            layer.events.data.connect(self._on_labels_data_changed)
            layer.events.paint.connect(self._on_labels_data_changed)
            # Labels colors are managed via colormap; listen for colormap changes.
            layer.events.colormap.connect(self._on_labels_color_changed)
        if self._source_mode == "labels":
            self._update_plot_from_labels()

    def set_ref_layers(self, layers: list | None) -> None:
        """Set the reference image layers to extract signals from.

        Parameters
        ----------
        layers : list or None
            Specific Image layers to use, or None to use all image layers
            that have an x-axis dimension.
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
            displayed_dims = set(self._viewer.dims.displayed)
            result = []
            for layer in self._viewer.layers:
                if layer._type_string != "image" or getattr(layer, "rgb", False):
                    continue
                # Include layers that have at least one non-displayed dimension
                # with more than one element (a plottable signal axis). Spatial-only
                # maps (tSNR, CV, …) have no such dimension and are excluded.
                da = layer.metadata.get("xarray")
                if da is not None:
                    has_signal_dim = any(
                        i not in displayed_dims and da.shape[i] > 1
                        for i in range(da.ndim)
                    )
                    if has_signal_dim:
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

    def _set_export_signals(self, signals: list[ExportSignal]) -> None:
        """Store the currently plotted signals for export."""
        self._export_signals = prepare_export_signal(signals)
        self._export_button.setEnabled(bool(self._export_signals))

    # -- Live signal registration ------------------------------------------------

    def _register_mouse_live_signal(self) -> None:
        """Register a single live signal for the mouse cursor."""
        if self._signals_store is None:
            return
        colors = self._get_colors()
        self._signals_store.register_live_signals(
            [
                LiveSignal(
                    id="mouse-0",
                    name="Cursor",
                    color=colors["accent"],
                    visible=True,
                    source_type="mouse",
                    source_id=None,
                ),
            ]
        )

    def _register_points_live_signals(self) -> None:
        """Register live signals from the current Points layer."""
        if self._signals_store is None or self._points_layer is None:
            return
        n_points = len(self._points_layer.data)
        signals = []
        for point_index in range(n_points):
            color = mpl_colors.to_hex(self._points_layer.face_color[point_index])
            signals.append(
                LiveSignal(
                    id=f"point-{point_index}",
                    name=f"Point {point_index}",
                    color=color,
                    visible=True,
                    source_type="point",
                    source_id=point_index,
                )
            )
        self._signals_store.register_live_signals(signals)

    def _register_labels_live_signals(self, unique_labels: np.ndarray) -> None:
        """Register live signals from unique label IDs.

        Parameters
        ----------
        unique_labels : numpy.ndarray
            Array of nonzero label IDs found in the Labels layer.
        """
        if self._signals_store is None:
            return
        signals = []
        for lid in unique_labels:
            lid_int = int(lid)
            color = mpl_colors.to_hex(self._get_label_color(lid_int))
            signals.append(
                LiveSignal(
                    id=f"label-{lid_int}",
                    name=f"Label {lid_int}",
                    color=color,
                    visible=True,
                    source_type="label",
                    source_id=lid_int,
                )
            )
        self._signals_store.register_live_signals(signals)

    # -- Bidirectional color sync -----------------------------------------------

    def _sync_live_colors_to_layers(self) -> None:
        """Push live signals color overrides from the store to napari layers."""
        if self._syncing_color or self._signals_store is None:
            return
        self._syncing_color = True
        try:
            for signals in self._signals_store.live_signals():
                if signals.source_type == "point" and self._points_layer is not None:
                    point_index = signals.source_id
                    if point_index is not None and point_index < len(
                        self._points_layer.data
                    ):
                        current = mpl_colors.to_hex(
                            self._points_layer.face_color[point_index]
                        )
                        if current != signals.color:
                            rgba = transform_color(signals.color)[0]
                            # napari's face_color getter returns a copy, so
                            # indexing into it doesn't update the layer. We
                            # must write back the full array via the setter.
                            colors = self._points_layer.face_color.copy()
                            colors[point_index] = rgba
                            self._points_layer.face_color = colors
                elif signals.source_type == "label" and self._labels_layer is not None:
                    lid = signals.source_id
                    if lid is not None:
                        current = mpl_colors.to_hex(self._get_label_color(lid))
                        if current != signals.color:
                            rgba = np.asarray(transform_color(signals.color)[0])
                            lid_int = int(lid)

                            # Collect labels that have colors in the current colormap,
                            # avoiding expensive full array read.
                            unique_labels = set()
                            for i in range(1, 51):
                                if self._labels_layer.get_color(i) is not None:
                                    unique_labels.add(i)

                            # Get default colors - use layer's colormap if valid,
                            # otherwise use matplotlib's categorical colormap.
                            default_colors = self._get_default_label_colors(50)

                            # Build color dict preserving all existing label colors
                            # and adding default colors for potential future labels.
                            color_dict: dict = {}
                            for other_lid in unique_labels:
                                if other_lid == lid_int:
                                    color_dict[other_lid] = rgba
                                else:
                                    other_color = self._labels_layer.get_color(
                                        other_lid
                                    )
                                    if other_color is not None:
                                        color_dict[other_lid] = other_color

                            # Add default colors for potential labels (up to 50) so they
                            # don't appear transparent when painted later.
                            # Always include at least 10 potential labels so newly painted labels
                            # have visible colors.
                            max_label = max(
                                10, lid_int, max(unique_labels) if unique_labels else 0
                            )
                            for i in range(1, min(max_label + 1, 50)):
                                if i not in color_dict:
                                    color_dict[i] = default_colors[i]

                            # Ensure background has a visible color (not transparent).
                            if None not in color_dict:
                                color_dict[None] = default_colors[0]

                            self._labels_layer.colormap = DirectLabelColormap(
                                color_dict=color_dict
                            )
                            self._labels_layer.refresh()
        finally:
            self._syncing_color = False

    def _on_points_color_changed(self, event=None) -> None:
        """Sync point colors from the layer to the store."""
        if self._syncing_color or self._signals_store is None:
            return
        if self._points_layer is None:
            return
        self._syncing_color = True
        try:
            for point_index in range(len(self._points_layer.data)):
                sid = f"point-{point_index}"
                live = self._signals_store.get_live_signal(sid)
                if live is None:
                    continue
                new_color = mpl_colors.to_hex(
                    self._points_layer.face_color[point_index]
                )
                if new_color != live.color:
                    self._signals_store.set_live_signal_color(sid, new_color)
        finally:
            self._syncing_color = False

    def _on_labels_color_changed(self, event=None) -> None:
        """Sync label colors from the layer's colormap to the store."""
        if self._syncing_color or self._signals_store is None:
            return
        if self._labels_layer is None:
            return
        self._syncing_color = True
        try:
            for signal in self._signals_store.live_signals():
                if signal.source_type != "label" or signal.source_id is None:
                    continue
                new_color = mpl_colors.to_hex(self._get_label_color(signal.source_id))
                if new_color != signal.color:
                    self._signals_store.set_live_signal_color(signal.id, new_color)
        finally:
            self._syncing_color = False

    def _clear_export_signals(self) -> None:
        """Clear the currently exported signal."""
        self._set_export_signals([])

    def _visible_imported_signals(self) -> list[ImportedSignal]:
        """Return visible imported signals from the shared store."""
        if self._signals_store is None:
            return []
        return self._signals_store.visible_imported_signals()

    def _mouse_coord_string(self) -> str:
        """Return the current mouse coordinates formatted for the plot title.

        Formats coordinates as "dim=value" pairs. String coordinates are shown as-is,
        numeric coordinates are formatted to one decimal place.
        """
        if self._current_layer is None or self._cursor_pos is None:
            return ""

        da = self._current_layer.metadata.get("xarray")
        if da is None:
            # Fallback: just format cursor positions without dimension names.
            xaxis_index = self._xaxis_dim_index(self._current_layer)
            spatial_coords = [
                c for i, c in enumerate(self._cursor_pos) if i != xaxis_index
            ]
            return ", ".join(f"{c:.1f}" for c in spatial_coords)

        # Map cursor position to data indices, then to coordinate values.
        xaxis_index = self._xaxis_dim_index(self._current_layer)
        coord_parts = []

        # Convert all world positions to data indices once.
        data_indices = self._current_layer.world_to_data(self._cursor_pos)

        for i, (dim, world_pos) in enumerate(zip(da.dims, self._cursor_pos)):
            if i == xaxis_index:
                continue  # Skip the x-axis dimension (signals dimension).

            # Get the data index for this dimension.
            data_index = int(round(data_indices[i]))

            if dim in da.coords and data_index < da.sizes[dim]:
                coord_val = da.coords[dim].values[data_index]
                # Check if coordinate is string-like.
                if isinstance(coord_val, str) or (
                    hasattr(coord_val, "dtype") and coord_val.dtype.kind in "UO"
                ):
                    coord_parts.append(f"{dim}={coord_val}")
                else:
                    coord_parts.append(f"{dim}={coord_val:.1f}")
            else:
                # No coordinate metadata, use world position.
                coord_parts.append(f"{dim}={world_pos:.1f}")

        return ", ".join(coord_parts)

    def _imported_plot_signals(self) -> list[PlotSignal]:
        """Return imported signals prepared for plotting and export."""
        plotted = []
        for signals in self._visible_imported_signals():
            x_values = np.asarray(signals.x).copy()
            y_values = np.asarray(signals.y, dtype=float).copy()
            if self._zscore:
                y_values = self._apply_zscore(y_values)
            plotted.append(PlotSignal(signals.name, x_values, y_values, signals.color))
        return plotted

    def _plot_signal_lines(
        self,
        signals: list[PlotSignal],
        *,
        linewidth: float = 1.5,
        alpha: float = 1.0,
    ) -> None:
        """Plot a list of signals onto the current axes."""
        for s in signals:
            kwargs = {"linewidth": linewidth, "alpha": alpha, "label": s.label}
            if s.color is not None:
                kwargs["color"] = s.color
            self._axes.plot(s.x_values, s.y_values, **kwargs)

    def _imported_signature(self, signals: list[PlotSignal]) -> tuple[str, ...]:
        """Return a lightweight signature for imported signals already on the axes."""
        return tuple(s.label for s in signals)

    def _legend_signature(
        self, live_label: str, imported_signals: list[PlotSignal]
    ) -> tuple[str, ...]:
        """Return the legend content signature for the current mouse plot."""
        return (live_label, *self._imported_signature(imported_signals))

    @staticmethod
    def _plot_signal_to_export(signals: list[PlotSignal]) -> list[ExportSignal]:
        """Convert PlotSignal to ExportSignal for CSV/TSV export.

        Parameters
        ----------
        signals : list[PlotSignal]
            Signals with color information.

        Returns
        -------
        list[ExportSignal]
            Signals without color for export.
        """
        return [ExportSignal(s.label, s.x_values, s.y_values) for s in signals]

    def _try_fast_mouse_update(
        self,
        *,
        live_label: str,
        live_color: str,
        x_values: np.ndarray,
        ts: np.ndarray,
        xlabel: str,
        title: str,
        saved_xlim,
        saved_ylim,
        imported_signals: list[PlotSignal],
        colors: dict,
    ) -> bool:
        """Update the mouse plot without rebuilding imported overlay lines."""
        if self._mouse_plot_dirty or self._mouse_live_line is None:
            return False
        if self._mouse_imported_signature != self._imported_signature(imported_signals):
            return False

        self._mouse_live_line.set_data(x_values, ts)
        self._mouse_live_line.set_label(live_label)
        self._mouse_live_line.set_color(live_color)
        self._set_export_signals(
            [ExportSignal(live_label, x_values, ts)]
            + self._plot_signal_to_export(imported_signals)
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

        self._update_legend_if_needed(colors, live_label, imported_signals)

        self._axes.relim()
        self._restore_view(saved_xlim, saved_ylim)
        if self._autoscale:
            self._axes.autoscale_view(scalex=False, scaley=True)

        self._has_plot = True
        self._prev_ts_valid = True
        self._canvas.draw_idle()
        return True

    def _update_legend_if_needed(
        self, colors: dict, live_label: str, imported_signals: list[PlotSignal]
    ) -> None:
        """Update the legend only when its entries actually changed."""
        signature = self._legend_signature(live_label, imported_signals)
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
        """Render only imported signals when no live data can be drawn."""
        imported_signals = self._imported_plot_signals()
        if not imported_signals:
            return False

        colors = self._get_colors()
        self._axes.clear()
        self._plot_signal_lines(imported_signals, linewidth=1.4, alpha=0.95)
        self._set_export_signals(self._plot_signal_to_export(imported_signals))
        self._mouse_live_line = None
        self._mouse_imported_signature = self._imported_signature(imported_signals)
        self._mouse_legend_signature = self._imported_signature(imported_signals)
        self._mouse_plot_dirty = True
        self._xaxis_coords = None
        self._style_valid_axes(
            colors,
            "Time",
            "Z-score" if self._zscore else "Value",
            "Imported Signals",
            with_legend=len(imported_signals) > 1,
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
        """Show a message when no imported signals are visible, else plot imports only."""
        if self._render_imported_only(saved_xlim=saved_xlim, saved_ylim=saved_ylim):
            return
        self._show_message(text)

    def _write_current_plot_delimited(self, path: Path, delimiter: str) -> None:
        """Write the currently plotted signals to a delimited text file."""
        write_delimited_signals(path, self._export_signals, delimiter=delimiter)

    def _save_current_plot(self) -> None:
        """Open a save dialog and export the current plot as CSV or TSV."""
        if not self._export_signals:
            show_error("No signals plot available to save.")
            return

        export_selection = prompt_delimited_export_path(
            self, "Export Signals", str(Path.home() / "signals.tsv")
        )
        if export_selection is None:
            return

        path, delimiter = export_selection

        try:
            self._write_current_plot_delimited(path, delimiter)
        except Exception as exc:
            show_error(str(exc))
            return

        show_info(f"Exported signals to {path}")

    def _apply_zscore(self, ts: np.ndarray) -> np.ndarray:
        """Normalise a signals to zero mean and unit variance.

        NaN values are ignored when computing the mean and standard deviation so that
        signals with missing values (imported files with partial columns) are z-scored
        correctly rather than becoming all-NaN.
        """
        mean = np.nanmean(ts)
        std = np.nanstd(ts)
        return (ts - mean) / std if std > 0 else ts - mean

    def _get_default_label_colors(self, count: int = 20) -> np.ndarray:
        """Return default label colors.

        First tries to use the Labels layer's colormap if it has valid colors,
        otherwise falls back to matplotlib's tab20 colormap.
        """
        if self._labels_layer is not None:
            cmap = self._labels_layer.colormap
            if hasattr(cmap, "colors") and len(cmap.colors) >= count:
                return cmap.colors

        # Fallback: use matplotlib's tab20 colormap
        return mpl_colormaps["tab20"](np.linspace(0, 1, 20))

    def _get_label_color(self, label_id: int) -> tuple:
        """Return a matplotlib-compatible RGBA color for a label ID.

        Tries to read the color from the Labels layer's colormap; falls back
        to the default label colors.
        """
        if self._labels_layer is not None:
            try:
                color = self._labels_layer.get_color(label_id)
                if color is not None:
                    return tuple(float(c) for c in color)
            except (AttributeError, KeyError, IndexError):
                pass
        defaults = self._get_default_label_colors(20)
        return tuple(defaults[int(label_id - 1) % 20])

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
                self._world_to_xaxis(self._cursor_world),
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
    def _signal_label(base: str, img_layer, ref_layers: list) -> str:
        """Build a signal label, prefixing the layer name when multiple layers are shown.

        Parameters
        ----------
        base : str
            Base label (e.g. `"Point 0"` or `"Label 3"`).
        img_layer : napari image layer
            Reference image layer contributing this signal.
        ref_layers : list
            All active reference image layers.

        Returns
        -------
        str
            `"<layer name> | <base>"` when more than one reference layer is active,
            otherwise just `base`.
        """
        return f"{img_layer.name} | {base}" if len(ref_layers) > 1 else base

    def _finalize_multi_signals_plot(
        self,
        export_signals: list,
        source_layer_name: str,
        ref_layers: list,
        colors: dict,
        saved_xlim,
        saved_ylim,
    ) -> None:
        """Overlay imported signals, style the axes, and redraw after a multi-signals plot.

        Shared by `_update_plot_from_points` and `_update_plot_from_labels` once their
        per-signal loops have completed successfully.

        Parameters
        ----------
        export_signals : list[ExportSignal]
            Signals already plotted by the caller.
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
        imported_signals = self._imported_plot_signals()
        self._plot_signal_lines(imported_signals, linewidth=1.4, alpha=0.95)
        self._set_export_signals(
            export_signals + self._plot_signal_to_export(imported_signals)
        )
        xlabel = self._get_xaxis_label(ref_layers[0])
        ylabel = "Z-score" if self._zscore else "Intensity"
        title_ref = ref_layers[0].name if len(ref_layers) == 1 else "All layers"
        self._style_valid_axes(
            colors,
            xlabel,
            ylabel,
            f"{title_ref} — {source_layer_name}",
            with_legend=len(self._export_signals) > 1,
        )
        self._restore_view(saved_xlim, saved_ylim)
        self._canvas.draw_idle()

    def _update_plot_from_points(self) -> None:
        """Plot signals for every point in the Points layer."""
        if self._updating_plot:
            return
        self._updating_plot = True
        try:
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

            # Register live signals so the manager can track them.
            self._register_points_live_signals()

            # Pre-compute x-axis coordinates per layer: they don't depend on individual
            # points.
            layer_xaxis_coords = [self._get_xaxis_coords(layer) for layer in ref_layers]

            self._axes.clear()
            has_any = False
            export_signals: list[ExportSignal] = []

            for point_index in range(n_points):
                # Check store for visibility/name/color overrides.
                live = (
                    self._signals_store.get_live_signal(f"point-{point_index}")
                    if self._signals_store is not None
                    else None
                )
                if live is not None and not live.visible:
                    continue

                pt_data = np.asarray(self._points_layer.data[point_index], dtype=float)
                n_pt = len(pt_data)
                # napari may pad layer.scale/translate to the viewer's ndim, so use the last
                # n_pt elements to match the point's dimensionality.
                pts_scale = np.asarray(self._points_layer.scale, dtype=float)[-n_pt:]
                pts_translate = np.asarray(self._points_layer.translate, dtype=float)[
                    -n_pt:
                ]
                # data → world (spatial, n_pt dims)
                pt_world = pt_data * pts_scale + pts_translate

                pt_name = live.name if live is not None else f"Point {point_index}"
                pt_color = (
                    live.color
                    if live is not None
                    else mpl_colors.to_hex(self._points_layer.face_color[point_index])
                )

                for layer_index, img_layer in enumerate(ref_layers):
                    img_ndim = img_layer.data.ndim
                    if n_pt < img_ndim:
                        # 3D point in a 4D image: pad the world coord with 0 at the
                        # front so world_to_data receives the correct ndim. The
                        # x-axis value (0) is irrelevant — _extract_signals replaces
                        # it with slice(None).
                        padded = np.zeros(img_ndim)
                        padded[-n_pt:] = pt_world
                        pt_world_img = padded
                    else:
                        pt_world_img = pt_world
                    try:
                        ts = self._extract_signals(img_layer, pt_world_img)
                    except Exception:  # noqa: BLE001
                        ts = None
                    if ts is None:
                        continue

                    ts = np.asarray(ts, dtype=float)
                    if self._zscore:
                        ts = self._apply_zscore(ts)

                    xaxis_coords = layer_xaxis_coords[layer_index]
                    x = (
                        xaxis_coords
                        if (xaxis_coords is not None and len(xaxis_coords) == len(ts))
                        else np.arange(len(ts))
                    )
                    label = self._signal_label(pt_name, img_layer, ref_layers)
                    linestyle = _LAYER_LINESTYLES[layer_index % len(_LAYER_LINESTYLES)]
                    self._axes.plot(
                        x,
                        ts,
                        color=pt_color,
                        linewidth=1.5,
                        linestyle=linestyle,
                        label=label,
                    )
                    export_signals.append(
                        ExportSignal(label, np.asarray(x), np.asarray(ts))
                    )
                    has_any = True
                    # Keep x-axis coords from the last successful layer for cursor mapping.
                    self._xaxis_coords = xaxis_coords
                    self._current_layer = img_layer

            if has_any:
                self._finalize_multi_signals_plot(
                    export_signals,
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
        finally:
            self._updating_plot = False

    def _update_plot_from_labels(self) -> None:
        """Plot mean signal for each unique label in the Labels layer."""
        if self._updating_plot:
            return
        self._updating_plot = True
        try:
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
            export_signals: list[ExportSignal] = []
            # Collect all unique labels across layers for registration.
            all_unique_labels: set[int] = set()

            for layer_index, img_layer in enumerate(ref_layers):
                xaxis_index = self._xaxis_dim_index(img_layer)
                img_data = img_layer.data
                img_spatial = tuple(
                    s for i, s in enumerate(img_data.shape) if i != xaxis_index
                )

                # Labels can have the full image shape (napari creates labels with the
                # same shape as the reference image) or the spatial shape only. Collapse
                # the x-axis in the former case.
                if labels_data.shape == img_data.shape:
                    labels_spatial = np.max(labels_data, axis=xaxis_index)
                elif labels_data.shape == img_spatial:
                    labels_spatial = labels_data
                else:
                    shape_mismatch = True
                    continue

                unique_labels = np.unique(labels_spatial)
                unique_labels = unique_labels[unique_labels != 0]  # exclude background
                if len(unique_labels) == 0:
                    continue

                all_unique_labels.update(int(lid) for lid in unique_labels)

                # Move x-axis to last axis so spatial boolean indexing works cleanly:
                # img_arr[mask] → (N_voxels, N_xaxis).
                img_arr = np.moveaxis(np.asarray(img_data), xaxis_index, -1)

                xaxis_coords = self._get_xaxis_coords(img_layer)
                x = (
                    xaxis_coords
                    if xaxis_coords is not None
                    else np.arange(img_arr.shape[-1])
                )

                for lid in unique_labels:
                    lid_int = int(lid)
                    # Check store for visibility/name/color overrides.
                    live = (
                        self._signals_store.get_live_signal(f"label-{lid_int}")
                        if self._signals_store is not None
                        else None
                    )
                    if live is not None and not live.visible:
                        continue

                    mask = labels_spatial == lid
                    ts = np.asarray(img_arr[mask].mean(axis=0), dtype=float)  # (T,)

                    if self._zscore:
                        ts = self._apply_zscore(ts)

                    base_name = live.name if live is not None else f"Label {lid_int}"
                    lid_color = (
                        live.color
                        if live is not None
                        else mpl_colors.to_hex(self._get_label_color(lid_int))
                    )
                    label = self._signal_label(base_name, img_layer, ref_layers)
                    linestyle = _LAYER_LINESTYLES[layer_index % len(_LAYER_LINESTYLES)]
                    self._axes.plot(
                        x,
                        ts,
                        color=lid_color,
                        linewidth=1.5,
                        linestyle=linestyle,
                        label=label,
                    )
                    export_signals.append(
                        ExportSignal(label, np.asarray(x), np.asarray(ts))
                    )
                    has_any = True

                # Keep time coords from the last successful layer for cursor mapping.
                self._xaxis_coords = xaxis_coords
                self._current_layer = img_layer

            # Register live signal after we know all unique labels.
            if all_unique_labels:
                self._register_labels_live_signals(np.array(sorted(all_unique_labels)))

            if has_any:
                self._finalize_multi_signals_plot(
                    export_signals,
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
        finally:
            self._updating_plot = False

    def closeEvent(self, a0) -> None:
        """Clean up when widget is closed.

        Parameters
        ----------
        a0 : QCloseEvent
            The close event (unused; accepted unconditionally).
        """
        if self._on_mouse_move in self._viewer.mouse_move_callbacks:
            self._viewer.mouse_move_callbacks.remove(self._on_mouse_move)
        if self._signals_store is not None:
            try:
                self._signals_store.plot_data_changed.disconnect(
                    self._on_plot_data_changed
                )
            except (RuntimeError, TypeError):
                pass
            try:
                self._signals_store.changed.disconnect(self._refresh_plot)
            except (RuntimeError, TypeError):
                pass
            try:
                self._signals_store.changed.disconnect(self._sync_live_colors_to_layers)
            except (RuntimeError, TypeError):
                pass
        self.set_points_layer(None)
        self.set_labels_layer(None)
        self._labels_debounce.stop()
        a0.accept()
