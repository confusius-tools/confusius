"""Time series plotter widget for the bottom dock."""

from __future__ import annotations

import napari
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from confusius._napari._utils import napari_colors, recolor_toolbar_icons


class TimeSeriesPlotter(QWidget):
    """Bottom dock widget for live time series plotting.

    This widget displays live time series plots when hovering over image layers while
    holding the Shift key. The first dimension of the image is treated as time.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = viewer
        self._cursor_pos: np.ndarray | None = None
        self._current_layer = None

        # Plot settings
        self._ylim_min: float | None = None
        self._ylim_max: float | None = None
        self._autoscale: bool = True
        self._show_grid: bool = True
        self._zscore: bool = False

        # Whether at least one time series has been plotted (used to decide whether to
        # restore the zoom state on the next _update_plot call).
        self._has_plot: bool = False

        # Time cursor blitting state
        self._show_cursor: bool = False
        self._vline = None
        self._bg = None  # Saved pixel buffer (without vline).
        self._cursor_frame: float = 0.0

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
        self._apply_theme()

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
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._axes = self._figure.add_subplot(111)
        # Save background after each full redraw for blitting the time cursor.
        self._canvas.mpl_connect("draw_event", self._on_draw)
        self._show_instructions()

    def _setup_callbacks(self) -> None:
        """Set up napari event callbacks."""
        self._viewer.mouse_move_callbacks.append(self._on_mouse_move)
        self._viewer.layers.events.inserted.connect(self._on_layer_change)
        self._viewer.layers.events.removed.connect(self._on_layer_change)
        self._viewer.layers.selection.events.active.connect(self._on_layer_change)

    def _get_colors(self) -> dict:
        """Get current theme colors."""
        return napari_colors(self._viewer.theme)

    def _apply_theme(self) -> None:
        """Apply napari theme to matplotlib figure."""
        colors = self._get_colors()

        self._figure.patch.set_facecolor(colors["bg"])
        self._axes.set_facecolor(colors["bg"])

        self._toolbar.setStyleSheet(f"background: {colors['bg']}; border: none;")
        recolor_toolbar_icons(self._toolbar, colors["fg"])

        for spine in self._axes.spines.values():
            spine.set_edgecolor(colors["fg"])
        self._axes.tick_params(colors=colors["fg"])
        self._axes.xaxis.label.set_color(colors["fg"])
        self._axes.yaxis.label.set_color(colors["fg"])
        self._axes.title.set_color(colors["fg"])

        self._canvas.draw()

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
        self._update_plot()

    def set_autoscale(self, enabled: bool) -> None:
        """Enable or disable Y-axis autoscaling.

        Parameters
        ----------
        enabled : bool
            Whether to autoscale the Y-axis.
        """
        self._autoscale = enabled
        self._update_plot()

    def set_show_grid(self, enabled: bool) -> None:
        """Enable or disable grid display.

        Parameters
        ----------
        enabled : bool
            Whether to show the grid.
        """
        self._show_grid = enabled
        self._update_plot()

    def set_zscore(self, enabled: bool) -> None:
        """Enable or disable Z-scoring of time series.

        Parameters
        ----------
        enabled : bool
            Whether to Z-score the time series.
        """
        self._zscore = enabled
        self._update_plot()

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
        self._update_plot()

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

    def _flush_cursor(self) -> None:
        """Perform the actual blit for the current cursor position."""
        if self._vline is not None and self._bg is not None:
            try:
                self._canvas.restore_region(self._bg)
                self._vline.set_xdata([self._cursor_frame, self._cursor_frame])
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

    def _show_instructions(self) -> None:
        """Show initial instructions."""
        self._axes.clear()
        colors = self._get_colors()
        self._axes.text(
            0.5,
            0.5,
            'Hold "Shift" while moving the cursor\n'
            "over an image layer to plot\n"
            "voxel time series.",
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
        self._canvas.draw()

    def _update_plot(self) -> None:
        """Update the time series plot with current data."""
        if self._current_layer is None or self._cursor_pos is None:
            return

        # Preserve zoom state across voxel changes. X limits (time range) are always
        # restored so the user can zoom in and keep exploring. Y limits are only
        # restored when autoscale is off, so the axis still adjusts to the new voxel's
        # amplitude when autoscale is on.
        saved_xlim = self._axes.get_xlim() if self._has_plot else None
        saved_ylim = (
            self._axes.get_ylim() if self._has_plot and not self._autoscale else None
        )

        colors = self._get_colors()
        self._axes.clear()

        ts = self._extract_time_series(self._current_layer, self._cursor_pos)
        if ts is not None:
            if self._zscore:
                ts_mean = np.mean(ts)
                ts_std = np.std(ts)
                if ts_std > 0:
                    ts = (ts - ts_mean) / ts_std
                else:
                    ts = ts - ts_mean

            time_points = np.arange(len(ts))
            self._axes.plot(
                time_points,
                ts,
                linewidth=1.5,
                color=colors["accent"],
            )

        if ts is not None:
            # Show actual cursor coordinates (not rounded to voxel), excluding the time
            # dimension.
            t_idx = self._time_dim_index(self._current_layer)
            spatial_coords = [c for i, c in enumerate(self._cursor_pos) if i != t_idx]
            coord_str = ", ".join(f"{c:.1f}" for c in spatial_coords)

            self._axes.set_xlabel("Time Frame", color=colors["fg"], fontsize=9)
            if self._zscore:
                self._axes.set_ylabel("Z-score", color=colors["fg"], fontsize=9)
            else:
                self._axes.set_ylabel("Intensity", color=colors["fg"], fontsize=9)
            self._axes.set_title(
                f"{self._current_layer.name} — ({coord_str})",
                color=colors["fg"],
                fontsize=10,
            )

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

            if saved_xlim is not None:
                self._axes.set_xlim(saved_xlim)
            if saved_ylim is not None:
                self._axes.set_ylim(saved_ylim)

            self._has_plot = True

            if self._show_cursor:
                self._vline = self._axes.axvline(
                    self._cursor_frame,
                    color=colors["cursor"],
                    linewidth=1.2,
                    alpha=0.85,
                    zorder=10,
                    animated=True,
                )
            else:
                self._vline = None
        else:
            self._vline = None
            # No valid data at the cursor position
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

        self._canvas.draw()

    def _time_dim_index(self, layer) -> int:
        """Return the data dimension index that corresponds to time.

        Reads `layer.metadata["xarray"]` when available; falls back to 0.
        """
        da = layer.metadata.get("xarray")
        if da is not None and "time" in da.dims:
            return list(da.dims).index("time")
        return 0

    def _extract_time_series(self, layer, cursor_pos: np.ndarray) -> np.ndarray | None:
        """Extract time series at cursor position from layer.

        Always uses the nearest voxel to the cursor position.
        """
        data = layer.data
        ind = list(int(round(x)) for x in layer.world_to_data(cursor_pos))

        if not all(0 <= i < max_i for i, max_i in zip(ind, data.shape)):
            return None

        t_idx = self._time_dim_index(layer)
        ind[t_idx] = slice(None)  # type: ignore[call-overload]
        return data[tuple(ind)]

    def closeEvent(self, a0) -> None:
        """Clean up when widget is closed."""
        if self._on_mouse_move in self._viewer.mouse_move_callbacks:
            self._viewer.mouse_move_callbacks.remove(self._on_mouse_move)
        a0.accept()
