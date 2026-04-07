"""Bottom dock widget showing QC plots (DVARS and carpet plot)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import napari
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import QSize, QTimer, Signal
from qtpy.QtWidgets import QSizePolicy, QTabWidget, QVBoxLayout, QWidget

from confusius._napari._export import (
    ExportSignal,
    prepare_export_signal,
    prompt_delimited_export_path,
    write_delimited_signals,
)
from confusius._napari._theme import (
    create_export_button,
    get_napari_colors,
    style_export_button,
    style_plot_toolbar,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    import xarray as xr


class QCPlotsWidget(QWidget):
    """Tabbed matplotlib canvas shown in the napari bottom dock area.

    Tabs
    ----
    DVARS
        Line plot of the DVARS metric, with a navigation toolbar and a blitted vertical
        cursor tracking the napari time slider.
    Carpet plot
        Raster image of voxel intensities over time, with a matching cursor.

    Parameters
    ----------
    viewer : napari.Viewer
        Used to read the current theme whenever plots are (re)drawn.
    """

    time_clicked = Signal(float)
    """Emitted when the user left-clicks on a QC plot axes.

    The payload is the x-axis time value (or frame index) at the click position.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = viewer
        self.setMinimumHeight(150)
        # Cached data for theme-change redraws.
        self._dvars_da: xr.DataArray | None = None
        self._dvars_layer_name: str = ""
        self._dvars_export_signal: list[ExportSignal] = []
        self._data_da: dict | None = None  # pre-computed carpet dict
        self._carpet_layer_name: str = ""
        # Blitting state per plot.
        self._dvars_ax: Axes | None = None
        self._dvars_vline = None
        self._dvars_bg = None  # saved pixel buffer (no vline)
        self._carpet_ax: Axes | None = None
        self._carpet_vline = None
        self._carpet_bg = None
        # Last known time value so vlines are restored correctly after replot.
        self._current_time_val: float | None = None
        # Throttle blit calls to ~60 fps (see SignalsPlotter for rationale).
        self._cursor_timer = QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.setInterval(16)  # ms → ~60 fps
        self._cursor_timer.timeout.connect(self._flush_cursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._setup_ui()

    def sizeHint(self) -> QSize:
        """Return the preferred initial size of the widget.

        Returns
        -------
        QSize
            Preferred size of 800 × 320 pixels.
        """
        return QSize(800, 320)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        _exp = QSizePolicy.Policy.Expanding
        self._tabs = QTabWidget()
        self._tabs.setSizePolicy(_exp, _exp)

        # --- DVARS tab (added lazily on first update_dvars call) ----------
        self._dvars_fig = Figure(tight_layout=True)
        self._dvars_canvas = FigureCanvas(self._dvars_fig)
        self._dvars_canvas.setSizePolicy(_exp, _exp)
        # draw_event fires after every full redraw; use it to save the background
        # (without the animated vline) ready for blitting.
        self._dvars_canvas.mpl_connect("draw_event", self._on_dvars_draw)
        self._dvars_canvas.mpl_connect("button_press_event", self._on_dvars_click)

        self._dvars_tab = QWidget()
        dvars_layout = QVBoxLayout(self._dvars_tab)
        dvars_layout.setContentsMargins(4, 4, 4, 4)
        dvars_layout.setSpacing(0)
        self._dvars_toolbar = NavigationToolbar(self._dvars_canvas, self._dvars_tab)
        self._dvars_export_button = create_export_button(
            self._dvars_toolbar, "dvarsExportButton", self._save_dvars
        )
        self._dvars_export_button.setEnabled(False)
        self._dvars_toolbar.addWidget(self._dvars_export_button)
        dvars_layout.addWidget(self._dvars_toolbar)
        dvars_layout.addWidget(self._dvars_canvas)
        # Tab is added to self._tabs only when first populated.

        # --- Carpet plot tab (added lazily on first update_carpet call) ---
        self._carpet_fig = Figure(tight_layout=True)
        self._carpet_canvas = FigureCanvas(self._carpet_fig)
        self._carpet_canvas.setSizePolicy(_exp, _exp)
        self._carpet_canvas.mpl_connect("draw_event", self._on_carpet_draw)
        self._carpet_canvas.mpl_connect("button_press_event", self._on_carpet_click)

        self._carpet_tab = QWidget()
        carpet_layout = QVBoxLayout(self._carpet_tab)
        carpet_layout.setContentsMargins(4, 4, 4, 4)
        carpet_layout.setSpacing(0)
        self._carpet_toolbar = NavigationToolbar(self._carpet_canvas, self._carpet_tab)
        carpet_layout.addWidget(self._carpet_toolbar)
        carpet_layout.addWidget(self._carpet_canvas)
        # Tab is added to self._tabs only when first populated.

        layout.addWidget(self._tabs)

    # ------------------------------------------------------------------
    # Theme helpers
    # ------------------------------------------------------------------

    def _style_toolbar(self, toolbar: NavigationToolbar, colors: dict) -> None:
        """Apply the napari background to the toolbar and recolor its icons."""
        style_plot_toolbar(toolbar, colors)
        if toolbar is self._dvars_toolbar:
            style_export_button(self._dvars_export_button, colors)

    def _set_dvars_export_signal(self, signal: list[ExportSignal]) -> None:
        """Store the currently plotted DVARS signal for export."""
        self._dvars_export_signal = prepare_export_signal(signal)
        self._dvars_export_button.setEnabled(bool(self._dvars_export_signal))

    def _write_dvars_delimited(self, path: Path, delimiter: str) -> None:
        """Write the current DVARS trace to a delimited text file."""
        write_delimited_signals(path, self._dvars_export_signal, delimiter=delimiter)

    def _save_dvars(self) -> None:
        """Open a save dialog and export the DVARS trace as CSV or TSV."""
        if not self._dvars_export_signal:
            show_error("No DVARS plot available to export.")
            return

        export_selection = prompt_delimited_export_path(
            self, "Export DVARS", str(Path.home() / "dvars.tsv")
        )
        if export_selection is None:
            return

        path, delimiter = export_selection

        try:
            self._write_dvars_delimited(path, delimiter)
        except Exception as exc:
            show_error(str(exc))
            return

        show_info(f"Exported DVARS to {path}")

    # ------------------------------------------------------------------
    # Blitting helpers
    # ------------------------------------------------------------------

    def _on_dvars_draw(self, event) -> None:
        """After each full DVARS draw, save the background and blit the vline."""
        if self._dvars_vline is None or self._dvars_ax is None:
            return
        try:
            self._dvars_bg = self._dvars_canvas.copy_from_bbox(self._dvars_fig.bbox)
            self._dvars_ax.draw_artist(self._dvars_vline)
            self._dvars_canvas.blit(self._dvars_fig.bbox)
        except Exception:  # noqa: BLE001
            self._dvars_bg = None

    def _on_carpet_draw(self, event) -> None:
        """After each full carpet draw, save the background and blit the vline."""
        if self._carpet_vline is None or self._carpet_ax is None:
            return
        try:
            self._carpet_bg = self._carpet_canvas.copy_from_bbox(self._carpet_fig.bbox)
            self._carpet_ax.draw_artist(self._carpet_vline)
            self._carpet_canvas.blit(self._carpet_fig.bbox)
        except Exception:  # noqa: BLE001
            self._carpet_bg = None

    # ------------------------------------------------------------------
    # Click-to-navigate helpers
    # ------------------------------------------------------------------

    def _on_dvars_click(self, event) -> None:
        """Handle left-click on DVARS axes: emit `time_clicked`."""
        if event.inaxes is not self._dvars_ax:
            return
        if event.button != 1:
            return
        if self._dvars_toolbar.mode:
            return
        self.time_clicked.emit(event.xdata)

    def _on_carpet_click(self, event) -> None:
        """Handle left-click on carpet plot axes: emit `time_clicked`."""
        if event.inaxes is not self._carpet_ax:
            return
        if event.button != 1:
            return
        if self._carpet_toolbar.mode:
            return
        self.time_clicked.emit(event.xdata)

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def update_dvars(self, dvars_da: xr.DataArray, layer_name: str = "") -> None:
        """Redraw the DVARS signal using the current napari theme.

        Parameters
        ----------
        dvars_da : xarray.DataArray
            1D DVARS signal, optionally with a `"time"` coordinate.
        layer_name : str, optional
            Name of the source layer, shown as the plot title.
        """
        import numpy as np

        self._dvars_da = dvars_da
        self._dvars_layer_name = layer_name
        # Invalidate blitting state before clearing the figure.
        self._dvars_ax = None
        self._dvars_vline = None
        self._dvars_bg = None

        # Add the tab on first call; it stays for the lifetime of the widget.
        if self._tabs.indexOf(self._dvars_tab) == -1:
            self._tabs.insertTab(0, self._dvars_tab, "DVARS")

        colors = get_napari_colors(self._viewer.theme)

        self._dvars_fig.clear()
        self._dvars_fig.patch.set_facecolor(colors["bg"])

        ax = self._dvars_fig.add_subplot(111)
        ax.set_facecolor(colors["bg"])

        time_coord = dvars_da.coords.get("time")
        if time_coord is not None:
            x = time_coord.values
        else:
            x = np.arange(len(dvars_da), dtype=float)
        xlabel = "Time (s)" if time_coord is not None else "Frame"
        self._set_dvars_export_signal([ExportSignal("DVARS", x, dvars_da.values)])

        ax.plot(x, dvars_da.values, color=colors["accent"], linewidth=1.2)
        ax.set_xlabel(xlabel, color=colors["fg"], fontsize=9)
        ax.set_ylabel("DVARS", color=colors["fg"], fontsize=9)
        if self._dvars_layer_name:
            ax.set_title(self._dvars_layer_name, color=colors["fg"], fontsize=10)
        ax.tick_params(colors=colors["fg"], labelsize=8)
        ax.set_xlim(x[0], x[-1])
        for spine in ax.spines.values():
            spine.set_edgecolor(colors["fg"])

        # animated=True excludes the vline from the normal draw cycle so only the static
        # content is rendered during canvas.draw(). The draw_event handler then saves
        # that background and blits the vline.
        t0 = (
            self._current_time_val
            if self._current_time_val is not None
            else float(x[0])
        )
        self._dvars_vline = ax.axvline(
            t0,
            color=colors["cursor"],
            linewidth=1.2,
            alpha=0.85,
            zorder=10,
            animated=True,
        )
        self._dvars_ax = ax

        self._style_toolbar(self._dvars_toolbar, colors)
        self._dvars_canvas.draw()  # → triggers _on_dvars_draw
        self._tabs.setCurrentWidget(self._dvars_tab)

    def update_carpet(self, carpet_data: dict, layer_name: str = "") -> None:
        """Redraw the carpet plot from pre-computed data.

        Parameters
        ----------
        carpet_data : dict
            Pre-computed dict returned by `_precompute_carpet`, with keys `signals`,
            `vmin`, `vmax`, `xlabel`, and `time_coord`.
        layer_name : str, optional
            Name of the source layer, shown as the plot title.
        """
        from confusius.plotting.image import _draw_carpet

        self._data_da = carpet_data
        self._carpet_layer_name = layer_name
        self._carpet_ax = None
        self._carpet_vline = None
        self._carpet_bg = None

        # Add the tab on first call; it stays for the lifetime of the widget.
        if self._tabs.indexOf(self._carpet_tab) == -1:
            self._tabs.addTab(self._carpet_tab, "Carpet plot")

        colors = get_napari_colors(self._viewer.theme)

        self._carpet_fig.clear()
        self._carpet_fig.patch.set_facecolor(colors["bg"])

        ax = self._carpet_fig.add_subplot(111)
        _draw_carpet(carpet_data, ax=ax, black_bg=colors["is_dark"])

        if self._carpet_layer_name:
            ax.set_title(self._carpet_layer_name, color=colors["fg"], fontsize=10)

        time_coord = carpet_data["time_coord"]
        t0 = self._current_time_val
        if t0 is None:
            t0 = float(time_coord[0]) if time_coord is not None else 0.0
        self._carpet_vline = ax.axvline(
            t0,
            color=colors["cursor"],
            linewidth=1.2,
            alpha=0.85,
            zorder=10,
            animated=True,
        )
        self._carpet_ax = ax

        self._style_toolbar(self._carpet_toolbar, colors)
        self._carpet_canvas.draw()  # → triggers _on_carpet_draw

    def set_time_cursor(self, time_val: float) -> None:
        """Schedule a blitted time-cursor update.

        The blit is deferred to a ~60 fps timer so that rapid step events from the
        napari time slider do not block the main thread waiting for the docked canvas to
        repaint.

        Parameters
        ----------
        time_val : float
            Physical time value (or frame index) to position the cursor at.
        """
        self._current_time_val = time_val
        if not self._cursor_timer.isActive():
            self._cursor_timer.start()

    def _flush_cursor(self) -> None:
        """Perform the actual blit for the current cursor position."""
        time_val = self._current_time_val
        if time_val is None:
            return

        if (
            self._dvars_vline is not None
            and self._dvars_bg is not None
            and self._dvars_ax is not None
        ):
            try:
                self._dvars_canvas.restore_region(self._dvars_bg)
                self._dvars_vline.set_xdata([time_val, time_val])
                self._dvars_ax.draw_artist(self._dvars_vline)
                self._dvars_canvas.blit(self._dvars_fig.bbox)
            except Exception:  # noqa: BLE001
                self._dvars_bg = None  # Force a full redraw next time.

        if (
            self._carpet_vline is not None
            and self._carpet_bg is not None
            and self._carpet_ax is not None
        ):
            try:
                self._carpet_canvas.restore_region(self._carpet_bg)
                self._carpet_vline.set_xdata([time_val, time_val])
                self._carpet_ax.draw_artist(self._carpet_vline)
                self._carpet_canvas.blit(self._carpet_fig.bbox)
            except Exception:  # noqa: BLE001
                self._carpet_bg = None

    def replot(self) -> None:
        """Redraw all cached plots with the current napari theme colours."""
        if self._dvars_da is not None:
            self.update_dvars(self._dvars_da)
        if self._data_da is not None:
            self.update_carpet(self._data_da)
