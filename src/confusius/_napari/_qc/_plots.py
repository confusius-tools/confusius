"""Bottom dock widget showing the QC carpet plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import napari
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from qtpy.QtCore import QSize, QTimer, Signal
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from confusius._napari._theme import get_napari_colors, style_plot_toolbar

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class QCPlotsWidget(QWidget):
    """Matplotlib canvas shown in the napari bottom dock area.

    The widget shows a carpet plot of voxel intensities over time, with a
    navigation toolbar and a blitted vertical cursor tracking the napari time slider.

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
        self._data_da: dict | None = None  # pre-computed carpet dict
        self._carpet_layer_name: str = ""
        # Blitting state.
        self._carpet_ax: Axes | None = None
        self._carpet_vline = None
        self._carpet_bg = None  # saved pixel buffer (no vline)
        # Last known time value so vlines are restored correctly after replot.
        self._current_time_val: float | None = None
        self._flushing_cursor: bool = False
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
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        _exp = QSizePolicy.Policy.Expanding
        self._carpet_fig = Figure(tight_layout=True)
        self._carpet_canvas = FigureCanvas(self._carpet_fig)
        self._carpet_canvas.setSizePolicy(_exp, _exp)
        # draw_event fires after every full redraw; use it to save the background
        # (without the animated vline) ready for blitting.
        self._carpet_canvas.mpl_connect("draw_event", self._on_carpet_draw)
        self._carpet_canvas.mpl_connect("button_press_event", self._on_carpet_click)

        self._carpet_toolbar = NavigationToolbar(self._carpet_canvas, self)
        layout.addWidget(self._carpet_toolbar)
        layout.addWidget(self._carpet_canvas)

    # ------------------------------------------------------------------
    # Theme helpers
    # ------------------------------------------------------------------

    def _style_toolbar(self, toolbar: NavigationToolbar, colors: dict) -> None:
        """Apply the napari background to the toolbar and recolor its icons."""
        style_plot_toolbar(toolbar, colors)

    # ------------------------------------------------------------------
    # Blitting helpers
    # ------------------------------------------------------------------

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

        colors = get_napari_colors(self._viewer.theme)

        self._carpet_fig.clear()
        self._carpet_fig.patch.set_facecolor(colors["bg"])

        ax = self._carpet_fig.add_subplot(111)
        _draw_carpet(carpet_data, ax=ax, bg_color=colors["bg"], fg_color=colors["fg"])

        if self._carpet_layer_name:
            ax.set_title(self._carpet_layer_name, color=colors["fg"], fontsize=10)

        time_coord = carpet_data["time_coord"]
        t0 = self._current_time_val
        if t0 is None:
            t0 = float(time_coord[0]) if time_coord is not None else 0.0
        self._carpet_vline = ax.axvline(
            t0,
            color=colors["accent"],
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
            World time value (or frame index) to position the cursor at.
        """
        self._current_time_val = time_val
        if not self._cursor_timer.isActive():
            self._cursor_timer.start()

    def _flush_cursor(self) -> None:
        """Perform the actual blit for the current cursor position."""
        if self._flushing_cursor:
            return
        self._flushing_cursor = True
        # Skip overlapping flushes: GUI events can trigger a new flush while the
        # previous one is still in progress (i.e. animation running and user clicks
        # in the plot to jump to that frame, or basically anything that will trigger a
        # redraw while animation is running)
        try:
            self._flush_cursor_impl()
        finally:
            self._flushing_cursor = False

    def _flush_cursor_impl(self) -> None:
        time_val = self._current_time_val
        if time_val is None:
            return

        if (
            self._carpet_vline is not None
            and self._carpet_bg is not None
            and self._carpet_ax is not None
        ):
            try:
                self._carpet_canvas.restore_region(self._carpet_bg)
                self._carpet_vline.set_xdata([time_val, time_val])
                self._carpet_ax.draw_artist(self._carpet_vline)
                # Use update() (async) instead of blit() (sync repaint) to
                # avoid blocking the Qt event loop during rapid animation.
                self._carpet_canvas.update()
            except Exception:  # noqa: BLE001
                self._carpet_bg = None  # Force a full redraw next time.

    def replot(self) -> None:
        """Redraw the cached carpet plot with the current napari theme colours."""
        if self._data_da is not None:
            self.update_carpet(self._data_da)
