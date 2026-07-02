"""Bottom-dock widget that plots the registration optimizer metric.

Mirrors the [`SignalPlotter`][confusius._napari._signals._plotter.SignalPlotter]
layout—a small matplotlib figure in the bottom dock—but stays deliberately simple: a
single line chart of the per-iteration metric value. The widget is created lazily by
`RegistrationPanel` when a registration starts, and torn down on completion so the dock
returns to its pre-run layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from qtpy.QtCore import QSize, QTimer
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from confusius._napari._theme import get_napari_colors, style_plot_toolbar

if TYPE_CHECKING:
    from napari import Viewer


class RegistrationMetricPlotter(QWidget):
    """Bottom-dock widget that plots the per-iteration optimizer metric.

    The widget is intentionally minimal: a single matplotlib axes, a navigation toolbar,
    and a thin status footer. Layout decisions (e.g. y-axis limits, line width) follow
    the same conventions as
    [`SignalPlotter`][confusius._napari._signals._plotter.SignalPlotter] for visual
    consistency between the two bottom-dock tabs.

    Parameters
    ----------
    viewer : napari.Viewer
        Active napari viewer, used to detect theme changes.
    """

    def __init__(self, viewer: "Viewer") -> None:
        super().__init__()
        self._viewer = viewer
        self._metric_values: list[float] = []
        self._metric_line = None
        # Throttle redraws to ~60 fps so rapid iteration events (and the
        # arrival of their queued Qt signals) don't flood the GUI thread.
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(16)
        self._redraw_timer.timeout.connect(self._render)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumHeight(300)
        self._setup_ui()
        self._apply_theme()
        self._viewer.events.theme.connect(lambda *_: self._apply_theme())

    def sizeHint(self) -> QSize:
        """Return the preferred initial size of the widget.

        Returns
        -------
        QSize
            Preferred initial size of 800 x 370 pixels.
        """
        return QSize(800, 370)

    def _setup_ui(self) -> None:
        """Build the matplotlib canvas and toolbar."""
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
        self._metric_line = self._axes.plot([], [], color="#e94b5f", linewidth=1.4)[0]
        self._axes.set_xlabel("Iteration")
        self._axes.set_ylabel("Metric value")
        self._axes.set_title("Registration metric")
        self._axes.grid(True, alpha=0.3)

    def _apply_theme(self) -> None:
        """Re-style the axes and toolbar to match the current napari theme."""
        colors = get_napari_colors(self._viewer.theme)
        self._figure.patch.set_facecolor(colors["bg"])
        self._axes.set_facecolor(colors["bg"])
        for spine in self._axes.spines.values():
            spine.set_edgecolor(colors["fg"])
        self._axes.tick_params(colors=colors["fg"])
        self._axes.xaxis.label.set_color(colors["fg"])
        self._axes.yaxis.label.set_color(colors["fg"])
        self._axes.title.set_color(colors["fg"])
        if self._metric_line is not None:
            self._metric_line.set_color(colors["accent"])
        style_plot_toolbar(self._toolbar, colors)
        self._canvas.draw_idle()

    def add_metric(self, value: float) -> None:
        """Append a metric value and schedule a redraw.

        Called from the GUI thread via the
        `NapariRegistrationProgressPlotterBridge.metric_updated` signal. Rapid iteration
        events are coalesced through a single-shot timer so the canvas is redrawn at
        most once per ~16 ms regardless of the worker-side event rate.

        Parameters
        ----------
        value : float
            Optimizer metric value at the current iteration.
        """
        self._metric_values.append(float(value))
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    def reset(self) -> None:
        """Clear the metric buffer and redraw an empty plot.

        Called before each new registration run so the curve starts from
        scratch instead of overlaying the previous run's data.
        """
        self._redraw_timer.stop()
        self._metric_values.clear()
        if self._metric_line is not None:
            self._metric_line.set_data([], [])
        self._axes.relim()
        self._axes.autoscale_view()
        self._canvas.draw_idle()

    def _render(self) -> None:
        """Redraw the metric line with the buffered values."""
        if self._metric_line is None:
            return
        n = len(self._metric_values)
        self._metric_line.set_data(np.arange(1, n + 1), self._metric_values)
        self._axes.relim()
        self._axes.autoscale_view()
        self._canvas.draw_idle()

    @property
    def metric_values(self) -> list[float]:
        """Copy of the metric value buffer.

        Returns
        -------
        list of float
            Optimizer metric value recorded at each iteration.
        """
        return list(self._metric_values)
