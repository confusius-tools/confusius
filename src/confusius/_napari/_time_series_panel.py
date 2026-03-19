"""Time series control panel for the ConfUSIus sidebar."""

from __future__ import annotations

from typing import TYPE_CHECKING

import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from confusius._napari._time_series_plotter import TimeSeriesPlotter


class TimeSeriesPanel(QWidget):
    """Right-side panel for configuring time series plots.

    The actual plots are rendered in a bottom dock widget that is created
    lazily. If the user closes the dock, clicking "Show plot" re-docks
    the widget.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = viewer
        self._plotter: TimeSeriesPlotter | None = None
        self._setup_ui()
        viewer.events.theme.connect(self._on_theme_changed)

    def _setup_ui(self) -> None:
        """Set up the control panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Axis limits group
        limits_group = QGroupBox("Axis Limits")
        limits_layout = QVBoxLayout(limits_group)
        limits_layout.setSpacing(4)

        # Autoscale checkbox. QCheckBox does not support rich text, so we pair a
        # text-less checkbox with a clickable QLabel to get the italic "y".
        autoscale_row = QHBoxLayout()
        self._autoscale_check = QCheckBox()
        self._autoscale_check.setChecked(True)
        self._autoscale_check.toggled.connect(self._on_autoscale_changed)
        autoscale_label = QLabel("Autoscale <i>y</i>-axis")
        autoscale_label.setTextFormat(Qt.TextFormat.RichText)
        autoscale_label.mousePressEvent = lambda _e: self._autoscale_check.toggle()  # type: ignore[method-assign]
        autoscale_row.addWidget(self._autoscale_check)
        autoscale_row.addWidget(autoscale_label)
        autoscale_row.addStretch()
        limits_layout.addLayout(autoscale_row)

        # Y-axis limits
        y_layout = QHBoxLayout()
        ymin_label = QLabel("<i>y</i> min:")
        ymin_label.setTextFormat(Qt.TextFormat.RichText)
        y_layout.addWidget(ymin_label)
        self._ymin_spin = QDoubleSpinBox()
        self._ymin_spin.setRange(-1e9, 1e9)
        self._ymin_spin.setSpecialValueText("Auto")
        self._ymin_spin.setValue(0)  # 0 = Auto (special value)
        self._ymin_spin.valueChanged.connect(self._apply_settings)
        y_layout.addWidget(self._ymin_spin)
        ymax_label = QLabel("<i>y</i> max:")
        ymax_label.setTextFormat(Qt.TextFormat.RichText)
        y_layout.addWidget(ymax_label)
        self._ymax_spin = QDoubleSpinBox()
        self._ymax_spin.setRange(-1e9, 1e9)
        self._ymax_spin.setSpecialValueText("Auto")
        self._ymax_spin.setValue(0)  # 0 = Auto (special value)
        self._ymax_spin.valueChanged.connect(self._apply_settings)
        y_layout.addWidget(self._ymax_spin)
        limits_layout.addLayout(y_layout)

        # Apply initial autoscale state so spinboxes start disabled.
        self._on_autoscale_changed(True)

        layout.addWidget(limits_group)

        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(4)

        self._grid_check = QCheckBox("Show grid")
        self._grid_check.setChecked(True)
        self._grid_check.toggled.connect(self._apply_settings)
        display_layout.addWidget(self._grid_check)

        self._zscore_check = QCheckBox("Z-score time series")
        self._zscore_check.setChecked(False)
        self._zscore_check.toggled.connect(self._apply_settings)
        display_layout.addWidget(self._zscore_check)

        self._cursor_check = QCheckBox("Show time cursor")
        self._cursor_check.setChecked(False)
        self._cursor_check.setToolTip(
            "Draw a vertical cursor on the plot tracking the napari time slider.\n"
            "Uses blitting for low overhead, but keep disabled if you notice lag."
        )
        self._cursor_check.toggled.connect(self._on_cursor_toggled)
        display_layout.addWidget(self._cursor_check)

        layout.addWidget(display_group)

        # Show plot button — disabled while the dock is visible.
        self._show_btn = QPushButton("Show Time Series Plot")
        self._show_btn.setObjectName("primary_btn")
        self._show_btn.clicked.connect(self._show_plot)
        layout.addWidget(self._show_btn)

        layout.addStretch()

    def _on_autoscale_changed(self, checked: bool) -> None:
        """Enable/disable manual Y-axis controls based on autoscale setting."""
        self._ymin_spin.setEnabled(not checked)
        self._ymax_spin.setEnabled(not checked)
        self._apply_settings()

    def _ensure_plotter(self) -> TimeSeriesPlotter:
        """Return the bottom-dock TimeSeriesPlotter.

        Creates and docks the widget on first call. If the dock was closed (the
        plotter's parent becomes None after napari removes it), re-docks it.
        When the plotter is already in a live dock this is a no-op.
        """
        from confusius._napari._time_series_plotter import TimeSeriesPlotter

        if self._plotter is None:
            self._plotter = TimeSeriesPlotter(self._viewer)

        if self._plotter.parent() is None:
            # Widget is not docked — create (or re-create) the dock.
            dock = self._viewer.window.add_dock_widget(
                self._plotter, name="Time Series Plot", area="bottom"
            )
            dock.setFloating(False)

            # Disable the button while the dock is visible; re-enable on hide/close.
            self._show_btn.setEnabled(False)
            dock.visibilityChanged.connect(
                lambda visible: self._show_btn.setEnabled(not visible)
            )

        # Always sync panel settings to the plotter (covers the case where settings
        # like the time cursor were configured before the plotter was first created).
        self._apply_settings()
        if self._cursor_check.isChecked():
            self._plotter.set_time_cursor(self._current_frame())

        return self._plotter

    def _show_plot(self) -> None:
        """Show or re-dock the time series plot widget."""
        # If the plotter is already in a live dock, just raise it.
        if self._plotter is not None and self._plotter.parent() is not None:
            parent = self._plotter.parent()
            if isinstance(parent, QWidget):
                parent.show()
                parent.raise_()
            return
        self._ensure_plotter()

    def _apply_settings(self) -> None:
        """Apply current settings to the plotter."""
        plotter = self._plotter
        if plotter is None:
            return

        autoscale = self._autoscale_check.isChecked()
        plotter.set_autoscale(autoscale)

        if not autoscale:
            ymin = self._ymin_spin.value()
            ymax = self._ymax_spin.value()
            ymin_val = ymin if ymin != 0 else None
            ymax_val = ymax if ymax != 0 else None
            plotter.set_ylim(ymin_val, ymax_val)

        plotter.set_show_grid(self._grid_check.isChecked())
        plotter.set_zscore(self._zscore_check.isChecked())
        plotter.set_show_cursor(self._cursor_check.isChecked())

    def _time_dim_index(self) -> int:
        """Return the viewer dimension index for time.

        Reads xarray metadata from the first layer that has a ``"time"`` dim;
        falls back to 0.
        """
        for layer in self._viewer.layers:
            da = layer.metadata.get("xarray")
            if da is not None and "time" in da.dims:
                return list(da.dims).index("time")
        return 0

    def _current_frame(self) -> float:
        """Return the current time frame index from the viewer's step."""
        current_step = self._viewer.dims.current_step
        t_idx = self._time_dim_index()
        return float(current_step[t_idx]) if t_idx < len(current_step) else 0.0

    def _on_cursor_toggled(self, checked: bool) -> None:
        """Connect or disconnect the time-step event and update the plotter."""
        if checked:
            self._viewer.dims.events.current_step.connect(self._on_time_step_changed)
        else:
            self._viewer.dims.events.current_step.disconnect(self._on_time_step_changed)
        if self._plotter is not None:
            self._plotter.set_show_cursor(checked)
            if checked:
                self._plotter.set_time_cursor(self._current_frame())

    def _on_time_step_changed(self, event) -> None:
        """Forward the current napari time step to the time cursor."""
        if self._plotter is None:
            return
        current_step = event.value
        t_idx = self._time_dim_index()
        if t_idx < len(current_step):
            self._plotter.set_time_cursor(float(current_step[t_idx]))

    def _on_theme_changed(self) -> None:
        """Handle napari theme change."""
        if self._plotter is not None:
            self._plotter.on_theme_changed()
