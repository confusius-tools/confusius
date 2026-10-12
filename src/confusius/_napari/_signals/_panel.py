"""Signals control panel for the ConfUSIus sidebar."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import napari
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QDoubleRangeSlider

from confusius._dims import POSE_DIM, TIME_DIM, VOXEL_DIMS, WORLD_DIMS
from confusius._napari._qt import find_main_window
from confusius._napari._signals._manager import SignalsManagerDialog
from confusius._napari._signals._plotter import SignalPlotter
from confusius._napari._signals._store import SignalStore

if TYPE_CHECKING:
    from confusius._napari._events._store import EventStore


class SignalPanel(QWidget):
    """Right-side panel for configuring signal plots.

    The actual plots are rendered in a bottom dock widget that is created lazily. If the
    user closes the dock, clicking "Show plot" re-docks the widget.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.
    event_store : EventStore, optional
        Shared store of temporal events whose intervals are shaded on the plot.
    signal_store : SignalStore, optional
        Shared store of stored/live signals. If not provided, a private store is
        created. Pass the same instance used elsewhere (e.g. the QC panel) so
        stored signals are available across panels.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        event_store: EventStore | None = None,
        signal_store: SignalStore | None = None,
    ) -> None:
        super().__init__()
        self._viewer = viewer
        self._event_store = event_store
        self._plotter: SignalPlotter | None = None
        self._signals_manager: SignalsManagerDialog | None = None
        self._signals_store = (
            signal_store if signal_store is not None else SignalStore(self)
        )
        self._live_sources = self._signals_store.live_sources(viewer)
        self._cached_xaxis_dim_index: int | None = None
        self._setup_ui()
        self._live_sources.sources_changed.connect(self._sync_source_controls)
        if event_store is not None:
            event_store.changed.connect(self._refresh_baseline_events)
        self._refresh_baseline_events()
        self._sync_references()
        self._sync_source_controls()
        self._apply_settings()
        viewer.events.theme.connect(self._on_theme_changed)

    def _setup_ui(self) -> None:
        """Set up the control panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Align source selectors independently of checkbox label widths.
        source_group = QGroupBox("Source")
        self._source_group = source_group
        source_layout = QGridLayout(source_group)
        source_layout.setHorizontalSpacing(6)
        source_layout.setVerticalSpacing(4)
        source_layout.setColumnStretch(1, 1)

        # Mouse row.
        self._mouse_check = QCheckBox("Mouse (Shift + hover)")
        self._mouse_check.setChecked(True)
        source_layout.addWidget(self._mouse_check, 0, 0)
        source_layout.addWidget(QLabel("Radius"), 0, 1)
        self._radius_spin = QDoubleSpinBox()
        self._radius_spin.setRange(0.0, 1000.0)
        self._radius_spin.setDecimals(2)
        self._radius_spin.setSingleStep(0.1)
        self._radius_spin.setSuffix(" mm")
        self._radius_spin.setAccessibleName("Shared mouse radius")
        self._radius_spin.setToolTip(
            "Shared mouse sphere radius. Zero selects one voxel; FC uses the same trace."
        )
        self._radius_spin.valueChanged.connect(self._live_sources.set_radius)
        source_layout.addWidget(self._radius_spin, 0, 2, 1, 2)

        # All ROI layers are detected by the shared controller.
        self._points_check = QCheckBox("All Points layers")
        source_layout.addWidget(self._points_check, 1, 0, 1, 3)
        self._new_points_btn = QPushButton("+")
        self._new_points_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._new_points_btn.setToolTip(
            "Create a new 3D Points layer (no time axis).\n"
            "Points will be visible at all time steps."
        )
        self._new_points_btn.clicked.connect(self._create_points_layer)
        source_layout.addWidget(self._new_points_btn, 1, 3)

        # Labels row.
        self._labels_check = QCheckBox("All Labels layers")
        source_layout.addWidget(self._labels_check, 2, 0, 1, 3)
        self._new_labels_btn = QPushButton("+")
        self._new_labels_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._new_labels_btn.setToolTip(
            "Create a new 3D Labels layer (no time axis).\n"
            "Labels will be visible at all time steps."
        )
        self._new_labels_btn.clicked.connect(self._create_labels_layer)
        source_layout.addWidget(self._new_labels_btn, 2, 3)

        # The same reference filter applies to mouse, point and label traces.
        self._ref_label = QLabel("Reference")
        self._ref_label.setEnabled(True)
        source_layout.addWidget(self._ref_label, 3, 0)
        self._ref_combo = QComboBox()
        self._ref_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._ref_combo.setEnabled(True)
        source_layout.addWidget(self._ref_combo, 3, 1, 1, 3)

        layout.addWidget(source_group)

        self._mouse_check.toggled.connect(
            lambda checked: self._on_source_enabled_changed(0, checked)
        )
        self._points_check.toggled.connect(
            lambda checked: self._on_source_enabled_changed(1, checked)
        )
        self._labels_check.toggled.connect(
            lambda checked: self._on_source_enabled_changed(2, checked)
        )
        self._ref_combo.currentTextChanged.connect(self._on_source_selection_changed)

        self._viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self._viewer.layers.events.removed.connect(self._refresh_source_combos)
        self._viewer.layers.events.renamed.connect(self._refresh_source_combos)
        self._viewer.layers.selection.events.active.connect(
            self._on_active_layer_changed
        )
        self._refresh_source_combos()

        # Axis parameters group.
        axis_group = QGroupBox("Axis Parameters")
        self._axis_group = axis_group
        axis_layout = QVBoxLayout(axis_group)
        axis_layout.setSpacing(4)

        # X-axis dimension selection.
        xaxis_row = QHBoxLayout()
        xaxis_label = QLabel("<i>x</i>-axis")
        xaxis_label.setTextFormat(Qt.TextFormat.RichText)
        xaxis_row.addWidget(xaxis_label)
        self._xaxis_combo = QComboBox()
        self._xaxis_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._xaxis_combo.setToolTip(
            "Select which dimension to use for the plot's x-axis. "
            "Defaults to 'time' when available, otherwise the first axis."
        )
        self._xaxis_combo.currentTextChanged.connect(self._on_xaxis_changed)
        xaxis_row.addWidget(self._xaxis_combo, stretch=1)
        axis_layout.addLayout(xaxis_row)

        # Autoscale checkbox. QCheckBox does not support rich text, so we pair a
        # text-less checkbox with a clickable QLabel to get the italic "y".
        autoscale_row = QHBoxLayout()
        self._autoscale_check = QCheckBox()
        self._autoscale_check.setChecked(True)
        self._autoscale_check.toggled.connect(self._on_autoscale_changed)
        autoscale_label = QLabel("Autoscale <i>y</i>-axis")
        autoscale_label.setTextFormat(Qt.TextFormat.RichText)
        cast("Any", autoscale_label).mousePressEvent = lambda _e: (
            self._autoscale_check.toggle()
        )
        autoscale_row.addWidget(self._autoscale_check)
        autoscale_row.addWidget(autoscale_label)
        autoscale_row.addStretch()
        axis_layout.addLayout(autoscale_row)

        yminmax_row = QHBoxLayout()
        spinbox: list[QDoubleSpinBox] = []
        for lim in ("min", "max"):
            ylim_label = QLabel(f"<i>y</i> {lim}")
            ylim_label.setTextFormat(Qt.TextFormat.RichText)
            yminmax_row.addWidget(ylim_label)
            spin = QDoubleSpinBox()
            spin.setObjectName(f"y{lim}_spin")
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Ignored horizontal policy: the minimum size hint spans the
            # widest possible value ("-1000000000.00"), which overflowed the
            # panel in issue #183. The layout stretch shares the row width.
            spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            spin.setRange(-1e9, 1e9)
            spin.setValue(-1.0 if lim == "min" else 1.0)
            spin.valueChanged.connect(self._apply_settings)
            spinbox.append(spin)
            yminmax_row.addWidget(spin, stretch=1)
        axis_layout.addLayout(yminmax_row)

        self._ymin_spin, self._ymax_spin = spinbox

        # Apply initial autoscale state so spinboxes start disabled.
        self._on_autoscale_changed(True)

        layout.addWidget(axis_group)

        # Initialize the x-axis combo after the widget is created.
        self._refresh_xaxis_combo()

        # Display options.
        display_group = QGroupBox("Display Options")
        self._display_group = display_group
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(4)

        display_layout.addWidget(QLabel("Standardization"))
        self._standardization_combo = QComboBox()
        self._standardization_combo.addItem("None", "raw")
        self._standardization_combo.addItem("Z-score", "zscore")
        self._standardization_combo.addItem("Percent signal change", "psc")
        self._standardization_combo.currentIndexChanged.connect(self._apply_settings)
        display_layout.addWidget(self._standardization_combo)
        self._baseline_group = QGroupBox("PSC baseline")
        baseline_layout = QVBoxLayout(self._baseline_group)
        self._baseline_source_combo = QComboBox()
        self._baseline_source_combo.addItem("Manual intervals", "manual")
        self._baseline_source_combo.addItem("Events", "events")
        self._baseline_source_combo.currentIndexChanged.connect(self._apply_settings)
        baseline_layout.addWidget(self._baseline_source_combo)
        self._baseline_events_widget = QWidget()
        events_layout = QVBoxLayout(self._baseline_events_widget)
        events_layout.setContentsMargins(0, 0, 0, 0)
        self._baseline_events_list = QListWidget()
        self._baseline_events_list.setMaximumHeight(120)
        self._baseline_events_list.setToolTip(
            "Select event occurrences whose union defines the baseline."
        )
        self._baseline_events_list.itemChanged.connect(self._apply_settings)
        events_layout.addWidget(self._baseline_events_list)
        self._baseline_outside_check = QCheckBox("Outside selected events")
        self._baseline_outside_check.toggled.connect(self._apply_settings)
        events_layout.addWidget(self._baseline_outside_check)
        self._baseline_events_status = QLabel()
        self._baseline_events_status.setWordWrap(True)
        events_layout.addWidget(self._baseline_events_status)
        baseline_layout.addWidget(self._baseline_events_widget)
        self._manual_baseline_widget = QWidget()
        baseline_layout.addWidget(self._manual_baseline_widget)
        self._baseline_layout = QVBoxLayout(self._manual_baseline_widget)
        self._baseline_layout.setContentsMargins(0, 0, 0, 0)
        self._baseline_layout.addWidget(QLabel("Intervals (x-axis units)"))
        self._baseline_bounds: tuple[float, float] | None = None
        self._baseline_last_bounds: tuple[float, float] | None = None
        self._baseline_rows: list[tuple[QWidget, QLabel, QDoubleRangeSlider]] = []
        self._add_baseline_btn = QPushButton("Add interval")
        self._add_baseline_btn.clicked.connect(self._add_baseline_interval)
        self._baseline_layout.addWidget(self._add_baseline_btn)
        self._add_baseline_interval()
        display_layout.addWidget(self._baseline_group)
        self._viewer.dims.events.current_step.connect(self._on_xaxis_step_changed)

        layout.addWidget(display_group)

        # Show plot button, disabled while the dock is visible.
        self._show_btn = QPushButton("Show Signal Plot")
        self._show_btn.setObjectName("primary_btn")
        self._show_btn.clicked.connect(self.show_plot)
        layout.addWidget(self._show_btn)

        self._manage_btn = QPushButton("Manage Signals")
        self._manage_btn.clicked.connect(self._show_signals_manager)
        layout.addWidget(self._manage_btn)

        layout.addStretch()

    def _on_autoscale_changed(self, checked: bool) -> None:
        """Enable/disable manual Y-axis controls based on autoscale setting."""
        if not checked and self._plotter is not None:
            # When disabling autoscale, capture current y-limits from the plot
            current_ylim = self._plotter.get_ylim()
            if current_ylim is not None:
                self._ymin_spin.setValue(current_ylim[0])
                self._ymax_spin.setValue(current_ylim[1])
        self._ymin_spin.setEnabled(not checked)
        self._ymax_spin.setEnabled(not checked)
        self._apply_settings()

    def _ensure_plotter_instance(self) -> SignalPlotter:
        """Return the plot renderer, constructing it on first use.

        Returns
        -------
        SignalPlotter
            Renderer consuming the shared controller.
        """
        if self._plotter is None:
            self._plotter = SignalPlotter(
                self._viewer,
                store=self._signals_store,
                event_store=self._event_store,
            )
            self._plotter.frame_clicked.connect(self._on_frame_clicked)
            self._plotter.axis_range_changed.connect(self._update_baseline_bounds)
            self._update_baseline_bounds(self._plotter.get_axis_range())
        return self._plotter

    def _ensure_plotter(self) -> SignalPlotter:
        """Return the bottom-dock SignalPlotter.

        Creates and docks the widget on first call. If the dock was closed (the
        plotter's parent becomes None after napari removes it), re-docks it. When the
        plotter is already in a live dock this is a no-op.
        """
        plotter = self._ensure_plotter_instance()

        if plotter.parent() is None:
            # Widget is not docked, create (or re-create) the dock.
            dock = self._viewer.window.add_dock_widget(
                plotter, name="Signal Plot", area="bottom"
            )

            # Disable the button while the dock is visible; re-enable on hide/close.
            self._show_btn.setEnabled(False)
            dock.visibilityChanged.connect(
                lambda visible: self._show_btn.setEnabled(not visible)
            )

            # Defer a resizeDocks call so Qt can settle the layout at the correct DPI
            # before the canvas first paints. This mirrors the pattern used in the QC
            # panel and prevents the HiDPI click-offset bug.
            def _settle_layout() -> None:
                main_win = find_main_window(dock)
                if main_win is None:
                    return
                # Zero minimum sizes on the central widget and all its children so the
                # bottom dock splitter can be dragged freely.  This also forces Qt to
                # recompute the window's device-pixel geometry, which fixes the HiDPI
                # click-offset bug triggered by adding a dock widget (mirrors the fix in
                # _ensure_qc_plots).
                from qtpy.QtCore import QSize

                central = main_win.centralWidget()
                if central is None:
                    return
                central.setMinimumSize(QSize(0, 0))
                for w in central.findChildren(QWidget):
                    w.setMinimumSize(QSize(0, 0))
                for side_dock in main_win.findChildren(QDockWidget):
                    if side_dock is not dock:
                        side_dock.setMinimumHeight(0)
                        widget = side_dock.widget()
                        if widget is not None:
                            widget.setMinimumSize(QSize(0, 0))

                # Ensure the window is tall enough that the initial dock height leaves
                # room for the user to resize upward.
                current = main_win.size()
                if current.height() < 800:
                    main_win.resize(current.width(), 800)

                main_win.resizeDocks([dock], [300], Qt.Orientation.Vertical)

            QTimer.singleShot(200, _settle_layout)

        # Always sync panel settings to the plotter (covers the case where settings like
        # the x-axis cursor were configured before the plotter was first created).
        self._refresh_xaxis_combo()
        self._apply_settings()
        self._sync_references()
        plotter.set_xaxis_cursor(self._current_xaxis_world())

        return plotter

    def show_plot(self) -> None:
        """Show or re-dock the signal plot widget."""
        # If the plotter is already in a live dock, just raise it.
        if self._plotter is not None and self._plotter.parent() is not None:
            parent = self._plotter.parent()
            if isinstance(parent, QWidget):
                parent.show()
                parent.raise_()
            return
        self._ensure_plotter()

    def _show_signals_manager(self) -> None:
        """Show the floating signals manager window."""
        if self._signals_manager is None:
            self._signals_manager = SignalsManagerDialog(
                self._signals_store, self, self._live_sources
            )
            self._signals_manager.apply_theme(self._viewer.theme)

        self._signals_manager.show()
        self._signals_manager.raise_()
        self._signals_manager.activateWindow()

    def _apply_settings(self) -> None:
        """Apply current settings to the plotter."""
        if hasattr(self, "_manual_baseline_widget"):
            self._baseline_group.setVisible(
                self._standardization_combo.currentData() == "psc"
            )
            events = self._baseline_source_combo.currentData() == "events"
            self._manual_baseline_widget.setVisible(not events)
            self._baseline_events_widget.setVisible(events)
            self._baseline_source_combo.setEnabled(self._event_store is not None)
            self._baseline_events_status.setText(
                "Select at least one positive-duration event."
                if not self._selected_baseline_events()
                else "Events use seconds and half-open intervals [start, end)."
            )
        plotter = self._plotter
        if plotter is None:
            return

        autoscale = self._autoscale_check.isChecked()
        plotter.set_autoscale(autoscale)

        if not autoscale:
            plotter.set_ylim(self._ymin_spin.value(), self._ymax_spin.value())

        intervals = (
            [tuple(slider.value()) for _, _, slider in self._baseline_rows]
            if self._baseline_bounds is not None
            else []
        )
        source = self._baseline_source_combo.currentData()
        if source == "events":
            intervals = [
                (onset, onset + duration)
                for onset, duration, _ in self._selected_baseline_events()
            ]
        plotter.set_standardization(
            self._standardization_combo.currentData(),
            intervals,
            baseline_source=source,
            outside=self._baseline_outside_check.isChecked(),
        )

        # Set the x-axis dimension.
        xaxis_dim = self._xaxis_combo.currentText()
        if xaxis_dim:
            plotter.set_xaxis_dim(xaxis_dim)

    def _selected_baseline_events(self) -> list[tuple[float, float, str]]:
        """Return checked event occurrences with positive duration.

        Returns
        -------
        list[tuple[float, float, str]]
            Event onset, duration and type, with times in seconds.
        """
        return [
            tuple(item.data(Qt.ItemDataRole.UserRole))
            for index in range(self._baseline_events_list.count())
            if (item := self._baseline_events_list.item(index)) is not None
            and item.checkState() == Qt.CheckState.Checked
            and item.data(Qt.ItemDataRole.UserRole)[1] > 0
        ]

    def _refresh_baseline_events(self) -> None:
        """Synchronize selectable event occurrences without changing annotations."""
        selected = set(self._selected_baseline_events())
        self._baseline_events_list.blockSignals(True)
        self._baseline_events_list.clear()
        if self._event_store is not None:
            for onset, duration, trial_type in self._event_store.iter_events():
                event = (float(onset), float(duration), str(trial_type))
                item = QListWidgetItem(
                    f"{trial_type}: {onset:g}–{onset + duration:g} s"
                )
                item.setData(Qt.ItemDataRole.UserRole, event)
                if duration > 0:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if event in selected
                        else Qt.CheckState.Unchecked
                    )
                else:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                    item.setToolTip(
                        "A zero-duration event cannot define a baseline period."
                    )
                self._baseline_events_list.addItem(item)
        self._baseline_events_list.blockSignals(False)
        self._apply_settings()

    def _add_baseline_interval(self) -> None:
        """Add a two-handle interval selector within the plotted axis range."""
        row = QWidget()
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        label = QLabel()
        header.addWidget(label, stretch=1)
        remove = QPushButton("−")
        remove.setToolTip("Remove this baseline interval")
        remove.clicked.connect(lambda: self._remove_baseline_interval(row))
        header.addWidget(remove)
        layout.addLayout(header)
        slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        slider.setMinimumWidth(0)
        slider.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        slider.setRange(*(self._baseline_bounds or (0.0, 1.0)))
        slider.setValue(self._baseline_bounds or (0.0, 1.0))
        slider.setEnabled(self._baseline_bounds is not None)
        slider.setToolTip("Inclusive baseline interval; shaded on the signal plot.")
        slider.valuesChanged.connect(self._on_baseline_changed)
        layout.addWidget(slider)
        self._baseline_rows.append((row, label, slider))
        self._baseline_layout.insertWidget(len(self._baseline_rows), row)
        self._on_baseline_changed()

    def _remove_baseline_interval(self, row: QWidget) -> None:
        """Remove an interval while retaining at least one baseline selector.

        Parameters
        ----------
        row : qtpy.QtWidgets.QWidget
            Interval row to remove.
        """
        if len(self._baseline_rows) == 1:
            return
        self._baseline_rows = [
            entry for entry in self._baseline_rows if entry[0] is not row
        ]
        self._baseline_layout.removeWidget(row)
        row.deleteLater()
        self._on_baseline_changed()

    def _update_baseline_bounds(self, bounds: tuple[float, float] | None) -> None:
        """Keep baseline sliders within finite numeric plot coordinates.

        Parameters
        ----------
        bounds : tuple[float, float], optional
            Axis extent. If not provided, disable interval selection.
        """
        previous = self._baseline_last_bounds
        self._baseline_bounds = bounds
        if bounds is not None:
            self._baseline_last_bounds = bounds
        for _, _, slider in self._baseline_rows:
            slider.blockSignals(True)
            if bounds is not None:
                start, end = slider.value()
                full = previous is None
                slider.setRange(*bounds)
                slider.setValue(
                    bounds
                    if full
                    else (
                        max(bounds[0], min(start, bounds[1])),
                        max(bounds[0], min(end, bounds[1])),
                    )
                )
            slider.setEnabled(bounds is not None)
            slider.blockSignals(False)
        self._add_baseline_btn.setEnabled(bounds is not None)
        self._on_baseline_changed()

    def _on_baseline_changed(self) -> None:
        """Update endpoint labels and the display-only baseline selection."""
        for row, label, slider in self._baseline_rows:
            start, end = slider.value()
            label.setText(
                f"{start:g} – {end:g}"
                if self._baseline_bounds is not None
                else "No numeric trace axis"
            )
            row.findChild(QPushButton).setEnabled(len(self._baseline_rows) > 1)
        self._apply_settings()

    def _xaxis_dim_index(self) -> int:
        """Return the viewer dimension index for the x-axis dimension.

        Uses a cached value when available. The cache is invalidated when the
        x-axis combo selection changes or layers are added/removed.
        """
        if self._cached_xaxis_dim_index is not None:
            return self._cached_xaxis_dim_index

        xaxis_dim = self._xaxis_combo.currentText()

        # First try the configured x-axis dimension.
        if xaxis_dim:
            for layer in self._viewer.layers:
                da = layer.metadata.get("xarray")
                if da is not None and xaxis_dim in da.dims:
                    self._cached_xaxis_dim_index = list(da.dims).index(xaxis_dim)
                    return self._cached_xaxis_dim_index

        # Fall back to looking for 'time'.
        for layer in self._viewer.layers:
            da = layer.metadata.get("xarray")
            if da is not None and TIME_DIM in da.dims:
                self._cached_xaxis_dim_index = list(da.dims).index(TIME_DIM)
                return self._cached_xaxis_dim_index

        self._cached_xaxis_dim_index = 0
        return 0

    def _current_xaxis_world(self) -> float:
        """Return the current world coordinate along the x-axis dimension."""
        xaxis_index = self._xaxis_dim_index()
        dims_point = self._viewer.dims.point
        return float(dims_point[xaxis_index]) if xaxis_index < len(dims_point) else 0.0

    def _on_xaxis_step_changed(self, event) -> None:
        """Forward the current napari x-axis world coordinate to the cursor."""
        if self._plotter is None:
            return
        xaxis_index = self._xaxis_dim_index()
        dims_point = self._viewer.dims.point
        if xaxis_index < len(dims_point):
            self._plotter.set_xaxis_cursor(float(dims_point[xaxis_index]))

    def _on_frame_clicked(self, frame: float) -> None:
        """Navigate the viewer to the clicked x-axis coordinate.

        `frame` is the x-axis plot value (a world coordinate, e.g. time
        in seconds).  Using `dims.set_point` avoids the double-conversion
        bug that occurs when setting `current_step` directly — the step
        index depends on `dims.range.step`, which changes when a video
        layer with a different time scale is loaded.
        """
        xaxis_index = self._xaxis_dim_index()
        if xaxis_index < len(self._viewer.dims.point):
            self._viewer.dims.set_point(xaxis_index, frame)

    def _on_theme_changed(self) -> None:
        """Handle napari theme change."""
        if self._plotter is not None:
            self._plotter.on_theme_changed()
        if self._signals_manager is not None:
            self._signals_manager.apply_theme(self._viewer.theme)

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def _refresh_source_combos(self, _event=None) -> None:
        """Repopulate all source combo boxes from the current viewer layers."""
        self._cached_xaxis_dim_index = None
        # Preserve current selections.
        cur_ref = self._ref_combo.currentData()

        self._ref_combo.blockSignals(True)
        try:
            self._ref_combo.clear()

            self._ref_combo.addItem("All image layers")

            references = self._live_sources.references()
            for layer in self._viewer.layers:
                if layer in references:
                    self._ref_combo.addItem(layer.name, layer)

            for combo, prev in ((self._ref_combo, cur_ref),):
                text_index = combo.findData(prev)
                if text_index >= 0:
                    combo.setCurrentIndex(text_index)
        finally:
            self._ref_combo.blockSignals(False)

        # Defer the plotter sync to the next event-loop iteration. When called from an
        # `inserted` event, this ensures napari has finished setting up the new layer
        # (including computing contrast limits via its async slice worker) before we
        # trigger a canvas redraw in the plotter.
        QTimer.singleShot(0, self._sync_references)

    def _on_source_enabled_changed(self, btn_id: int, checked: bool) -> None:
        """Enable independent source kinds without deleting registrations.

        Parameters
        ----------
        btn_id : int
            Mouse, Points or Labels control index.
        checked : bool
            Whether to enable this source kind.
        """
        kind = ("mouse", "point", "label")[btn_id]
        for source in self._live_sources.sources():
            if source.kind == kind:
                self._live_sources.set_enabled(source.id, checked)
        self._radius_spin.setEnabled(self._mouse_check.isChecked())
        self._sync_references()

    def _on_source_selection_changed(self) -> None:
        """Update the reference filter without adding or deleting ROI sources."""
        self._sync_references()

    def _sync_source_controls(self) -> None:
        """Reflect source enablement changed through the manager."""
        sources = self._live_sources.sources()
        for kind, check in (
            ("mouse", self._mouse_check),
            ("point", self._points_check),
            ("label", self._labels_check),
        ):
            enabled = any(s.kind == kind and s.enabled for s in sources)
            check.blockSignals(True)
            check.setChecked(enabled)
            check.setEnabled(any(s.kind == kind for s in sources))
            check.blockSignals(False)
        self._radius_spin.setEnabled(self._mouse_check.isChecked())

    def _sync_references(self) -> None:
        """Update shared reference configuration independently of plot visibility."""
        ref_layer = self._ref_combo.currentData()
        self._live_sources.set_references(
            [ref_layer] if ref_layer is not None else None
        )

    def _on_active_layer_changed(self, event) -> None:
        """Update image axes without disrupting ROI editing."""
        # A transient empty selection can suspend plotting before ROI focus arrives.
        self._refresh_xaxis_combo()

    def _on_layer_inserted(self, event) -> None:
        """Handle layer insertion to refresh source combos and x-axis options.

        Also refreshes the x-axis combo when an image layer with xarray metadata
        is inserted, since the metadata may be attached after the layer is added
        to the viewer (e.g., via plot_napari).
        """
        self._cached_xaxis_dim_index = None
        self._refresh_source_combos()

        # Defer the x-axis combo refresh to the next event-loop iteration so that
        # any metadata attached after the layer is inserted (e.g., by plot_napari)
        # is available when we check for it.
        QTimer.singleShot(0, self._deferred_xaxis_refresh)

    def _deferred_xaxis_refresh(self) -> None:
        """Refresh x-axis combo after deferring to allow metadata attachment."""
        # Check the active layer for xarray metadata.
        layer = self._viewer.layers.selection.active
        if (
            layer is not None
            and layer._type_string == "image"
            and "xarray" in layer.metadata
        ):
            self._refresh_xaxis_combo()

    def _get_available_xaxis_dims(self) -> list[str]:
        """Return list of available x-axis dimensions from the active layer.

        Returns the dimension names from the active image layer's xarray
        metadata if available, restricted to the **non-displayed** (slider)
        axes with more than one element. Displayed axes are shown on screen and
        have no slider, so they are not valid x-axis choices; singleton axes
        cannot form a meaningful x-axis. This mirrors the criterion used by
        [_refresh_source_combos][confusius._napari._signals._panel.SignalPanel._refresh_source_combos]
        to identify signal-bearing layers.

        When the active layer lacks xarray metadata (e.g., a video layer),
        falls back to the first image layer in the viewer that does have
        xarray metadata.  Returns an empty list when no suitable layer is
        found.
        """
        layer = self._viewer.layers.selection.active
        if layer is None:
            return []
        if layer._type_string != "image":
            references = self._live_sources.references()
            if not references:
                return []
            reference = self._ref_combo.currentData()
            layer = reference if reference in references else references[0]
        if layer.ndim <= self._viewer.dims.ndisplay:
            return []

        da = layer.metadata.get("xarray")

        # If the active layer has no xarray metadata (e.g., video layer),
        # search for the first image layer that does.
        if da is None:
            for other in self._viewer.layers:
                if other._type_string == "image":
                    da = other.metadata.get("xarray")
                    if da is not None:
                        layer = other
                        break

        if da is not None:
            # Offer the non-displayed (slider) axes with more than one element.
            # Viewer indices include leading dimensions missing from this layer.
            offset = self._viewer.dims.ndim - layer.ndim
            displayed_dims = {index - offset for index in self._viewer.dims.displayed}
            return [
                str(dim)
                for i, dim in enumerate(da.dims)
                if i not in displayed_dims and da.shape[i] > 1
            ]

        return []

    def _refresh_xaxis_combo(self) -> None:
        """Repopulate the x-axis dimension dropdown.

        Preserves the current selection if it's still available.
        Defaults to 'time' if present, otherwise the first available dimension.
        """
        dims = self._get_available_xaxis_dims()
        if not dims:
            self._xaxis_combo.clear()
            self._xaxis_combo.setEnabled(False)
            if self._plotter is not None:
                self._plotter.set_xaxis_dim("")
            return

        current = self._xaxis_combo.currentText()
        self._xaxis_combo.blockSignals(True)
        try:
            self._xaxis_combo.clear()
            for dim in dims:
                self._xaxis_combo.addItem(dim)

            # Try to restore previous selection, or default to 'time', or first item.
            if current and current in dims:
                self._xaxis_combo.setCurrentText(current)
            elif TIME_DIM in dims:
                self._xaxis_combo.setCurrentText(TIME_DIM)
            else:
                self._xaxis_combo.setCurrentIndex(0)
        finally:
            self._xaxis_combo.blockSignals(False)

        self._xaxis_combo.setEnabled(True)
        # Notify plotter of the (potentially new) selection.
        self._on_xaxis_changed()

    def _on_xaxis_changed(self) -> None:
        """Handle x-axis dimension selection change."""
        self._cached_xaxis_dim_index = None
        dim = self._xaxis_combo.currentText()
        if dim:
            self._live_sources.set_display_axis(dim)
        self._apply_settings()
        self._refresh_source_combos()

    def _spatial_info(self) -> tuple[tuple[int, ...], dict[str, Any]] | None:
        """Return the spatial shape and layer geometry of the first signal layer.

        Spatial axes are those named `z`, `y`, or `x` in the xarray metadata.
        This always covers the full spatial volume regardless of which dims are
        currently displayed or used as sliders.

        Returns
        -------
        shape : tuple[int, ...]
            Spatial shape of the reference layer.
        geometry : dict[str, Any]
            `scale`, `translate`, `axis_labels`, and `units` restricted to the
            spatial axes, ready to pass to a napari `add_*` method. All values come
            from the *same* layer so they are guaranteed to be consistent, and
            copying `units`/`axis_labels` keeps napari's world units consistent
            across layers (otherwise it warns and stops rendering units).

        Returns `None` when no suitable layer is found.
        """
        for layer in self._viewer.layers:
            if layer._type_string != "image":
                continue
            da = layer.metadata.get("xarray")
            if da is not None:
                spatial_indices = [
                    i
                    for i, dim in enumerate(da.dims)
                    if dim in (POSE_DIM, *VOXEL_DIMS, *WORLD_DIMS)
                ]
                if not spatial_indices:
                    continue
            elif layer.data.ndim >= 4:
                # Fallback: treat dim 0 as the signal axis for 4D+ layers without
                # xarray.
                spatial_indices = list(range(1, layer.data.ndim))
            else:
                continue
            shape = tuple(layer.data.shape[i] for i in spatial_indices)
            geometry = {
                "scale": tuple(float(layer.scale[i]) for i in spatial_indices),
                "translate": tuple(float(layer.translate[i]) for i in spatial_indices),
                "axis_labels": tuple(layer.axis_labels[i] for i in spatial_indices),
                "units": tuple(layer.units[i] for i in spatial_indices),
            }
            return shape, geometry
        return None

    def _create_points_layer(self) -> None:
        """Add a new 3D Points layer (no time axis) to the viewer.

        The layer is initialised with no points and a rescaling projection mode so
        that points stay visible on neighbouring slices. Geometry is copied from the
        reference image so the layer is aligned and brush/point sizes match the data.
        """
        import numpy as np

        info = self._spatial_info()
        ndim = 3
        kwargs: dict[str, Any] = {}
        if info is not None:
            shape, kwargs = info
            ndim = len(shape)
            # napari's default point size is 10 world-units, which is enormous for
            # mm-scale data.
            kwargs["size"] = 2.0
        self._viewer.add_points(  # type: ignore
            np.empty((0, ndim)),
            name="Points",
            ndim=ndim,
            projection_mode="rescale_linear",
            **kwargs,
        )

    def _create_labels_layer(self) -> None:
        """Add a new 3D Labels layer (no time axis) to the viewer.

        Shape and geometry are derived from the first image layer with xarray
        metadata so the labels are pixel-aligned with the image and the paint brush
        maps to the correct voxel positions.
        """
        import numpy as np

        info = self._spatial_info()
        shape, kwargs = info if info is not None else ((64, 64, 64), {})
        self._viewer.add_labels(  # type: ignore
            np.zeros(shape, dtype=np.int32),
            name="Labels",
            **kwargs,
        )
