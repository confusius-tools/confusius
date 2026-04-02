"""Signals control panel for the ConfUSIus sidebar."""

from __future__ import annotations

import napari
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from confusius._dims import SPATIAL_DIMS_WITH_POSE, TIME_DIM
from confusius._napari._signals._manager import SignalsManagerDialog
from confusius._napari._signals._plotter import SignalPlotter
from confusius._napari._signals._store import SignalStore


class SignalPanel(QWidget):
    """Right-side panel for configuring signal plots.

    The actual plots are rendered in a bottom dock widget that is created lazily. If the
    user closes the dock, clicking "Show plot" re-docks the widget.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = viewer
        self._plotter: SignalPlotter | None = None
        self._signals_manager: SignalsManagerDialog | None = None
        self._signals_store = SignalStore(self)
        self._setup_ui()
        viewer.events.theme.connect(self._on_theme_changed)

    def _setup_ui(self) -> None:
        """Set up the control panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Source group.
        source_group = QGroupBox("Source")
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(4)

        self._source_btn_group = QButtonGroup(self)

        # Mouse row.
        self._radio_mouse = QRadioButton("Mouse (Shift + hover)")
        self._radio_mouse.setChecked(True)
        self._source_btn_group.addButton(self._radio_mouse, 0)
        source_layout.addWidget(self._radio_mouse)

        # Points row. Text is part of the radio button (same pattern as the Mouse row)
        # so the indicator and label are always flush with no gap.
        points_row = QHBoxLayout()
        self._radio_points = QRadioButton("Points:")
        self._source_btn_group.addButton(self._radio_points, 1)
        points_row.addWidget(self._radio_points)
        self._points_combo = QComboBox()
        self._points_combo.setEnabled(False)
        points_row.addWidget(self._points_combo, stretch=1)
        self._new_points_btn = QPushButton("+")
        self._new_points_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._new_points_btn.setToolTip(
            "Create a new 3D Points layer (no time axis).\n"
            "Points will be visible at all time steps."
        )
        self._new_points_btn.clicked.connect(self._create_points_layer)
        points_row.addWidget(self._new_points_btn)
        source_layout.addLayout(points_row)

        # Labels row.
        labels_row = QHBoxLayout()
        self._radio_labels = QRadioButton("Labels:")
        self._source_btn_group.addButton(self._radio_labels, 2)
        labels_row.addWidget(self._radio_labels)
        self._labels_combo = QComboBox()
        self._labels_combo.setEnabled(False)
        labels_row.addWidget(self._labels_combo, stretch=1)
        self._new_labels_btn = QPushButton("+")
        self._new_labels_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._new_labels_btn.setToolTip(
            "Create a new 3D Labels layer (no time axis).\n"
            "Labels will be visible at all time steps."
        )
        self._new_labels_btn.clicked.connect(self._create_labels_layer)
        labels_row.addWidget(self._new_labels_btn)
        source_layout.addLayout(labels_row)

        # Reference image (enabled in points/labels mode).
        ref_row = QHBoxLayout()
        self._ref_label = QLabel("Reference:")
        self._ref_label.setEnabled(False)
        ref_row.addWidget(self._ref_label)
        self._ref_combo = QComboBox()
        self._ref_combo.setEnabled(False)
        ref_row.addWidget(self._ref_combo, stretch=1)
        source_layout.addLayout(ref_row)

        layout.addWidget(source_group)

        self._source_btn_group.idToggled.connect(self._on_source_mode_changed)
        self._points_combo.currentTextChanged.connect(self._on_source_selection_changed)
        self._labels_combo.currentTextChanged.connect(self._on_source_selection_changed)
        self._ref_combo.currentTextChanged.connect(self._on_source_selection_changed)

        self._viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self._viewer.layers.events.removed.connect(self._refresh_source_combos)
        self._viewer.layers.selection.events.active.connect(
            self._on_active_layer_changed
        )
        self._refresh_source_combos()

        # Axis parameters group.
        axis_group = QGroupBox("Axis Parameters")
        axis_layout = QVBoxLayout(axis_group)
        axis_layout.setSpacing(4)

        # X-axis dimension selection.
        xaxis_row = QHBoxLayout()
        xaxis_label = QLabel("<i>x</i>-axis:")
        xaxis_label.setTextFormat(Qt.TextFormat.RichText)
        xaxis_row.addWidget(xaxis_label)
        self._xaxis_combo = QComboBox()
        self._xaxis_combo.setToolTip(
            "Select which dimension to use for the plot's x-axis. "
            "Defaults to 'time' when available, otherwise the first axis."
        )
        self._xaxis_combo.currentTextChanged.connect(self._on_xaxis_changed)
        xaxis_row.addWidget(self._xaxis_combo, stretch=1)
        axis_layout.addLayout(xaxis_row)

        # Y-axis limits.
        y_layout = QHBoxLayout()
        ymin_label = QLabel("<i>y</i> min:")
        ymin_label.setTextFormat(Qt.TextFormat.RichText)
        y_layout.addWidget(ymin_label)
        self._ymin_spin = QDoubleSpinBox()
        self._ymin_spin.setRange(-1e9, 1e9)
        self._ymin_spin.setValue(-1.0)
        self._ymin_spin.valueChanged.connect(self._apply_settings)
        y_layout.addWidget(self._ymin_spin)
        ymax_label = QLabel("<i>y</i> max:")
        ymax_label.setTextFormat(Qt.TextFormat.RichText)
        y_layout.addWidget(ymax_label)
        self._ymax_spin = QDoubleSpinBox()
        self._ymax_spin.setRange(-1e9, 1e9)
        self._ymax_spin.setValue(1.0)
        self._ymax_spin.valueChanged.connect(self._apply_settings)
        y_layout.addWidget(self._ymax_spin)
        axis_layout.addLayout(y_layout)

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
        axis_layout.addLayout(autoscale_row)

        # Apply initial autoscale state so spinboxes start disabled.
        self._on_autoscale_changed(True)

        layout.addWidget(axis_group)

        # Initialize the x-axis combo after the widget is created.
        self._refresh_xaxis_combo()

        # Display options.
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(4)

        self._grid_check = QCheckBox("Show grid")
        self._grid_check.setChecked(True)
        self._grid_check.toggled.connect(self._apply_settings)
        display_layout.addWidget(self._grid_check)

        self._cursor_check = QCheckBox("Show x-axis cursor")
        self._cursor_check.setChecked(False)
        self._cursor_check.setToolTip(
            "Draw a vertical cursor on the plot tracking the napari x-axis slider."
        )
        self._cursor_check.toggled.connect(self._on_cursor_toggled)
        display_layout.addWidget(self._cursor_check)

        self._zscore_check = QCheckBox("Z-score signal")
        self._zscore_check.setChecked(False)
        self._zscore_check.toggled.connect(self._apply_settings)
        display_layout.addWidget(self._zscore_check)

        layout.addWidget(display_group)

        # Show plot button, disabled while the dock is visible.
        self._show_btn = QPushButton("Show Signal Plot")
        self._show_btn.setObjectName("primary_btn")
        self._show_btn.clicked.connect(self._show_plot)
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

    def _ensure_plotter(self) -> SignalPlotter:
        """Return the bottom-dock SignalPlotter.

        Creates and docks the widget on first call. If the dock was closed (the
        plotter's parent becomes None after napari removes it), re-docks it. When the
        plotter is already in a live dock this is a no-op.
        """
        if self._plotter is None:
            self._plotter = SignalPlotter(self._viewer, store=self._signals_store)

        if self._plotter.parent() is None:
            # Widget is not docked, create (or re-create) the dock.
            dock = self._viewer.window.add_dock_widget(
                self._plotter, name="Signal Plot", area="bottom"
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
                main_win = self._find_main_window(dock)
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
        self._apply_settings()
        self._sync_source_to_plotter()
        if self._cursor_check.isChecked():
            self._plotter.set_xaxis_cursor(self._current_xaxis_world())

        return self._plotter

    def _show_plot(self) -> None:
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
            self._signals_manager = SignalsManagerDialog(self._signals_store, self)
            self._signals_manager.apply_theme(self._viewer.theme)

        self._signals_manager.show()
        self._signals_manager.raise_()
        self._signals_manager.activateWindow()

    def _apply_settings(self) -> None:
        """Apply current settings to the plotter."""
        plotter = self._plotter
        if plotter is None:
            return

        autoscale = self._autoscale_check.isChecked()
        plotter.set_autoscale(autoscale)

        if not autoscale:
            plotter.set_ylim(self._ymin_spin.value(), self._ymax_spin.value())

        plotter.set_show_grid(self._grid_check.isChecked())
        plotter.set_zscore(self._zscore_check.isChecked())
        plotter.set_show_cursor(self._cursor_check.isChecked())

        # Set the x-axis dimension.
        xaxis_dim = self._xaxis_combo.currentText()
        if xaxis_dim:
            plotter.set_xaxis_dim(xaxis_dim)

    def _xaxis_dim_index(self) -> int:
        """Return the viewer dimension index for the x-axis dimension.

        Uses the configured x-axis dimension from the dropdown when available,
        otherwise falls back to looking for a 'time' dimension, then to 0.
        """
        xaxis_dim = self._xaxis_combo.currentText()

        # First try the configured x-axis dimension.
        if xaxis_dim:
            for layer in self._viewer.layers:
                da = layer.metadata.get("xarray")
                if da is not None and xaxis_dim in da.dims:
                    return list(da.dims).index(xaxis_dim)

        # Fall back to looking for 'time'.
        for layer in self._viewer.layers:
            da = layer.metadata.get("xarray")
            if da is not None and TIME_DIM in da.dims:
                return list(da.dims).index(TIME_DIM)

        return 0

    def _current_xaxis_world(self) -> float:
        """Return the current world coordinate along the x-axis dimension."""
        xaxis_index = self._xaxis_dim_index()
        dims_point = self._viewer.dims.point
        return float(dims_point[xaxis_index]) if xaxis_index < len(dims_point) else 0.0

    def _on_cursor_toggled(self, checked: bool) -> None:
        """Connect or disconnect the x-axis step event and update the plotter."""
        if checked:
            self._viewer.dims.events.current_step.connect(self._on_xaxis_step_changed)
        else:
            try:
                self._viewer.dims.events.current_step.disconnect(
                    self._on_xaxis_step_changed
                )
            except RuntimeError:
                pass
        if self._plotter is not None:
            self._plotter.set_show_cursor(checked)
            if checked:
                self._plotter.set_xaxis_cursor(self._current_xaxis_world())

    def _on_xaxis_step_changed(self, event) -> None:
        """Forward the current napari x-axis world coordinate to the cursor."""
        if self._plotter is None:
            return
        xaxis_index = self._xaxis_dim_index()
        dims_point = self._viewer.dims.point
        if xaxis_index < len(dims_point):
            self._plotter.set_xaxis_cursor(float(dims_point[xaxis_index]))

    def _on_theme_changed(self) -> None:
        """Handle napari theme change."""
        if self._plotter is not None:
            self._plotter.on_theme_changed()
        if self._signals_manager is not None:
            self._signals_manager.apply_theme(self._viewer.theme)

    def _find_main_window(self, widget: QWidget) -> QMainWindow | None:
        """Traverse up the widget hierarchy to find the QMainWindow.

        Parameters
        ----------
        widget : QWidget
            Starting widget to search from.

        Returns
        -------
        QMainWindow | None
            The main window if found, None otherwise.
        """
        parent = widget.parent()
        while parent is not None:
            if isinstance(parent, QMainWindow):
                return parent
            parent = parent.parent()
        return None

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def _refresh_source_combos(self, _event=None) -> None:
        """Repopulate all source combo boxes from the current viewer layers."""
        # Preserve current selections.
        cur_points = self._points_combo.currentText()
        cur_labels = self._labels_combo.currentText()
        cur_ref = self._ref_combo.currentText()

        self._points_combo.blockSignals(True)
        self._labels_combo.blockSignals(True)
        self._ref_combo.blockSignals(True)
        try:
            self._points_combo.clear()
            self._labels_combo.clear()
            self._ref_combo.clear()

            self._ref_combo.addItem("All image layers")

            displayed_dims = set(self._viewer.dims.displayed)
            for layer in self._viewer.layers:
                t = layer._type_string
                if t == "points":
                    self._points_combo.addItem(layer.name)
                elif t == "labels":
                    self._labels_combo.addItem(layer.name)
                elif t == "image" and not getattr(layer, "rgb", False):
                    # Include layers with at least one non-displayed signal
                    # dimension so spatial maps (tSNR, CV, ...) are excluded.
                    da = layer.metadata.get("xarray")
                    if da is not None:
                        has_signal_dim = any(
                            i not in displayed_dims and da.shape[i] > 1
                            for i in range(da.ndim)
                        )
                    else:
                        has_signal_dim = layer.data.ndim >= 4
                    if has_signal_dim:
                        self._ref_combo.addItem(layer.name)

            for combo, prev in (
                (self._points_combo, cur_points),
                (self._labels_combo, cur_labels),
                (self._ref_combo, cur_ref),
            ):
                text_index = combo.findText(prev)
                if text_index >= 0:
                    combo.setCurrentIndex(text_index)
        finally:
            self._points_combo.blockSignals(False)
            self._labels_combo.blockSignals(False)
            self._ref_combo.blockSignals(False)

        # Defer the plotter sync to the next event-loop iteration. When called from an
        # `inserted` event, this ensures napari has finished setting up the new layer
        # (including computing contrast limits via its async slice worker) before we
        # trigger a canvas redraw in the plotter.
        QTimer.singleShot(0, self._sync_source_to_plotter)

    def _on_source_mode_changed(self, btn_id: int, checked: bool) -> None:
        """Handle radio button toggling between source modes."""
        if not checked:
            return
        is_mouse = btn_id == 0
        self._points_combo.setEnabled(btn_id == 1)
        self._labels_combo.setEnabled(btn_id == 2)
        self._ref_combo.setEnabled(not is_mouse)
        self._ref_label.setEnabled(not is_mouse)
        self._sync_source_to_plotter()

    def _on_source_selection_changed(self) -> None:
        """Handle combo box selection changes."""
        self._sync_source_to_plotter()

    def _sync_source_to_plotter(self) -> None:
        """Push the current source mode and layer selections to the plotter."""
        plotter = self._plotter
        if plotter is None:
            return

        btn_id = self._source_btn_group.checkedId()
        mode = {0: "mouse", 1: "points", 2: "labels"}.get(btn_id, "mouse")

        # Reference layers.
        ref_text = self._ref_combo.currentText()
        if ref_text == "All image layers" or not ref_text:
            plotter.set_ref_layers(None)
        else:
            try:
                plotter.set_ref_layers([self._viewer.layers[ref_text]])
            except KeyError:
                plotter.set_ref_layers(None)

        # Points layer.
        points_text = self._points_combo.currentText()
        if points_text:
            try:
                plotter.set_points_layer(self._viewer.layers[points_text])
            except KeyError:
                plotter.set_points_layer(None)
        else:
            plotter.set_points_layer(None)

        # Labels layer.
        labels_text = self._labels_combo.currentText()
        if labels_text:
            try:
                plotter.set_labels_layer(self._viewer.layers[labels_text])
            except KeyError:
                plotter.set_labels_layer(None)
        else:
            plotter.set_labels_layer(None)

        # Mode last, triggers a replot with the already-updated layers.
        plotter.set_source_mode(mode)

    def _on_active_layer_changed(self, event) -> None:
        """Handle active layer change to update x-axis dropdown options."""
        self._refresh_xaxis_combo()

    def _on_layer_inserted(self, event) -> None:
        """Handle layer insertion to refresh source combos and x-axis options.

        Also refreshes the x-axis combo when an image layer with xarray metadata
        is inserted, since the metadata may be attached after the layer is added
        to the viewer (e.g., via plot_napari).
        """
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
        metadata if available, excluding:
        - Spatial dimensions (including `pose`) which are never valid signal axes
        - Dimensions with only 1 element (can't create meaningful x-axis)

        Falls back to dimension indices based on the layer's data shape if no
        xarray metadata is present.
        """
        layer = self._viewer.layers.selection.active
        if layer is None or layer._type_string != "image":
            return []

        da = layer.metadata.get("xarray")
        if da is not None:
            # Filter out spatial dimensions and single-element dimensions.
            return [
                dim
                for i, dim in enumerate(da.dims)
                if dim not in SPATIAL_DIMS_WITH_POSE and da.shape[i] > 1
            ]

        # Fallback: generate generic dimension names based on data shape.
        # Without xarray metadata we cannot identify spatial dims by name,
        # so fall back to excluding displayed dims.
        displayed_dims = set(self._viewer.dims.displayed)
        ndim = layer.data.ndim
        return [
            f"dim_{i}"
            for i in range(ndim)
            if i not in displayed_dims and layer.data.shape[i] > 1
        ]

    def _refresh_xaxis_combo(self) -> None:
        """Repopulate the x-axis dimension dropdown.

        Preserves the current selection if it's still available.
        Defaults to 'time' if present, otherwise the first available dimension.
        """
        dims = self._get_available_xaxis_dims()
        if not dims:
            self._xaxis_combo.clear()
            self._xaxis_combo.setEnabled(False)
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
        self._apply_settings()

    def _spatial_info(
        self,
    ) -> tuple[
        tuple[int, ...] | None, tuple[float, ...] | None, tuple[float, ...] | None
    ]:
        """Return (shape, scale, translate) for the spatial axes of the first signal layer.

        Spatial axes are those named `z`, `y`, or `x` in the xarray metadata.
        This always covers the full spatial volume regardless of which dims are
        currently displayed or used as sliders.

        All three values come from the *same* layer so they are guaranteed to
        be consistent.  Returns `(None, None, None)` when no suitable layer
        is found.
        """
        for layer in self._viewer.layers:
            if layer._type_string != "image":
                continue
            da = layer.metadata.get("xarray")
            if da is not None:
                spatial_indices = [
                    i for i, dim in enumerate(da.dims) if dim in SPATIAL_DIMS_WITH_POSE
                ]
                if not spatial_indices:
                    continue
                shape = tuple(da.shape[i] for i in spatial_indices)
                scale = tuple(float(layer.scale[i]) for i in spatial_indices)
                translate = tuple(float(layer.translate[i]) for i in spatial_indices)
                return shape, scale, translate
            # Fallback: treat dim 0 as the signal axis for 4D+ layers without xarray.
            if layer.data.ndim >= 4:
                return (
                    layer.data.shape[1:],
                    tuple(float(s) for s in layer.scale[1:]),
                    tuple(float(t) for t in layer.translate[1:]),
                )
        return None, None, None

    def _create_points_layer(self) -> None:
        """Add a new 3D Points layer (no time axis) to the viewer.

        The layer is initialised with no points and `out_of_slice_display` enabled so
        that added points are always visible regardless of the current time step. Scale
        and translate are copied from the reference image so the layer is aligned and
        brush/point sizes match the data.
        """
        import numpy as np

        shape, scale, translate = self._spatial_info()
        ndim = len(shape) if shape is not None else 3
        kwargs: dict = {}
        if scale is not None:
            kwargs["scale"] = scale
            # napari's default point size is 10 world-units, which is enormous for
            # mm-scale data.
            kwargs["size"] = 2.0
        if translate is not None:
            kwargs["translate"] = translate
        layer = self._viewer.add_points(
            np.empty((0, ndim)),
            name="Points (3D)",
            ndim=ndim,
            **kwargs,
        )
        layer.out_of_slice_display = True

    def _create_labels_layer(self) -> None:
        """Add a new 3D Labels layer (no time axis) to the viewer.

        Shape, scale, and translate are derived from the first image layer with
        xarray metadata so the labels are pixel-aligned with the image and the
        paint brush maps to the correct voxel positions.
        """
        import numpy as np

        shape, scale, translate = self._spatial_info()
        shape = shape or (64, 64, 64)
        kwargs: dict = {}
        if scale is not None:
            kwargs["scale"] = scale
        if translate is not None:
            kwargs["translate"] = translate
        self._viewer.add_labels(
            np.zeros(shape, dtype=np.int32),
            name="Labels (3D)",
            **kwargs,
        )
