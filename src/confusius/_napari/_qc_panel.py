"""QC configuration panel for the ConfUSIus napari plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import napari
from napari.qt.threading import thread_worker
from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from confusius._napari._utils import show_napari_error

if TYPE_CHECKING:
    import xarray as xr

    from confusius._napari._qc_plots import QCPlotsWidget


@thread_worker
def _compute_qc_metrics(
    da: xr.DataArray,
    do_dvars: bool,
    do_tsnr: bool,
    do_cv: bool,
    do_carpet: bool,
) -> dict:
    """Compute QC metrics in a background thread.

    All expensive numpy/xarray operations run here so the napari event loop stays
    responsive and the progress bar can animate. Matplotlib drawing and napari layer
    additions happen in the main-thread callback.
    """
    results: dict = {}
    if do_dvars:
        from confusius.qc import compute_dvars

        results["dvars"] = compute_dvars(da)
    if do_tsnr:
        from confusius.qc import compute_tsnr

        results["tsnr"] = compute_tsnr(da)
    if do_cv:
        from confusius.qc import compute_cv

        results["cv"] = compute_cv(da)
    if do_carpet:
        from confusius.plotting.image import _prepare_carpet_data

        results["carpet"] = _prepare_carpet_data(da)
    return results


class QCPanel(QWidget):
    """Right-side panel for computing QC metrics and displaying plots.

    Time-series metrics (DVARS, Carpet plot) are rendered in a bottom dock widget that
    is created lazily. If the user closes the dock, clicking "Show plots" or "Compute"
    re-docks the widget (cached plots are preserved). Spatial map metrics (tSNR, CV) are
    added as new napari layers with correct scale and translate derived from the
    DataArray's coordinates.

    DVARS and spatial computations run in a background thread via
    `napari.qt.threading.thread_worker` so the UI remains responsive.

    The DataArray is read from `layer.metadata["xarray"]`, populated by both the Data
    panel and the npe2 file readers.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        # Cached reference to the inner plot widget (survives dock closure because
        # napari re-parents it to None rather than destroying it).
        self._qc_plots: QCPlotsWidget | None = None
        self._setup_ui()
        viewer.layers.events.inserted.connect(self._refresh_layers)
        viewer.layers.events.removed.connect(self._refresh_layers)
        viewer.events.theme.connect(self._on_theme_changed)
        viewer.dims.events.current_step.connect(self._on_time_step_changed)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Layer"))
        self._layer_combo = QComboBox()
        layout.addWidget(self._layer_combo)

        # --- Time-series metrics (bottom dock) ---------------------------
        ts_group = QGroupBox("Time-series metrics")
        ts_layout = QVBoxLayout(ts_group)
        ts_layout.setSpacing(4)
        self._dvars_check = QCheckBox("DVARS")
        self._dvars_check.setChecked(True)
        self._carpet_check = QCheckBox("Carpet plot")
        self._carpet_check.setChecked(True)
        ts_layout.addWidget(self._dvars_check)
        ts_layout.addWidget(self._carpet_check)
        layout.addWidget(ts_group)

        # --- Spatial map metrics (new viewer layers) ---------------------
        maps_group = QGroupBox("Spatial maps")
        maps_layout = QVBoxLayout(maps_group)
        maps_layout.setSpacing(4)
        self._tsnr_check = QCheckBox("tSNR")
        self._tsnr_check.setChecked(False)
        self._cv_check = QCheckBox("CV")
        self._cv_check.setChecked(False)
        maps_layout.addWidget(self._tsnr_check)
        maps_layout.addWidget(self._cv_check)
        layout.addWidget(maps_group)

        # --- Actions -----------------------------------------------------
        self._compute_btn = QPushButton("Compute")
        self._compute_btn.setObjectName("primary_btn")
        self._compute_btn.clicked.connect(self._compute)
        layout.addWidget(self._compute_btn)

        # "Show plots" is only useful once something has been computed.
        self._show_btn = QPushButton("Show QC plots")
        self._show_btn.setEnabled(False)
        self._show_btn.setToolTip("Show the QC plots dock.")
        self._show_btn.clicked.connect(self._show_plots)
        layout.addWidget(self._show_btn)

        # Thin indeterminate progress bar; animates while the thread runs.
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumHeight(4)
        self._progress.hide()
        layout.addWidget(self._progress)

        layout.addStretch()
        self._refresh_layers()

    # ------------------------------------------------------------------
    # Layer list helpers
    # ------------------------------------------------------------------

    def _refresh_layers(self) -> None:
        """Repopulate the layer combo box from the current viewer layers."""
        current = self._layer_combo.currentText()
        self._layer_combo.clear()
        for layer in self.viewer.layers:
            self._layer_combo.addItem(layer.name)
        idx = self._layer_combo.findText(current)
        if idx >= 0:
            self._layer_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Status / busy helpers
    # ------------------------------------------------------------------

    def _begin_work(self) -> None:
        self._compute_btn.setEnabled(False)
        self._compute_btn.setText("Computing…")
        self._progress.show()
        # Force a UI repaint so the button state and progress bar are visible before the
        # main thread may be briefly occupied by the carpet draw.
        QApplication.processEvents()

    def _end_work(self) -> None:
        self._compute_btn.setEnabled(True)
        self._compute_btn.setText("Compute")
        self._progress.hide()

    # ------------------------------------------------------------------
    # Bottom dock management
    # ------------------------------------------------------------------

    def _ensure_qc_plots(self) -> QCPlotsWidget:
        """Return the bottom-dock QCPlotsWidget.

        napari's `remove_dock_widget` re-parents the inner widget to None
        rather than destroying it, so the widget survives dock closure and
        can be re-docked by calling `add_dock_widget` again with the same
        object (cached plots are preserved).
        """
        from confusius._napari._qc_plots import QCPlotsWidget

        if self._qc_plots is not None:
            try:
                self._qc_plots.isVisible()  # Raises RuntimeError if deleted.
            except RuntimeError:
                self._qc_plots = None

        if self._qc_plots is None:
            self._qc_plots = QCPlotsWidget(self.viewer)

        if self._qc_plots.parent() is None:
            # Widget is not currently docked (brand-new or orphaned after dock closure).
            # Re-dock it; existing plot data is preserved.
            dock = self.viewer.window.add_dock_widget(
                self._qc_plots, name="QC Plots", area="bottom"
            )

            # Deferred so the layout is fully settled before we touch it.
            def _resize_dock() -> None:
                # Find the QMainWindow by walking up the dock's parent chain.
                main_win: QMainWindow | None = None
                p = dock.parent()
                while p is not None:
                    if isinstance(p, QMainWindow):
                        main_win = p
                        break
                    p = p.parent()
                if main_win is None:
                    return

                # The bottom dock, the central widget, and the side docks all share the
                # same vertical band. Qt's dock area layout uses the maximum of the
                # central widget's minimumSize and the tallest side-dock's minimumSize
                # as the floor for the middle band, which caps how tall the bottom dock
                # can grow.
                #
                # Fix: zero the minimum heights of the central widget, all its QWidget
                # descendants, AND all side dock widgets so the user can freely drag the
                # bottom dock splitter upward.
                from qtpy.QtWidgets import QDockWidget

                central = main_win.centralWidget()
                if central is None:
                    return
                central.setMinimumSize(QSize(0, 0))
                for w in central.findChildren(QWidget):
                    w.setMinimumSize(QSize(0, 0))
                for side_dock in main_win.findChildren(QDockWidget):
                    if side_dock is not dock:
                        side_dock.setMinimumHeight(0)
                        side_dock.widget().setMinimumSize(
                            QSize(0, 0)
                        ) if side_dock.widget() else None

                # Ensure the window is tall enough that the initial dock height leaves
                # room for the user to resize upward.
                current = main_win.size()
                if current.height() < 800:
                    main_win.resize(current.width(), 800)

                main_win.resizeDocks([dock], [300], Qt.Orientation.Vertical)

            QTimer.singleShot(200, _resize_dock)
        else:
            # Already in a live dock; make sure it's visible.
            parent = self._qc_plots.parent()
            if isinstance(parent, QWidget):
                parent.show()
                parent.raise_()

        return self._qc_plots

    def _show_plots(self) -> None:
        """Re-show the QC plots dock without recomputing."""
        self._ensure_qc_plots()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_theme_changed(self) -> None:
        """Replot cached data when the napari theme changes."""
        if self._qc_plots is None:
            return
        try:
            self._qc_plots.replot()
        except RuntimeError:
            self._qc_plots = None

    def _time_val_from_da(self, da: xr.DataArray) -> float | None:
        """Return the current time value for *da* based on the viewer step.

        Maps the viewer's current step index for the time dimension to the corresponding
        coordinate value (or bare index when no coordinate exists). Returns None when
        the time dimension cannot be resolved.
        """
        if "time" not in da.dims:
            return None
        current_step = self.viewer.dims.current_step
        time_dim_idx = list(da.dims).index("time")
        if time_dim_idx >= len(current_step):
            return None
        time_idx = current_step[time_dim_idx]
        time_coord = da.coords.get("time")
        if time_coord is not None and time_idx < len(time_coord):
            return float(time_coord[time_idx])
        return float(time_idx)

    def _on_time_step_changed(self, event) -> None:
        """Forward the current napari time step to the time cursor."""
        if self._qc_plots is None:
            return
        try:
            self._qc_plots.isVisible()
        except RuntimeError:
            self._qc_plots = None
            return

        layer_name = self._layer_combo.currentText()
        if not layer_name:
            return
        try:
            layer = self.viewer.layers[layer_name]
        except KeyError:
            return

        da = layer.metadata.get("xarray")
        if da is None:
            return
        time_val = self._time_val_from_da(da)
        if time_val is not None:
            self._qc_plots.set_time_cursor(time_val)

    # ------------------------------------------------------------------
    # Compute callbacks
    # ------------------------------------------------------------------

    def _on_compute_returned(
        self,
        results: dict,
        da: xr.DataArray,
        layer_name: str,
    ) -> None:
        """Main-thread callback: draw plots and add spatial layers."""
        try:
            if "dvars" in results or "carpet" in results:
                qc_widget = self._ensure_qc_plots()

                if "dvars" in results:
                    qc_widget.update_dvars(results["dvars"])

                if "carpet" in results:
                    qc_widget.update_carpet(results["carpet"])

                # Sync cursor to the current slider position so it does not start at t=0
                # when plots are first drawn.
                time_val = self._time_val_from_da(da)
                if time_val is not None:
                    qc_widget.set_time_cursor(time_val)

                self._show_btn.setEnabled(True)

            if "tsnr" in results or "cv" in results:
                from confusius.plotting.image import plot_napari

                # plot_napari sets axis_labels from the map's dims (z, y, x), which
                # would clobber the viewer's "time" label on dim 0. Save and restore to
                # keep the existing label intact.
                saved_labels = tuple(self.viewer.dims.axis_labels)

                if "tsnr" in results:
                    plot_napari(
                        results["tsnr"],
                        viewer=self.viewer,
                        name=f"{layer_name} — tSNR",
                    )

                if "cv" in results:
                    plot_napari(
                        results["cv"],
                        viewer=self.viewer,
                        name=f"{layer_name} — CV",
                    )

                self.viewer.dims.axis_labels = saved_labels
        except Exception as exc:  # noqa: BLE001
            show_napari_error(str(exc))
        finally:
            self._end_work()

    def _on_compute_error(self, exc: Exception) -> None:
        self._end_work()
        show_napari_error(str(exc))

    def _compute(self) -> None:
        layer_name = self._layer_combo.currentText()
        if not layer_name:
            show_napari_error("No layer selected.")
            return

        layer = self.viewer.layers[layer_name]
        da = layer.metadata.get("xarray")
        if da is None:
            show_napari_error(
                "Selected layer has no DataArray metadata. "
                "Load the file using the Data panel or the File menu."
            )
            return

        self._begin_work()

        do_dvars = self._dvars_check.isChecked()
        do_carpet = self._carpet_check.isChecked()
        do_tsnr = self._tsnr_check.isChecked()
        do_cv = self._cv_check.isChecked()

        worker = _compute_qc_metrics(da, do_dvars, do_tsnr, do_cv, do_carpet)
        worker.returned.connect(
            lambda results: self._on_compute_returned(results, da, layer_name)
        )
        worker.errored.connect(self._on_compute_error)
        worker.start()
