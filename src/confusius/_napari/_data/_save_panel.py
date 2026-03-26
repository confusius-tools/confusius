"""Data saving panel for the ConfUSIus napari plugin."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from confusius._napari._io._writers import _compute_dataarray_from_layer

if TYPE_CHECKING:
    import napari
    from napari.layers import Layer

_SUPPORTED_EXTENSIONS = (".nii.gz", ".nii", ".zarr")


@thread_worker
def _save_data(da: xr.DataArray, path: Path) -> Path:
    """Save *da* to *path* in a background thread."""
    from confusius.io import save

    save(da, path)
    return path


class SavePanel(QWidget):
    """Panel for saving napari layers to ConfUSIus-supported formats.

    Supports two save modes:

    - **Direct**: the layer carries a ConfUSIus DataArray in its metadata
      (loaded via the reader), so it is saved verbatim.
    - **Reconstruct**: coordinates are inferred from the layer's `scale`,
      `translate`, `axis_labels`, and `units` (typical for user-drawn
      labels layers).
    - **Template**: coordinates are borrowed from a second layer that *does*
      carry a DataArray (e.g. the fUSI image on which the labels were drawn).
      This preserves the full physical coordinate system and all DataArray
      attributes. When the layer has fewer dimensions than the template (e.g. a
      3D labels layer with a 4D image template), the trailing spatial dimensions
      of the template are used.

    The output format is inferred from the file extension (`.nii` /
    `.nii.gz` for NIfTI, `.zarr` for Zarr).

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self._setup_ui()
        self._connect_viewer_events()
        self._refresh_layer_combos()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # --- Layer selection ------------------------------------------
        layer_group = QGroupBox("Layer")
        layer_form = QFormLayout(layer_group)
        layer_form.setSpacing(6)

        self._layer_combo = QComboBox()
        self._layer_combo.setToolTip("Layer to save.")

        self._template_combo = QComboBox()
        self._template_combo.setPlaceholderText("None")
        self._template_combo.setToolTip(
            "Optional: borrow coordinates and metadata from this layer.\n"
            "Useful when saving labels drawn on top of a ConfUSIus image layer\n"
            "to preserve physical coordinates. The layer must have been loaded\n"
            "via the ConfUSIus reader."
        )

        layer_form.addRow("Save layer:", self._layer_combo)
        layer_form.addRow("Coordinates from:", self._template_combo)
        layout.addWidget(layer_group)

        # --- Output ------------------------------------------------------
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)
        output_layout.setSpacing(6)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Output path (.nii.gz or .zarr) …")
        self._path_edit.returnPressed.connect(self._save)

        self._browse_btn = QPushButton("Browse")
        self._browse_btn.clicked.connect(self._browse)

        path_row = QHBoxLayout()
        path_row.addWidget(self._path_edit)
        path_row.addWidget(self._browse_btn)

        output_layout.addLayout(path_row)
        layout.addWidget(output_group)

        # --- Progress + save button -----------------------------------
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumHeight(4)
        self._progress.hide()
        layout.addWidget(self._progress)

        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("primary_btn")
        self._save_btn.clicked.connect(self._save)
        layout.addWidget(self._save_btn)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Viewer event wiring
    # ------------------------------------------------------------------

    def _connect_viewer_events(self) -> None:
        self.viewer.layers.events.inserted.connect(self._refresh_layer_combos)
        self.viewer.layers.events.removed.connect(self._refresh_layer_combos)
        self.viewer.layers.events.changed.connect(self._refresh_layer_combos)

    def _refresh_layer_combos(self, event: object = None) -> None:
        """Repopulate both dropdowns from the current viewer layer list."""
        current_layer = self._layer_combo.currentText()
        current_template = self._template_combo.currentText()

        self._layer_combo.blockSignals(True)
        self._template_combo.blockSignals(True)
        self._layer_combo.clear()
        self._template_combo.clear()

        # List layers in reverse so the most-recently added appears first.
        for layer in reversed(self.viewer.layers):
            self._layer_combo.addItem(layer.name)
            if layer.metadata.get("xarray") is not None:
                self._template_combo.addItem(layer.name)

        text_index = self._layer_combo.findText(current_layer)
        if text_index >= 0:
            self._layer_combo.setCurrentIndex(text_index)

        text_index = self._template_combo.findText(current_template)
        # -1 means "no selection" (placeholder shown), which is the correct default.
        self._template_combo.setCurrentIndex(text_index)

        self._layer_combo.blockSignals(False)
        self._template_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Browse
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        start = self._path_edit.text().strip() or str(Path.home())
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Layer",
            start,
            "NIfTI (*.nii.gz *.nii);;Zarr store (*.zarr);;All files (*)",
        )
        if path_str:
            self._path_edit.setText(path_str)

    # ------------------------------------------------------------------
    # DataArray construction
    # ------------------------------------------------------------------

    def _find_layer(self, name: str) -> Layer | None:
        matches = [layer for layer in self.viewer.layers if layer.name == name]
        return matches[0] if matches else None

    def _build_da(self) -> xr.DataArray | None:
        """Assemble the DataArray to be saved from the selected layers."""
        layer_name = self._layer_combo.currentText()
        if not layer_name:
            show_error("No layer selected.")
            return None

        layer = self._find_layer(layer_name)
        if layer is None:
            show_error(f"Layer {layer_name!r} not found.")
            return None

        template_name = self._template_combo.currentText()
        use_template = bool(template_name)

        if use_template:
            template_layer = self._find_layer(template_name)
            if template_layer is None:
                show_error(f"Template layer {template_name!r} not found.")
                return None
            template_da: xr.DataArray = template_layer.metadata["xarray"]
            layer_shape = layer.data.shape
            ndim = len(layer_shape)

            if layer_shape == template_da.shape:
                # Exact match: use all template dimensions.
                return xr.DataArray(
                    layer.data,
                    dims=template_da.dims,
                    coords=template_da.coords,
                    attrs=template_da.attrs,
                )
            elif layer_shape == template_da.shape[-ndim:]:
                # Layer has fewer dimensions (e.g. 3D labels vs 4D template):
                # use the trailing spatial dimensions of the template.
                spatial_dims = template_da.dims[-ndim:]
                spatial_coords = {
                    d: template_da.coords[d]
                    for d in spatial_dims
                    if d in template_da.coords
                }
                return xr.DataArray(
                    layer.data,
                    dims=spatial_dims,
                    coords=spatial_coords,
                    attrs=template_da.attrs,
                )
            else:
                show_error(
                    f"Shape mismatch: '{layer_name}' has shape {layer_shape} "
                    f"but template '{template_name}' has shape {template_da.shape}."
                )
                return None

        # No template: use the DataArray from layer metadata when available,
        # otherwise reconstruct from scale/translate/axis_labels/units.
        da: xr.DataArray | None = layer.metadata.get("xarray")
        if da is not None:
            return da

        axis_labels = (
            list(layer.axis_labels) if getattr(layer, "axis_labels", None) else None
        )
        units = list(layer.units) if getattr(layer, "units", None) else None
        meta = {
            "metadata": {},
            "axis_labels": axis_labels,
            "scale": list(layer.scale),
            "translate": list(layer.translate),
            "units": units,
        }
        return _compute_dataarray_from_layer(layer.data, meta)

    # ------------------------------------------------------------------
    # Work management
    # ------------------------------------------------------------------

    def _begin_work(self) -> None:
        self._save_btn.setEnabled(False)
        self._browse_btn.setEnabled(False)
        self._path_edit.setEnabled(False)
        self._progress.show()
        QApplication.processEvents()

    def _end_work(self) -> None:
        self._save_btn.setEnabled(True)
        self._browse_btn.setEnabled(True)
        self._path_edit.setEnabled(True)
        self._progress.hide()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _save(self) -> None:
        path_str = self._path_edit.text().strip()
        if not path_str:
            show_error("Please specify an output path.")
            return

        if not any(path_str.endswith(ext) for ext in _SUPPORTED_EXTENSIONS):
            show_error(
                f"Unsupported file extension. Use one of: "
                f"{', '.join(_SUPPORTED_EXTENSIONS)}."
            )
            return

        da = self._build_da()
        if da is None:
            return

        self._begin_work()
        worker = _save_data(da, Path(path_str))
        worker.returned.connect(self._on_save_returned)
        worker.errored.connect(self._on_save_error)
        worker.start()

    def _on_save_returned(self, path: Path) -> None:
        self._end_work()
        show_info(f"Saved to {path}")

    def _on_save_error(self, exc: Exception) -> None:
        self._end_work()
        show_error(str(exc))
