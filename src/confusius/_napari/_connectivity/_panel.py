"""Functional-connectivity panel for the ConfUSIus sidebar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QDoubleRangeSlider

from confusius._dims import TIME_DIM, VOXEL_DIMS
from confusius._napari._signals._store import SignalStore, StoredSignal
from confusius.plotting.napari import plot_napari

if TYPE_CHECKING:
    import napari
    from napari.layers import Image

_MOUSE_SOURCE = "Mouse (Shift + hover)"


@dataclass(slots=True)
class _PreparedLayer:
    """Cached data needed to compute seed correlations quickly."""

    layer: Image
    data: xr.DataArray
    flat: np.ndarray
    centered: np.ndarray
    norms: np.ndarray
    spatial_shape: tuple[int, ...]
    time_axis: int
    window: slice


class FunctionalConnectivityPanel(QWidget):
    """Right-side panel for interactive seed-based connectivity maps."""

    def __init__(self, viewer: napari.Viewer, signal_store: SignalStore) -> None:
        super().__init__()
        self._viewer = viewer
        self._signal_store = signal_store
        self._prepared: _PreparedLayer | None = None
        self._map_layer = None
        self._last_voxel: tuple[int, ...] | None = None
        self._mouse_active = False
        self._mouse_seed_created = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Seed-based maps")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._data_combo = self._make_narrow_combo()
        self._seed_combo = self._make_narrow_combo()
        self._radius_spin = QDoubleSpinBox()
        self._radius_spin.setRange(0.0, 1000.0)
        self._radius_spin.setDecimals(1)
        self._radius_spin.setSingleStep(0.1)
        self._radius_spin.setValue(0.0)
        self._radius_spin.setSuffix(" mm")
        form.addRow("Data", self._data_combo)
        form.addRow("Seed", self._seed_combo)
        form.addRow("Radius", self._radius_spin)
        layout.addLayout(form)

        time_header = QHBoxLayout()
        self._time_start_label = QLabel("Start: —")
        self._time_end_label = QLabel("End: —")
        self._time_end_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        time_header.addWidget(self._time_start_label)
        time_header.addWidget(self._time_end_label)
        self._time_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        self._time_slider.setRange(0.0, 1.0)
        self._time_slider.setValue((0.0, 1.0))
        self._time_slider.setMinimumWidth(0)
        self._time_slider.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self._time_slider.valuesChanged.connect(self._on_time_window_changed)
        layout.addLayout(time_header)
        layout.addWidget(self._time_slider)

        self._compute_btn = QPushButton("Compute map")
        self._compute_btn.setObjectName("primary_btn")
        self._compute_btn.clicked.connect(self._compute_clicked)
        self._seed_combo.currentIndexChanged.connect(self._on_seed_changed)
        self._radius_spin.valueChanged.connect(self._on_radius_changed)
        layout.addWidget(self._compute_btn)

        self._status = QLabel(
            "Use a stored signal, or choose mouse and hold Shift over the data."
        )
        self._status.setWordWrap(True)
        self._status.setObjectName("placeholder")
        layout.addWidget(self._status)
        layout.addStretch()

        self._viewer.layers.events.inserted.connect(self._refresh_layers)
        self._viewer.layers.events.removed.connect(self._refresh_layers)
        self._viewer.mouse_move_callbacks.append(self._on_mouse_move)
        self._signal_store.changed.connect(self._refresh_seeds)
        self._refresh_layers()
        self._refresh_seeds()
        self._on_seed_changed()

    def _make_narrow_combo(self) -> QComboBox:
        """Return a combo box that does not force the sidebar wider."""
        combo = QComboBox()
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(8)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return combo

    def _refresh_layers(self, event=None) -> None:
        """Refresh selectable image layers containing a `time` dimension."""
        current = self._data_combo.currentText()
        prepared_layer = self._prepared.layer if self._prepared is not None else None
        self._data_combo.clear()
        for layer in self._viewer.layers:
            data = getattr(layer, "metadata", {}).get("xarray")
            if isinstance(data, xr.DataArray) and TIME_DIM in data.dims:
                self._data_combo.addItem(layer.name)
        index = self._data_combo.findText(current)
        if index >= 0:
            self._data_combo.setCurrentIndex(index)
        if (
            prepared_layer is not None
            and prepared_layer.name == self._data_combo.currentText()
        ):
            return
        self._prepared = None
        self._last_voxel = None

    def _set_time_controls(self, data: xr.DataArray) -> None:
        """Match the time-window controls to the current data time coordinate."""
        times = np.asarray(data[TIME_DIM].values, dtype=float)
        start = float(times[0])
        end = float(times[-1])
        step = abs(float(np.median(np.diff(times)))) if times.size > 1 else 1.0
        self._time_slider.blockSignals(True)
        self._time_slider.setRange(start, end)
        self._time_slider.setSingleStep(step)
        self._time_slider.setValue((start, end))
        self._time_slider.blockSignals(False)
        self._update_time_labels(data, start, end)

    def _refresh_seeds(self) -> None:
        """Refresh mouse/stored signal seed choices."""
        current_id = self._seed_combo.currentData()
        self._seed_combo.clear()
        self._seed_combo.addItem(_MOUSE_SOURCE, None)
        for signal in self._signal_store.stored_signals():
            self._seed_combo.addItem(signal.name, signal.id)
        index = self._seed_combo.findData(current_id)
        if index >= 0:
            self._seed_combo.setCurrentIndex(index)

    def _on_seed_changed(self) -> None:
        """Update controls for the selected seed source."""
        mouse = self._seed_combo.currentData() is None
        self._mouse_active = mouse and self._mouse_seed_created
        self._radius_spin.setEnabled(mouse)
        self._compute_btn.setVisible(not mouse or not self._mouse_seed_created)
        self._compute_btn.setText(
            "Create mouse map" if mouse else "Compute map"
        )
        self._status.setText(
            "Hold Shift over the data layer."
            if mouse and self._mouse_seed_created
            else "Use a stored signal, or choose mouse and hold Shift over the data."
        )

    def _on_radius_changed(self) -> None:
        """Update the mouse seed map after a radius change."""
        if self._last_voxel is None or self._prepared is None:
            return
        self._show_map(
            self._corr_map(
                self._mean_seed_trace(self._prepared, self._last_voxel), self._prepared
            ),
            "Mouse seed map",
        )

    def _on_time_window_changed(self, value: tuple[float, float]) -> None:
        """Recompute cached correlations for the selected time window."""
        start, end = sorted(float(v) for v in value)
        if start == end:
            end = min(end + self._time_slider.singleStep(), self._time_slider.maximum())
            start = min(start, end - self._time_slider.singleStep())
            self._time_slider.blockSignals(True)
            self._time_slider.setValue((start, end))
            self._time_slider.blockSignals(False)
        try:
            prepared = self._prepare_layer() if self._prepared is None else self._prepared
            self._update_time_labels(prepared.data, start, end)
            self._apply_time_window(prepared)
            self._recompute_current_map(prepared)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(str(exc))

    def _update_time_labels(self, data: xr.DataArray, start: float, end: float) -> None:
        """Show start and end times next to the range slider."""
        unit = data.coords[TIME_DIM].attrs.get("units", "")
        suffix = f" {unit}" if unit else ""
        self._time_start_label.setText(f"Start: {start:g}{suffix}")
        self._time_end_label.setText(f"End: {end:g}{suffix}")

    def _compute_clicked(self) -> None:
        """Compute one seed map from the selected source."""
        if self._seed_combo.currentData() is None:
            self._mouse_seed_created = True
            self._mouse_active = True
            self._on_seed_changed()
            return

        self._mouse_active = False
        signal = self._selected_signal()
        if signal is None:
            show_error("Select an existing stored signal first.")
            return
        try:
            prepared = self._prepare_layer()
            seed = self._align_signal(signal, prepared.data)[prepared.window]
            self._show_map(self._corr_map(seed, prepared), f"Seed map: {signal.name}")
            show_info(f"Computed seed map from {signal.name}.")
        except Exception as exc:  # noqa: BLE001
            show_error(str(exc))

    def _selected_signal(self) -> StoredSignal | None:
        """Return the selected stored signal, if any."""
        signal_id = self._seed_combo.currentData()
        for signal in self._signal_store.stored_signals():
            if signal.id == signal_id:
                return signal
        return None

    def _prepare_layer(self) -> _PreparedLayer:
        """Cache centered voxel time series for the selected layer."""
        name = self._data_combo.currentText()
        if not name:
            raise ValueError("Select a data layer with a time dimension.")
        layer = self._viewer.layers[name]
        data = layer.metadata.get("xarray")
        if not isinstance(data, xr.DataArray) or TIME_DIM not in data.dims:
            raise ValueError(
                "Selected layer must come from a VoxelData array with time."
            )
        if self._prepared is not None and self._prepared.layer is layer:
            return self._prepared

        time_axis = data.get_axis_num(TIME_DIM)
        values = np.moveaxis(np.asarray(data.data, dtype=float), time_axis, 0)
        self._set_time_controls(data)
        spatial_shape = values.shape[1:]
        flat = values.reshape(values.shape[0], -1)
        self._prepared = _PreparedLayer(
            layer,
            data,
            flat,
            flat,
            np.ones(flat.shape[1], dtype=float),
            spatial_shape,
            time_axis,
            slice(0, values.shape[0]),
        )
        self._apply_time_window(self._prepared)
        return self._prepared

    def _apply_time_window(self, prepared: _PreparedLayer) -> None:
        """Update centered data and norms for the selected time window."""
        times = np.asarray(prepared.data[TIME_DIM].values, dtype=float)
        start, end = sorted(float(v) for v in self._time_slider.value())
        indices = np.flatnonzero((times >= start) & (times <= end))
        if indices.size < 2:
            raise ValueError("Select at least two time points.")
        prepared.window = slice(int(indices[0]), int(indices[-1]) + 1)
        window = prepared.flat[prepared.window]
        prepared.centered = window - window.mean(axis=0)
        prepared.norms = np.sqrt(np.sum(prepared.centered * prepared.centered, axis=0))

    def _mean_seed_trace(
        self, prepared: _PreparedLayer, spatial_indices: tuple[int, ...]
    ) -> np.ndarray:
        """Return the mean trace inside the selected mouse sphere."""
        radius = self._radius_spin.value()
        if radius <= 0:
            return prepared.centered[
                :, np.ravel_multi_index(spatial_indices, prepared.spatial_shape)
            ]

        spacings = [abs(float(prepared.data.fusi.spacing[dim])) for dim in VOXEL_DIMS]
        grids = np.ogrid[tuple(slice(0, n) for n in prepared.spatial_shape)]
        dist2 = sum(
            ((grid - index) * spacing) ** 2
            for grid, index, spacing in zip(grids, spatial_indices, spacings)
        )
        mask = dist2 <= radius * radius
        return prepared.centered[:, mask.ravel()].mean(axis=1)

    def _align_signal(self, signal: StoredSignal, data: xr.DataArray) -> np.ndarray:
        """Return a stored signal sampled on `data`'s time grid."""
        y = np.asarray(signal.y, dtype=float)
        if y.size == data.sizes[TIME_DIM]:
            return y
        if TIME_DIM not in data.coords:
            raise ValueError("Signal length does not match the data time dimension.")
        return np.interp(np.asarray(data[TIME_DIM], dtype=float), signal.x, y)

    def _corr_map(self, seed: np.ndarray, prepared: _PreparedLayer) -> xr.DataArray:
        """Compute a Pearson correlation map from a seed signal."""
        seed = np.asarray(seed, dtype=float)
        seed = seed - seed.mean()
        seed_norm = float(np.sqrt(np.sum(seed * seed)))
        denom = prepared.norms * seed_norm
        corr = np.divide(
            prepared.centered.T @ seed,
            denom,
            out=np.zeros_like(prepared.norms, dtype=float),
            where=denom != 0,
        )
        template = prepared.data.isel({TIME_DIM: 0}, drop=True)
        return template.copy(data=corr.reshape(prepared.spatial_shape)).assign_attrs(
            long_name="Pearson r"
        )

    def _recompute_current_map(self, prepared: _PreparedLayer) -> None:
        """Update the visible seed map after a time-window change."""
        if self._map_layer is None or self._map_layer.name not in self._viewer.layers:
            return
        signal = self._selected_signal()
        if signal is not None:
            seed = self._align_signal(signal, prepared.data)[prepared.window]
            self._show_map(self._corr_map(seed, prepared), f"Seed map: {signal.name}")
        elif self._last_voxel is not None:
            self._show_map(
                self._corr_map(self._mean_seed_trace(prepared, self._last_voxel), prepared),
                "Mouse seed map",
            )

    def _show_map(self, data: xr.DataArray, name: str) -> None:
        """Add or update the seed-map layer."""
        if self._map_layer is None or self._map_layer.name not in self._viewer.layers:
            _viewer, self._map_layer = plot_napari(
                data,
                viewer=self._viewer,
                name=name,
                colormap="twilight",
                contrast_limits=(-1, 1),
                blending="translucent",
                opacity=0.7,
                show_colorbar=False,
                show_scale_bar=False,
            )
        else:
            self._map_layer.name = name
            self._map_layer.data = np.asarray(data.data)
            self._map_layer.metadata["xarray"] = data

    def _on_mouse_move(self, viewer, event) -> None:
        """Update the seed map from the voxel under the mouse."""
        if not self._mouse_active or "Shift" not in event.modifiers:
            return
        try:
            prepared = self._prepare_layer()
            indices = [
                round(v) for v in prepared.layer.world_to_data(viewer.cursor.position)
            ]
            spatial_indices = tuple(
                indices[i] for i in range(len(indices)) if i != prepared.time_axis
            )
            if len(spatial_indices) != len(prepared.spatial_shape):
                return
            if not all(
                0 <= i < n for i, n in zip(spatial_indices, prepared.spatial_shape)
            ):
                return
            if spatial_indices == self._last_voxel:
                return
            self._last_voxel = spatial_indices
            self._show_map(
                self._corr_map(
                    self._mean_seed_trace(prepared, spatial_indices), prepared
                ),
                "Mouse seed map",
            )
        except Exception as exc:  # noqa: BLE001
            self._status.setText(str(exc))

    def closeEvent(self, event) -> None:
        """Disconnect callbacks owned by this panel."""
        if self._on_mouse_move in self._viewer.mouse_move_callbacks:
            self._viewer.mouse_move_callbacks.remove(self._on_mouse_move)
        super().closeEvent(event)
