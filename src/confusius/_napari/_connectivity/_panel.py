"""Functional-connectivity panel consuming explicitly selected shared traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr
from napari.layers import Image
from napari.utils.notifications import show_error
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QDoubleRangeSlider

from confusius._dims import TIME_DIM
from confusius._napari._signals._store import SignalStore
from confusius._utils.signals import align_signal_to_time
from confusius.plotting.napari import plot_napari
from confusius.validation import validate_voxeldata

if TYPE_CHECKING:
    import napari
    from qtpy.QtGui import QCloseEvent


@dataclass(slots=True)
class _PreparedLayer:
    """Cached target voxel time series.

    Attributes
    ----------
    layer : napari.layers.Image
        Explicit target recording.
    data : xarray.DataArray
        Target VoxelData array.
    flat : numpy.ndarray
        Full time by voxel matrix.
    centered : numpy.ndarray
        Centered matrix for the selected time window.
    norms : numpy.ndarray
        Voxel norms for the selected window.
    spatial_shape : tuple[int, ...]
        Output map shape.
    window : slice
        Selected target time window.
    revision : int
        Recording revision used to invalidate cached target values.
    """

    layer: Image
    data: xr.DataArray
    flat: np.ndarray
    centered: np.ndarray
    norms: np.ndarray
    spatial_shape: tuple[int, ...]
    window: slice
    revision: int


class FunctionalConnectivityPanel(QWidget):
    """Compute maps from shared live or stored seeds and an explicit target.

    Parameters
    ----------
    viewer : napari.Viewer
        Viewer containing recordings and output maps.
    signal_store : SignalStore
        Shared sources and stored snapshots.
    """

    def __init__(self, viewer: napari.Viewer, signal_store: SignalStore) -> None:
        super().__init__()
        self._viewer = viewer
        self._signal_store = signal_store
        self._live_sources = signal_store.live_sources(viewer)
        self._prepared: _PreparedLayer | None = None
        self._map_layer: Image | None = None
        self._active_seed: str | None = None
        self._active_target: Image | None = None
        self._stored_payload: tuple[np.ndarray, np.ndarray] | None = None
        self._computing = False
        self._closed = False
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(16)
        self._update_timer.timeout.connect(self._update_active_map)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        title = QLabel("Seed-based maps")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._data_combo = self._make_narrow_combo()
        self._seed_combo = self._make_narrow_combo()
        form.addRow("Target recording", self._data_combo)
        form.addRow("Seed trace", self._seed_combo)
        self._seed_description = QLabel("Select a seed and its reference recording.")
        self._seed_description.setWordWrap(True)
        form.addRow(self._seed_description)
        layout.addLayout(form)
        header = QHBoxLayout()
        self._time_start_label = QLabel("Start: —")
        self._time_end_label = QLabel("End: —")
        self._time_end_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        header.addWidget(self._time_start_label)
        header.addWidget(self._time_end_label)
        layout.addLayout(header)
        self._time_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        self._time_slider.setRange(0.0, 1.0)
        self._time_slider.setValue((0.0, 1.0))
        self._time_slider.setMinimumWidth(0)
        self._time_slider.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self._time_slider)
        self._compute_btn = QPushButton("Compute map")
        self._compute_btn.setObjectName("primary_btn")
        self._compute_btn.clicked.connect(self._compute_clicked)
        layout.addWidget(self._compute_btn)
        self._status = QLabel("Select an explicit seed trace; pinning is optional.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()
        self._data_combo.currentIndexChanged.connect(self._on_target_changed)
        self._seed_combo.currentIndexChanged.connect(self._on_seed_changed)
        self._time_slider.valuesChanged.connect(self._on_time_window_changed)
        viewer.layers.events.inserted.connect(self._refresh_layers)
        viewer.layers.events.removed.connect(self._refresh_layers)
        viewer.layers.events.renamed.connect(self._refresh_layers)
        signal_store.changed.connect(self._refresh_seeds)
        self._live_sources.traces_changed.connect(self._on_traces_changed)
        self._live_sources.sources_changed.connect(self._refresh_layers)
        self._refresh_layers()
        self._refresh_seeds()

    def _make_narrow_combo(self) -> QComboBox:
        """Create a sidebar-friendly selector.

        Returns
        -------
        qtpy.QtWidgets.QComboBox
            Narrow combo box.
        """
        combo = QComboBox()
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(8)
        return combo

    def _refresh_layers(self, event=None) -> None:
        """Refresh targets by object identity, never by name.

        Parameters
        ----------
        event : object, optional
            Layer-list event.
        """
        current = self._data_combo.currentData()
        self._data_combo.blockSignals(True)
        self._data_combo.clear()
        self._data_combo.addItem("Select target…", None)
        for layer in self._live_sources.references():
            data = layer.metadata["xarray"]
            if (
                TIME_DIM in data.dims
                and TIME_DIM in data.coords
                and data.sizes[TIME_DIM] > 1
            ):
                self._data_combo.addItem(layer.name, layer)
        if current is not None:
            index = self._data_combo.findData(current)
            if index < 0:
                self._data_combo.addItem(f"{current.name} (removed)", current)
                index = self._data_combo.count() - 1
            self._data_combo.setCurrentIndex(index)
        self._data_combo.blockSignals(False)
        self._on_seed_changed(mark_stale=False)

    def _refresh_seeds(self) -> None:
        """Preserve seed selection even when hidden, disabled or removed."""
        current = self._seed_combo.currentData()
        previous_name = self._seed_combo.currentText()
        self._seed_combo.blockSignals(True)
        self._seed_combo.clear()
        self._seed_combo.addItem("Select seed…", None)
        for signal in self._signal_store.live_signals():
            if (
                signal.reference_layer is None
                or TIME_DIM not in signal.reference_layer.metadata["xarray"].dims
            ):
                continue
            suffix = (
                " (unavailable)" if not signal.enabled or not signal.available else ""
            )
            self._seed_combo.addItem(signal.display_name + suffix, signal.id)
        for signal in self._signal_store.stored_signals():
            if signal.coordinate_dim == TIME_DIM:
                self._seed_combo.addItem(
                    f"Stored | {signal.name} [{signal.id}]", signal.id
                )
                self._seed_combo.setItemData(
                    self._seed_combo.count() - 1,
                    str(signal.file_path)
                    if signal.file_path is not None
                    else signal.source_label,
                    Qt.ItemDataRole.ToolTipRole,
                )
        if current is not None:
            index = self._seed_combo.findData(current)
            if index < 0:
                self._seed_combo.addItem(
                    previous_name.removesuffix(" (removed)") + " (removed)", current
                )
                index = self._seed_combo.count() - 1
            self._seed_combo.setCurrentIndex(index)
        self._seed_combo.blockSignals(False)
        self._on_seed_changed(mark_stale=False)
        if current == self._active_seed and self._stored_payload is not None:
            stored = next(
                (s for s in self._signal_store.stored_signals() if s.id == current),
                None,
            )
            if stored is not None and (
                stored.x is not self._stored_payload[0]
                or stored.y is not self._stored_payload[1]
            ):
                self._on_traces_changed()

    def _on_target_changed(self) -> None:
        """Require explicit computation after a target change."""
        self._prepared = None
        target = self._data_combo.currentData()
        self._mark_stale()
        if isinstance(target, Image) and target in self._viewer.layers:
            try:
                self._set_time_controls(target.metadata["xarray"])
            except ValueError as exc:
                self._status.setText(str(exc))
                self._compute_btn.setEnabled(False)
                return
        self._on_seed_changed(mark_stale=False)

    def _on_seed_changed(self, *, mark_stale: bool = True) -> None:
        """Describe the reference independently of the target.

        Parameters
        ----------
        mark_stale : bool, default: True
            Whether to disarm automatic updates after a selection change.
        """
        if mark_stale:
            self._mark_stale()
        sid = self._seed_combo.currentData()
        live = self._signal_store.get_live_signal(sid)
        target = self._data_combo.currentData()
        stored = next(
            (s for s in self._signal_store.stored_signals() if s.id == sid), None
        )
        exists = live is not None or (
            stored is not None and stored.coordinate_dim == TIME_DIM
        )
        enabled = exists and (live is None or live.enabled and live.available)
        valid_target = isinstance(target, Image) and target in self._viewer.layers
        self._compute_btn.setEnabled(bool(enabled and valid_target))
        if (
            self._map_layer is not None
            and self._active_seed == sid
            and not (enabled and valid_target)
        ):
            self._map_layer.metadata["seed_stale"] = True
            self._status.setText(
                "Map paused/stale: the selected seed or target is unavailable."
            )
        if live is not None:
            reference = live.reference_layer
            text = (
                f"Seed reference: {reference.name}."
                if reference is not None
                else "Missing reference recording."
            )
            if live.source_type == "mouse":
                text += f" Shared mouse radius: {self._live_sources.radius:g} mm."
            if target is not None and target is not reference:
                text += " Cross-recording: align the reference trace to target times; do not re-extract from target."
            self._seed_description.setText(text)
        else:
            self._seed_description.setText(
                f"Stored seed: {stored.source_label} [{stored.id}]. Aligned by its time coordinates."
                if stored is not None
                else "Select an available seed trace."
            )

    def _mark_stale(self) -> None:
        """Disarm live updates when changing the selected seed or target."""
        self._active_seed = None
        self._active_target = None
        self._stored_payload = None
        if self._map_layer is not None:
            self._map_layer.metadata["seed_stale"] = True
            self._status.setText(
                "Map is stale. Compute to apply the selected seed and target."
            )

    def _set_time_controls(self, data: xr.DataArray) -> None:
        """Initialise the target time window.

        Parameters
        ----------
        data : xarray.DataArray
            Target recording.
        """
        times = np.asarray(data[TIME_DIM].values, dtype=float)
        align_signal_to_time(times, np.zeros(times.size), times)
        self._time_slider.blockSignals(True)
        self._time_slider.setRange(float(times[0]), float(times[-1]))
        self._time_slider.setSingleStep(float(np.median(np.diff(times))))
        self._time_slider.setValue((float(times[0]), float(times[-1])))
        self._time_slider.blockSignals(False)
        self._update_time_labels(data)

    def _update_time_labels(self, data: xr.DataArray) -> None:
        """Show target time bounds and units.

        Parameters
        ----------
        data : xarray.DataArray
            Target recording.
        """
        start, end = self._time_slider.value()
        unit = data[TIME_DIM].attrs.get("units", "")
        suffix = f" {unit}" if unit else ""
        self._time_start_label.setText(f"Start: {start:g}{suffix}")
        self._time_end_label.setText(f"End: {end:g}{suffix}")

    def _prepare_layer(self) -> _PreparedLayer:
        """Prepare target data once, preserving its VoxelData geometry.

        Returns
        -------
        _PreparedLayer
            Cached time-by-voxel values.

        Raises
        ------
        ValueError
            If no supported target or valid time coordinates are selected.
        """
        layer = self._data_combo.currentData()
        if not isinstance(layer, Image) or layer not in self._viewer.layers:
            raise ValueError("Select an existing target recording.")
        data = layer.metadata["xarray"]
        validate_voxeldata(data, require_time=True, allow_pose=False)
        if data.ndim != 4:
            raise ValueError(
                "FC targets require time, k, j, i without extra dimensions."
            )
        times = np.asarray(data[TIME_DIM].values, dtype=float)
        align_signal_to_time(times, np.zeros(times.size), times)
        revision = self._live_sources.revision(layer)
        if (
            self._prepared is None
            or self._prepared.data is not data
            or self._prepared.revision != revision
        ):
            values = np.asarray(layer.data, dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(
                    "FC target contains non-finite values; preprocess it first."
                )
            flat = values.reshape(values.shape[0], -1)
            self._prepared = _PreparedLayer(
                layer,
                data,
                flat,
                flat,
                np.ones(flat.shape[1]),
                values.shape[1:],
                slice(0, len(times)),
                revision,
            )
        self._apply_time_window(self._prepared)
        return self._prepared

    def _apply_time_window(self, prepared: _PreparedLayer) -> None:
        """Update centered target data for the chosen window.

        Parameters
        ----------
        prepared : _PreparedLayer
            Cached target data.

        Raises
        ------
        ValueError
            If fewer than two samples are selected.
        """
        times = np.asarray(prepared.data[TIME_DIM], dtype=float)
        start, end = sorted(self._time_slider.value())
        indices = np.flatnonzero((times >= start) & (times <= end))
        if len(indices) < 2:
            raise ValueError("Select at least two target time points.")
        window = slice(int(indices[0]), int(indices[-1]) + 1)
        if (
            self._prepared is not None
            and window == prepared.window
            and prepared.centered is not prepared.flat
        ):
            return
        prepared.window = window
        values = prepared.flat[window]
        prepared.centered = values - values.mean(axis=0)
        prepared.norms = np.linalg.norm(prepared.centered, axis=0)

    def _seed_values(self, prepared: _PreparedLayer) -> tuple[str, np.ndarray]:
        """Resolve exactly one seed and align it to the selected target window.

        Parameters
        ----------
        prepared : _PreparedLayer
            Target data.

        Returns
        -------
        name : str
            Selected seed display name.
        values : numpy.ndarray
            Seed values on target window times.

        Raises
        ------
        ValueError
            If the seed is unavailable, uncovered or not a supported time trace.
        """
        sid = self._seed_combo.currentData()
        live = self._signal_store.get_live_signal(sid)
        if live is not None:
            if live.reference_layer is None:
                raise ValueError("Selected seed has no reference recording.")
            reference = live.reference_layer.metadata["xarray"]
            validate_voxeldata(reference, require_time=True, allow_pose=False)
            if reference.ndim != 4:
                raise ValueError(
                    "FC seeds require a time, k, j, i reference recording."
                )
            x, y = self._live_sources.get_trace(sid)
            name = live.display_name
            if reference[TIME_DIM].attrs.get("units") != prepared.data[
                TIME_DIM
            ].attrs.get("units"):
                raise ValueError("Seed and target time units must match.")
        else:
            stored = next(
                (s for s in self._signal_store.stored_signals() if s.id == sid), None
            )
            if stored is None or stored.coordinate_dim != TIME_DIM:
                raise ValueError("Select an existing, unambiguous time seed trace.")
            x, y, name = stored.x, stored.y, stored.name
        times = np.asarray(prepared.data[TIME_DIM].values)[prepared.window]
        seed = align_signal_to_time(x, y, times)
        if not np.isfinite(seed).all():
            raise ValueError(
                "Seed must provide finite values covering the selected target time window; extrapolation is not allowed."
            )
        return name, seed

    def _corr_map(self, seed: np.ndarray, prepared: _PreparedLayer) -> xr.DataArray:
        """Compute Pearson correlations using cached target centering.

        Parameters
        ----------
        seed : numpy.ndarray
            Aligned seed values.
        prepared : _PreparedLayer
            Prepared target window.

        Returns
        -------
        xarray.DataArray
            VoxelData correlation map preserving target geometry.
        """
        centered = seed - seed.mean()
        denominator = prepared.norms * np.linalg.norm(centered)
        corr = np.divide(
            prepared.centered.T @ centered,
            denominator,
            out=np.zeros_like(prepared.norms),
            where=denominator != 0,
        )
        template = prepared.data.isel({TIME_DIM: 0}, drop=True)
        return template.copy(data=corr.reshape(prepared.spatial_shape)).assign_attrs(
            long_name="Pearson r"
        )

    def _compute_clicked(self) -> None:
        """Create a map and explicitly arm updates for this seed/target pair."""
        self._active_seed = self._seed_combo.currentData()
        self._active_target = self._data_combo.currentData()
        self._update_active_map()

    def _on_traces_changed(self, source=None) -> None:
        """Coalesce relevant source changes without another mouse callback.

        Parameters
        ----------
        source : object, optional
            Changed layer, mouse sentinel or a general invalidation.
        """
        live = (
            self._signal_store.get_live_signal(self._active_seed)
            if self._active_seed is not None
            else None
        )
        if source == "mouse" and (live is None or live.source_type != "mouse"):
            return
        if (
            isinstance(source, Image)
            and source is not self._active_target
            and (live is None or source is not live.reference_layer)
        ):
            return
        if self._active_seed is not None and not self._update_timer.isActive():
            self._update_timer.start()

    def _on_time_window_changed(self, value: tuple[float, float]) -> None:
        """Refresh an explicitly created map after changing its time window.

        Parameters
        ----------
        value : tuple[float, float]
            Target time bounds.
        """
        target = self._data_combo.currentData()
        if isinstance(target, Image):
            self._update_time_labels(target.metadata["xarray"])
        self._on_traces_changed()

    def _update_active_map(self) -> None:
        """Update only the armed seed/target pair; never substitute another seed."""
        if self._computing or self._active_seed is None:
            return
        if (
            self._seed_combo.currentData() != self._active_seed
            or self._data_combo.currentData() is not self._active_target
        ):
            self._mark_stale()
            return
        self._computing = True
        try:
            prepared = self._prepare_layer()
            name, seed = self._seed_values(prepared)
            result = self._corr_map(seed, prepared)
            title = f"Seed map: {name} → {prepared.layer.name}"
            if self._map_layer is None or self._map_layer not in self._viewer.layers:
                _, layer = plot_napari(
                    result,
                    viewer=self._viewer,
                    name=title,
                    colormap="twilight",
                    contrast_limits=(-1, 1),
                    blending="translucent",
                    opacity=0.7,
                    show_colorbar=False,
                    show_scale_bar=False,
                )
                self._map_layer = cast("Image", layer)
            else:
                self._map_layer.name = title
                self._map_layer.data = np.asarray(result.data)  # ty: ignore[invalid-assignment]
                self._map_layer.metadata["xarray"] = result
            live = self._signal_store.get_live_signal(self._active_seed)
            reference = live.reference_layer if live is not None else None
            self._map_layer.metadata.update(
                seed_id=self._active_seed,
                seed_stale=False,
                seed_reference_id=self._live_sources.layer_id(reference)
                if reference is not None
                else None,
                target_id=self._live_sources.layer_id(prepared.layer),
            )
            stored = next(
                (
                    s
                    for s in self._signal_store.stored_signals()
                    if s.id == self._active_seed
                ),
                None,
            )
            self._stored_payload = (stored.x, stored.y) if stored is not None else None
            self._status.setText(f"Active map: {name} → {prepared.layer.name}.")
        except (ValueError, KeyError) as exc:
            self._status.setText(f"Map paused/stale: {exc}")
            if self._map_layer is not None:
                self._map_layer.metadata["seed_stale"] = True
            if self._map_layer is None:
                show_error(str(exc))
        finally:
            self._computing = False

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Disconnect panel subscriptions without closing shared sources.

        Parameters
        ----------
        a0 : qtpy.QtGui.QCloseEvent, optional
            Qt close event.
        """
        if self._closed:
            super().closeEvent(a0)
            return
        self._closed = True
        self._update_timer.stop()
        self._signal_store.changed.disconnect(self._refresh_seeds)
        self._live_sources.traces_changed.disconnect(self._on_traces_changed)
        self._live_sources.sources_changed.disconnect(self._refresh_layers)
        self._viewer.layers.events.inserted.disconnect(self._refresh_layers)
        self._viewer.layers.events.removed.disconnect(self._refresh_layers)
        self._viewer.layers.events.renamed.disconnect(self._refresh_layers)
        super().closeEvent(a0)
