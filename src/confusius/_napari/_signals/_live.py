"""Shared, viewer-scoped mouse, point and label signal sources."""

from __future__ import annotations

import colorsys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib.colors import to_hex, to_rgb
from napari.layers import Image, Labels, Layer, Points
from napari.utils.key_bindings import coerce_keybinding
from qtpy.QtCore import QObject, QTimer, Signal

from confusius._dims import TIME_DIM, VOXEL_DIMS
from confusius._napari._signals._store import LiveSignal, SignalStore
from confusius._napari._utils import (
    CATEGORICAL_COLORS,
    extract_label_mean_trace,
    extract_voxel_trace,
)
from confusius._utils.geometry import get_voxel_to_world_affine
from confusius.validation import validate_voxeldata

if TYPE_CHECKING:
    import napari

_POINT_ID = "confusius_signal_id"
"""Points feature preserving item identity through movement and deletion."""


@dataclass(slots=True)
class LiveSource:
    """One independently enabled spatial source.

    Attributes
    ----------
    id : str
        Session-stable source identifier.
    kind : str
        Mouse, point or label source.
    layer : napari.layers.Layer or None
        Backing ROI layer; absent for the shared cursor.
    enabled : bool
        Whether to supply live traces.
    """

    id: str
    kind: Literal["mouse", "point", "label"]
    layer: Layer | None = None
    enabled: bool = True


class LiveSources(QObject):
    """Own source lifecycle and extraction independently of plotting.

    Parameters
    ----------
    viewer : napari.Viewer
        Viewer providing layers and the shared cursor.
    store : SignalStore
        Store providing presentation metadata.
    """

    traces_changed = Signal(object)
    sources_changed = Signal()

    def __init__(self, viewer: napari.Viewer, store: SignalStore) -> None:
        super().__init__(store)
        self.viewer = viewer
        self.store = store
        self._sources = {"mouse": LiveSource("mouse", "mouse")}
        self._layers: dict[int, str] = {}
        self._counter = 0
        self._point_counter = 0
        self._revisions: dict[str, int] = {}
        self._references: list[Image] | None = None
        self._colors: dict[str, str] = {}
        self._tones: dict[tuple[str, int], int] = {}
        self._label_ids: dict[str, set[int]] = {}
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._position: np.ndarray | None = None
        self._mouse_sample: tuple | None = None
        self._radius = 0.0
        self._display_axis = TIME_DIM
        self._replacing_points: set[int] = set()
        self._syncing = False
        self._closed = False
        self._shift_bound = False
        self._previous_shift = None
        self._subscribed: list[Layer] = []
        self._mouse_timer = QTimer(self)
        self._mouse_timer.setSingleShot(True)
        self._mouse_timer.setInterval(16)
        self._mouse_timer.timeout.connect(self._mouse_changed)
        self._roi_timer = QTimer(self)
        self._roi_timer.setSingleShot(True)
        self._roi_timer.setInterval(150)
        self._roi_timer.timeout.connect(self._data_changed)
        viewer.layers.events.inserted.connect(self._layers_changed)
        viewer.layers.events.removed.connect(self._layers_changed)
        viewer.layers.events.renamed.connect(self._names_changed)
        viewer.mouse_move_callbacks.append(self._on_mouse_move)
        self._layers_changed()
        self._bind_shift()

    def layer_id(self, layer: Layer) -> str:
        """Return a session-stable layer identifier.

        Parameters
        ----------
        layer : napari.layers.Layer
            Layer whose identity is requested.

        Returns
        -------
        str
            Identifier independent of the display name.
        """
        key = id(layer)
        if key not in self._layers:
            self._layers[key] = f"layer-{self._counter}"
            self._counter += 1
        return self._layers[key]

    def sources(self) -> list[LiveSource]:
        """Return registered sources.

        Returns
        -------
        list[LiveSource]
            Sources in insertion order.
        """
        return list(self._sources.values())

    def add_source(self, layer: Points | Labels) -> str:
        """Register a Points or Labels layer without replacing other sources.

        Parameters
        ----------
        layer : napari.layers.Points or napari.layers.Labels
            Backing source layer.

        Returns
        -------
        str
            Stable source identifier.

        Raises
        ------
        ValueError
            If the layer is not a supported source in this viewer.
        """
        if not isinstance(layer, (Points, Labels)) or layer not in self.viewer.layers:
            raise ValueError("Select a Points or Labels layer in this viewer.")
        sid = self.layer_id(layer)
        if sid not in self._sources:
            self._sources[sid] = LiveSource(
                sid, "point" if isinstance(layer, Points) else "label", layer
            )
        self._data_changed()
        return sid

    def remove_source(self, source_id: str) -> None:
        """Remove an ROI source; mouse can be disabled instead.

        Parameters
        ----------
        source_id : str
            Registered source identifier.

        Raises
        ------
        ValueError
            If the source is unknown or is the mouse source.
        """
        if source_id == "mouse" or source_id not in self._sources:
            raise ValueError("Only registered ROI sources can be removed.")
        del self._sources[source_id]
        self._label_ids.pop(source_id, None)
        self._data_changed()

    def set_enabled(self, source_id: str, enabled: bool) -> None:
        """Enable extraction without removing source identity or overrides.

        Parameters
        ----------
        source_id : str
            Source identifier.
        enabled : bool
            Whether to enable extraction.
        """
        self._sources[source_id].enabled = enabled
        if source_id == "mouse":
            self._bind_shift() if enabled else self._unbind_shift()
        self._reconcile()
        self.sources_changed.emit()
        self.traces_changed.emit(None)

    def set_references(self, layers: list[Image] | None) -> None:
        """Configure reference recordings, retaining excluded trace identities.

        Parameters
        ----------
        layers : list[napari.layers.Image], optional
            Reference layers. If not provided, use all eligible recordings.
        """
        self._references = layers
        self._reconcile()
        self.sources_changed.emit()
        self.traces_changed.emit(None)

    def references(self) -> list[Image]:
        """Return eligible reference recordings.

        Returns
        -------
        list[napari.layers.Image]
            Valid VoxelData images with time.
        """
        result = []
        for layer in self.viewer.layers:
            if not isinstance(layer, Image) or layer.rgb:
                continue
            data = layer.metadata.get("xarray")
            if data is None or not any(
                axis in data.dims and data.sizes[axis] > 1
                for axis in (TIME_DIM, self._display_axis)
            ):
                continue
            try:
                validate_voxeldata(data)
            except (TypeError, ValueError):
                continue
            result.append(layer)
        return result

    def set_display_axis(self, axis: str) -> None:
        """Keep Signals' non-time axis support separate from FC time extraction.

        Parameters
        ----------
        axis : str
            Display axis requested by Signals.
        """
        if axis != self._display_axis:
            self._display_axis = axis
            self._data_changed()

    @property
    def radius(self) -> float:
        """Return the shared physical mouse radius.

        Returns
        -------
        float
            Sphere radius in millimetres; zero selects one voxel.
        """
        return self._radius

    def set_radius(self, radius: float) -> None:
        """Set the shared mouse sphere radius.

        Parameters
        ----------
        radius : float
            Nonnegative physical radius in millimetres.

        Raises
        ------
        ValueError
            If the radius is negative or non-finite.
        """
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("Mouse radius must be finite and nonnegative.")
        self._radius = float(radius)
        self._cache = {
            key: value
            for key, value in self._cache.items()
            if not key[0].startswith("mouse-")
        }
        self._reconcile()
        self.traces_changed.emit("mouse")

    def base_color(self, layer: Image) -> str:
        """Return the recording's signal colour, independent of its colormap.

        Parameters
        ----------
        layer : napari.layers.Image
            Reference recording.

        Returns
        -------
        str
            Hex signal colour.
        """
        lid = self.layer_id(layer)
        if lid not in self._colors:
            self._colors[lid] = CATEGORICAL_COLORS[
                len(self._colors) % len(CATEGORICAL_COLORS)
            ]
        return self._colors[lid]

    def set_base_color(self, layer: Image, color: str) -> None:
        """Set a recording's base signal colour.

        Parameters
        ----------
        layer : napari.layers.Image
            Reference recording.
        color : str
            Matplotlib-compatible colour.
        """
        self._colors[self.layer_id(layer)] = to_hex(color)
        self._reconcile()
        self.sources_changed.emit()

    def _item_ids(self, source: LiveSource) -> Sequence[int | None]:
        """Find stable ROI item identifiers.

        Parameters
        ----------
        source : LiveSource
            Registered source.

        Returns
        -------
        list[int or None]
            Stable point IDs, label values or the mouse sentinel.
        """
        if source.kind == "mouse":
            return [None]
        layer = source.layer
        if isinstance(layer, Points):
            features = layer.features.copy()
            ids = (
                features[_POINT_ID].tolist()
                if _POINT_ID in features
                else [None] * len(layer.data)
            )
            seen = set()
            fixed = []
            for value in ids:
                if value is None or not np.isfinite(value) or int(value) in seen:
                    value = self._point_counter
                    self._point_counter += 1
                value = int(value)
                self._point_counter = max(self._point_counter, value + 1)
                seen.add(value)
                fixed.append(value)
            if _POINT_ID not in features or ids != fixed:
                features[_POINT_ID] = fixed
                layer.features = features
            return fixed
        if isinstance(layer, Labels):
            current = {int(v) for v in np.unique(np.asarray(layer.data)) if v != 0}
            known = self._label_ids.setdefault(source.id, set())
            known.update(current)
            return [int(value) for value in sorted(known)]
        return []

    def _reconcile(self) -> None:
        """Reconcile source/reference pairs without relying on rendered traces."""
        if self._syncing or self._closed:
            return
        self._syncing = True
        try:
            recordings = self.references()
            selected = recordings if self._references is None else self._references
            signals = []
            for source in self._sources.values():
                items = self._item_ids(source)
                current_labels = (
                    set(np.unique(np.asarray(source.layer.data)))
                    if isinstance(source.layer, Labels)
                    else set()
                )
                for layer in recordings:
                    base = self.base_color(layer)
                    for item in items:
                        sid = f"{source.kind}-{source.id}-{item}-{self.layer_id(layer)}"
                        if source.kind == "mouse":
                            description = f"Mouse (radius {self._radius:g} mm)"
                            color = base
                        else:
                            assert source.layer is not None and item is not None
                            description = (
                                f"{source.layer.name} | {source.kind.title()} {item}"
                            )
                            hue, lightness, saturation = colorsys.rgb_to_hls(
                                *to_rgb(base)
                            )
                            tone = self._tones.setdefault(
                                (source.id, item), len(self._tones)
                            )
                            lightness = 0.3 + (tone % 5) * 0.1
                            color = to_hex(
                                colorsys.hls_to_rgb(hue, lightness, saturation)
                            )
                        name = f"{layer.name} | {description}"
                        signals.append(
                            LiveSignal(
                                id=sid,
                                name=name,
                                default_name=name,
                                color=color,
                                visible=True,
                                source_type=source.kind,
                                source_id=item,
                                source_key=source.id,
                                reference_layer=layer,
                                backing_layer=source.layer,
                                layer_name=source.layer.name
                                if source.layer is not None
                                else None,
                                enabled=source.enabled and layer in selected,
                                available=(
                                    item in current_labels
                                    if source.kind == "label"
                                    else True
                                ),
                            )
                        )
            self.store.register_live_signals(signals)
        finally:
            self._syncing = False

    def get_trace(
        self, signal_id: str, axis: str = TIME_DIM, reference: Image | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract raw values for an explicitly identified live trace.

        Parameters
        ----------
        signal_id : str
            Stable trace identifier including its reference recording.
        axis : str, default: "time"
            Axis kept as a full trace for Signals display.
        reference : napari.layers.Image, optional
            Explicit reference override for preprocessing ROI confounds.

        Returns
        -------
        x : numpy.ndarray
            Actual axis coordinates.
        y : numpy.ndarray
            Raw trace values.

        Raises
        ------
        ValueError
            If the source is unavailable, outside the recording, or incompatible.
        """
        live = self.store.get_live_signal(signal_id)
        if live is None or not live.available:
            raise ValueError("Selected live source is unavailable.")
        source = self._sources.get(live.source_key)
        if source is None:
            raise ValueError("Selected source is no longer registered.")
        if not source.enabled or (reference is None and not live.enabled):
            raise ValueError("Selected live source is disabled.")
        layer = reference if reference is not None else live.reference_layer
        if not isinstance(layer, Image) or layer not in self.viewer.layers:
            raise ValueError("The seed reference recording was removed.")
        data = layer.metadata["xarray"]
        validate_voxeldata(data)
        if tuple(layer.data.shape) != data.shape:
            raise ValueError(
                "Recording data and VoxelData metadata must have matching shapes."
            )
        if axis not in data.dims or (axis == TIME_DIM and axis not in data.coords):
            raise ValueError(f"Reference recording requires {axis!r} coordinates.")
        axis_index = data.get_axis_num(axis)
        context = tuple(self.viewer.dims.point)
        key = (signal_id, self.layer_id(layer), axis, context)
        if key in self._cache:
            return self._cache[key]
        if live.source_type == "mouse":
            if self._position is None:
                raise ValueError(
                    "Hold Shift over the recording to position the mouse seed."
                )
            y = self._mouse_trace(layer, axis_index)
        elif isinstance(live.backing_layer, Points):
            points = live.backing_layer
            ids = points.features[_POINT_ID].to_numpy()
            indices = np.flatnonzero(ids == live.source_id)
            if indices.size != 1:
                raise ValueError("Selected point no longer exists.")
            position = np.asarray(
                points.data_to_world(points.data[int(indices[0])]), dtype=float
            )
            position = np.pad(position, (max(0, layer.ndim - len(position)), 0))
            y = extract_voxel_trace(layer, position, axis_index)
        elif isinstance(live.backing_layer, Labels):
            labels = live.backing_layer
            spatial_axes = [i for i in range(layer.ndim) if i != axis_index]
            # Equal shapes alone do not establish a shared voxel grid.
            if labels.ndim == layer.ndim:
                label_axes = spatial_axes
            else:
                label_axes = list(range(labels.ndim))
            if len(label_axes) != len(spatial_axes):
                raise ValueError("Labels and recording must share the same voxel grid.")
            origin = np.asarray(labels.data_to_world(np.zeros(labels.ndim)))[label_axes]
            image_origin = np.asarray(layer.data_to_world(np.zeros(layer.ndim)))[
                spatial_axes
            ]
            labels_basis = np.stack(
                [
                    np.asarray(labels.data_to_world(np.eye(labels.ndim)[i]))[label_axes]
                    - origin
                    for i in label_axes
                ]
            )
            image_basis = np.stack(
                [
                    np.asarray(layer.data_to_world(np.eye(layer.ndim)[i]))[spatial_axes]
                    - image_origin
                    for i in spatial_axes
                ]
            )
            if not np.allclose(origin, image_origin) or not np.allclose(
                labels_basis, image_basis
            ):
                raise ValueError("Labels and recording must share the same voxel grid.")
            assert live.source_id is not None
            y = extract_label_mean_trace(
                layer, np.asarray(labels.data), live.source_id, axis_index
            )
        else:
            raise ValueError("Live source has no backing layer.")
        if y is None:
            raise ValueError("No valid data at the selected source position or ROI.")
        result = (np.asarray(data[axis].values), np.asarray(y, dtype=float))
        self._cache = {k: v for k, v in self._cache.items() if k[:3] != key[:3]}
        self._cache[key] = result
        return result

    def _mouse_trace(self, layer: Image, axis_index: int) -> np.ndarray | None:
        """Extract the shared mouse voxel or physical sphere mean.

        Parameters
        ----------
        layer : napari.layers.Image
            Reference recording.
        axis_index : int
            Trace axis.

        Returns
        -------
        numpy.ndarray or None
            Raw trace, or no data outside the recording.

        Raises
        ------
        ValueError
            If a sphere is requested for an unsupported geometry or unit.
        """
        assert self._position is not None
        if self._radius == 0:
            return extract_voxel_trace(layer, self._position, axis_index)
        data = layer.metadata["xarray"]
        if data.dims != (TIME_DIM, *VOXEL_DIMS) or axis_index != 0:
            raise ValueError(
                "Mouse spheres require a time, k, j, i recording without extra dimensions."
            )
        indices = tuple(round(v) for v in layer.world_to_data(self._position))[1:]
        if not all(0 <= v < n for v, n in zip(indices, layer.data.shape[1:])):
            return None
        unit = data.coords["x"].attrs["units"]
        factors = {
            "m": 1000.0,
            "mm": 1.0,
            "um": 0.001,
            "µm": 0.001,
            "micrometer": 0.001,
            "millimeter": 1.0,
            "meter": 1000.0,
        }
        if unit not in factors:
            raise ValueError(
                f"Cannot express a millimetre radius in spatial units {unit!r}."
            )
        affine = np.asarray(get_voxel_to_world_affine(data))[:3, :3] * factors[unit]
        # Inverse-affine row norms bound a physical sphere even for sheared grids.
        bounds = self._radius * np.linalg.norm(np.linalg.inv(affine), axis=1)
        native = [np.asarray(data[dim].values, dtype=float) for dim in VOXEL_DIMS]
        slices = []
        for coordinate, index, bound in zip(native, indices, bounds):
            candidates = np.flatnonzero(np.abs(coordinate - coordinate[index]) <= bound)
            slices.append(slice(int(candidates[0]), int(candidates[-1]) + 1))
        grids = np.meshgrid(
            *(
                coordinate[s] - coordinate[i]
                for coordinate, s, i in zip(native, slices, indices)
            ),
            indexing="ij",
        )
        offsets = np.stack(grids, axis=-1)
        distances = np.linalg.norm(offsets @ affine.T, axis=-1)
        values = np.asarray(layer.data[(slice(None), *slices)], dtype=float)
        return values[:, distances <= self._radius].mean(axis=1)

    def pin_origin(self, signal_id: str, axis: str = TIME_DIM) -> str:
        """Return a snapshot origin including mouse position and radius.

        Parameters
        ----------
        signal_id : str
            Trace identifier.
        axis : str, default: "time"
            Axis represented by the snapshot.

        Returns
        -------
        str
            Snapshot identity.
        """
        live = self.store.get_live_signal(signal_id)
        if (
            live is not None
            and live.source_type == "mouse"
            and self._position is not None
        ):
            assert live.reference_layer is not None
            indices = tuple(
                round(v) for v in live.reference_layer.world_to_data(self._position)
            )
            axis_index = live.reference_layer.metadata["xarray"].get_axis_num(axis)
            context = tuple(v for i, v in enumerate(indices) if i != axis_index)
            return f"{signal_id}:axis={axis}:{context}:radius={self._radius:g}"
        return f"{signal_id}:axis={axis}"

    def _subscribe(self, layer: Layer, disconnect: bool = False) -> None:
        """Connect or disconnect extraction invalidation events.

        Parameters
        ----------
        layer : napari.layers.Layer
            Layer to observe.
        disconnect : bool, default: False
            Whether to disconnect rather than connect.
        """
        for name in (
            "data",
            "scale",
            "translate",
            "rotate",
            "shear",
            "affine",
            "features",
        ):
            emitter = getattr(layer.events, name, None)
            if emitter is not None:
                if disconnect:
                    emitter.disconnect(self._on_data_event)
                else:
                    emitter.connect(self._on_data_event)
        if isinstance(layer, Labels):
            if disconnect:
                layer.events.paint.disconnect(self._on_paint)
            else:
                layer.events.paint.connect(self._on_paint)

    def _layers_changed(self, event=None) -> None:
        """Reconcile additions/removals and release removed layer subscriptions.

        Parameters
        ----------
        event : object, optional
            Napari layer-list event.
        """
        for layer in list(self._subscribed):
            if layer not in self.viewer.layers:
                self._subscribe(layer, disconnect=True)
                self._subscribed.remove(layer)
                self._layers.pop(id(layer), None)
        for layer in self.viewer.layers:
            if layer not in self._subscribed and isinstance(
                layer, (Image, Points, Labels)
            ):
                self._subscribe(layer)
                self._subscribed.append(layer)
        self._sources = {
            sid: s
            for sid, s in self._sources.items()
            if s.layer is None or s.layer in self.viewer.layers
        }
        self._data_changed()
        # Plotting attaches metadata after the layer insertion event.
        QTimer.singleShot(0, self._data_changed)

    def _names_changed(self, event=None) -> None:
        """Update display names without invalidating values or identities.

        Parameters
        ----------
        event : object, optional
            Napari rename event.
        """
        self._reconcile()
        self.sources_changed.emit()

    def _on_data_event(self, event=None) -> None:
        """Invalidate immediately after data or transform edits.

        Parameters
        ----------
        event : object, optional
            Napari layer event.
        """
        if not self._syncing:
            layer = getattr(event, "source", None)
            if isinstance(layer, Image) and layer not in self.references():
                return
            action = getattr(event, "action", None)
            if action in ("changing", "adding", "removing"):
                if isinstance(layer, Points) and action == "changing":
                    self._replacing_points.add(id(layer))
                return
            if isinstance(layer, Points) and id(layer) in self._replacing_points:
                self._replacing_points.remove(id(layer))
                # A bulk data replacement cannot establish which old point survived.
                features = layer.features.copy()
                features[_POINT_ID] = np.arange(
                    self._point_counter, self._point_counter + len(layer.data)
                )
                self._point_counter += len(layer.data)
                self._syncing = True
                try:
                    layer.features = features
                finally:
                    self._syncing = False
            if isinstance(layer, Image):
                lid = self.layer_id(layer)
                self._revisions[lid] = self._revisions.get(lid, 0) + 1
            self._data_changed(layer)

    def revision(self, layer: Image) -> int:
        """Return the recording data/geometry revision for consumer caches.

        Parameters
        ----------
        layer : napari.layers.Image
            Reference or target recording.

        Returns
        -------
        int
            Monotonically increasing revision.
        """
        return self._revisions.get(self.layer_id(layer), 0)

    def _on_paint(self, event=None) -> None:
        """Debounce extraction during a brush stroke.

        Parameters
        ----------
        event : object, optional
            Napari paint event.
        """
        self._cache.clear()
        self._roi_timer.start()

    def _data_changed(self, layer: Layer | None = None) -> None:
        """Invalidate cached values and reconcile source item availability.

        Parameters
        ----------
        layer : napari.layers.Layer, optional
            Changed backing layer, when known.
        """
        if self._closed:
            return
        if isinstance(layer, Image):
            lid = self.layer_id(layer)
            self._cache = {
                key: value for key, value in self._cache.items() if key[1] != lid
            }
        elif layer is not None:
            affected = {
                s.id for s in self.store.live_signals() if s.backing_layer is layer
            }
            self._cache = {
                key: value
                for key, value in self._cache.items()
                if key[0] not in affected
            }
        else:
            self._cache.clear()
        self._reconcile()
        self.sources_changed.emit()
        self.traces_changed.emit(layer)

    def _bind_shift(self) -> None:
        """Install the only ConfUSIus Shift binding while mouse is enabled."""
        if not self._shift_bound:
            self._previous_shift = self.viewer.keymap.get(coerce_keybinding("Shift"))
            self.viewer.bind_key("Shift", self._on_shift_pressed, overwrite=True)
            self._shift_bound = True

    def _unbind_shift(self) -> None:
        """Restore the displaced binding when mouse is disabled or closed."""
        if self._shift_bound:
            self.viewer.bind_key("Shift", self._previous_shift, overwrite=True)
            self._shift_bound = False

    def _on_shift_pressed(self, viewer) -> None:
        """Update at a stationary cursor when Shift is pressed.

        Parameters
        ----------
        viewer : napari.Viewer
            Viewer supplying the cursor.
        """
        self._position = np.asarray(viewer.cursor.position, dtype=float)
        self._mouse_changed()

    def _on_mouse_move(self, viewer, event) -> None:
        """Coalesce Shift-hover movement for both consumers.

        Parameters
        ----------
        viewer : napari.Viewer
            Viewer supplying the cursor.
        event : object
            Mouse event with modifiers.
        """
        if self._sources["mouse"].enabled and "Shift" in event.modifiers:
            self._position = np.asarray(viewer.cursor.position, dtype=float)
            if not self._mouse_timer.isActive():
                self._mouse_timer.start()

    def _mouse_changed(self) -> None:
        """Update changed mouse voxels only, retaining cached ROI means."""
        if self._position is None:
            return
        sample = (
            self._radius,
            tuple(
                (
                    self.layer_id(layer),
                    tuple(round(v) for v in layer.world_to_data(self._position)),
                )
                for layer in self.references()
            ),
        )
        if sample == self._mouse_sample:
            return
        self._mouse_sample = sample
        self._cache = {
            key: value
            for key, value in self._cache.items()
            if not key[0].startswith("mouse-")
        }
        self.traces_changed.emit("mouse")

    def close(self) -> None:
        """Release viewer callbacks without destroying stored snapshots."""
        if self._closed:
            return
        self._closed = True
        self._mouse_timer.stop()
        self._roi_timer.stop()
        self._unbind_shift()
        if self._on_mouse_move in self.viewer.mouse_move_callbacks:
            self.viewer.mouse_move_callbacks.remove(self._on_mouse_move)
        self.viewer.layers.events.inserted.disconnect(self._layers_changed)
        self.viewer.layers.events.removed.disconnect(self._layers_changed)
        self.viewer.layers.events.renamed.disconnect(self._names_changed)
        for layer in self._subscribed:
            self._subscribe(layer, disconnect=True)
        self._subscribed.clear()
