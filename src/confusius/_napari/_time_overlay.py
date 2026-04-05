"""Manage a napari text overlay that displays the current time coordinate value."""

from __future__ import annotations

import napari
import napari.layers
import numpy as np


class _TimeOverlay:
    """Manage the viewer text overlay that displays the current time coordinate.

    The overlay reads the time value from a *reference layer* so that non-uniform
    coordinates and multi-recording setups (different time origins) are handled
    correctly.  The reference layer is resolved as follows:

    * Starts as `None`; on activation the first layer whose `axis_labels`
      contain `"time"` is used.
    * When the user selects exactly one layer that has a `"time"` axis, that
      layer becomes the new reference.  Selecting zero or multiple time-aware
      layers leaves the reference unchanged.
    * If the reference layer is removed, the reference resets to `None` and a
      new one is picked on the next activation cycle.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        self._viewer = viewer
        self._active: bool = False
        self._time_idx: int | None = None
        self._units: str | None = None
        self._ref_layer: napari.layers.Layer | None = None

        viewer.layers.events.inserted.connect(self.check)
        viewer.layers.events.removed.connect(self._on_layer_removed)
        viewer.dims.events.current_step.connect(self.update)
        viewer.dims.events.ndisplay.connect(self.check)
        viewer.dims.events.axis_labels.connect(self.check)
        viewer.layers.selection.events.changed.connect(self._on_selection_changed)

    # -- helpers ----------------------------------------------------------

    def _find_time_dim_index(self) -> int | None:
        """Return the viewer axis index for "time", or `None` if absent.

        napari does not propagate `axis_labels` from layers to
        `viewer.dims`, so we inspect each layer's labels directly and
        map the layer-local index to the viewer axis (layers are
        right-aligned in the viewer dims).
        """
        for layer in self._viewer.layers:
            labels = layer.axis_labels
            if "time" in labels:
                layer_idx = list(labels).index("time")
                offset = self._viewer.dims.ndim - layer.ndim
                return offset + layer_idx
        return None

    def _read_time_units(self) -> str | None:
        """Read time units from the reference layer's metadata."""
        if self._ref_layer is not None:
            da = self._ref_layer.metadata.get("xarray")
            if da is not None and "time" in da.coords:
                return da.coords["time"].attrs.get("units", "s")
            # Fallback for non-xarray layers (e.g., video).
            return self._ref_layer.metadata.get("time_units")
        return None

    def _read_time_value(self) -> float | None:
        """Read the actual time coordinate from the reference layer.

        Maps the viewer's world coordinate to the layer's data index via
        `world_to_data` so that layers with different time origins or
        scales are resolved correctly.  The data index is then used to
        look up the true xarray coordinate, avoiding napari's linear
        scale/translate approximation for non-uniform spacing.

        For layers without xarray metadata (e.g., video layers), returns
        `None` so that the caller falls back to ``dims.point`` which is
        correct as long as the layer's time scale is set properly.
        """
        if self._ref_layer is None:
            return None
        da = self._ref_layer.metadata.get("xarray")
        if da is None or "time" not in da.coords:
            return None

        # Map the viewer world coordinate to this layer's data space.
        world_point = np.array(self._viewer.dims.point)
        offset = self._viewer.dims.ndim - self._ref_layer.ndim
        layer_world_point = world_point[offset:]
        data_point = self._ref_layer.world_to_data(layer_world_point)

        time_local_idx = list(da.dims).index("time")
        step = int(np.round(data_point[time_local_idx]))

        coords = da.coords["time"].values
        if 0 <= step < len(coords):
            return float(coords[step])
        return None

    # -- lifecycle --------------------------------------------------------

    def _activate(self) -> None:
        """Cache time axis index, units, and configure overlay appearance."""
        self._time_idx = self._find_time_dim_index()

        # Pick a default reference layer when none is set.
        if self._ref_layer is None:
            for layer in self._viewer.layers:
                if "time" in layer.axis_labels:
                    self._ref_layer = layer
                    break

        self._units = self._read_time_units()

        overlay = self._viewer.text_overlay
        overlay.position = "bottom_left"
        overlay.font_size = 14
        overlay.color = "white"
        overlay.opacity = 0.6
        self._active = True

    def _deactivate(self) -> None:
        """Hide the overlay and clear cached state."""
        self._viewer.text_overlay.visible = False
        self._viewer.text_overlay.text = ""
        self._active = False
        self._time_idx = None

    # -- public event handlers --------------------------------------------

    def _on_layer_removed(self, event=None) -> None:
        """Reset reference layer if it was removed, then re-check."""
        if event is not None and event.value is self._ref_layer:
            self._ref_layer = None
        self.check()

    def _on_selection_changed(self) -> None:
        """Update the reference layer from the current selection.

        If exactly one selected layer has a `"time"` axis it becomes the new reference.
        Zero or multiple time-aware selections leave the reference unchanged.
        """
        selected_with_time = [
            layer
            for layer in self._viewer.layers.selection
            if "time" in layer.axis_labels
        ]
        if len(selected_with_time) == 1:
            self._ref_layer = selected_with_time[0]
            self._units = self._read_time_units()
            self.update()

    def check(self) -> None:
        """Activate or deactivate the overlay based on current dims."""
        time_idx = self._find_time_dim_index()
        is_sliced = time_idx is not None and time_idx not in self._viewer.dims.displayed

        if is_sliced:
            # Re-activate to refresh cached index/units (layers may have changed).
            self._activate()
            self.update()
        elif self._active:
            self._deactivate()

    def update(self) -> None:
        """Set the overlay text to the current time value."""
        if not self._active or self._time_idx is None:
            return
        time_val = self._read_time_value()
        if time_val is None:
            # Fall back to napari's linear approximation when no xarray metadata is
            # available.
            time_val = float(self._viewer.dims.point[self._time_idx])
        self._viewer.text_overlay.text = (
            f"{time_val:.2f} {self._units if self._units else ''}"
        )
        self._viewer.text_overlay.visible = True
