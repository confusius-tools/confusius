"""Manage a napari text overlay that displays the current time coordinate value."""

from __future__ import annotations

from typing import TYPE_CHECKING

import napari
import napari.layers

from confusius._napari._timeaxis import (
    find_time_dim_index,
    read_time_units,
    read_time_value,
)

if TYPE_CHECKING:
    from napari.layers import Layer

    from confusius._napari._events._store import EventStore


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
        self._ref_layer: Layer | None = None
        self._event_store: EventStore | None = None

        viewer.layers.events.inserted.connect(self.check)
        viewer.layers.events.removed.connect(self._on_layer_removed)
        viewer.dims.events.current_step.connect(self.update)
        viewer.dims.events.ndisplay.connect(self.check)
        viewer.dims.events.axis_labels.connect(self.check)
        viewer.layers.selection.events.changed.connect(self._on_selection_changed)

    # -- helpers ----------------------------------------------------------

    def _find_time_dim_index(self) -> int | None:
        """Return the viewer axis index for "time", or `None` if absent."""
        return find_time_dim_index(self._viewer)

    def _read_time_units(self) -> str | None:
        """Read time units from the reference layer's metadata."""
        return read_time_units(self._ref_layer)

    def _read_time_value(self) -> float | None:
        """Read the actual time coordinate from the reference layer."""
        return read_time_value(self._ref_layer, self._viewer)

    # -- events -----------------------------------------------------------

    def set_event_store(self, store: EventStore | None) -> None:
        """Attach an event store whose active events annotate the overlay.

        Parameters
        ----------
        store : EventStore, optional
            Store of temporal events. When provided, the trial types of the
            events active at the current time are appended to the overlay text.
        """
        self._event_store = store
        self.update()

    def _event_status(self, time_val: float) -> str:
        """Return a suffix naming the events active at *time_val*.

        Parameters
        ----------
        time_val : float
            Current time coordinate value.

        Returns
        -------
        str
            A leading separator and comma-separated trial types of the active
            events, or an empty string when no events are active or the store is
            disabled.
        """
        store = self._event_store
        if store is None or not store.show_in_overlay:
            return ""
        active = store.active_events(time_val)
        if not active:
            return ""
        # Preserve order while removing duplicate trial types.
        names = list(dict.fromkeys(event.trial_type for event in active))
        return "  ●  " + ", ".join(names)

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
        base = f"{time_val:.2f} {self._units if self._units else ''}".rstrip()
        self._viewer.text_overlay.text = base + self._event_status(time_val)
        self._viewer.text_overlay.visible = True
