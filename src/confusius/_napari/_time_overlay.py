"""Manage a napari text overlay that displays the current time coordinate value."""

from __future__ import annotations

import napari


class _TimeOverlay:
    """Manage the viewer text overlay that displays the current time coordinate.

    Caches the time axis index and units on activation so that per-step updates
    are cheap.

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

        viewer.layers.events.inserted.connect(self.check)
        viewer.layers.events.removed.connect(self.check)
        viewer.dims.events.current_step.connect(self.update)
        viewer.dims.events.ndisplay.connect(self.check)
        viewer.dims.events.axis_labels.connect(self.check)

    # -- helpers ----------------------------------------------------------

    def _find_time_dim_index(self) -> int | None:
        """Return the viewer axis index for "time", or ``None`` if absent.

        napari does not propagate ``axis_labels`` from layers to
        ``viewer.dims``, so we inspect each layer's labels directly and
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
        """Read time units from the first layer carrying xarray metadata."""
        for layer in self._viewer.layers:
            da = layer.metadata.get("xarray")
            if da is not None and "time" in da.coords:
                return da.coords["time"].attrs.get("units", "s")
        return None

    def _read_time_value(self, step: int) -> float | None:
        """Read the actual time coordinate value for a given step index.

        Uses xarray metadata so non-uniform coordinates are resolved correctly
        instead of relying on napari's linear scale/translate approximation.
        """
        for layer in self._viewer.layers:
            da = layer.metadata.get("xarray")
            if da is not None and "time" in da.coords:
                coords = da.coords["time"].values
                if 0 <= step < len(coords):
                    return float(coords[step])
        return None

    # -- lifecycle --------------------------------------------------------

    def _activate(self) -> None:
        """Cache time axis index, units, and configure overlay appearance."""
        self._time_idx = self._find_time_dim_index()
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

    def check(self, event=None) -> None:
        """Activate or deactivate the overlay based on current dims."""
        time_idx = self._find_time_dim_index()
        is_sliced = time_idx is not None and time_idx not in self._viewer.dims.displayed

        if is_sliced:
            # Re-activate to refresh cached index/units (layers may have changed).
            self._activate()
            self.update()
        elif self._active:
            self._deactivate()

    def update(self, event=None) -> None:
        """Set the overlay text to the current time value."""
        if not self._active or self._time_idx is None:
            return
        step = int(self._viewer.dims.current_step[self._time_idx])
        time_val = self._read_time_value(step)
        if time_val is None:
            # Fall back to napari's linear approximation when no xarray
            # metadata is available.
            time_val = float(self._viewer.dims.point[self._time_idx])
        self._viewer.text_overlay.text = (
            f"{time_val:.2f} {self._units if self._units else ''}"
        )
        self._viewer.text_overlay.visible = True
