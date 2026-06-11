"""Integration tests for temporal events with the plotter and time overlay."""

from __future__ import annotations

import numpy as np
import xarray as xr

from confusius._napari._events._store import EventStore
from confusius._napari._signals._plotter import SignalPlotter
from confusius._napari._time_overlay import _TimeOverlay
from confusius.plotting import plot_napari


def _make_4d_da(rng):
    """Create a minimal 4D DataArray with a time coordinate in seconds."""
    shape = (5, 4, 6, 8)
    data = rng.random(shape).astype(np.float32)
    return xr.DataArray(
        data,
        dims=["time", "z", "y", "x"],
        coords={
            "time": xr.DataArray(
                np.arange(5) * 1.0, dims=["time"], attrs={"units": "s"}
            ),
            "z": xr.DataArray(np.arange(4) * 0.2, dims=["z"]),
            "y": xr.DataArray(np.arange(6) * 0.1, dims=["y"]),
            "x": xr.DataArray(np.arange(8) * 0.05, dims=["x"]),
        },
    )


def test_plotter_shades_events(rng, make_napari_viewer):
    """The plotter draws one background band per event over the signal plot."""
    viewer = make_napari_viewer()
    _, layer = plot_napari(
        _make_4d_da(rng), viewer=viewer, show_colorbar=False, show_scale_bar=False
    )
    store = EventStore()
    store.add_event(1.0, 2.0, "stim")

    plotter = SignalPlotter(viewer, event_store=store)
    # Drive a mouse-mode plot so the axes have data lines.
    plotter._current_layer = layer
    plotter._cursor_pos = np.array(viewer.dims.point)
    plotter._update_plot()

    assert len(plotter._event_spans) == 1

    # Disabling shading removes the band on the next store change.
    store.set_shade_signals(False)
    assert plotter._event_spans == []


def test_overlay_names_active_event(rng, make_napari_viewer):
    """The time overlay appends the active event's trial type."""
    viewer = make_napari_viewer()
    plot_napari(
        _make_4d_da(rng), viewer=viewer, show_colorbar=False, show_scale_bar=False
    )
    store = EventStore()
    store.add_event(1.0, 2.0, "stim")  # active over [1, 3) seconds

    overlay = _TimeOverlay(viewer)
    overlay.set_event_store(store)
    overlay.check()

    # Move to time = 2.0 s (frame 2), which is inside the event.
    viewer.dims.set_current_step(overlay._time_idx, 2)
    assert "stim" in viewer.text_overlay.text

    # Move to time = 4.0 s (frame 4), outside the event.
    viewer.dims.set_current_step(overlay._time_idx, 4)
    assert "stim" not in viewer.text_overlay.text
