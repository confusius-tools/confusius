"""Integration tests for temporal events with the plotter and time overlay."""

from __future__ import annotations

import numpy as np
import xarray as xr
from napari.utils.key_bindings import coerce_keybinding

from confusius._napari._events._panel import EventPanel
from confusius._napari._events._store import EventStore
from confusius._napari._signals._plotter import SignalPlotter
from confusius._napari._time_overlay import _TimeOverlay
from confusius.plotting import plot_napari


def _make_4d_da(rng, time_attrs=None):
    """Create a minimal 4D DataArray with a time coordinate in seconds."""
    shape = (5, 4, 6, 8)
    data = rng.random(shape).astype(np.float32)
    return xr.DataArray(
        data,
        dims=["time", "z", "y", "x"],
        coords={
            "time": xr.DataArray(
                np.arange(5) * 1.0,
                dims=["time"],
                attrs={"units": "s", **(time_attrs or {})},
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
    assert "stim [1.00 s, 3.00 s)" in viewer.text_overlay.text

    # Move to time = 4.0 s (frame 4), outside the event.
    viewer.dims.set_current_step(overlay._time_idx, 4)
    assert "stim" not in viewer.text_overlay.text


def test_overlay_snaps_unsampled_event_to_next_frame(rng, make_napari_viewer):
    """An event no frame samples appears on the next frame's overlay."""
    viewer = make_napari_viewer()
    plot_napari(
        _make_4d_da(rng), viewer=viewer, show_colorbar=False, show_scale_bar=False
    )
    store = EventStore()
    store.add_event(1.2, 0.1, "stim")  # entirely between frames at 1.0 and 2.0
    store.add_event(-0.5, 0.0, "blip")  # before the recording
    store.add_event(1.0, 0.0, "tick")  # exactly at frame 1's timestamp

    overlay = _TimeOverlay(viewer)
    overlay.set_event_store(store)
    overlay.check()

    viewer.dims.set_current_step(overlay._time_idx, 0)
    assert "blip [-0.50 s, -0.50 s)" in viewer.text_overlay.text

    viewer.dims.set_current_step(overlay._time_idx, 1)
    assert "stim" not in viewer.text_overlay.text
    assert "tick" in viewer.text_overlay.text

    viewer.dims.set_current_step(overlay._time_idx, 2)
    assert "stim [1.20 s, 1.30 s)" in viewer.text_overlay.text
    # Regression: an event at exactly a frame's timestamp must not be repeated
    # on the following frame.
    assert "tick" not in viewer.text_overlay.text

    viewer.dims.set_current_step(overlay._time_idx, 3)
    assert "stim" not in viewer.text_overlay.text


def test_overlay_uses_acquisition_window(rng, make_napari_viewer):
    """Events overlapping the frame's acquisition window are reported on it."""
    viewer = make_napari_viewer()
    da = _make_4d_da(
        rng,
        time_attrs={
            "volume_acquisition_reference": "start",
            "volume_acquisition_duration": 0.8,
        },
    )
    plot_napari(da, viewer=viewer, show_colorbar=False, show_scale_bar=False)
    store = EventStore()
    store.add_event(2.5, 0.1, "stim")  # inside frame 2's window [2.0, 2.8)
    store.add_event(2.85, 0.05, "gap")  # in the dead time between frames 2 and 3

    overlay = _TimeOverlay(viewer)
    overlay.set_event_store(store)
    overlay.check()

    viewer.dims.set_current_step(overlay._time_idx, 2)
    assert "stim" in viewer.text_overlay.text
    assert "gap" not in viewer.text_overlay.text

    viewer.dims.set_current_step(overlay._time_idx, 3)
    assert "gap [2.85 s, 2.90 s)" in viewer.text_overlay.text
    assert "stim" not in viewer.text_overlay.text


def test_overlay_window_falls_back_to_time_spacing(rng, make_napari_viewer):
    """With a reference but no duration, the frame spacing tiles the windows.

    A short event between two frame timestamps is then attributed to the frame
    whose acquisition covers it, not snapped to the next frame.
    """
    viewer = make_napari_viewer()
    da = _make_4d_da(rng, time_attrs={"volume_acquisition_reference": "start"})
    plot_napari(da, viewer=viewer, show_colorbar=False, show_scale_bar=False)
    store = EventStore()
    store.add_event(1.5, 0.0, "dirac")  # between frames 1 and 2

    overlay = _TimeOverlay(viewer)
    overlay.set_event_store(store)
    overlay.check()

    # Frames are 1 s apart, so frame 1's window is [1.0, 2.0) and covers the event.
    viewer.dims.set_current_step(overlay._time_idx, 1)
    assert "dirac [1.50 s, 1.50 s)" in viewer.text_overlay.text

    viewer.dims.set_current_step(overlay._time_idx, 2)
    assert "dirac" not in viewer.text_overlay.text


def test_panel_binds_keys_while_visible(make_napari_viewer):
    """Showing the panel binds S/E/Escape in the viewer keymap; hiding removes them."""
    viewer = make_napari_viewer()
    panel = EventPanel(viewer, EventStore())
    keys = [coerce_keybinding(key) for key in ("S", "E", "Escape")]

    panel.show()
    assert all(key in viewer.keymap for key in keys)

    panel.hide()
    assert all(key not in viewer.keymap for key in keys)
