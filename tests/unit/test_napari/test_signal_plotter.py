"""Signals rendering tests using public source/store APIs and observable lines."""

import numpy as np
import pytest
from napari.utils.key_bindings import coerce_keybinding
from numpy.testing import assert_allclose

from confusius._napari._signals._plotter import SignalPlotter
from confusius._napari._signals._store import SignalStore
from confusius.plotting.napari import plot_napari


@pytest.fixture
def plot_setup(make_napari_viewer, sample_voxeldata_3dt, qtbot):
    """Create a plot consuming a shared recording source."""
    viewer = make_napari_viewer()
    _, recording = plot_napari(
        sample_voxeldata_3dt,
        viewer=viewer,
        name="data",
        show_colorbar=False,
        show_scale_bar=False,
    )
    store = SignalStore()
    controller = store.live_sources(viewer)
    plotter = SignalPlotter(viewer, store)
    qtbot.addWidget(plotter)
    viewer.cursor.position = recording.data_to_world((0, 1, 1, 1))
    viewer.keymap[coerce_keybinding("Shift")](viewer)
    yield viewer, recording, store, controller, plotter
    plotter.close()
    controller.close()


def test_plot_shares_mouse_trace_and_radius(plot_setup):
    _, recording, store, controller, plotter = plot_setup
    seed = store.live_signals()[0]
    line = plotter._axes.lines[0]
    assert_allclose(line.get_xdata(), recording.metadata["xarray"].time)
    assert_allclose(line.get_ydata(), recording.data[:, 1, 1, 1])
    controller.set_radius(0.11)
    assert_allclose(
        plotter._axes.lines[0].get_ydata(), controller.get_trace(seed.id)[1]
    )


def test_mouse_point_and_label_traces_render_together(plot_setup):
    viewer, recording, _, controller, plotter = plot_setup
    kwargs = {"scale": recording.scale[-3:], "translate": recording.translate[-3:]}
    points = viewer.add_points([[2, 2, 2]], **kwargs)
    labels = np.zeros(recording.data.shape[1:], dtype=np.int32)
    labels[0, 0, :2] = 4
    roi = viewer.add_labels(labels, **kwargs)
    controller.add_source(points)
    controller.add_source(roi)
    values = {line.get_label(): line.get_ydata() for line in plotter._axes.lines}
    assert len(values) == 3
    for name, y in values.items():
        expected = (
            recording.data[:, 2, 2, 2]
            if "Point" in name
            else recording.data[:, 0, 0, :2].mean(axis=1)
            if "Label" in name
            else recording.data[:, 1, 1, 1]
        )
        assert_allclose(y, expected)


def test_pin_is_raw_snapshot_and_survives_source_disable(plot_setup):
    _, recording, store, controller, plotter = plot_setup
    plotter.set_zscore(True)
    plotter._pin_button.click()
    pinned = store.stored_signals()[0]
    assert_allclose(pinned.y, recording.data[:, 1, 1, 1])
    controller.set_enabled("mouse", False)
    assert len(plotter._axes.lines) == 1
    assert_allclose(
        plotter._axes.lines[0].get_ydata(),
        (pinned.y - pinned.y.mean()) / pinned.y.std(),
    )
    assert not plotter._pin_button.isEnabled()
    assert plotter._export_button.isEnabled()


def test_pin_distinguishes_recording_position_and_radius(plot_setup):
    viewer, recording, store, controller, plotter = plot_setup
    plotter._pin_button.click()
    plotter._pin_button.click()
    assert len(store.stored_signals()) == 1
    controller.set_radius(0.11)
    plotter._pin_button.click()
    viewer.cursor.position = recording.data_to_world((0, 2, 2, 2))
    viewer.keymap[coerce_keybinding("Shift")](viewer)
    plotter._pin_button.click()
    assert len({s.pin_origin for s in store.stored_signals()}) == 3


def test_stored_visibility_color_and_rename_are_preserved(plot_setup):
    _, recording, store, controller, plotter = plot_setup
    times = recording.metadata["xarray"].time.values
    signal = store.pin_signal(
        "stored", "external", times, np.arange(len(times)), "#123456", "test"
    )
    controller.set_enabled("mouse", False)
    store.rename_signal(signal.id, "renamed")
    line = plotter._axes.lines[0]
    assert line.get_label() == "renamed"
    assert line.get_color() == "#123456"
    assert_allclose(line.get_ydata(), signal.y)
    store.set_signal_visible(signal.id, False)
    assert len(plotter._axes.lines) == 0
    assert not plotter._export_button.isEnabled()


def test_mouse_updates_preserve_user_zoom_and_manual_ylim(plot_setup):
    viewer, recording, _, _, plotter = plot_setup
    plotter._axes.set_xlim(10.5, 11.5)
    plotter.set_autoscale(False)
    plotter.set_ylim(-3, 3)
    viewer.cursor.position = recording.data_to_world((0, 2, 2, 2))
    viewer.keymap[coerce_keybinding("Shift")](viewer)
    assert_allclose(plotter._axes.get_xlim(), (10.5, 11.5))
    assert_allclose(plotter.get_ylim(), (-3, 3))


def test_plot_close_does_not_remove_shared_sources_or_shift(plot_setup):
    viewer, _, store, controller, plotter = plot_setup
    seed = store.live_signals()[0]
    plotter.close()
    assert viewer.keymap.get(coerce_keybinding("Shift")) is not None
    assert_allclose(controller.get_trace(seed.id)[1], plotter._live_plot_signals[0].y)


def test_zscore_constant_and_missing_values_are_display_only(plot_setup):
    _, recording, store, controller, plotter = plot_setup
    controller.set_enabled("mouse", False)
    y = np.full(recording.data.shape[0], 3.0)
    y[2] = np.nan
    signal = store.pin_signal(
        "constant",
        "constant",
        recording.metadata["xarray"].time.values,
        y,
        "#123456",
        "test",
    )
    plotter.set_zscore(True)
    expected = np.zeros_like(y)
    expected[2] = np.nan
    assert_allclose(plotter._axes.lines[0].get_ydata(), expected)
    assert_allclose(store.stored_signals()[0].y, signal.y)


def test_axis_cursor_uses_coordinate_values(plot_setup):
    _, _, _, _, plotter = plot_setup
    plotter.set_show_cursor(True)
    plotter.set_xaxis_cursor(11.25)
    plotter._canvas.draw()
    assert_allclose(plotter._vline.get_xdata(), (11.25, 11.25))


def test_non_time_volume_axis_remains_supported(
    make_napari_viewer, sample_voxeldata_3d, qtbot
):
    """The lifecycle refactor must preserve plotting a volume slider axis."""
    viewer = make_napari_viewer()
    _, recording = plot_napari(
        sample_voxeldata_3d, viewer=viewer, show_colorbar=False, show_scale_bar=False
    )
    store = SignalStore()
    plotter = SignalPlotter(viewer, store)
    qtbot.addWidget(plotter)
    plotter.set_xaxis_dim("k")
    viewer.cursor.position = recording.data_to_world((1, 1, 1))
    viewer.keymap[coerce_keybinding("Shift")](viewer)
    assert_allclose(
        np.asarray(plotter._axes.lines[0].get_ydata(), dtype=float),
        np.asarray(recording.data, dtype=float)[:, 1, 1],
    )
    plotter._pin_button.click()
    assert store.stored_signals()[0].coordinate_dim == "k"
    store.live_sources(viewer).close()


def test_reopened_plot_resumes_shared_live_updates(plot_setup):
    """Closing the plot suspends rendering, not the live source lifecycle."""
    _, _, store, controller, plotter = plot_setup
    seed = store.live_signals()[0]
    plotter.close()
    controller.set_radius(0.11)
    plotter.show()
    assert_allclose(
        plotter._axes.lines[0].get_ydata(), controller.get_trace(seed.id)[1]
    )
