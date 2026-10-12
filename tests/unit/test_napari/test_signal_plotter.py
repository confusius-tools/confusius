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
    viewer, recording, _, _controller, plotter = plot_setup
    kwargs = {"scale": recording.scale[-3:], "translate": recording.translate[-3:]}
    viewer.add_points([[2, 2, 2]], **kwargs)
    labels = np.zeros(recording.data.shape[1:], dtype=np.int32)
    labels[0, 0, :2] = 4
    viewer.add_labels(labels, **kwargs)
    values = {
        line.get_label(): line.get_ydata()
        for line in plotter._axes.lines
        if not line.get_label().startswith("_")
    }
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


def test_mouse_name_color_and_default_display(plot_setup):
    from confusius._utils.colors import RED

    viewer, recording, store, _, plotter = plot_setup
    signal = store.live_signals()[0]
    world = recording.data_to_world((0, 1, 1, 1))
    assert signal.display_name == (
        f"data | z={world[-3]:.2f}, y={world[-2]:.2f}, x={world[-1]:.2f} mm (radius 0 mm)"
    )
    assert signal.color == RED
    assert plotter._vline is not None
    assert any(line.get_visible() for line in plotter._axes.get_xgridlines())
    assert not any(action.text() == "Pin" for action in plotter._toolbar.actions())
    viewer.cursor.position = recording.data_to_world((0, 2, 2, 2))
    viewer.keymap[coerce_keybinding("Shift")](viewer)
    assert store.live_signals()[0].display_name != signal.display_name


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
    viewer, recording, store, controller, plotter = plot_setup
    seed = store.live_signals()[0]
    plotter.close()
    assert viewer.keymap.get(coerce_keybinding("Shift")) is not None
    assert_allclose(controller.get_trace(seed.id)[1], recording.data[:, 1, 1, 1])


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
    store.live_sources(viewer).close()


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        (None, [-50, 0, 50, np.nan]),
        ([(10, 10)], [0, 100, 200, np.nan]),
        ([(10, 11), (11, 12)], [-50, 0, 50, np.nan]),
        ([(20, 30)], [np.nan] * 4),
        ([(13, 13)], [np.nan] * 4),
    ],
)
def test_psc_baselines_display_export_and_preserve_raw(plot_setup, intervals, expected):
    _, _, store, controller, plotter = plot_setup
    controller.set_enabled("mouse", False)
    raw = np.array([2.0, 4.0, 6.0, np.nan])
    store.pin_signal("psc", "psc", np.arange(10, 14), raw, "#123456", "test")
    plotter.set_standardization("psc", intervals)
    assert_allclose(plotter._axes.lines[0].get_ydata(), expected)
    assert_allclose(plotter._export_signals[0].y_values, expected)
    assert_allclose(store.stored_signals()[0].y, raw)
    assert plotter._axes.get_ylabel() == "Signal change (%)"
    if not np.isfinite(expected).any():
        assert "PSC undefined" in plotter._baseline_warning.get_text()
    else:
        assert plotter._baseline_warning is None
    assert plotter._baseline_spans


@pytest.mark.parametrize("raw", [[0.0, 0.0], [-2.0, -4.0], [np.nan, np.inf]])
def test_psc_zero_negative_and_nonfinite_baselines(plot_setup, raw):
    _, _, store, controller, plotter = plot_setup
    controller.set_enabled("mouse", False)
    store.pin_signal("psc", "psc", np.arange(2), np.array(raw), "#123456", "test")
    plotter.set_standardization("psc")
    expected = [100 / 3, -100 / 3] if raw[0] == -2 else [np.nan, np.nan]
    assert_allclose(plotter._axes.lines[0].get_ydata(), expected)


@pytest.mark.parametrize("intervals", [[(2, 1)], [(np.nan, 2)], [(0, np.inf)]])
def test_invalid_psc_intervals_are_rejected(plot_setup, intervals):
    plotter = plot_setup[-1]
    with pytest.raises(ValueError, match="finite ordered"):
        plotter.set_standardization("psc", intervals)


@pytest.mark.parametrize("outside", [False, True])
def test_event_baselines_use_half_open_union_or_its_complement(plot_setup, outside):
    _, _, store, controller, plotter = plot_setup
    controller.set_enabled("mouse", False)
    raw = np.arange(2.0, 14.0, 2.0)
    store.pin_signal("events", "events", np.arange(6), raw, "#123456", "test")
    plotter.set_standardization(
        "psc", [(1.0, 3.0), (2.0, 4.0)], baseline_source="events", outside=outside
    )
    mean = raw[[0, 4, 5]].mean() if outside else raw[1:4].mean()
    expected = (raw - mean) / mean * 100
    assert_allclose(plotter._axes.lines[0].get_ydata(), expected)
    assert_allclose(plotter._export_signals[0].y_values, expected)
    spans = plotter._baseline_spans
    bounds = [(span.get_x(), span.get_x() + span.get_width()) for span in spans]
    assert bounds == ([(0.0, 1.0), (4.0, 5.0)] if outside else [(1.0, 4.0)])
    assert_allclose(store.stored_signals()[0].y, raw)


@pytest.mark.parametrize("outside", [False, True])
def test_empty_event_selection_is_explicitly_undefined(plot_setup, outside):
    _, _, _, _, plotter = plot_setup
    plotter.set_standardization("psc", [], baseline_source="events", outside=outside)
    assert not np.isfinite(plotter._axes.lines[0].get_ydata()).any()
    assert "selected periods" in plotter._baseline_warning.get_text()
    assert not plotter._baseline_spans


def test_event_baseline_rejects_non_seconds_time_coordinates(plot_setup):
    _, recording, _, _, plotter = plot_setup
    recording.metadata["xarray"].time.attrs["units"] = "ms"
    plotter.set_standardization("psc", [(10.0, 12.0)], baseline_source="events")
    assert not np.isfinite(plotter._axes.lines[0].get_ydata()).any()
    assert "seconds time axis" in plotter._baseline_warning.get_text()


def test_disabling_grid_does_not_warn_or_enable_it(plot_setup):
    plotter = plot_setup[-1]
    plotter.set_show_grid(False)
    assert not any(line.get_visible() for line in plotter._axes.get_xgridlines())


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
