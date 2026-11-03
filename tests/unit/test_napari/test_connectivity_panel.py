"""Numerical regressions for explicit stored and live FC seed selection."""

import numpy as np
import pytest
from napari.utils.key_bindings import coerce_keybinding
from numpy.testing import assert_allclose

from confusius._napari._connectivity._panel import FunctionalConnectivityPanel
from confusius._napari._signals._store import SignalStore
from confusius.plotting.napari import plot_napari


@pytest.fixture
def setup_fc(make_napari_viewer, sample_voxeldata_3dt, qtbot):
    """Create FC controls backed by shared live sources."""
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
    panel = FunctionalConnectivityPanel(viewer, store)
    qtbot.addWidget(panel)
    panel._data_combo.setCurrentIndex(panel._data_combo.findData(recording))
    yield viewer, recording, store, controller, panel
    panel.close()
    controller.close()


def expected_map(values, seed):
    """Compute reference voxel-wise Pearson correlations with NumPy."""
    flat = values.reshape(len(seed), -1)
    return np.array([np.corrcoef(seed, column)[0, 1] for column in flat.T]).reshape(
        values.shape[1:]
    )


def choose_seed(panel, sid):
    """Explicitly select a seed and press Compute."""
    panel._seed_combo.setCurrentIndex(panel._seed_combo.findData(sid))
    panel._compute_btn.click()


def test_no_implicit_seed_and_labels_cannot_be_targets(setup_fc):
    viewer, recording, store, _controller, panel = setup_fc
    viewer.add_labels(
        np.zeros(recording.data.shape[1:], dtype=np.int32), metadata=recording.metadata
    )
    assert panel._seed_combo.currentData() is None
    assert not panel._compute_btn.isEnabled()
    assert [
        panel._data_combo.itemData(i) for i in range(1, panel._data_combo.count())
    ] == [recording]
    assert len(store.live_signals()) == 1


def test_stored_seed_map_and_time_window_use_actual_coordinates(setup_fc, qtbot):
    _, recording, store, _, panel = setup_fc
    seed = recording.data[:, 0, 0, 0]
    signal = store.pin_signal(
        "seed",
        "seed",
        recording.metadata["xarray"].time.values,
        seed,
        "#ffffff",
        "test",
    )
    choose_seed(panel, signal.id)
    assert_allclose(panel._map_layer.data, expected_map(recording.data, seed))
    panel._time_slider.setValue((10.5, 11.5))
    qtbot.wait(40)
    assert_allclose(panel._map_layer.data, expected_map(recording.data[1:4], seed[1:4]))
    assert panel._time_start_label.text() == "Start: 10.5 s"
    assert panel._time_end_label.text() == "End: 11.5 s"


def test_equal_sample_counts_do_not_bypass_time_alignment(setup_fc):
    _, recording, store, _, panel = setup_fc
    times = recording.metadata["xarray"].time.values
    seed = np.arange(len(times), dtype=float) ** 2
    signal = store.pin_signal(
        "shifted", "shifted", times - 0.25, seed, "#ffffff", "test"
    )
    panel._time_slider.setValue((times[0], times[-2]))
    choose_seed(panel, signal.id)
    aligned = np.interp(times[:-1], times - 0.25, seed)
    assert_allclose(panel._map_layer.data, expected_map(recording.data[:-1], aligned))
    panel._time_slider.setValue((times[0], times[-1]))
    panel._compute_btn.click()
    assert panel._map_layer.metadata["seed_stale"]
    assert "extrapolation" in panel._status.text()


def test_mouse_map_creation_preserves_focus_and_signal_axis(setup_fc, qtbot):
    """Creating a derived map never steals focus or relabels a time trace."""
    from confusius._napari._signals._panel import SignalPanel

    viewer, recording, store, _, panel = setup_fc
    signals = SignalPanel(viewer, signal_store=store)
    qtbot.addWidget(signals)
    signals.show_plot()
    assert signals._plotter is not None
    viewer.cursor.position = recording.data_to_world((0, 1, 1, 1))
    viewer.keymap[coerce_keybinding("Shift")](viewer)
    selected = set(viewer.layers.selection)
    seed = store.live_signals()[0]
    choose_seed(panel, seed.id)
    assert panel._map_layer is not None
    assert viewer.layers.selection.active is recording
    assert set(viewer.layers.selection) == selected
    qtbot.wait(30)
    assert signals._xaxis_combo.currentText() == "time"
    assert_allclose(
        np.asarray(signals._plotter._axes.lines[0].get_ydata(), dtype=float),
        np.asarray(recording.data)[:, 1, 1, 1],
    )
    signals._plotter.close()


def test_visibility_never_changes_selected_live_seed(setup_fc, qtbot):
    viewer, recording, store, controller, panel = setup_fc
    viewer.cursor.position = recording.data_to_world((0, 1, 1, 1))
    viewer.keymap[coerce_keybinding("Shift")](viewer)
    seed = store.live_signals()[0]
    choose_seed(panel, seed.id)
    expected = expected_map(recording.data, recording.data[:, 1, 1, 1])
    store.set_live_signal_visible(seed.id, False)
    recording.visible = False
    qtbot.wait(40)
    assert panel._seed_combo.currentData() == seed.id
    assert_allclose(panel._map_layer.data, expected)
    controller.set_radius(0.11)
    qtbot.wait(40)
    assert_allclose(
        panel._map_layer.data,
        expected_map(recording.data, controller.get_trace(seed.id)[1]),
    )


def test_label_seed_automatically_updates_after_painting(setup_fc, qtbot):
    viewer, recording, store, controller, panel = setup_fc
    labels = np.zeros(recording.data.shape[1:], dtype=np.int32)
    labels[0, 0, :2] = 5
    layer = viewer.add_labels(
        labels, scale=recording.scale[-3:], translate=recording.translate[-3:]
    )
    controller.add_source(layer)
    seed = next(s for s in store.live_signals() if s.source_type == "label")
    choose_seed(panel, seed.id)
    assert_allclose(
        panel._map_layer.data,
        expected_map(recording.data, recording.data[:, 0, 0, :2].mean(axis=1)),
    )
    layer.data[:] = 0
    layer.data[1, 1, :3] = 5
    layer.events.paint()
    qtbot.wait(210)
    assert_allclose(
        panel._map_layer.data,
        expected_map(recording.data, recording.data[:, 1, 1, :3].mean(axis=1)),
    )
    layer.data = np.zeros_like(labels)
    qtbot.wait(40)
    assert panel._seed_combo.currentData() == seed.id
    assert panel._map_layer.metadata["seed_stale"]


def test_point_seed_automatically_updates_after_movement(setup_fc, qtbot):
    viewer, recording, store, controller, panel = setup_fc
    points = viewer.add_points(
        [[1, 1, 1]], scale=recording.scale[-3:], translate=recording.translate[-3:]
    )
    controller.add_source(points)
    seed = next(s for s in store.live_signals() if s.source_type == "point")
    choose_seed(panel, seed.id)
    points.data[0] = [2, 2, 2]
    points.events.data()
    qtbot.wait(40)
    assert panel._seed_combo.currentData() == seed.id
    assert_allclose(
        panel._map_layer.data, expected_map(recording.data, recording.data[:, 2, 2, 2])
    )


def test_multiple_recordings_have_distinct_explicit_seeds(setup_fc, qtbot):
    viewer, recording, store, controller, panel = setup_fc
    data = recording.metadata["xarray"]
    other_data = data.copy(data=np.flip(data.values, axis=0))
    _, other = plot_napari(
        other_data,
        viewer=viewer,
        name="other",
        show_colorbar=False,
        show_scale_bar=False,
    )
    controller.set_references(None)
    viewer.cursor.position = recording.data_to_world((0, 1, 1, 1))
    viewer.keymap[coerce_keybinding("Shift")](viewer)
    seeds = [s for s in store.live_signals() if s.source_type == "mouse"]
    assert len({s.id for s in seeds}) == 2
    for signal in seeds:
        store.rename_live_signal(signal.id, "same alias")
    labels = [
        panel._seed_combo.itemText(panel._seed_combo.findData(s.id)) for s in seeds
    ]
    assert len(set(labels)) == 2
    assert all("same alias" in label for label in labels)
    assert any("other" in label for label in labels)
    assert any("data" in label for label in labels)
    seed = next(s for s in seeds if s.reference_layer is other)
    choose_seed(panel, seed.id)
    assert_allclose(
        panel._map_layer.data,
        expected_map(recording.data, np.asarray(other.data)[:, 1, 1, 1]),
    )
    assert "Cross-recording" in panel._seed_description.text()
    other.name = "renamed reference"
    recording.name = "renamed target"
    assert panel._seed_combo.currentData() == seed.id
    assert panel._data_combo.currentData() is recording
    qtbot.wait(40)
    viewer.layers.remove(other)
    qtbot.wait(40)
    assert panel._seed_combo.currentData() == seed.id
    assert panel._map_layer.metadata["seed_stale"]
    assert "removed" in panel._seed_combo.currentText()


def test_seed_and_target_changes_require_explicit_compute(setup_fc, qtbot):
    _, recording, store, _, panel = setup_fc
    times = recording.metadata["xarray"].time.values
    a = store.pin_signal("a", "same", times, recording.data[:, 0, 0, 0], "#fff", "test")
    b = store.pin_signal("b", "same", times, recording.data[:, 1, 1, 1], "#fff", "test")
    choose_seed(panel, a.id)
    first = panel._map_layer.data.copy()
    panel._seed_combo.setCurrentIndex(panel._seed_combo.findData(b.id))
    qtbot.wait(40)
    assert panel._map_layer.metadata["seed_stale"]
    assert_allclose(panel._map_layer.data, first)
    panel._compute_btn.click()
    assert_allclose(panel._map_layer.data, expected_map(recording.data, b.y))


def test_selected_snapshot_updates_without_changing_seed_identity(setup_fc, qtbot):
    """Re-pinning updates an active stored seed; removing it never selects another."""
    _, recording, store, _, panel = setup_fc
    times = recording.metadata["xarray"].time.values
    snapshot = store.pin_signal(
        "snapshot", "snapshot", times, recording.data[:, 0, 0, 0], "#ffffff", "test"
    )
    choose_seed(panel, snapshot.id)
    store.set_signal_visible(snapshot.id, False)
    replacement = store.pin_signal(
        "snapshot", "ignored name", times, recording.data[:, 1, 1, 1], "#ffffff", "test"
    )
    qtbot.wait(40)
    assert replacement.id == snapshot.id == panel._seed_combo.currentData()
    assert_allclose(panel._map_layer.data, expected_map(recording.data, replacement.y))
    store.remove_signals([snapshot.id])
    assert panel._seed_combo.currentData() == snapshot.id
    assert panel._map_layer.metadata["seed_stale"]
    assert not panel._compute_btn.isEnabled()
