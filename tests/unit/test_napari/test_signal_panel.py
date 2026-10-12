"""Unit tests for the SignalPanel x-axis dimension selection.

The x-axis combo must offer exactly the *non-displayed* (slider) axes with more
than one element. Displayed axes (the two on screen) and singleton axes are
never valid x-axis choices. These tests use the `make_napari_viewer` fixture
so `viewer.dims.displayed` is populated by a real layer.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture
def panel(viewer):
    from confusius._napari._signals._panel import SignalPanel

    return SignalPanel(viewer)


class TestAvailableXaxisDims:
    """`_get_available_xaxis_dims` returns the non-displayed, multi-element axes."""

    def test_lists_slider_axis_for_3d_volume(self, viewer, panel, sample_voxeldata_3d):
        # (z, y, x): napari displays (y, x); only z is a slider axis.
        viewer.add_image(
            sample_voxeldata_3d.values, metadata={"xarray": sample_voxeldata_3d}
        )
        assert panel._get_available_xaxis_dims() == ["k"]

    def test_lists_all_slider_axes_for_4dt_volume(
        self, viewer, panel, sample_voxeldata_3dt
    ):
        # (time, z, y, x): napari displays (y, x); both time and z are sliders.
        viewer.add_image(
            sample_voxeldata_3dt.values, metadata={"xarray": sample_voxeldata_3dt}
        )
        assert panel._get_available_xaxis_dims() == ["time", "k"]

    def test_volume_axes_use_layer_local_indices_in_4d_viewer(
        self, viewer, panel, sample_voxeldata_3dt, sample_voxeldata_3d
    ):
        viewer.add_image(
            sample_voxeldata_3dt.values, metadata={"xarray": sample_voxeldata_3dt}
        )
        viewer.add_image(
            sample_voxeldata_3d.values, metadata={"xarray": sample_voxeldata_3d}
        )
        assert panel._get_available_xaxis_dims() == ["k"]
        assert panel._xaxis_combo.currentText() == "k"

    def test_2d_image_clears_live_plot_and_recording_restores_it(
        self, viewer, panel, sample_voxeldata_3dt
    ):
        from napari.utils.key_bindings import coerce_keybinding
        from numpy.testing import assert_allclose

        from confusius.plotting.napari import plot_napari

        _, recording = plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        panel.show_plot()
        viewer.cursor.position = recording.data_to_world((0, 1, 1, 1))
        viewer.keymap[coerce_keybinding("Shift")](viewer)
        assert_allclose(
            panel._plotter._axes.lines[0].get_ydata(),
            sample_voxeldata_3dt.values[:, 1, 1, 1],
        )
        viewer.add_image(np.ones((3, 4)))
        assert panel._xaxis_combo.count() == 0
        assert len(panel._plotter._axes.lines) == 0
        assert panel._plotter._axes.get_xlabel() == "No plottable axis"
        viewer.layers.selection.active = recording
        assert panel._xaxis_combo.currentText() == "time"
        assert_allclose(
            panel._plotter._axes.lines[0].get_ydata(),
            sample_voxeldata_3dt.values[:, 1, 1, 1],
        )
        panel._plotter.close()

    def test_excludes_singleton_slider_axis(self, viewer, panel):
        # time is a singleton slider axis and must not be offered.
        da = xr.DataArray(np.zeros((1, 4, 6, 8)), dims=["time", "k", "j", "i"])
        viewer.add_image(da.values, metadata={"xarray": da})
        assert panel._get_available_xaxis_dims() == ["k"]

    def test_combo_defaults_to_time_when_present(
        self, viewer, panel, sample_voxeldata_3dt
    ):
        viewer.add_image(
            sample_voxeldata_3dt.values, metadata={"xarray": sample_voxeldata_3dt}
        )
        panel._refresh_xaxis_combo()
        items = [
            panel._xaxis_combo.itemText(i) for i in range(panel._xaxis_combo.count())
        ]
        assert items == ["time", "k"]
        assert panel._xaxis_combo.currentText() == "time"


def test_psc_sliders_shade_baselines_without_losing_mouse_trace(
    viewer, panel, sample_voxeldata_3dt
):
    """Baseline selection changes displayed values, not the mouse source."""
    from napari.utils.key_bindings import coerce_keybinding
    from numpy.testing import assert_allclose
    from qtpy.QtWidgets import QPushButton

    from confusius.plotting.napari import plot_napari

    data = sample_voxeldata_3dt.copy(data=np.abs(sample_voxeldata_3dt.values) + 2)
    _, layer = plot_napari(
        data, viewer=viewer, show_colorbar=False, show_scale_bar=False
    )
    panel.show_plot()
    viewer.cursor.position = layer.data_to_world((0, 1, 1, 1))
    viewer.keymap[coerce_keybinding("Shift")](viewer)
    seed = panel._signals_store.live_signals()[0]
    x, raw = panel._live_sources.get_trace(seed.id)
    assert panel._standardization_combo.currentText() == "None"
    assert panel._baseline_group.isHidden()
    panel._standardization_combo.setCurrentText("Percent signal change")
    assert not panel._baseline_group.isHidden()
    slider = panel._baseline_rows[0][2]
    assert_allclose(slider.value(), [x[0], x[-1]])
    slider.setValue((x[0], x[1]))
    mean = raw[:2].mean()
    assert_allclose(
        panel._plotter._axes.lines[0].get_ydata(), (raw - mean) / abs(mean) * 100
    )
    assert_allclose(panel._live_sources.get_trace(seed.id)[1], raw)
    assert panel._signals_store.get_live_signal(seed.id).enabled
    assert panel._plotter._baseline_warning is None
    assert len(panel._plotter._baseline_spans) == 1
    span = panel._plotter._baseline_spans[0]
    assert_allclose([span.get_x(), span.get_x() + span.get_width()], [x[0], x[1]])
    panel._add_baseline_btn.click()
    panel._baseline_rows[1][2].setValue((x[-2], x[-1]))
    assert len(panel._plotter._baseline_spans) == 2
    legend_labels = [
        text.get_text() for text in panel._plotter._axes.get_legend().get_texts()
    ]
    assert legend_labels.count("PSC baseline") == 1
    panel._baseline_rows[1][0].findChild(QPushButton).click()
    assert len(panel._baseline_rows) == 1
    panel._standardization_combo.setCurrentText("None")
    assert panel._baseline_group.isHidden()
    assert not panel._plotter._baseline_spans
    assert "PSC baseline" not in [
        text.get_text() for text in panel._plotter._axes.get_legend().get_texts()
    ]
    assert_allclose(panel._plotter._axes.lines[0].get_ydata(), raw)
    panel._plotter.close()


def test_manual_baseline_survives_layer_focus_and_range_changes(
    viewer, panel, sample_voxeldata_3dt
):
    from numpy.testing import assert_allclose

    from confusius.plotting.napari import plot_napari

    plot_napari(
        sample_voxeldata_3dt, viewer=viewer, show_colorbar=False, show_scale_bar=False
    )
    panel.show_plot()
    panel._live_sources.set_enabled("mouse", False)
    store = panel._signals_store
    original = store.pin_signal(
        "original",
        "original",
        np.array([10.0, 20.0]),
        np.array([2.0, 4.0]),
        "red",
        "test",
    )
    panel._standardization_combo.setCurrentText("Percent signal change")
    slider = panel._baseline_rows[0][2]
    assert_allclose(slider.value(), (10.0, 20.0))
    wider = store.pin_signal(
        "wider", "wider", np.array([0.0, 30.0]), np.array([2.0, 4.0]), "blue", "test"
    )
    assert_allclose(slider.value(), (10.0, 20.0))
    slider.setValue((11.0, 19.0))
    viewer.layers.selection.clear()
    panel._new_points_btn.click()
    viewer.layers.selection.active.add([1, 1, 1])
    assert_allclose(slider.value(), (11.0, 19.0))
    store.pin_signal(
        "narrow",
        "narrow",
        np.array([12.0, 18.0]),
        np.array([2.0, 4.0]),
        "green",
        "test",
    )
    roi = viewer.layers.selection.active
    source = next(s for s in panel._live_sources.sources() if s.layer is roi)
    panel._live_sources.set_enabled(source.id, False)
    store.remove_signals([original.id, wider.id])
    assert_allclose(slider.value(), (12.0, 18.0))
    panel._plotter.close()


def test_event_baseline_selection_tracks_event_unions_and_preserves_manual_intervals(
    viewer, sample_voxeldata_3dt, qtbot
):
    from napari.utils.key_bindings import coerce_keybinding
    from numpy.testing import assert_allclose
    from qtpy.QtCore import Qt

    from confusius._napari._events._store import EventStore
    from confusius._napari._signals._panel import SignalPanel
    from confusius.plotting.napari import plot_napari

    data = sample_voxeldata_3dt.copy(
        data=np.arange(10)[:, None, None, None]
        + np.ones(sample_voxeldata_3dt.shape) * 2
    )
    _, layer = plot_napari(
        data, viewer=viewer, show_colorbar=False, show_scale_bar=False
    )
    events = EventStore()
    events.add_event(10.0, 1.0, "first")
    events.add_event(10.5, 1.5, "second")
    events.add_event(11.0, 0.0, "instant")
    panel = SignalPanel(viewer, event_store=events)
    qtbot.addWidget(panel)
    panel.show_plot()
    viewer.cursor.position = layer.data_to_world((0, 1, 1, 1))
    viewer.keymap[coerce_keybinding("Shift")](viewer)
    panel._standardization_combo.setCurrentText("Percent signal change")
    slider = panel._baseline_rows[0][2]
    slider.setValue((10.5, 12.0))
    panel._baseline_source_combo.setCurrentText("Events")
    assert panel._manual_baseline_widget.isHidden()
    assert not panel._baseline_events_widget.isHidden()
    listing = panel._baseline_events_list
    first, second, instant = listing.item(0), listing.item(1), listing.item(2)
    assert first is not None and second is not None and instant is not None
    first.setCheckState(Qt.CheckState.Checked)
    second.setCheckState(Qt.CheckState.Checked)
    assert not (instant.flags() & Qt.ItemFlag.ItemIsEnabled)
    plotter = panel._plotter
    assert plotter is not None
    raw = np.asarray(data.values[:, 1, 1, 1])
    expected = (raw - raw[:4].mean()) / raw[:4].mean() * 100
    assert_allclose(
        np.asarray(plotter._axes.lines[0].get_ydata(), dtype=float), expected
    )
    panel._baseline_outside_check.setChecked(True)
    expected = (raw - raw[4:].mean()) / raw[4:].mean() * 100
    assert_allclose(
        np.asarray(plotter._axes.lines[0].get_ydata(), dtype=float), expected
    )
    events.add_event(9.0, 0.5, "earlier")
    assert len(panel._selected_baseline_events()) == 2
    assert_allclose(
        np.asarray(plotter._axes.lines[0].get_ydata(), dtype=float), expected
    )
    events.set_shade_signals(False)
    assert_allclose(
        np.asarray(plotter._axes.lines[0].get_ydata(), dtype=float), expected
    )
    events.remove_events([2])
    expected = (raw - raw[2:].mean()) / raw[2:].mean() * 100
    assert_allclose(
        np.asarray(plotter._axes.lines[0].get_ydata(), dtype=float), expected
    )
    panel._baseline_source_combo.setCurrentText("Manual intervals")
    assert_allclose(slider.value(), (10.5, 12.0))
    mean = raw[1:5].mean()
    assert_allclose(
        np.asarray(plotter._axes.lines[0].get_ydata(), dtype=float),
        (raw - mean) / mean * 100,
    )
    plotter.close()


def test_existing_roi_layers_are_detected_and_disabled_sources_stay_disabled(
    viewer, sample_voxeldata_3dt
):
    """Detection covers existing layers without resetting manager overrides."""
    from confusius._napari._signals._store import SignalStore

    viewer.add_image(
        sample_voxeldata_3dt.values, metadata={"xarray": sample_voxeldata_3dt}
    )
    points = viewer.add_points([[1, 1, 1]])
    labels = viewer.add_labels(np.ones(sample_voxeldata_3dt.shape[-3:], dtype=int))
    store = SignalStore()
    controller = store.live_sources(viewer)
    assert {s.layer for s in controller.sources() if s.layer is not None} == {
        points,
        labels,
    }
    sid = next(s.id for s in controller.sources() if s.layer is points)
    controller.set_enabled(sid, False)
    viewer.add_points([[2, 2, 2]])
    assert not next(s.enabled for s in controller.sources() if s.id == sid)
    assert sum(s.kind == "point" for s in controller.sources()) == 2
    controller.close()


class TestCreateLayers:
    """New Points/Labels layers must share the reference image's units and axis labels."""

    @pytest.fixture
    def image_layer(self, viewer, sample_voxeldata_3dt):
        from confusius.plotting.napari import plot_napari

        _, layer = plot_napari(sample_voxeldata_3dt, viewer=viewer)
        return layer

    @pytest.mark.parametrize("kind", ["point", "label"])
    def test_drawing_in_new_roi_restores_reference_axis(
        self, viewer, panel, image_layer, kind, qtbot
    ):
        from numpy.testing import assert_allclose

        panel.show_plot()
        # Napari clears the active layer while selecting a newly created ROI.
        viewer.layers.selection.clear()
        button = panel._new_points_btn if kind == "point" else panel._new_labels_btn
        button.click()
        roi = viewer.layers.selection.active
        if kind == "point":
            roi.add([1, 1, 1])
        else:
            roi.data[1, 1, 1] = 1
            roi.events.paint()
            qtbot.wait(180)
        assert panel._ref_combo.currentData() is None
        assert panel._xaxis_combo.currentText() == "time"
        signals = [
            s for s in panel._signals_store.live_signals() if s.backing_layer is roi
        ]
        assert len(signals) == 1
        line = next(
            line
            for line in panel._plotter._axes.lines
            if line.get_label() == signals[0].display_name
        )
        assert_allclose(line.get_xdata(), image_layer.metadata["xarray"].time.values)
        assert_allclose(line.get_ydata(), image_layer.data[:, 1, 1, 1])
        panel._plotter.close()

    @pytest.mark.parametrize("method", ["_create_points_layer", "_create_labels_layer"])
    def test_new_layer_matches_reference_geometry(
        self, viewer, panel, image_layer, method
    ):
        getattr(panel, method)()
        new_layer = viewer.layers[-1]
        assert new_layer.name == (
            "Points" if method == "_create_points_layer" else "Labels"
        )
        assert new_layer.units == image_layer.units[1:]
        assert new_layer.axis_labels == image_layer.axis_labels[1:]
        assert new_layer.scale.tolist() == image_layer.scale[1:].tolist()
        assert new_layer.translate.tolist() == image_layer.translate[1:].tolist()
        # Consistent units keep napari from dropping units for rendering.
        assert viewer.layers.extent.units is not None
