"""Regression tests for shared live source identity and numerical extraction."""

from types import SimpleNamespace

import numpy as np
import pytest
from napari.utils.key_bindings import coerce_keybinding
from numpy.testing import assert_allclose, assert_array_equal

from confusius._napari._signals._store import SignalStore
from confusius._utils.geometry import get_voxel_to_world_affine
from confusius.plotting.napari import plot_napari


@pytest.fixture
def live_setup(make_napari_viewer, sample_voxeldata_3dt):
    """Create a correctly positioned recording and shared source controller."""
    viewer = make_napari_viewer()
    _, recording = plot_napari(
        sample_voxeldata_3dt,
        viewer=viewer,
        name="recording",
        show_colorbar=False,
        show_scale_bar=False,
    )
    store = SignalStore()
    controller = store.live_sources(viewer)
    yield viewer, recording, store, controller
    controller.close()


def add_rois(viewer, recording):
    """Create voxel-aligned point and label layers for the recording."""
    kwargs = {"scale": recording.scale[-3:], "translate": recording.translate[-3:]}
    points = viewer.add_points([[1, 1, 1], [2, 2, 2]], name="points", **kwargs)
    labels_data = np.zeros(recording.data.shape[1:], dtype=np.int32)
    labels_data[0, 0, :2] = 3
    labels = viewer.add_labels(labels_data, name="labels", **kwargs)
    return points, labels


def trace(store, kind, recording, item=None):
    """Select exactly one explicitly qualified live trace."""
    matches = [
        s
        for s in store.live_signals()
        if s.source_type == kind
        and s.reference_layer is recording
        and (item is None or s.source_id == item)
    ]
    assert len(matches) == 1
    return matches[0]


def position_mouse(viewer, recording, indices=(1, 1, 1)):
    """Exercise the installed Shift handler at a stationary cursor."""
    viewer.cursor.position = recording.data_to_world((0, *indices))
    viewer.keymap[coerce_keybinding("Shift")](viewer)


def test_sources_coexist_and_keep_overrides_when_disabled(live_setup):
    viewer, recording, store, controller = live_setup
    points, labels = add_rois(viewer, recording)
    point_source = controller.add_source(points)
    label_source = controller.add_source(labels)
    point = next(s for s in store.live_signals() if s.source_type == "point")
    store.rename_live_signal(point.id, "custom")
    store.set_live_signal_color(point.id, "#123456")
    ids = {s.id for s in store.live_signals()}
    for sid in ("mouse", point_source, label_source):
        controller.set_enabled(sid, False)
        assert {s.id for s in store.live_signals()} == ids
        controller.set_enabled(sid, True)
    assert {s.source_type for s in store.live_signals()} == {"mouse", "point", "label"}
    assert store.get_live_signal(point.id).name == "custom"
    assert store.get_live_signal(point.id).color == "#123456"
    x, y = controller.get_trace(point.id)
    assert_array_equal(x, recording.metadata["xarray"].time)
    assert_allclose(y, recording.data[:, 1, 1, 1])


def test_reference_changes_do_not_reassign_or_delete_trace(live_setup):
    viewer, recording, store, controller = live_setup
    _, other = plot_napari(
        recording.metadata["xarray"] * 2,
        viewer=viewer,
        name="other",
        show_colorbar=False,
        show_scale_bar=False,
    )
    controller.set_references(None)
    first = trace(store, "mouse", recording)
    second = trace(store, "mouse", other)
    position_mouse(viewer, recording)
    controller.set_references([other])
    assert store.get_live_signal(first.id).reference_layer is recording
    assert not store.get_live_signal(first.id).enabled
    assert store.get_live_signal(second.id).enabled
    with pytest.raises(ValueError, match="disabled"):
        controller.get_trace(first.id)
    controller.set_references(None)
    assert_allclose(controller.get_trace(first.id)[1], recording.data[:, 1, 1, 1])


@pytest.mark.parametrize("kind", ["point", "label"])
def test_roi_color_sync_and_per_recording_style_overrides(live_setup, kind):
    from matplotlib.colors import to_hex
    from napari.utils.colormaps import DirectLabelColormap

    viewer, recording, store, controller = live_setup
    points, labels = add_rois(viewer, recording)
    _, second = plot_napari(
        recording.metadata["xarray"],
        viewer=viewer,
        show_colorbar=False,
        show_scale_bar=False,
    )
    source = points if kind == "point" else labels
    seeds = [s for s in store.live_signals() if s.backing_layer is source]
    seed = next(s for s in seeds if s.reference_layer is recording)
    if kind == "point":
        points.face_color = ["red", "blue"]
        assert store.get_live_signal(seed.id).color == "#ff0000"
        other_before = to_hex(points.face_color[1])
    else:
        labels.colormap = DirectLabelColormap(
            color_dict={None: "gray", 0: "transparent", 3: "green"}
        )
        assert store.get_live_signal(seed.id).color == "#008000"
    raw_before = controller.get_trace(seed.id)[1].copy()
    store.set_live_signal_color(seed.id, "#123456")
    assert store.get_live_signal(seed.id).color == "#123456"
    assert store.get_live_signal(seed.id).color_overridden
    other = next(
        s
        for s in seeds
        if s.reference_layer is second and s.source_id == seed.source_id
    )
    assert store.get_live_signal(other.id).color != "#123456"
    assert store.get_live_signal(seed.id).line_style == "-"
    assert store.get_live_signal(other.id).line_style == "--"
    store.set_signal_line_style(seed.id, ":")
    if kind == "point":
        assert to_hex(points.face_color[0]) == "#ff0000"
        assert to_hex(points.face_color[1]) == other_before
        points.face_color = ["yellow", "blue"]
    else:
        assert to_hex(labels.get_color(3)) == "#008000"
        labels.colormap = DirectLabelColormap(
            color_dict={None: "gray", 0: "transparent", 3: "yellow"}
        )
    assert store.get_live_signal(seed.id).color == "#123456"
    assert store.get_live_signal(other.id).color == "#ffff00"
    assert store.get_live_signal(seed.id).line_style == ":"
    store.set_live_signal_color_sync(seed.id, True)
    assert store.get_live_signal(seed.id).color == "#ffff00"
    assert not store.get_live_signal(seed.id).color_overridden
    assert_allclose(controller.get_trace(seed.id)[1], raw_before)
    assert any(
        s.reference_layer is second and s.color == "#ffff00"
        for s in store.live_signals()
        if s.backing_layer is source
    )


def test_hidden_labels_keep_their_color_in_traces(live_setup):
    from matplotlib.colors import to_hex

    viewer, recording, store, _ = live_setup
    _, labels = add_rois(viewer, recording)
    expected = to_hex(labels.get_color(3))
    labels.selected_label = 1
    labels.show_selected_label = True
    seed = trace(store, "label", recording, 3)
    assert seed.color == expected


def test_rename_preserves_identity_color_and_custom_name(live_setup):
    viewer, recording, store, controller = live_setup
    _, labels = add_rois(viewer, recording)
    controller.add_source(labels)
    original = trace(store, "label", recording, 3)
    controller.set_base_color(recording, "#5f91b4")
    original = store.get_live_signal(original.id)
    colormap = recording.colormap
    label_colormap = labels.colormap
    recording.name = "renamed recording"
    labels.name = "renamed ROI"
    renamed = store.get_live_signal(original.id)
    assert renamed.color == original.color
    assert "renamed recording" in renamed.name and "renamed ROI" in renamed.name
    assert recording.colormap is colormap
    assert labels.colormap is label_colormap
    store.rename_live_signal(original.id, "my seed")
    store.set_live_signal_color(original.id, "#abcdef")
    recording.name = "renamed again"
    controller.set_base_color(recording, "#a56731")
    assert store.get_live_signal(original.id).name == "my seed"
    assert store.get_live_signal(original.id).color == "#abcdef"


def test_label_repainting_and_erasure_refresh_values(live_setup, qtbot):
    viewer, recording, store, controller = live_setup
    _, labels = add_rois(viewer, recording)
    controller.add_source(labels)
    seed = trace(store, "label", recording, 3)
    assert_allclose(
        controller.get_trace(seed.id)[1], recording.data[:, 0, 0, :2].mean(axis=1)
    )
    labels.data[:] = 0
    labels.data[1, 1, :3] = 3
    labels.events.paint()
    qtbot.wait(180)
    assert_allclose(
        controller.get_trace(seed.id)[1], recording.data[:, 1, 1, :3].mean(axis=1)
    )
    labels.data = np.zeros_like(labels.data)
    assert not store.get_live_signal(seed.id).available
    with pytest.raises(ValueError, match="unavailable"):
        controller.get_trace(seed.id)
    new = labels.data.copy()
    new[2, 2, 2] = 3
    labels.data = new
    assert_allclose(controller.get_trace(seed.id)[1], recording.data[:, 2, 2, 2])


def test_point_edits_and_deletion_preserve_item_identity(live_setup):
    viewer, recording, store, controller = live_setup
    points, _ = add_rois(viewer, recording)
    controller.add_source(points)
    seeds = [s for s in store.live_signals() if s.source_type == "point"]
    first, second = seeds
    points.selected_data = {0}
    points.remove_selected()
    assert store.get_live_signal(first.id) is None
    assert store.get_live_signal(second.id).source_id == second.source_id
    assert_allclose(controller.get_trace(second.id)[1], recording.data[:, 2, 2, 2])
    points.data[0] = [3, 3, 3]
    points.events.data()
    assert_allclose(controller.get_trace(second.id)[1], recording.data[:, 3, 3, 3])


def test_layer_removal_keeps_other_sources_and_snapshots(live_setup):
    viewer, recording, store, controller = live_setup
    points, labels = add_rois(viewer, recording)
    controller.add_source(points)
    controller.add_source(labels)
    label = trace(store, "label", recording, 3)
    x, y = controller.get_trace(label.id)
    snapshot = store.pin_signal(label.id, "snapshot", x, y, label.color, "test")
    viewer.layers.remove(labels)
    assert store.get_live_signal(label.id) is None
    assert any(s.source_type == "point" for s in store.live_signals())
    assert_allclose(store.stored_signals()[0].y, snapshot.y)
    viewer.layers.remove(recording)
    assert not store.live_signals()
    assert len(store.stored_signals()) == 1


def test_shared_mouse_radius_uses_physical_geometry(live_setup):
    viewer, recording, store, controller = live_setup
    position_mouse(viewer, recording)
    seed = trace(store, "mouse", recording)
    x0, y0 = controller.get_trace(seed.id)
    assert_allclose(y0, recording.data[:, 1, 1, 1])
    controller.set_radius(0.11)
    x, y = controller.get_trace(seed.id)
    affine = get_voxel_to_world_affine(recording.metadata["xarray"])[:3, :3]
    offsets = np.indices(recording.data.shape[1:]).transpose(1, 2, 3, 0) - 1
    mask = np.linalg.norm(offsets @ affine.T, axis=-1) <= 0.11
    assert_allclose(y, recording.data[:, mask].mean(axis=1))
    assert_array_equal(x, x0)
    assert controller.get_trace(seed.id)[1] is y
    origin = controller.pin_origin(seed.id)
    controller.set_radius(0)
    assert controller.pin_origin(seed.id) != origin


def test_mouse_shift_binding_is_shared_and_restored(make_napari_viewer):
    viewer = make_napari_viewer()
    previous = lambda viewer: None
    viewer.bind_key("Shift", previous)
    store = SignalStore()
    controller = store.live_sources(viewer)
    assert store.live_sources(viewer) is controller
    controller.set_enabled("mouse", False)
    assert viewer.keymap[coerce_keybinding("Shift")] is previous
    controller.set_enabled("mouse", True)
    assert viewer.keymap[coerce_keybinding("Shift")] != previous
    controller.close()
    assert viewer.keymap[coerce_keybinding("Shift")] is previous


def test_mouse_outside_recording_does_not_return_previous_trace(live_setup, qtbot):
    viewer, recording, store, controller = live_setup
    position_mouse(viewer, recording)
    seed = trace(store, "mouse", recording)
    controller.get_trace(seed.id)
    viewer.cursor.position = recording.data_to_world((0, -20, -20, -20))
    for callback in viewer.mouse_move_callbacks:
        callback(viewer, SimpleNamespace(modifiers={"Shift"}))
    qtbot.wait(30)
    with pytest.raises(ValueError, match="No valid data"):
        controller.get_trace(seed.id)


def test_labels_follow_world_space_across_recording_transforms(live_setup):
    viewer, recording, store, controller = live_setup
    _, labels = add_rois(viewer, recording)
    _, second = plot_napari(
        recording.metadata["xarray"],
        viewer=viewer,
        show_colorbar=False,
        show_scale_bar=False,
    )
    affine = np.eye(second.ndim + 1)
    affine[-2, -1] = second.scale[-1]
    second.affine = affine
    first_seed = trace(store, "label", recording, 3)
    second_seed = trace(store, "label", second, 3)
    assert_allclose(
        controller.get_trace(first_seed.id)[1], recording.data[:, 0, 0, :2].mean(axis=1)
    )
    assert_allclose(
        controller.get_trace(second_seed.id)[1], np.asarray(second.data)[:, 0, 0, 0]
    )
    recording.affine = affine
    assert_allclose(controller.get_trace(first_seed.id)[1], recording.data[:, 0, 0, 0])
    recording.affine = np.eye(recording.ndim + 1)
    assert_allclose(
        controller.get_trace(first_seed.id)[1], recording.data[:, 0, 0, :2].mean(axis=1)
    )
    assert_array_equal(labels.data[0, 0, :2], [3, 3])


def test_labels_resample_into_recording_with_different_voxel_spacing(live_setup):
    viewer, recording, store, controller = live_setup
    _, labels = add_rois(viewer, recording)
    labels.data[:] = 0
    labels.data[0, 0, 2:4] = 3
    labels.events.data()
    strided = recording.metadata["xarray"].isel(i=slice(None, None, 2))
    _, second = plot_napari(
        strided, viewer=viewer, show_colorbar=False, show_scale_bar=False
    )
    seed = trace(store, "label", second, 3)
    assert_allclose(
        controller.get_trace(seed.id)[1], np.asarray(strided.data)[:, 0, 0, 1]
    )


def test_transformed_labels_outside_recording_are_unavailable(live_setup):
    viewer, recording, store, controller = live_setup
    _, labels = add_rois(viewer, recording)
    labels.translate = np.asarray(labels.translate) + 100
    seed = trace(store, "label", recording, 3)
    with pytest.raises(ValueError, match="No valid data"):
        controller.get_trace(seed.id)


def test_bulk_point_replacement_does_not_reassign_existing_seed(live_setup):
    """An unidentifiable replacement invalidates old point identities."""
    viewer, recording, store, controller = live_setup
    points, _ = add_rois(viewer, recording)
    controller.add_source(points)
    original = {s.id for s in store.live_signals() if s.source_type == "point"}
    points.data = np.array([[3, 3, 3], [1, 2, 3]])
    replacement = {s.id for s in store.live_signals() if s.source_type == "point"}
    assert original.isdisjoint(replacement)
    for sid in original:
        with pytest.raises(ValueError, match="unavailable"):
            controller.get_trace(sid)


def test_panel_controls_do_not_drop_other_sources_or_resurrect_removed_ones(
    live_setup, qtbot
):
    """Adding, disabling and filtering sources never replaces another source."""
    from confusius._napari._signals._panel import SignalPanel

    viewer, recording, store, controller = live_setup
    points, _labels = add_rois(viewer, recording)
    panel = SignalPanel(viewer, signal_store=store)
    qtbot.addWidget(panel)
    assert panel._points_check.isChecked()
    assert panel._labels_check.isChecked()
    original = {s.id for s in store.live_signals()}
    panel._points_check.setChecked(False)
    assert {s.id for s in store.live_signals()} == original
    assert all(s.enabled for s in store.live_signals() if s.source_type == "label")
    panel._points_check.setChecked(True)
    point_source = next(s for s in controller.sources() if s.layer is points)
    controller.remove_source(point_source.id)
    assert not any(s.source_type == "point" for s in store.live_signals())
    assert any(s.source_type == "label" for s in store.live_signals())
    assert panel._plotter is None


def test_mouse_sphere_respects_retained_native_indices(
    make_napari_viewer, sample_voxeldata_3dt
):
    """A strided VoxelData view keeps its actual physical voxel spacing."""
    viewer = make_napari_viewer()
    data = sample_voxeldata_3dt.isel(
        k=slice(None, None, 2), j=slice(None, None, 2), i=slice(None, None, 2)
    )
    _, recording = plot_napari(
        data, viewer=viewer, show_colorbar=False, show_scale_bar=False
    )
    store = SignalStore()
    controller = store.live_sources(viewer)
    position_mouse(viewer, recording)
    controller.set_radius(0.25)
    seed = trace(store, "mouse", recording)
    native = np.stack(
        np.meshgrid(*(data[dim].values for dim in ("k", "j", "i")), indexing="ij"),
        axis=-1,
    )
    center = np.array([data[dim].values[1] for dim in ("k", "j", "i")])
    affine = get_voxel_to_world_affine(data)[:3, :3]
    mask = np.linalg.norm((native - center) @ affine.T, axis=-1) <= 0.25
    assert_allclose(controller.get_trace(seed.id)[1], data.values[:, mask].mean(axis=1))
    controller.close()


def test_mouse_moves_within_one_voxel_reuse_shared_values(live_setup, qtbot):
    """Sub-voxel movement must not repeatedly extract or recompute the same seed."""
    viewer, recording, store, controller = live_setup
    qtbot.wait(20)
    position_mouse(viewer, recording)
    seed = trace(store, "mouse", recording)
    cached = controller.get_trace(seed.id)[1]
    changes = []
    controller.traces_changed.connect(changes.append)
    viewer.cursor.position = recording.data_to_world((0, 1.2, 1.2, 1.2))
    for callback in viewer.mouse_move_callbacks:
        callback(viewer, SimpleNamespace(modifiers={"Shift"}))
    qtbot.wait(30)
    assert changes == []
    assert controller.get_trace(seed.id)[1] is cached
    assert_allclose(cached, recording.data[:, 1, 1, 1])
    viewer.cursor.position = recording.data_to_world((0, 2, 2, 2))
    for callback in viewer.mouse_move_callbacks:
        callback(viewer, SimpleNamespace(modifiers={"Shift"}))
    qtbot.wait(30)
    assert changes == ["mouse"]
    assert_allclose(controller.get_trace(seed.id)[1], recording.data[:, 2, 2, 2])
