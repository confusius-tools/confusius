"""Tests for the transform panel's logic and observable widget and layer state."""

import numpy as np
import pytest
import SimpleITK as sitk
from napari.layers import Labels
from napari.qt import get_stylesheet
from napari.settings import get_settings
from qtpy.QtCore import QPoint, QPointF, Qt
from qtpy.QtGui import QWheelEvent
from qtpy.QtWidgets import QApplication

from confusius._napari._registration._panel import RegistrationPanel
from confusius._napari._registration._transform_payloads import (
    get_affine_transform_from_payload,
    load_transform_payload,
)
from confusius._napari._transform import _panel as transform_panel_module
from confusius._napari._transform._model import (
    TransformState,
    format_matrix,
    parse_matrix,
)
from confusius._napari._transform._panel import TransformPanel
from confusius._utils.napari import build_napari_layer_data
from confusius.registration import apply_zoom
from confusius.xarray import create_voxeldata

AXES = ("z", "y", "x")


@pytest.fixture
def volume():
    """Return a smooth, anisotropic, offset 3D volume."""
    grids = np.meshgrid(*[np.linspace(0, 3, n) for n in (8, 16, 20)], indexing="ij")
    return create_voxeldata(
        np.sin(grids[0]) + np.cos(grids[1]) * np.sin(grids[2]) + 3.0,
        dims=("k", "j", "i"),
        spacing=(0.4, 0.1, 0.1),
        origin=(1.0, 2.0, 3.0),
        name="volume",
    )


def _add_layer(viewer, data, **kwargs):
    """Add `data` to `viewer` the way the plugin readers and `plot_napari` do."""
    array, layer_kwargs, layer_type = build_napari_layer_data(data, **kwargs)
    return getattr(viewer, f"add_{layer_type}")(array, **layer_kwargs)


@pytest.fixture
def viewer(make_napari_viewer):
    """Return a napari viewer."""
    return make_napari_viewer()


@pytest.fixture
def layer(viewer, volume):
    """Return an image layer built from `volume`."""
    return _add_layer(viewer, volume)


@pytest.fixture
def panel(qtbot, viewer, layer):
    """Return a transform panel targeting `layer`."""
    widget = TransformPanel(viewer)
    qtbot.addWidget(widget)
    return widget


def _select(combo, data):
    """Select the combo entry holding `data`."""
    index = combo.findData(data)
    assert index >= 0
    combo.setCurrentIndex(index)


def _drag(row, value_int, *, release=True):
    """Drag a row's slider handle to an integer position, as the mouse would."""
    row.slider.setSliderDown(True)
    row.slider.setSliderPosition(value_int)
    if release:
        row.slider.setSliderDown(False)


def _set_clipboard_text(text):
    """Put `text` on the clipboard."""
    clipboard = QApplication.clipboard()
    assert clipboard is not None
    clipboard.setText(text)


def _angle_to_int(row, angle):
    """Return the slider position for a rotation angle."""
    return round(angle / row.resolution)


@pytest.mark.parametrize("axis", range(3))
def test_plus_moves_one_data_unit_on_anisotropic_voxels(panel, layer, axis):
    start = layer.translate.copy()
    _select(panel._translate_step_combo, 1.0)

    panel._translate_rows[AXES[axis]].plus_btn.click()

    expected = start.copy()
    expected[axis] += 1.0
    np.testing.assert_allclose(layer.translate, expected)
    np.testing.assert_allclose(layer.scale, (0.4, 0.1, 0.1))


def test_minus_moves_the_other_way(panel, layer):
    start = layer.translate.copy()
    _select(panel._translate_step_combo, 0.1)

    panel._translate_rows["y"].minus_btn.click()

    np.testing.assert_allclose(layer.translate - start, (0.0, -0.1, 0.0))


def test_one_voxel_step_follows_the_axis_spacing(panel, layer):
    start = layer.translate.copy()
    _select(panel._translate_step_combo, "voxel")

    panel._translate_rows["z"].plus_btn.click()
    panel._translate_rows["x"].plus_btn.click()

    np.testing.assert_allclose(layer.translate - start, (0.4, 0.0, 0.1))


def test_slider_spin_and_buttons_stay_in_sync(panel, monkeypatch):
    row = panel._translate_rows["z"]
    calls = []
    session = panel._target.session
    show = session.show
    monkeypatch.setattr(session, "show", lambda m: (calls.append(m), show(m)))

    row.spin.setValue(0.5)

    assert row.slider.value() == round(0.5 / row.resolution)
    assert len(calls) == 1

    _select(panel._translate_step_combo, 0.1)
    row.plus_btn.click()

    assert row.spin.value() == pytest.approx(0.6)
    assert row.slider.value() == round(0.6 / row.resolution)
    assert len(calls) == 2


def test_slider_step_outside_a_drag_updates_the_spin(panel):
    row = panel._translate_rows["y"]

    row.slider.setValue(row.slider.value() + 100)

    assert row.spin.value() == pytest.approx(100 * row.resolution)
    assert panel.state.translation[1] == pytest.approx(100 * row.resolution)


def test_typed_value_is_exact(panel):
    panel._translate_rows["x"].spin.setValue(1.2345)

    assert panel.state.translation[2] == pytest.approx(1.2345)


def test_translation_drag_moves_the_layer_and_is_one_undo_step(panel, layer):
    start = layer.translate.copy()
    row = panel._translate_rows["z"]

    row.slider.setSliderDown(True)
    row.slider.setSliderPosition(100)
    row.slider.setSliderPosition(200)
    row.slider.setSliderDown(False)

    np.testing.assert_allclose(layer.translate[0] - start[0], 200 * row.resolution)
    panel._undo_btn.click()
    np.testing.assert_allclose(layer.translate, start)
    assert panel.state.is_identity()


def test_rotation_release_keeps_rotation_and_the_dial_stays(panel, layer, volume):
    row = panel._rotate_rows["x"]
    position = _angle_to_int(row, 20.0)
    start = panel.state

    _drag(row, position)

    angle = position * row.resolution
    expected = start.rotated("x", angle, frame="global")
    np.testing.assert_allclose(panel.state.matrix(), expected.matrix())
    assert row.slider.value() == position
    assert row.spin.value() == pytest.approx(20.0)
    # Released on the tight grid `plot_napari` would use.
    array, kwargs, _ = build_napari_layer_data(
        volume.fusi.affine.apply(expected.matrix()),
        resample_fill_value=panel._target.session.fill_value,
    )
    assert layer.data.shape == array.shape
    np.testing.assert_allclose(layer.translate, kwargs["translate"])


def test_drag_shows_the_value_in_the_spin(panel):
    row = panel._rotate_rows["x"]

    _drag(row, _angle_to_int(row, 12.0), release=False)

    assert row.spin.value() == pytest.approx(12.0)


def test_drag_holds_one_grid(panel, layer):
    row = panel._rotate_rows["x"]
    row.slider.setSliderDown(True)
    shape = layer.data.shape
    translate = layer.translate.copy()

    for angle in (5.0, 25.0, -30.0):
        row.slider.setSliderPosition(_angle_to_int(row, angle))
        panel._flush_update()

        assert layer.data.shape == shape
        np.testing.assert_array_equal(layer.translate, translate)


def test_3d_drag_caps_the_grid_and_release_restores_it(
    panel, viewer, layer, volume, monkeypatch
):
    monkeypatch.setattr(transform_panel_module, "MAX_3D_DRAG_VOXELS", 500)
    viewer.dims.ndisplay = 3
    row = panel._rotate_rows["x"]

    _drag(row, _angle_to_int(row, 20.0), release=False)

    assert np.prod(layer.data.shape) <= 1000

    row.slider.setSliderDown(False)

    expected = _fresh_display(panel, panel.state.matrix())
    assert layer.data.shape == expected.shape


def test_2d_drag_keeps_full_resolution(panel, layer, monkeypatch):
    monkeypatch.setattr(transform_panel_module, "MAX_3D_DRAG_VOXELS", 500)
    scale = layer.scale.copy()
    row = panel._rotate_rows["x"]

    _drag(row, _angle_to_int(row, 20.0), release=False)

    assert np.prod(layer.data.shape) > 500
    np.testing.assert_allclose(layer.scale, scale)


def test_rotation_plus_applies_one_step(panel):
    _select(panel._rotate_step_combo, 5.0)
    start = panel.state

    panel._rotate_rows["y"].plus_btn.click()

    np.testing.assert_allclose(
        panel.state.matrix(), start.rotated("y", 5.0, frame="global").matrix()
    )


def test_typed_angle_sets_the_angle_turned(panel):
    row = panel._rotate_rows["z"]
    start = panel.state

    row.spin.setValue(7.5)

    np.testing.assert_allclose(
        panel.state.matrix(), start.rotated("z", 7.5, frame="global").matrix()
    )
    assert row.spin.value() == pytest.approx(7.5)

    row.spin.setValue(10.0)

    # Two steps of 7.5 and 2.5 degrees, against one of 10, up to round-off.
    np.testing.assert_allclose(
        panel.state.matrix(),
        start.rotated("z", 10.0, frame="global").matrix(),
        atol=1e-12,
    )
    assert row.spin.value() == pytest.approx(10.0)


def test_dial_drag_across_the_bottom_keeps_turning(panel):
    row = panel._rotate_rows["x"]

    row.slider.setSliderDown(True)
    row.slider.setSliderPosition(_angle_to_int(row, 170.0))
    row.slider.setSliderPosition(_angle_to_int(row, -170.0))
    row.slider.setSliderDown(False)

    expected = TransformState.identity(center=panel.state.center).rotated(
        "x", 190.0, frame="global"
    )
    np.testing.assert_allclose(panel.state.matrix(), expected.matrix())
    assert row.spin.value() == pytest.approx(-170.0)


def test_reset_and_the_matrix_tools_put_the_dials_back_to_zero(panel):
    row = panel._rotate_rows["x"]

    row.spin.setValue(30.0)
    panel._identity_btn.click()
    assert row.spin.value() == 0.0
    assert row.slider.value() == 0

    row.spin.setValue(30.0)
    panel._reset_btn.click()
    assert row.spin.value() == 0.0


def test_one_drag_is_one_undo_step(panel):
    row = panel._rotate_rows["x"]
    row.slider.setSliderDown(True)
    for angle in (5.0, 10.0, 15.0):
        row.slider.setSliderPosition(_angle_to_int(row, angle))
    row.slider.setSliderDown(False)
    assert not panel.state.is_identity()

    panel._undo_btn.click()

    assert panel.state.is_identity()


def test_undo_and_redo_restore_the_state_and_the_layer(panel, layer):
    original = layer.data
    panel._rotate_rows["x"].spin.setValue(10.0)
    rotated = panel.state.matrix()
    rotated_shape = layer.data.shape

    panel._undo_btn.click()

    assert panel.state.is_identity()
    assert layer.data is original

    panel._redo_btn.click()

    np.testing.assert_allclose(panel.state.matrix(), rotated)
    assert layer.data.shape == rotated_shape


def test_undo_keeps_the_current_center(panel):
    panel._center_geometry_btn.click()
    pivot = panel.state.pivot()
    panel._rotate_rows["x"].spin.setValue(30.0)

    panel._undo_btn.click()

    assert panel.state.is_identity()
    np.testing.assert_allclose(panel.state.pivot(), pivot)

    panel._redo_btn.click()

    np.testing.assert_allclose(panel.state.pivot(), pivot)


def test_a_failed_update_keeps_the_state(panel, monkeypatch):
    panel._rotate_rows["x"].spin.setValue(10.0)
    state = panel.state
    undo_enabled = panel._undo_btn.isEnabled()

    def fail(matrix):
        raise TypeError("cannot read")

    monkeypatch.setattr(panel._target.session, "show", fail)

    with pytest.raises(TypeError):
        panel._commit(state.rotated("x", 5.0, frame="global"))

    assert panel.state is state
    panel._target.history.undo()
    assert panel._target.history.redo() is state
    assert panel._undo_btn.isEnabled() == undo_enabled


def test_undo_buttons_follow_the_history(panel):
    assert not panel._undo_btn.isEnabled()
    assert not panel._redo_btn.isEnabled()

    panel._rotate_rows["x"].spin.setValue(10.0)

    assert panel._undo_btn.isEnabled()
    assert not panel._redo_btn.isEnabled()

    panel._undo_btn.click()

    assert not panel._undo_btn.isEnabled()
    assert panel._redo_btn.isEnabled()


def test_esc_cancels_a_drag(panel, layer):
    original = layer.data
    panel._rotate_rows["x"].spin.setValue(10.0)
    start = panel.state.matrix()
    row = panel._rotate_rows["y"]
    _drag(row, _angle_to_int(row, 20.0), release=False)

    panel._cancel_drag()

    np.testing.assert_allclose(panel.state.matrix(), start)
    assert row.slider.value() == 0
    assert row.spin.value() == 0.0

    # The mouse is still down: further moves and the release must not change anything.
    row.slider.setSliderPosition(_angle_to_int(row, 30.0))
    row.slider.setSliderDown(False)

    np.testing.assert_allclose(panel.state.matrix(), start)
    assert row.slider.value() == 0
    panel._undo_btn.click()
    assert panel.state.is_identity()
    assert layer.data is original


def test_esc_cancels_a_translation_drag(panel, layer):
    panel._translate_rows["z"].spin.setValue(0.5)
    start = layer.translate.copy()
    row = panel._translate_rows["z"]
    _drag(row, row.slider.value() + 100, release=False)

    panel._cancel_drag()

    np.testing.assert_allclose(layer.translate, start)
    assert row.spin.value() == pytest.approx(0.5)
    row.slider.setSliderPosition(row.slider.value() + 50)
    row.slider.setSliderDown(False)
    np.testing.assert_allclose(layer.translate, start)
    assert row.slider.value() == round(0.5 / row.resolution)


def test_esc_without_a_drag_does_nothing(panel):
    panel._rotate_rows["x"].spin.setValue(10.0)
    start = panel.state.matrix()

    panel._cancel_drag()

    np.testing.assert_array_equal(panel.state.matrix(), start)


def test_release_flushes_a_pending_update(panel, layer):
    row = panel._rotate_rows["x"]
    position = _angle_to_int(row, 18.0)

    # No event loop between the move and the release, so the timer cannot have fired.
    _drag(row, position)

    expected = TransformState.identity(center=panel.state.center).rotated(
        "x", position * row.resolution, frame="global"
    )
    np.testing.assert_allclose(panel.state.matrix(), expected.matrix())
    plane = np.asarray(layer.data)
    fresh = _fresh_display(panel, expected.matrix())
    np.testing.assert_allclose(plane, fresh, atol=1e-10)


def _fresh_display(panel, matrix):
    """Return what `plot_napari` displays for `matrix` on the panel's source."""
    array, _, _ = build_napari_layer_data(
        panel._target.session.source.fusi.affine.apply(matrix),
        resample_fill_value=panel._target.session.fill_value,
    )
    return np.asarray(array)


def test_updates_coalesce(panel, monkeypatch):
    row = panel._rotate_rows["x"]
    row.slider.setSliderDown(True)
    calls = []
    session = panel._target.session
    drag_to = session.drag_to
    monkeypatch.setattr(session, "drag_to", lambda m: (calls.append(m), drag_to(m)))

    for angle in range(1, 11):
        row.slider.setSliderPosition(_angle_to_int(row, float(angle)))

    assert calls == []

    panel._flush_update()

    assert len(calls) == 1
    expected = TransformState.identity(center=panel.state.center).rotated(
        "x", _angle_to_int(row, 10.0) * row.resolution, frame="global"
    )
    np.testing.assert_allclose(calls[0], expected.matrix())


def test_coalesced_update_runs_from_the_event_loop(panel, qtbot, monkeypatch):
    row = panel._rotate_rows["x"]
    row.slider.setSliderDown(True)
    calls = []
    monkeypatch.setattr(panel._target.session, "drag_to", calls.append)

    row.slider.setSliderPosition(_angle_to_int(row, 5.0))

    qtbot.waitUntil(lambda: len(calls) == 1, timeout=2000)


def test_global_local_toggle_changes_rotation_axes(panel):
    panel._rotate_rows["x"].spin.setValue(90.0)
    start = panel.state

    panel._frame_local_radio.setChecked(True)
    panel._rotate_rows["z"].spin.setValue(10.0)

    np.testing.assert_allclose(
        panel.state.matrix(), start.rotated("z", 10.0, frame="local").matrix()
    )

    panel._undo_btn.click()
    panel._frame_global_radio.setChecked(True)
    panel._rotate_rows["z"].spin.setValue(10.0)

    np.testing.assert_allclose(
        panel.state.matrix(), start.rotated("z", 10.0, frame="global").matrix()
    )


def test_local_frame_translation_rows_show_local_offsets(panel):
    panel._rotate_rows["x"].spin.setValue(90.0)
    panel._translate_rows["z"].spin.setValue(2.0)
    assert panel.state.translation[0] == pytest.approx(2.0)

    panel._frame_local_radio.setChecked(True)

    # +90 degrees about x maps local +y onto world -z.
    assert panel._translate_rows["y"].spin.value() == pytest.approx(-2.0)
    assert panel._translate_rows["z"].spin.value() == pytest.approx(0.0)

    panel._translate_rows["z"].spin.setValue(1.0)

    # Local +z is world +y.
    np.testing.assert_allclose(panel.state.translation, (2.0, 1.0, 0.0), atol=1e-12)


def test_reset_matches_the_original_layer(panel, layer):
    original = layer.data
    scale, translate = layer.scale.copy(), layer.translate.copy()
    panel._rotate_rows["x"].spin.setValue(30.0)
    panel._translate_rows["y"].spin.setValue(0.7)

    panel._reset_btn.click()

    assert layer.data is original
    np.testing.assert_array_equal(layer.scale, scale)
    np.testing.assert_array_equal(layer.translate, translate)
    assert panel.state.is_identity()
    assert panel._translate_rows["y"].spin.value() == 0.0
    assert not panel._undo_btn.isEnabled()


def test_target_excludes_pose_and_plain_layers(
    qtbot, viewer, layer, sample_voxeldata_3dt_pose
):
    plain = viewer.add_image(np.zeros((3, 4, 5)), name="plain")
    pose = viewer.add_image(
        np.zeros(sample_voxeldata_3dt_pose.shape),
        metadata={"xarray": sample_voxeldata_3dt_pose},
        name="pose",
    )
    points = viewer.add_points(np.zeros((1, 3)), name="points")

    widget = TransformPanel(viewer)
    qtbot.addWidget(widget)

    combo = widget._target_combo
    assert combo.findData(layer) >= 0
    for excluded in (plain, pose, points):
        assert combo.findData(excluded) < 0


def test_labels_layers_are_targets(qtbot, viewer, volume):
    labels = _add_layer(viewer, volume.copy(data=(volume.values > 3).astype(np.int32)))

    widget = TransformPanel(viewer)
    qtbot.addWidget(widget)

    assert widget._target_combo.findData(labels) >= 0


def test_painted_labels_show_a_message_and_disable_controls(qtbot, viewer, volume):
    labels = _add_layer(viewer, volume.copy(data=(volume.values > 3).astype(np.int32)))
    labels.data = labels.data.copy()
    labels.data[0, 0, 0] = 9

    widget = TransformPanel(viewer)
    qtbot.addWidget(widget)
    _select(widget._target_combo, labels)

    assert "paint" in widget._status_label.text()
    assert not widget._translate_rows["z"].isEnabled()
    assert not widget._rotate_rows["z"].isEnabled()


def test_no_target_disables_controls(qtbot, viewer):
    widget = TransformPanel(viewer)
    qtbot.addWidget(widget)

    assert widget._target_combo.count() == 0
    assert not widget._translate_rows["z"].isEnabled()
    assert not widget._reset_btn.isEnabled()


def test_a_layer_added_later_becomes_the_target(qtbot, viewer, volume):
    widget = TransformPanel(viewer)
    qtbot.addWidget(widget)

    added = _add_layer(viewer, volume)

    assert widget._target_combo.currentData() is added
    assert widget._translate_rows["z"].isEnabled()


def test_switching_target_keeps_each_layers_state(qtbot, panel, viewer, layer, volume):
    other = _add_layer(viewer, volume, name="other")
    panel._rotate_rows["x"].spin.setValue(10.0)
    rotated = panel.state.matrix()

    _select(panel._target_combo, other)

    assert panel.state.is_identity()
    assert not panel._undo_btn.isEnabled()

    panel._translate_rows["z"].spin.setValue(0.5)
    _select(panel._target_combo, layer)

    np.testing.assert_allclose(panel.state.matrix(), rotated)
    assert panel._undo_btn.isEnabled()
    assert panel._translate_rows["z"].spin.value() == 0.0

    _select(panel._target_combo, other)

    assert panel._translate_rows["z"].spin.value() == pytest.approx(0.5)


def test_removing_the_target_moves_on(panel, viewer, layer, volume):
    other = _add_layer(viewer, volume, name="other")
    panel._rotate_rows["x"].spin.setValue(10.0)

    viewer.layers.remove(layer)

    assert panel._target_combo.findData(layer) < 0
    assert panel._target_combo.currentData() is other
    assert panel.state.is_identity()


def test_not_applied_label_follows_the_state(panel):
    assert panel._not_applied_label.isHidden()

    panel._rotate_rows["x"].spin.setValue(10.0)

    assert not panel._not_applied_label.isHidden()

    panel._undo_btn.click()

    assert panel._not_applied_label.isHidden()


def test_panel_is_no_wider_than_the_registration_panel(qtbot, viewer, layer):
    transform_panel = TransformPanel(viewer)
    registration_panel = RegistrationPanel(viewer)
    qtbot.addWidget(transform_panel)
    qtbot.addWidget(registration_panel)
    stylesheet = get_stylesheet(get_settings().appearance.theme)
    transform_panel.setStyleSheet(stylesheet)
    registration_panel.setStyleSheet(stylesheet)
    QApplication.processEvents()

    assert (
        transform_panel.minimumSizeHint().width()
        <= registration_panel.minimumSizeHint().width()
    )


def _wait_for_worker(panel, qtbot):
    """Wait until the panel's worker has delivered its result."""
    qtbot.waitUntil(lambda: panel._worker is None, timeout=20000)


def test_rotation_keeps_the_center_fixed(panel, viewer):
    point = np.array([2.0, 2.5, 3.5])
    panel._translate_rows["y"].spin.setValue(0.3)
    viewer.cursor.position = tuple(point)
    panel._center_cursor_btn.click()
    before = panel.state.matrix()

    panel._rotate_rows["x"].spin.setValue(33.0)

    # Undoing the previous transform leaves the step alone, which must fix the point.
    step = panel.state.matrix() @ np.linalg.inv(before)
    np.testing.assert_allclose((step @ [*point, 1.0])[:3], point)
    np.testing.assert_allclose(panel.state.pivot(), point)


@pytest.mark.parametrize(
    "button", ["_center_origin_btn", "_center_cursor_btn", "_center_geometry_btn"]
)
def test_moving_the_center_does_not_move_the_volume(panel, layer, viewer, button):
    viewer.cursor.position = (1.5, 2.5, 3.5)
    panel._rotate_rows["x"].spin.setValue(25.0)
    panel._translate_rows["z"].spin.setValue(0.4)
    matrix = panel.state.matrix()
    translate = layer.translate.copy()
    view = layer.data

    getattr(panel, button).click()

    np.testing.assert_allclose(panel.state.matrix(), matrix)
    np.testing.assert_array_equal(layer.translate, translate)
    assert layer.data is view


def test_moving_the_center_is_not_an_undo_step(panel):
    panel._rotate_rows["x"].spin.setValue(25.0)
    matrix = panel.state.matrix()

    panel._center_geometry_btn.click()

    np.testing.assert_allclose(panel.state.matrix(), matrix)
    panel._undo_btn.click()
    assert panel.state.is_identity()


def test_center_starts_at_the_grid_origin(panel):
    np.testing.assert_allclose(panel.state.pivot(), (1.0, 2.0, 3.0))
    assert panel._center_readout.text().startswith("1")


def test_geometry_center_follows_the_transform(panel, volume):
    panel._translate_rows["z"].spin.setValue(1.0)

    panel._center_geometry_btn.click()

    expected = volume.fusi.affine.geometry_center.to_numpy() + (1.0, 0.0, 0.0)
    np.testing.assert_allclose(panel.state.pivot(), expected)


def test_grid_origin_follows_the_transform(panel):
    panel._center_geometry_btn.click()
    panel._rotate_rows["x"].spin.setValue(90.0)

    panel._center_origin_btn.click()

    origin = (panel.state.matrix() @ [1.0, 2.0, 3.0, 1.0])[:3]
    np.testing.assert_allclose(panel.state.pivot(), origin)


def test_center_readout_follows_the_pivot(panel):
    panel._translate_rows["z"].spin.setValue(1.0)

    # The center moves along with the volume, from z = 1.
    assert panel._center_readout.text().startswith("2")


def test_marker_follows_the_pivot(panel, viewer, layer):
    viewer.layers.selection.active = layer
    panel._center_marker_check.setChecked(True)
    marker = panel._marker_layer

    assert marker in viewer.layers
    assert viewer.layers.selection.active is layer
    np.testing.assert_allclose(marker.data[0], panel.state.pivot())

    panel._center_geometry_btn.click()
    np.testing.assert_allclose(marker.data[0], panel.state.pivot())

    panel._translate_rows["y"].spin.setValue(0.5)
    np.testing.assert_allclose(marker.data[0], panel.state.pivot())

    panel._center_marker_check.setChecked(False)
    assert marker not in viewer.layers


def test_deleting_the_marker_layer_unchecks_the_box(panel, viewer):
    panel._center_marker_check.setChecked(True)

    viewer.layers.remove(panel._marker_layer)

    assert not panel._center_marker_check.isChecked()
    panel._translate_rows["y"].spin.setValue(0.5)


def test_zoom_matches_apply_zoom(panel, volume):
    panel._zoom_spins["z"].setValue(2.0)
    panel._zoom_spins["y"].setValue(0.5)
    panel._zoom_spins["x"].setValue(1.25)

    expected = apply_zoom(volume, (2.0, 0.5, 1.25), center="origin")
    np.testing.assert_allclose(
        panel.state.matrix() @ volume.fusi.affine.voxel_to_world,
        expected.fusi.affine.voxel_to_world,
    )


def test_uniform_zoom_scales_the_layer(panel, layer):
    scale = layer.scale.copy()
    panel._zoom_lock_check.setChecked(True)

    panel._zoom_spins["y"].setValue(2.0)

    assert [spin.value() for spin in panel._zoom_spins.values()] == [2.0, 2.0, 2.0]
    np.testing.assert_allclose(layer.scale, scale * 2.0)


def test_zoom_is_one_undo_step(panel):
    panel._zoom_lock_check.setChecked(True)
    panel._zoom_spins["y"].setValue(2.0)

    panel._undo_btn.click()

    assert panel.state.is_identity()
    assert panel._zoom_spins["z"].value() == 1.0


def test_flip_matches_a_negative_zoom_factor(panel, volume):
    panel._flip_btns["y"].click()

    expected = apply_zoom(volume, (1.0, -1.0, 1.0), center="origin")
    np.testing.assert_allclose(
        panel.state.matrix() @ volume.fusi.affine.voxel_to_world,
        expected.fusi.affine.voxel_to_world,
    )


def test_zoom_section_starts_collapsed(panel):
    assert panel._zoom_content.isHidden()

    panel._zoom_toggle.setChecked(True)

    assert not panel._zoom_content.isHidden()


def test_apply_matches_affine_apply(panel, layer, volume, qtbot):
    panel._rotate_rows["x"].spin.setValue(20.0)
    panel._translate_rows["y"].spin.setValue(0.5)
    matrix = panel.state.matrix()
    previewed = np.asarray(layer.data)
    pivot = panel.state.pivot()

    panel._apply_btn.click()
    _wait_for_worker(panel, qtbot)

    expected = volume.fusi.affine.apply(matrix)
    source = layer.metadata["source_xarray"]
    np.testing.assert_array_equal(
        source.fusi.affine.voxel_to_world, expected.fusi.affine.voxel_to_world
    )
    np.testing.assert_array_equal(source.values, volume.values)
    # Nothing moves on screen, and the panel starts over from the new pose.
    np.testing.assert_allclose(np.asarray(layer.data), previewed, atol=1e-10)
    assert isinstance(layer.data, np.ndarray)
    assert panel.state.is_identity()
    np.testing.assert_allclose(panel.state.pivot(), pivot)
    assert panel._not_applied_label.isHidden()
    assert not panel._undo_btn.isEnabled()


def test_apply_re_expresses_stored_affines(qtbot, viewer, volume):
    stored = np.eye(4)
    stored[0, 3] = 5.0
    layer = _add_layer(viewer, volume.assign_attrs(affines={"to_lab": stored}))
    widget = TransformPanel(viewer)
    qtbot.addWidget(widget)
    _select(widget._target_combo, layer)
    widget._rotate_rows["x"].spin.setValue(20.0)
    assert widget.state is not None
    matrix = widget.state.matrix()

    widget._apply_btn.click()
    _wait_for_worker(widget, qtbot)

    np.testing.assert_allclose(
        layer.metadata["source_xarray"].attrs["affines"]["to_lab"],
        stored @ np.linalg.inv(matrix),
    )


def test_reset_after_apply_keeps_the_committed_pose(panel, layer, qtbot):
    panel._rotate_rows["x"].spin.setValue(20.0)
    panel._apply_btn.click()
    _wait_for_worker(panel, qtbot)
    committed = layer.data
    panel._translate_rows["y"].spin.setValue(0.5)

    panel._reset_btn.click()

    assert layer.data is committed


def test_target_locked_during_apply(panel, qtbot):
    panel._rotate_rows["x"].spin.setValue(20.0)

    panel._apply_btn.click()

    assert not panel._target_combo.isEnabled()
    assert not panel._apply_btn.isEnabled()
    assert not panel._rotate_group.isEnabled()
    _wait_for_worker(panel, qtbot)
    assert panel._target_combo.isEnabled()
    assert panel._rotate_group.isEnabled()


def test_apply_after_the_layer_was_removed_is_harmless(
    panel, viewer, layer, volume, qtbot
):
    other = _add_layer(viewer, volume, name="other")
    original = layer.metadata["source_xarray"]
    panel._rotate_rows["x"].spin.setValue(20.0)

    panel._apply_btn.click()
    viewer.layers.remove(layer)
    _wait_for_worker(panel, qtbot)

    assert layer.metadata["source_xarray"] is original
    assert panel._target_combo.currentData() is other
    assert panel.state.is_identity()
    assert other.metadata["source_xarray"] is not original


def test_apply_error_reports_and_re_enables_controls(panel, qtbot, monkeypatch):
    messages = []
    monkeypatch.setattr(transform_panel_module, "show_error", messages.append)

    def fail(*args, **kwargs):
        raise ValueError("no luck")

    monkeypatch.setattr(transform_panel_module, "build_applied_layer", fail)
    panel._rotate_rows["x"].spin.setValue(20.0)

    panel._apply_btn.click()
    _wait_for_worker(panel, qtbot)

    assert messages == ["no luck"]
    assert panel._apply_btn.isEnabled()
    assert not panel.state.is_identity()


def test_apply_refuses_paint_added_after_selecting_the_target(
    qtbot, viewer, volume, monkeypatch
):
    messages = []
    monkeypatch.setattr(transform_panel_module, "show_error", messages.append)
    # Float labels are cast for the layer, so the layer holds a copy of the source.
    labels = volume.copy(data=(volume.values > 3).astype(np.float64))
    layer = _add_layer(viewer, labels, layer_type="labels")
    widget = TransformPanel(viewer)
    qtbot.addWidget(widget)
    _select(widget._target_combo, layer)
    layer.data[2, 3, 4] = 9
    painted = layer.data
    source = layer.metadata["source_xarray"]
    widget._rotate_rows["x"].spin.setValue(20.0)

    widget._apply_btn.click()
    _wait_for_worker(widget, qtbot)

    assert len(messages) == 1
    assert "paint" in messages[0]
    assert layer.metadata["source_xarray"] is source
    widget._reset_btn.click()
    assert layer.data is painted
    assert np.count_nonzero(np.asarray(layer.data) == 9) == 1


def test_resample_refuses_paint_added_after_selecting_the_target(
    qtbot, viewer, volume, monkeypatch
):
    messages = []
    monkeypatch.setattr(transform_panel_module, "show_error", messages.append)
    labels = volume.copy(data=(volume.values > 3).astype(np.float64))
    layer = _add_layer(viewer, labels, layer_type="labels")
    widget = TransformPanel(viewer)
    qtbot.addWidget(widget)
    _select(widget._target_combo, layer)
    layer.data[2, 3, 4] = 9
    n_layers = len(viewer.layers)

    widget._resample_btn.click()
    _wait_for_worker(widget, qtbot)

    assert len(messages) == 1
    assert len(viewer.layers) == n_layers


def test_apply_without_a_transform_does_nothing(panel, layer):
    source = layer.metadata["source_xarray"]

    panel._apply_btn.click()

    assert panel._worker is None
    assert layer.metadata["source_xarray"] is source


def test_resample_adds_a_layer_and_leaves_the_original(panel, viewer, layer, qtbot):
    panel._rotate_rows["x"].spin.setValue(20.0)
    source = layer.metadata["source_xarray"]
    view = layer.data
    state = panel.state

    panel._resample_btn.click()
    _wait_for_worker(panel, qtbot)

    added = viewer.layers[-1]
    assert added is not layer
    assert added.name == f"{layer.name} (resampled)"
    assert added.metadata["source_xarray"].shape == added.data.shape
    np.testing.assert_allclose(added.data, np.asarray(view), atol=1e-10)
    assert tuple(added.contrast_limits) == tuple(layer.contrast_limits)
    assert layer.metadata["source_xarray"] is source
    assert layer.data is view
    assert panel.state is state
    assert panel._target_combo.currentData() is layer


def test_resample_labels_adds_a_labels_layer(qtbot, viewer, volume):
    labels = _add_layer(viewer, volume.copy(data=(volume.values > 3).astype(np.int32)))
    widget = TransformPanel(viewer)
    qtbot.addWidget(widget)
    _select(widget._target_combo, labels)
    widget._rotate_rows["x"].spin.setValue(20.0)

    widget._resample_btn.click()
    _wait_for_worker(widget, qtbot)

    added = viewer.layers[-1]
    assert isinstance(added, Labels)
    assert set(np.unique(added.data)) <= {0, 1}


def test_resample_uses_the_reference_grid(panel, viewer, qtbot):
    reference = create_voxeldata(
        np.zeros((5, 6, 7)),
        dims=("k", "j", "i"),
        spacing=(0.5, 0.2, 0.3),
        origin=(1.0, 2.0, 3.0),
    )
    reference_layer = _add_layer(viewer, reference, name="reference")
    _select(panel._reference_combo, reference_layer)
    panel._rotate_rows["x"].spin.setValue(5.0)

    panel._resample_btn.click()
    _wait_for_worker(panel, qtbot)

    added = viewer.layers[-1]
    assert added.data.shape == (5, 6, 7)
    np.testing.assert_allclose(added.scale, (0.5, 0.2, 0.3))


def test_reference_overlays_and_restores(panel, viewer, layer, volume):
    other = _add_layer(viewer, volume, name="other")
    other.visible = False
    layer.blending = "translucent"
    assert viewer.layers.index(other) > viewer.layers.index(layer)

    _select(panel._reference_combo, other)

    assert other.visible
    assert viewer.layers.index(other) < viewer.layers.index(layer)
    assert layer.blending == "additive"

    panel._reference_combo.setCurrentIndex(0)

    assert panel._reference_combo.currentData() is None
    assert layer.blending == "translucent"


def test_reference_combo_excludes_the_target(panel, viewer, layer, volume):
    other = _add_layer(viewer, volume, name="other")

    assert panel._reference_combo.findData(other) >= 0
    assert panel._reference_combo.findData(layer) < 0

    _select(panel._target_combo, other)

    assert panel._reference_combo.findData(layer) >= 0
    assert panel._reference_combo.findData(other) < 0


def test_matrix_view_shows_the_transform(panel):
    assert parse_matrix(panel._matrix_view.toPlainText()) == pytest.approx(np.eye(4))

    panel._rotate_rows["x"].spin.setValue(30.0)

    np.testing.assert_allclose(
        parse_matrix(panel._matrix_view.toPlainText()), panel.state.matrix(), atol=1e-4
    )


def test_copy_then_paste_reproduces_the_pose(panel):
    panel._rotate_rows["x"].spin.setValue(30.0)
    panel._translate_rows["y"].spin.setValue(0.25)
    panel._flip_btns["z"].click()
    expected = panel.state.matrix()
    panel._copy_btn.click()
    panel._reset_btn.click()
    assert panel.state.is_identity()

    panel._paste_btn.click()

    np.testing.assert_allclose(panel.state.matrix(), expected)
    assert not panel.state.is_sheared


def test_paste_is_one_undo_step(panel):
    _set_clipboard_text(format_matrix(np.diag([1.0, 2.0, 3.0, 1.0])))

    panel._paste_btn.click()
    assert not panel.state.is_identity()

    panel._undo_btn.click()
    assert panel.state.is_identity()


def test_pasting_a_sheared_matrix_enters_affine_mode(panel):
    matrix = np.eye(4)
    matrix[0, 1] = 0.3
    _set_clipboard_text(format_matrix(matrix))

    panel._paste_btn.click()

    np.testing.assert_array_equal(panel.state.matrix(), matrix)
    assert not panel._translate_group.isEnabled()
    assert not panel._rotate_group.isEnabled()
    assert not panel._zoom_group.isEnabled()
    assert "Affine" in panel._status_label.text()
    assert panel._apply_btn.isEnabled()

    panel._undo_btn.click()

    assert panel._translate_group.isEnabled()
    assert panel._status_label.isHidden()


def test_pasting_invalid_text_keeps_the_state(panel, monkeypatch):
    messages = []
    monkeypatch.setattr(transform_panel_module, "show_error", messages.append)
    panel._rotate_rows["x"].spin.setValue(30.0)
    expected = panel.state.matrix()
    _set_clipboard_text("not a matrix")

    panel._paste_btn.click()

    np.testing.assert_array_equal(panel.state.matrix(), expected)
    assert len(messages) == 1


def test_save_then_load_reproduces_the_pose(panel, tmp_path):
    panel._rotate_rows["x"].spin.setValue(30.0)
    panel._translate_rows["y"].spin.setValue(0.25)
    expected = panel.state.matrix()
    path = tmp_path / "pose.json"

    panel._save_matrix(path)
    panel._reset_btn.click()
    panel._load_matrix(path)

    np.testing.assert_allclose(panel.state.matrix(), expected)


def test_saved_payload_holds_the_pull_transform(panel, tmp_path):
    panel._translate_rows["y"].spin.setValue(0.25)
    path = tmp_path / "pose.json"

    panel._save_matrix(path)

    # The registration panel reads these files, and applies them as pull transforms.
    pull = get_affine_transform_from_payload(load_transform_payload(path))
    np.testing.assert_allclose(pull @ panel.state.matrix(), np.eye(4), atol=1e-12)


def test_save_time_series_stores_a_spatial_grid(qtbot, viewer, volume, tmp_path):
    series = create_voxeldata(
        np.stack([volume.values] * 3),
        dims=("time", "k", "j", "i"),
        time=np.arange(3, dtype=float),
        voxel_to_world=volume.fusi.affine.voxel_to_world,
    )
    layer = _add_layer(viewer, series, name="series")
    widget = TransformPanel(viewer)
    qtbot.addWidget(widget)
    _select(widget._target_combo, layer)
    path = tmp_path / "pose.json"

    widget._save_matrix(path)

    assert load_transform_payload(path)["output_grid"]["shape"] == list(volume.shape)


def test_loading_a_bad_file_keeps_the_state(panel, tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(transform_panel_module, "show_error", messages.append)
    path = tmp_path / "bad.json"
    path.write_text("{}")

    panel._load_matrix(path)

    assert panel.state.is_identity()
    assert len(messages) == 1


def test_itk_export_maps_the_same_points(panel, tmp_path):
    panel._rotate_rows["x"].spin.setValue(30.0)
    panel._translate_rows["y"].spin.setValue(0.25)
    path = tmp_path / "pose.tfm"
    source = np.array([1.0, 2.0, 3.0])
    target = (panel.state.matrix() @ [*source, 1.0])[:3]

    panel._save_matrix(path)

    def to_lps(point):
        return np.diag([-1.0, -1.0, 1.0]) @ point[::-1]

    mapped = sitk.ReadTransform(str(path)).TransformPoint(to_lps(target).tolist())
    np.testing.assert_allclose(mapped, to_lps(source))


def test_invert_composes_to_identity(panel):
    panel._rotate_rows["x"].spin.setValue(30.0)
    panel._translate_rows["y"].spin.setValue(0.25)
    matrix = panel.state.matrix()

    panel._invert_btn.click()

    np.testing.assert_allclose(panel.state.matrix() @ matrix, np.eye(4), atol=1e-12)


def test_identity_keeps_the_pivot_and_is_an_undo_step(panel, layer):
    original = layer.data
    panel._center_geometry_btn.click()
    pivot = panel.state.pivot()
    panel._rotate_rows["x"].spin.setValue(30.0)
    matrix = panel.state.matrix()

    panel._identity_btn.click()

    assert panel.state.is_identity()
    assert layer.data is original
    np.testing.assert_allclose(panel.state.pivot(), pivot)

    panel._undo_btn.click()
    np.testing.assert_allclose(panel.state.matrix(), matrix)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (Qt.Key.Key_Right, (0.0, 0.0, 1.0)),
        (Qt.Key.Key_Left, (0.0, 0.0, -1.0)),
        (Qt.Key.Key_Down, (0.0, 1.0, 0.0)),
        (Qt.Key.Key_Up, (0.0, -1.0, 0.0)),
        (Qt.Key.Key_PageUp, (1.0, 0.0, 0.0)),
        (Qt.Key.Key_PageDown, (-1.0, 0.0, 0.0)),
    ],
)
def test_keys_translate_along_the_displayed_axes(panel, qtbot, key, expected):
    _select(panel._translate_step_combo, 1.0)

    qtbot.keyClick(panel, key)

    np.testing.assert_allclose(panel.state.translation, expected)


def test_keys_follow_a_rolled_view(panel, viewer, qtbot):
    _select(panel._translate_step_combo, 1.0)
    # Displays (x, z), with y on the slider.
    viewer.dims.order = (1, 2, 0)

    qtbot.keyClick(panel, Qt.Key.Key_Right)
    np.testing.assert_allclose(panel.state.translation, (1.0, 0.0, 0.0))

    qtbot.keyClick(panel, Qt.Key.Key_Down)
    np.testing.assert_allclose(panel.state.translation, (1.0, 0.0, 1.0))

    qtbot.keyClick(panel, Qt.Key.Key_PageUp)
    np.testing.assert_allclose(panel.state.translation, (1.0, 1.0, 1.0))


def test_shift_and_alt_scale_the_key_step(panel, qtbot):
    _select(panel._translate_step_combo, 0.1)

    qtbot.keyClick(panel, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    assert panel.state.translation[2] == pytest.approx(1.0)

    qtbot.keyClick(panel, Qt.Key.Key_Right, Qt.KeyboardModifier.AltModifier)
    assert panel.state.translation[2] == pytest.approx(1.01)


def test_one_key_press_is_one_undo_step(panel, qtbot):
    qtbot.keyClick(panel, Qt.Key.Key_Right)

    panel._undo_btn.click()

    assert panel.state.is_identity()


def test_keys_do_nothing_without_a_target(qtbot, viewer):
    widget = TransformPanel(viewer)
    qtbot.addWidget(widget)

    qtbot.keyClick(widget, Qt.Key.Key_Right)

    assert widget.state is None


def test_keys_do_nothing_in_affine_mode(panel, qtbot):
    matrix = np.eye(4)
    matrix[0, 1] = 0.3
    _set_clipboard_text(format_matrix(matrix))
    panel._paste_btn.click()

    qtbot.keyClick(panel, Qt.Key.Key_Right)

    np.testing.assert_array_equal(panel.state.matrix(), matrix)


def test_wheel_over_a_slider_scrolls_the_sidebar_instead(panel):
    row = panel._rotate_rows["x"]
    event = QWheelEvent(
        QPointF(5.0, 5.0),
        QPointF(5.0, 5.0),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )

    QApplication.sendEvent(row.slider, event)

    assert not event.isAccepted()
    assert panel.state.is_identity()
    assert row.slider.value() == 0


def test_removing_the_reference_clears_it(panel, viewer, layer, volume):
    other = _add_layer(viewer, volume, name="other")
    layer.blending = "translucent"
    _select(panel._reference_combo, other)

    viewer.layers.remove(other)

    assert panel._reference_combo.currentData() is None
    assert layer.blending == "translucent"


def test_preview_keeps_the_displayed_slice(panel, viewer, layer):
    point = tuple(viewer.dims.point)
    row = panel._rotate_rows["x"]
    # napari snaps the slice to the planes of whichever grid is displayed, so the
    # position can move by up to half a voxel, never to another part of the volume.
    half_voxel = float(layer.scale[-3:].min()) / 2

    row.slider.setSliderDown(True)
    assert tuple(viewer.dims.point) == pytest.approx(point, abs=half_voxel)

    row.slider.setSliderPosition(_angle_to_int(row, 30.0))
    panel._flush_update()
    row.slider.setSliderDown(False)
    assert tuple(viewer.dims.point) == pytest.approx(point, abs=half_voxel)

    panel._reset_btn.click()
    assert tuple(viewer.dims.point) == pytest.approx(point, abs=half_voxel)


def test_value_boxes_fit_in_their_rows_under_the_stylesheet(panel):
    panel.setStyleSheet(get_stylesheet(get_settings().appearance.theme))
    panel.resize(380, panel.sizeHint().height())
    panel.show()
    for _ in range(5):
        QApplication.processEvents()

    for row in (*panel._translate_rows.values(), *panel._rotate_rows.values()):
        assert row.spin.geometry().right() <= row.width()


def test_rotation_dials_sit_side_by_side_with_their_boxes_below(panel):
    panel.setStyleSheet(get_stylesheet(get_settings().appearance.theme))
    panel.resize(380, panel.sizeHint().height())
    panel.show()
    for _ in range(5):
        QApplication.processEvents()

    rows = [panel._rotate_rows[name] for name in AXES]
    tops = [row.mapTo(panel, row.rect().topLeft()).y() for row in rows]
    lefts = [row.mapTo(panel, row.rect().topLeft()).x() for row in rows]
    assert len(set(tops)) == 1
    assert lefts == sorted(lefts) and len(set(lefts)) == 3
    for row in rows:
        assert row.spin.geometry().top() >= row.slider.geometry().bottom()
        assert row.plus_btn.geometry().left() >= row.slider.geometry().right()
        assert row.minus_btn.geometry().left() >= row.slider.geometry().right()
        assert row.plus_btn.geometry().bottom() <= row.minus_btn.geometry().top()
        assert abs(row.spin.geometry().left() - row.slider.geometry().left()) <= 1
        assert abs(row.spin.geometry().right() - row.plus_btn.geometry().right()) <= 1

