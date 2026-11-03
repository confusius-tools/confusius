"""Tests for the VideoPanel widget and its dedicated video viewer.

`_VideoArray` is tested as a pure-logic unit.  Panel tests use real napari
viewers via `make_napari_viewer` with a stubbed `VideoReaderNP`.

See https://napari.org/dev/plugins/testing_and_publishing/test.html
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtWidgets import QWidget

import confusius._napari._video._video_panel as panel_module
from confusius._napari._data._load_panel import DataPanel
from confusius._napari._timeaxis import find_time_dim_index, read_current_time
from confusius._napari._video._video_panel import VideoPanel
from confusius._napari._video._video_viewer import VideoViewer, _VideoArray
from confusius.plotting import plot_napari

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeVideo:
    """Minimal VideoReaderNP stub."""

    def __init__(self, n_frames=20, h=48, w=64, rgb=False):
        c = (3,) if rgb else ()
        self.number_of_frames = n_frames
        self.frame_rate = 30.0
        self.reads = []
        self._frames = np.random.default_rng(0).integers(
            0, 255, size=(n_frames, h, w, *c), dtype=np.uint8
        )
        self.frame_shape = self._frames.shape[1:]
        self.dtype = self._frames.dtype

    def __getitem__(self, idx):
        self.reads.append(idx)
        return self._frames[idx]


def _arr(fv, **kwargs):
    """Build a `_VideoArray` over a stub reader (untyped so `ty` accepts it)."""
    return _VideoArray(fv, **kwargs)


# ---------------------------------------------------------------------------
# _VideoArray
# ---------------------------------------------------------------------------


class TestVideoArray:
    def test_integer_index_matches_source_frame(self):
        fv = _FakeVideo(n_frames=5, h=4, w=6)
        arr = _arr(fv)
        np.testing.assert_array_equal(arr[3], fv[3])

    def test_slice_index_stacks_frames(self):
        fv = _FakeVideo(n_frames=10, h=4, w=6)
        arr = _arr(fv)
        np.testing.assert_array_equal(arr[2:5], fv[2:5])

    def test_tuple_index_with_spatial_slice(self):
        fv = _FakeVideo(n_frames=5, h=8, w=10)
        arr = _arr(fv)
        np.testing.assert_array_equal(arr[1, 2:4, 3:7], fv[1, 2:4, 3:7])

    def test_shape_dtype_len(self):
        arr = _arr(_FakeVideo(n_frames=7, h=4, w=6, rgb=True))
        assert arr.shape == (7, 4, 6, 3)
        assert arr.dtype == np.uint8
        assert arr.ndim == 4
        assert len(arr) == 7

    def test_step_subsamples_frames(self):
        fv = _FakeVideo(n_frames=10, h=4, w=6)
        arr = _arr(fv, step=3)
        assert arr.shape[0] == 4
        np.testing.assert_array_equal(arr[2], fv[6])

    def test_empty_slice(self):
        arr = _arr(_FakeVideo(n_frames=5, h=4, w=6))
        assert arr[3:3].shape == (0, 4, 6)

    def test_unbounded_time_slice_returns_single_frame(self):
        fv = _FakeVideo(n_frames=50, h=4, w=6)
        arr = _arr(fv)
        out = arr[:]
        assert out.shape == (1, 4, 6)
        np.testing.assert_array_equal(out[0], fv[0])

    def test_unsupported_index_raises(self):
        arr = _arr(_FakeVideo(n_frames=5, h=4, w=6))
        with pytest.raises(IndexError):
            arr[np.array([0, 1])]


# ---------------------------------------------------------------------------
# Panel integration
# ---------------------------------------------------------------------------


@pytest.fixture
def viewer(make_napari_viewer, sample_voxeldata_3dt):
    """Main viewer with the sample fUSI volume loaded (time axis 10-14.5 s)."""
    v = make_napari_viewer()
    plot_napari(
        sample_voxeldata_3dt, viewer=v, show_colorbar=False, show_scale_bar=False
    )
    return v


@pytest.fixture
def panel(viewer):
    p = VideoPanel(viewer)
    yield p
    # napari's fixture fails on any QtViewer alive at teardown.
    if p._video_viewer is not None:
        p._video_viewer.close()


@pytest.fixture
def add_video(panel, monkeypatch, tmp_path):
    """Return a function that adds a stubbed video through the video panel."""

    def _add(fv: _FakeVideo, name: str = "clip"):
        path = tmp_path / f"{name}.mp4"
        path.touch()
        monkeypatch.setattr(panel_module, "VideoReaderNP", lambda p, **kw: fv)
        panel.add_video(path)
        return panel._videos[-1]

    return _add


def _delete_video(panel, name: str) -> None:
    """Select the video layer called `name` and press the viewer's bin button."""
    video_viewer = panel._video_viewer
    layers = video_viewer.model.layers
    layers.selection = {layers[name]}
    video_viewer.layer_buttons.deleteButton.click()


def test_add_video_opens_docks_and_leaves_main_viewer_alone(viewer, panel, add_video):
    entry = add_video(_FakeVideo(n_frames=10, rgb=True))
    video_viewer = panel._video_viewer
    assert viewer.window.dock_widgets["Video"] is video_viewer.qt_viewer
    assert viewer.window.dock_widgets["video layer list"] is video_viewer.controls
    assert entry.layer.data.shape == (10, 48, 64, 3)
    assert entry.layer.rgb
    np.testing.assert_allclose(entry.layer.scale, (1 / 30, 1.0, 1.0))
    assert len(viewer.layers) == 1
    assert not viewer.canvas.grid.enabled
    assert panel.isAncestorOf(video_viewer.controls) is False


def test_add_video_does_not_move_main_time(viewer, panel, add_video):
    t_idx = find_time_dim_index(viewer)
    before = viewer.dims.point[t_idx]
    add_video(_FakeVideo(n_frames=300))
    assert viewer.dims.point[t_idx] == before


def test_missing_file_shows_error_and_opens_nothing(panel, tmp_path, monkeypatch):
    errors = []
    monkeypatch.setattr(panel_module, "show_error", errors.append)
    panel.add_video(tmp_path / "nope.mp4")
    assert errors and "not found" in errors[0]
    assert panel._video_viewer is None


def test_undecodable_video_shows_error_and_opens_nothing(panel, tmp_path, monkeypatch):
    class _UndecodableVideo(_FakeVideo):
        def __getitem__(self, idx):
            raise ValueError("cannot decode frame")

    errors = []
    monkeypatch.setattr(panel_module, "show_error", errors.append)
    monkeypatch.setattr(
        panel_module, "VideoReaderNP", lambda p, **kw: _UndecodableVideo()
    )
    path = tmp_path / "truncated.mp4"
    path.touch()
    panel.add_video(path)
    assert errors and "cannot decode frame" in errors[0]
    assert panel._video_viewer is None


def test_only_video_errors_show_notifications_on_video_canvas(
    viewer, panel, add_video, tmp_path
):
    from napari._qt.dialogs.qt_notification import NapariQtNotification

    add_video(_FakeVideo())
    canvas = panel._video_viewer.qt_viewer.canvas.native
    # Narrower than napari's 400 px toast.
    canvas.resize(300, 200)
    panel._video_viewer.frame_step_spin.setValue(2)
    assert not canvas.findChildren(NapariQtNotification)
    assert not viewer.window._qt_viewer.canvas.native.findChildren(NapariQtNotification)

    panel.add_video(tmp_path / "nope.mp4")
    toasts = canvas.findChildren(NapariQtNotification)
    assert len(toasts) == 1
    assert "File not found" in toasts[0].message.text()
    for toast in toasts:
        assert 0 <= toast.x()
        assert toast.geometry().right() < canvas.width()
    assert not viewer.window._qt_viewer.canvas.native.findChildren(NapariQtNotification)


@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("video_first", [False, True])
def test_irregular_time_warning(
    viewer, panel, add_video, monkeypatch, irregular, video_first
):
    from napari.utils.notifications import NotificationSeverity

    notify = MagicMock()
    monkeypatch.setattr(VideoViewer, "notify", notify)
    scan = viewer.layers[0].metadata["xarray"]
    times = scan.time.values.copy()
    if irregular:
        times[5:] += 1.0
    scan = scan.assign_coords(time=scan.time.copy(data=times))
    viewer.layers.clear()
    if video_first:
        add_video(_FakeVideo(n_frames=450))
    with pytest.warns(UserWarning, match="non-uniform") if irregular else nullcontext():
        plot_napari(scan, viewer=viewer, show_colorbar=False, show_scale_bar=False)
    if not video_first:
        notify.assert_not_called()
        add_video(_FakeVideo(n_frames=450))

    if irregular:
        notify.assert_called_once()
        message, severity = notify.call_args.args
        assert "not regularly sampled" in message
        assert "linear approximation and may drift" in message
        assert severity == NotificationSeverity.WARNING
    else:
        notify.assert_not_called()

    calls = notify.call_count
    viewer.dims.set_point(find_time_dim_index(viewer), 12.0)
    add_video(_FakeVideo(n_frames=450), "second")
    assert notify.call_count == calls


def test_two_videos_share_video_grid(viewer, panel, add_video):
    add_video(_FakeVideo(), "a")
    add_video(_FakeVideo(h=10, w=200), "b")
    model = panel._video_viewer.model
    assert [layer.name for layer in model.layers] == ["Video: a", "Video: b"]
    assert model.canvas.grid.enabled
    assert panel._video_viewer.layer_list.model().rowCount() == 2


def test_remove_last_video_closes_dock(viewer, panel, add_video):
    assert not panel._show_viewer_btn.isEnabled()
    assert not panel._show_layer_list_btn.isEnabled()
    add_video(_FakeVideo(), "a")
    add_video(_FakeVideo(), "b")
    assert not panel._show_viewer_btn.isEnabled()
    assert not panel._show_layer_list_btn.isEnabled()
    video_viewer = panel._video_viewer
    _delete_video(panel, "Video: a")
    assert [e.layer.name for e in panel._videos] == ["Video: b"]
    assert [layer.name for layer in video_viewer.model.layers] == ["Video: b"]

    _delete_video(panel, "Video: b")
    # The viewer is kept (opening another GL canvas later can crash) but hidden.
    assert panel._video_viewer is video_viewer
    assert video_viewer.dock.isHidden()
    assert video_viewer.controls_dock.isHidden()
    assert not panel._show_viewer_btn.isEnabled()
    assert not panel._show_layer_list_btn.isEnabled()

    add_video(_FakeVideo(), "c")
    assert panel._video_viewer is video_viewer
    assert not video_viewer.dock.isHidden()
    assert not video_viewer.controls_dock.isHidden()
    assert [layer.name for layer in video_viewer.model.layers] == ["Video: c"]


def test_closed_video_windows_can_be_reopened_from_data_io(viewer, panel, add_video):
    from qtpy.QtWidgets import QApplication

    window = viewer.window._qt_window
    window.setUpdatesEnabled(False)
    window.show()
    add_video(_FakeVideo())
    QApplication.processEvents()
    video_viewer = panel._video_viewer
    assert panel._show_layer_list_btn.text() == "Show video list"
    assert not panel._show_viewer_btn.isEnabled()
    assert not panel._show_layer_list_btn.isEnabled()

    controls_dock = video_viewer.controls_dock
    controls_dock.titleBarWidget().close_button.click()
    assert controls_dock.isHidden()
    assert panel._show_layer_list_btn.isEnabled()
    assert not panel._show_viewer_btn.isEnabled()
    # Closing hides the dock instead of deleting it, so it can come back.
    assert viewer.window.dock_widgets["video layer list"] is video_viewer.controls
    panel._show_layer_list_btn.click()
    assert not controls_dock.isHidden()
    assert not panel._show_layer_list_btn.isEnabled()

    video_dock = video_viewer.dock
    video_dock.titleBarWidget().close_button.click()
    assert video_dock.isHidden()
    assert panel._show_viewer_btn.isEnabled()
    assert not panel._show_layer_list_btn.isEnabled()
    assert viewer.window.dock_widgets["Video"] is video_viewer.qt_viewer
    panel._show_viewer_btn.click()
    assert not video_dock.isHidden()
    assert not panel._show_viewer_btn.isEnabled()


def test_layer_removed_from_the_model_is_forgotten(viewer, panel, add_video):
    a = add_video(_FakeVideo(), "a")
    add_video(_FakeVideo(), "b")
    # Any removal path (keyboard Delete, scripting) goes through the model.
    panel._video_viewer.model.layers.remove(a.layer)
    assert [e.layer.name for e in panel._videos] == ["Video: b"]
    assert panel._video_viewer is not None


def test_removing_longer_video_does_not_move_main_time(viewer, panel, add_video):
    add_video(_FakeVideo(n_frames=30), "short")
    longer = add_video(_FakeVideo(n_frames=450), "long")
    t_idx = find_time_dim_index(viewer)
    viewer.dims.set_point(t_idx, 12.0)

    panel._video_viewer.model.layers.remove(longer.layer)

    assert viewer.dims.point[t_idx] == pytest.approx(12.0)
    assert panel._video_viewer.model.dims.point[0] == pytest.approx(29 / 30)
    panel._video_viewer.model.layers.clear()
    assert viewer.dims.point[t_idx] == pytest.approx(12.0)


def test_frame_step_decodes_only_the_frame_at_the_kept_time(viewer, panel, add_video):
    video = _FakeVideo(n_frames=450)
    entry = add_video(video)
    viewer.dims.set_point(find_time_dim_index(viewer), 12.0)
    video.reads.clear()

    panel._video_viewer.frame_step_spin.setValue(3)

    assert video.reads == [360]
    assert entry.layer.visible
    np.testing.assert_array_equal(entry.layer.data[120], video._frames[360])


def test_frame_step_leaves_hidden_videos_hidden_and_undecoded(
    viewer, panel, add_video
):
    video = _FakeVideo(n_frames=450)
    entry = add_video(video)
    viewer.dims.set_point(find_time_dim_index(viewer), 12.0)
    entry.layer.visible = False
    video.reads.clear()

    panel._video_viewer.frame_step_spin.setValue(3)

    assert not entry.layer.visible
    assert video.reads == []

    entry.layer.visible = True
    assert video.reads == [360]


@pytest.mark.parametrize("old_step, new_step", [(1, 5), (5, 1)])
def test_frame_step_change_keeps_running_playback_in_place(
    viewer, panel, add_video, qtbot, old_step, new_step
):
    add_video(_FakeVideo(n_frames=1800, h=2, w=2))
    video_viewer = panel._video_viewer
    video_viewer.frame_step_spin.setValue(old_step)
    dims = video_viewer.model.dims
    dims.set_point(0, 12.0)
    qt_dims = video_viewer.qt_viewer.dims
    qt_dims.play(0, fps=100)
    # Frames queue up in the old step's units while the event loop is busy.
    time.sleep(0.05)

    video_viewer.frame_step_spin.setValue(new_step)
    # Tests draw nothing, and only a draw lets the next queued frame through.
    dims._play_ready = True
    qtbot.wait(20)
    qt_dims.stop()

    assert dims.point[0] == pytest.approx(12.0, abs=1.0)
    # Playback must loop at the new end, not at the old step's.
    assert qt_dims._animation_thread.max_point == dims.nsteps[0]


def test_fast_frame_step_changes_keep_playback_running(
    viewer, panel, add_video, qtbot
):
    add_video(_FakeVideo(n_frames=1800, h=2, w=2))
    video_viewer = panel._video_viewer
    qt_dims = video_viewer.qt_viewer.dims
    qt_dims.play(0, fps=100)
    time.sleep(0.05)

    for step in range(2, 40):
        video_viewer.frame_step_spin.setValue(step)

    qtbot.waitUntil(
        lambda: qt_dims.slider_widgets[0].play_button.property("playing") == "True"
    )
    assert qt_dims._animation_thread.isRunning()
    qt_dims.stop()


def test_closed_video_dock_does_not_decode_until_reopened(
    viewer, panel, add_video, qtbot
):
    video = _FakeVideo(n_frames=450)
    add_video(video)
    video_viewer = panel._video_viewer
    video_dims = video_viewer.qt_viewer.dims
    t_idx = find_time_dim_index(viewer)
    # Away from the time reopening must catch up with, which would decode nothing.
    viewer.dims.set_point(t_idx, 11.0)
    video_dims.play(0)
    qtbot.waitUntil(lambda: video_dims.is_playing)

    video_viewer.dock.titleBarWidget().close_button.click()
    assert not video_dims.is_playing

    video.reads.clear()
    viewer.dims.set_point(t_idx, 12.0)
    assert video.reads == []

    video_viewer.show_viewer()
    assert video_viewer.model.dims.point[0] == pytest.approx(12.0, abs=1 / 30)
    assert 360 in video.reads


def test_play_buttons_of_both_viewers_share_time_playback(
    viewer, panel, add_video, qtbot
):
    add_video(_FakeVideo(n_frames=450))
    main_dims = viewer.window._qt_viewer.dims
    video_dims = panel._video_viewer.qt_viewer.dims
    main_button = main_dims.slider_widgets[find_time_dim_index(viewer)].play_button
    video_button = video_dims.slider_widgets[0].play_button

    for first_button, first_dims, second_button, second_dims in (
        (video_button, video_dims, main_button, main_dims),
        (main_button, main_dims, video_button, video_dims),
    ):
        # `_on_click` is what the buttons' mouse release runs.
        first_button._on_click()
        qtbot.waitUntil(lambda dims=first_dims: dims.is_playing)
        second_button._on_click()
        assert not first_dims.is_playing
        assert not second_dims.is_playing
        qtbot.waitUntil(
            lambda button=first_button: button.property("playing") == "False"
        )


def test_play_on_another_main_axis_is_independent_of_the_video(
    viewer, panel, add_video, qtbot
):
    add_video(_FakeVideo(n_frames=450))
    main_dims = viewer.window._qt_viewer.dims
    video_dims = panel._video_viewer.qt_viewer.dims
    time_axis = find_time_dim_index(viewer)
    assert time_axis is not None

    video_dims.slider_widgets[0].play_button._on_click()
    qtbot.waitUntil(lambda: video_dims.is_playing)
    main_dims.slider_widgets[time_axis + 1].play_button._on_click()
    qtbot.waitUntil(lambda: main_dims.is_playing)
    assert video_dims.is_playing
    main_dims.stop()
    video_dims.stop()


def test_play_buttons_stay_shared_after_main_sliders_are_rebuilt(
    viewer, panel, add_video, qtbot
):
    add_video(_FakeVideo(n_frames=450))
    main_dims = viewer.window._qt_viewer.dims
    video_dims = panel._video_viewer.qt_viewer.dims
    scan = viewer.layers[0].metadata["xarray"]
    viewer.layers.clear()
    plot_napari(scan, viewer=viewer, show_colorbar=False, show_scale_bar=False)

    video_dims.slider_widgets[0].play_button._on_click()
    qtbot.waitUntil(lambda: video_dims.is_playing)
    main_dims.slider_widgets[find_time_dim_index(viewer)].play_button._on_click()
    assert not video_dims.is_playing
    assert not main_dims.is_playing


def test_closing_video_viewer_restores_main_play_buttons(
    viewer, panel, add_video, qtbot
):
    add_video(_FakeVideo(n_frames=450))
    main_dims = viewer.window._qt_viewer.dims
    panel._video_viewer.close()
    panel._video_viewer = None

    main_dims.slider_widgets[find_time_dim_index(viewer)].play_button._on_click()
    qtbot.waitUntil(lambda: main_dims.is_playing)
    main_dims.stop()


def test_frame_step_persists_across_viewer_reopen(viewer, panel, add_video):
    add_video(_FakeVideo(n_frames=30), "a")
    panel._video_viewer.frame_step_spin.setValue(3)
    _delete_video(panel, "Video: a")
    assert not panel._videos

    entry = add_video(_FakeVideo(n_frames=30), "b")
    assert panel._video_viewer.frame_step_spin.value() == 3
    assert entry.layer.data.shape[0] == 10


def test_frame_step_rebuilds_all_videos_and_keeps_time(viewer, panel, add_video):
    a = add_video(_FakeVideo(n_frames=450), "a")
    b = add_video(_FakeVideo(n_frames=450), "b")
    model = panel._video_viewer.model
    model.dims.set_point(0, 6.0)

    panel._video_viewer.frame_step_spin.setValue(3)

    for entry in (a, b):
        assert entry.layer.data.shape[0] == 150
        np.testing.assert_allclose(entry.layer.scale[0], 3 / 30)
    assert model.dims.point[0] == pytest.approx(6.0, abs=3 / 30)


@pytest.mark.parametrize("units, factor", [("s", 1), ("ms", 1e3), ("us", 1e6)])
@pytest.mark.parametrize("video_first", [False, True])
def test_time_sync_converts_main_time_units(
    viewer, panel, add_video, units, factor, video_first
):
    scan = viewer.layers[0].metadata["xarray"]
    scan = scan.assign_coords(time=scan.time * factor)
    scan.time.attrs["units"] = units
    viewer.layers.clear()
    video = _FakeVideo(n_frames=450)
    if video_first:
        entry = add_video(video)
    plot_napari(scan, viewer=viewer, show_colorbar=False, show_scale_bar=False)
    if not video_first:
        entry = add_video(video)
    t_idx = find_time_dim_index(viewer)
    model = panel._video_viewer.model
    assert model.dims.point[0] == pytest.approx(viewer.dims.point[t_idx] / factor)

    viewer.dims.set_point(t_idx, 12.0 * factor)

    assert model.dims.point[0] == pytest.approx(12.0)
    assert model.dims.current_step[0] == 360
    np.testing.assert_array_equal(entry.layer.data[360], video._frames[360])
    # 11.2 s falls between the 11.0 s and 11.5 s fUSI frames.
    model.dims.set_point(0, 11.2)
    assert viewer.dims.point[t_idx] == pytest.approx(11.2 * factor)
    assert read_current_time(viewer)[0] == pytest.approx(11.0 * factor)


def test_signal_cursor_follows_video_time(viewer, panel, add_video):
    from confusius._napari._signals._panel import SignalPanel

    add_video(_FakeVideo(n_frames=450))
    signals = SignalPanel(viewer)
    # A stub plotter: the real one docks a Matplotlib canvas this test does not need.
    signals._plotter = plotter = MagicMock()
    signals._cursor_check.setChecked(True)
    panel._video_viewer.model.dims.set_point(0, 11.2)
    assert plotter.set_xaxis_cursor.call_args.args[0] == pytest.approx(11.2)


def test_video_viewer_follows_theme(viewer, panel, add_video):
    add_video(_FakeVideo())
    viewer.theme = "light"
    assert panel._video_viewer.model.theme == "light"


def test_main_reset_view_also_resets_video_view(viewer, panel, add_video):
    from napari.components._viewer_key_bindings import reset_view

    add_video(_FakeVideo())
    video_viewer = panel._video_viewer
    camera = video_viewer.model.scene.camera
    main_camera = viewer.scene.camera
    camera.zoom = 123.0
    main_camera.zoom = 456.0
    # Same entry point as napari's Home button and Ctrl+R.
    reset_view(viewer)
    assert camera.zoom != 123.0
    assert main_camera.zoom != 456.0
    fitted = camera.zoom
    video_viewer.model.reset_view()
    assert camera.zoom == fitted

    # Closing restores the main viewer's own method.
    video_viewer.close()
    panel._video_viewer = None
    assert "reset_view" not in viewer.__dict__
    camera_zoom_before = main_camera.zoom
    main_camera.zoom = 456.0
    viewer.reset_view()
    assert main_camera.zoom == camera_zoom_before


def test_video_view_is_fitted_once_dock_is_laid_out(viewer, panel, add_video, qtbot):
    from qtpy.QtWidgets import QApplication

    window = viewer.window._qt_window
    window.setUpdatesEnabled(False)
    window.resize(1600, 900)
    window.show()
    QApplication.processEvents()
    add_video(_FakeVideo())
    model = panel._video_viewer.model
    # Discard napari's immediate fit, before the new dock has its final size.
    model.scene.camera.zoom = 123.0
    model.scene.camera.center = (0.0, -100.0, -100.0)
    QApplication.processEvents()

    expected_zoom = 0.95 * min(np.asarray(model.canvas.size) / (48, 64))
    qtbot.waitUntil(lambda: model.scene.camera.zoom == pytest.approx(expected_zoom))
    np.testing.assert_allclose(model.scene.camera.center[-2:], (23.5, 31.5))


def test_viewer_controls_act_on_video_viewer_only(viewer, panel, add_video):
    add_video(_FakeVideo(), "a")
    add_video(_FakeVideo(), "b")
    video_viewer = panel._video_viewer
    buttons = video_viewer.buttons
    controls = video_viewer.controls
    assert video_viewer.controls_dock.widget() is controls
    # Use napari's own layer panel as the width budget, not a pixel constant.
    for widget in [*controls.findChildren(QWidget), controls]:
        if widget.layout() is not None:
            widget.layout().activate()
    native_controls = viewer.window._qt_viewer.dockLayerList.widget()
    native_controls.layout().activate()
    assert (
        controls.minimumSizeHint().width() <= native_controls.minimumSizeHint().width()
    )
    assert controls.sizePolicy() == native_controls.sizePolicy()
    bottom_row = controls.layout().itemAt(controls.layout().count() - 1).widget()
    assert bottom_row is buttons
    assert buttons.layout().itemAt(buttons.layout().count() - 1).widget() is (
        video_viewer.frame_step_spin
    )
    for hidden in (
        buttons.consoleButton,
        buttons.ndisplayButton,
        buttons.rollDimsButton,
        buttons.resetViewButton,
        video_viewer.layer_buttons.newPointsButton,
        video_viewer.layer_buttons.newShapesButton,
        video_viewer.layer_buttons.newLabelsButton,
    ):
        assert hidden.isHidden()

    main_order = viewer.dims.order
    main_grid = viewer.canvas.grid.enabled
    buttons.transposeDimsButton.click()
    assert video_viewer.model.dims.order == (0, 2, 1)
    assert viewer.dims.order == main_order

    assert video_viewer.model.canvas.grid.enabled
    assert buttons.gridViewButton.isChecked()
    buttons.gridViewButton.click()
    assert not video_viewer.model.canvas.grid.enabled
    assert not buttons.gridViewButton.isChecked()
    assert viewer.canvas.grid.enabled == main_grid

    for name in ("Video: a", "Video: b"):
        _delete_video(panel, name)
    assert video_viewer.controls_dock.isHidden()


def test_layer_context_menu_acts_on_video_layers(viewer, panel, add_video):
    a = add_video(_FakeVideo(), "a")
    b = add_video(_FakeVideo(), "b")
    video_viewer = panel._video_viewer
    fusi = viewer.layers[0]
    viewer.layers.selection = {fusi}
    video_viewer.model.layers.selection = {a.layer}

    def trigger(text: str) -> None:
        menu = video_viewer.build_layer_context_menu()
        actions = {action.text(): action for action in menu.actions()}
        actions[text].trigger()

    trigger("Toggle visibility")
    assert (a.layer.visible, b.layer.visible) == (False, True)
    trigger("Toggle visibility")
    trigger("Hide other videos")
    assert (a.layer.visible, b.layer.visible) == (True, False)
    trigger("Show all videos")
    assert (a.layer.visible, b.layer.visible) == (True, True)
    assert fusi.visible is True

    trigger("Remove selected videos")
    assert [e.layer.name for e in panel._videos] == ["Video: b"]
    assert len(viewer.layers) == 1

    # Nothing napari offers that a video cannot run (duplicate, split RGB, ...).
    menu = video_viewer.build_layer_context_menu()
    assert [action.text() for action in menu.actions() if not action.isSeparator()] == [
        "Toggle visibility",
        "Show all videos",
        "Hide other videos",
        "Remove selected videos",
    ]


def test_layer_context_menu_is_reused(viewer, panel, add_video, monkeypatch):
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QMenu

    add_video(_FakeVideo())
    shown = []
    monkeypatch.setattr(QMenu, "exec", lambda menu, *args: shown.append(menu))
    delegate = panel._video_viewer.layer_list.itemDelegate()
    for _ in range(2):
        delegate.show_context_menu(None, None, QPoint(), None)
    assert len(shown) == 2
    assert shown[0] is shown[1]


def test_overlay_shows_video_time(viewer, panel, add_video):
    add_video(_FakeVideo(n_frames=450), "a")
    add_video(_FakeVideo(n_frames=450), "b")
    model = panel._video_viewer.model
    overlay = model.canvas.overlays.text
    model.dims.set_point(0, 11.0)
    assert overlay.visible
    assert overlay.text == "11.00 s"

    _delete_video(panel, "Video: a")
    assert overlay.visible
    assert overlay.text == "11.00 s"


def test_video_dock_sits_between_canvas_and_panel_dock(
    make_napari_viewer, sample_voxeldata_3dt, monkeypatch, tmp_path
):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication

    from confusius._napari._widget import ConfUSIusWidget

    viewer = make_napari_viewer()
    plot_napari(
        sample_voxeldata_3dt, viewer=viewer, show_colorbar=False, show_scale_bar=False
    )
    widget = ConfUSIusWidget(viewer)
    panel_dock = viewer.window.add_dock_widget(widget, name="ConfUSIus", area="right")
    panel = widget.findChild(VideoPanel)

    window = panel_dock.parentWidget()

    def relayout() -> None:
        # The window stays hidden, so Qt holds back its resize events and the
        # layout does not follow a resize by itself; on macOS, pending window
        # events can even undo a first resize back to Qt's default 640 x 480.
        # Flush them, then lay the window out at 1600 x 900 explicitly.
        QApplication.processEvents()
        window.resize(1600, 900)
        window.layout().invalidate()
        window.layout().activate()

    relayout()
    panel_width = panel_dock.width()
    layer_dock = viewer.window._qt_viewer.dockLayerList
    layer_width = layer_dock.width()

    path = tmp_path / "clip.mp4"
    path.touch()
    monkeypatch.setattr(panel_module, "VideoReaderNP", lambda p, **kw: _FakeVideo())
    load_panel = widget.findChild(DataPanel)
    load_panel._path_edit.setText(str(path))
    load_panel._load_btn.click()
    video_viewer = panel._video_viewer
    assert video_viewer is not None
    video_dock = video_viewer.dock
    controls_dock = video_viewer.controls_dock

    relayout()
    try:
        assert (
            window.dockWidgetArea(video_dock) == Qt.DockWidgetArea.RightDockWidgetArea
        )
        assert (
            window.dockWidgetArea(controls_dock) == Qt.DockWidgetArea.LeftDockWidgetArea
        )
        assert controls_dock.y() > layer_dock.y()
        assert controls_dock.height() < layer_dock.height()
        assert controls_dock.x() == layer_dock.x()
        # It shares the layer list's column instead of widening it.
        assert controls_dock.width() == layer_dock.width()
        assert layer_dock.width() == pytest.approx(layer_width, abs=4)
        assert not panel_dock.isHidden()
        assert video_dock.x() < panel_dock.x()
        assert video_dock.y() == panel_dock.y()
        # Full window height, not capped at the initial size hint.
        assert video_dock.height() >= 0.85 * window.height()
        # The video takes its space from the canvas, not from the panel.
        assert panel_dock.width() == pytest.approx(panel_width, abs=4)
        assert video_dock.width() == pytest.approx(panel_width, abs=4)

        # A user can widen the shared column and shrink it back again.
        window.resizeDocks([layer_dock], [layer_width + 150], Qt.Orientation.Horizontal)
        relayout()
        assert layer_dock.width() > layer_width
        window.resizeDocks([layer_dock], [layer_width], Qt.Orientation.Horizontal)
        relayout()
        assert layer_dock.width() == pytest.approx(layer_width, abs=4)
        assert controls_dock.width() == layer_dock.width()

        # Removing the last video hides the dock and gives its space back to
        # the canvas, not to the panel; re-adding shows it sized as before.
        _delete_video(panel, "Video: clip")
        relayout()
        assert video_dock.isHidden()
        assert controls_dock.isHidden()
        assert panel_dock.width() == pytest.approx(panel_width, abs=4)
        panel.add_video(path)
        relayout()
        assert not video_dock.isHidden()
        assert not controls_dock.isHidden()
        assert video_dock.x() < panel_dock.x()
        assert panel_dock.width() == pytest.approx(panel_width, abs=4)
        assert video_dock.width() == pytest.approx(panel_width, abs=4)

        # Both title-bar buttons also give the space back to the canvas, and
        # reopening takes it from the canvas again.
        canvas = window.centralWidget()
        canvas_width = canvas.width()
        for button_name in ("close_button", "hide_button"):
            getattr(video_dock.titleBarWidget(), button_name).click()
            relayout()
            assert video_dock.isHidden()
            assert panel_dock.width() == pytest.approx(panel_width, abs=4)
            assert canvas.width() > canvas_width
            video_viewer.show_viewer()
            relayout()
            assert not video_dock.isHidden()
            assert panel_dock.width() == pytest.approx(panel_width, abs=4)
            assert video_dock.width() == pytest.approx(panel_width, abs=4)
            assert canvas.width() == pytest.approx(canvas_width, abs=4)
    finally:
        video_viewer.close()
    relayout()
    assert not panel_dock.isHidden()
    assert panel_dock.width() == pytest.approx(panel_width, abs=4)


@pytest.mark.parametrize("show_before_load", [False, True])
def test_video_layer_list_starts_at_smallest_height(
    viewer, panel, add_video, show_before_load
):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication

    window = viewer.window._qt_window
    # Exercise the shown window's layout without painting an offscreen GL canvas.
    window.setUpdatesEnabled(False)
    window.resize(1600, 900)
    if show_before_load:
        window.show()
        QApplication.processEvents()
    add_video(_FakeVideo())
    window.show()
    QApplication.processEvents()
    controls_dock = panel._video_viewer.controls_dock
    layer_dock = viewer.window._qt_viewer.dockLayerList
    initial_height = controls_dock.height()
    window.resizeDocks(
        [layer_dock, controls_dock],
        [layer_dock.height() + initial_height, 1],
        Qt.Orientation.Vertical,
    )
    QApplication.processEvents()
    assert controls_dock.height() == initial_height
    window.resizeDocks(
        [layer_dock, controls_dock],
        [layer_dock.height() - 100, initial_height + 100],
        Qt.Orientation.Vertical,
    )
    QApplication.processEvents()
    assert controls_dock.height() > initial_height


def test_closing_panel_dock_closes_video_viewer(
    make_napari_viewer, sample_voxeldata_3dt, monkeypatch, tmp_path, qtbot
):
    from napari.utils._proxies import PublicOnlyProxy
    from qtpy.QtCore import QCoreApplication, QEvent

    viewer = make_napari_viewer()
    plot_napari(
        sample_voxeldata_3dt, viewer=viewer, show_colorbar=False, show_scale_bar=False
    )
    # napari hands plugin widgets a proxy of the viewer.
    panel = VideoPanel(PublicOnlyProxy(viewer))  # ty: ignore[invalid-argument-type]
    panel_dock = viewer.window.add_dock_widget(panel, name="ConfUSIus")
    path = tmp_path / "clip.mp4"
    path.touch()
    monkeypatch.setattr(panel_module, "VideoReaderNP", lambda p, **kw: _FakeVideo())
    panel.add_video(path)
    assert "Video" in viewer.window.dock_widgets
    assert "video layer list" in viewer.window.dock_widgets
    assert "reset_view" in viewer.__dict__

    # What the panel dock's close button does.
    viewer.window.remove_dock_widget(panel_dock)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qtbot.waitUntil(lambda: "Video" not in viewer.window.dock_widgets)
    assert "video layer list" not in viewer.window.dock_widgets
    assert "reset_view" not in viewer.__dict__


class TestTimeOverlayFallback:
    def test_overlay_uses_dims_point_for_layer_without_xarray(self, viewer):
        """A layer with fps metadata but no xarray falls back to `dims.point`."""
        from confusius._napari._time_overlay import _TimeOverlay

        overlay = _TimeOverlay(viewer)
        overlay.check()
        assert overlay._active

        fps = 50.0
        layer = viewer.add_image(
            np.zeros((10, 4, 6, 8), dtype=np.uint8),
            name="Video: test",
            scale=(1.0 / fps, 1.0, 1.0, 1.0),
            axis_labels=("time", "z", "y", "x"),
            metadata={"fps": fps, "time_units": "s"},
        )
        viewer.layers.selection = {layer}
        assert overlay._ref_layer is layer

        time_idx = overlay._time_idx
        viewer.dims.set_current_step(time_idx, 3)
        expected_text = f"{float(viewer.dims.point[time_idx]):.2f} s"
        assert viewer.canvas.overlays.text.text == expected_text
