"""Tests for the VideoPanel widget and its dedicated video viewer.

`_VideoArray` is tested as a pure-logic unit.  Panel tests use real napari
viewers via `make_napari_viewer` with a stubbed `VideoReaderNP`.

See https://napari.org/dev/plugins/testing_and_publishing/test.html
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtWidgets import QWidget

import confusius._napari._video._video_panel as panel_module
from confusius._napari._timeaxis import find_time_dim_index, read_current_time
from confusius._napari._video._video_panel import VideoPanel
from confusius._napari._video._video_viewer import _VideoArray
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
        self._frames = np.random.default_rng(0).integers(
            0, 255, size=(n_frames, h, w, *c), dtype=np.uint8
        )

    def __getitem__(self, idx):
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
    """Return a function that adds a stubbed video through the panel UI path."""

    def _add(fv: _FakeVideo, name: str = "clip"):
        path = tmp_path / f"{name}.mp4"
        path.touch()
        monkeypatch.setattr(panel_module, "VideoReaderNP", lambda p, **kw: fv)
        panel._path_edit.setText(str(path))
        panel._load_from_path()
        return panel._videos[-1]

    return _add


def _delete_video(panel, name: str) -> None:
    """Select the video layer called `name` and press the viewer's bin button."""
    video_viewer = panel._video_viewer
    layers = video_viewer.model.layers
    layers.selection = {layers[name]}
    video_viewer.layer_buttons.deleteButton.click()


def test_add_video_opens_dock_and_leaves_main_viewer_alone(viewer, panel, add_video):
    assert not panel._no_videos_label.isHidden()
    entry = add_video(_FakeVideo(n_frames=10, rgb=True))
    assert viewer.window.dock_widgets["Video"] is panel._video_viewer.qt_viewer
    assert entry.layer.data.shape == (10, 48, 64, 3)
    assert entry.layer.rgb
    np.testing.assert_allclose(entry.layer.scale, (1 / 30, 1.0, 1.0))
    assert len(viewer.layers) == 1
    assert not viewer.canvas.grid.enabled
    assert panel._no_videos_label.isHidden()
    assert panel.isAncestorOf(panel._video_viewer.layer_list)


def test_add_video_does_not_move_main_time(viewer, panel, add_video):
    t_idx = find_time_dim_index(viewer)
    before = viewer.dims.point[t_idx]
    add_video(_FakeVideo(n_frames=300))
    assert viewer.dims.point[t_idx] == before


def test_missing_file_shows_error_and_opens_nothing(panel, tmp_path, monkeypatch):
    errors = []
    monkeypatch.setattr(panel_module, "show_error", errors.append)
    panel._path_edit.setText(str(tmp_path / "nope.mp4"))
    panel._load_from_path()
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
    panel._path_edit.setText(str(path))
    panel._load_from_path()
    assert errors and "cannot decode frame" in errors[0]
    assert panel._video_viewer is None
    assert not panel._no_videos_label.isHidden()


def test_video_messages_show_on_video_canvas_within_it(
    viewer, panel, add_video, tmp_path
):
    from napari._qt.dialogs.qt_notification import NapariQtNotification

    add_video(_FakeVideo())
    canvas = panel._video_viewer.qt_viewer.canvas.native
    # Narrower than napari's 400 px toast.
    canvas.resize(300, 200)
    panel._path_edit.setText(str(tmp_path / "nope.mp4"))
    panel._load_from_path()
    panel._video_viewer.frame_step_spin.setValue(2)

    toasts = canvas.findChildren(NapariQtNotification)
    assert [toast.message.text().split()[0] for toast in toasts] == [
        "Video",
        "File",
        "Frame",
    ]
    for toast in toasts[1:]:
        assert 0 <= toast.x()
        assert toast.geometry().right() < canvas.width()
    assert not viewer.window._qt_viewer.canvas.native.findChildren(
        NapariQtNotification
    )


def test_two_videos_share_video_grid(viewer, panel, add_video):
    add_video(_FakeVideo(), "a")
    add_video(_FakeVideo(h=10, w=200), "b")
    model = panel._video_viewer.model
    assert [layer.name for layer in model.layers] == ["Video: a", "Video: b"]
    assert model.canvas.grid.enabled
    assert panel._video_viewer.layer_list.model().rowCount() == 2


def test_remove_last_video_closes_dock(viewer, panel, add_video):
    add_video(_FakeVideo(), "a")
    add_video(_FakeVideo(), "b")
    video_viewer = panel._video_viewer
    _delete_video(panel, "Video: a")
    assert [e.name for e in panel._videos] == ["Video: b"]
    assert [layer.name for layer in video_viewer.model.layers] == ["Video: b"]

    _delete_video(panel, "Video: b")
    # The viewer is kept (opening another GL canvas later can crash) but hidden.
    assert panel._video_viewer is video_viewer
    assert video_viewer.dock.isHidden()
    assert video_viewer.controls.isHidden()
    assert not panel._no_videos_label.isHidden()

    add_video(_FakeVideo(), "c")
    assert panel._video_viewer is video_viewer
    assert not video_viewer.dock.isHidden()
    assert not video_viewer.controls.isHidden()
    assert panel._no_videos_label.isHidden()
    assert [layer.name for layer in video_viewer.model.layers] == ["Video: c"]


def test_layer_removed_from_the_model_is_forgotten(viewer, panel, add_video):
    a = add_video(_FakeVideo(), "a")
    add_video(_FakeVideo(), "b")
    # Any removal path (keyboard Delete, scripting) goes through the model.
    panel._video_viewer.model.layers.remove(a.layer)
    assert [e.name for e in panel._videos] == ["Video: b"]
    assert panel._video_viewer is not None


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


def test_time_sync_main_to_video(viewer, panel, add_video):
    entry = add_video(_FakeVideo(n_frames=450))
    t_idx = find_time_dim_index(viewer)
    viewer.dims.set_point(t_idx, 12.0)
    model = panel._video_viewer.model
    assert model.dims.point[0] == pytest.approx(12.0, abs=1 / 30)
    assert model.dims.current_step[0] == 360
    assert entry.layer.data.shape[0] == 450


def test_time_sync_video_to_main(viewer, panel, add_video):
    add_video(_FakeVideo(n_frames=450))
    t_idx = find_time_dim_index(viewer)
    # 11.2 s falls between the 11.0 s and 11.5 s fUSI frames.
    panel._video_viewer.model.dims.set_point(0, 11.2)
    assert viewer.dims.point[t_idx] == pytest.approx(11.2)
    assert read_current_time(viewer)[0] == pytest.approx(11.0)


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
    add_video(_FakeVideo())
    model = panel._video_viewer.model
    qtbot.wait(20)
    camera = model.scene.camera
    zoom, center = camera.zoom, camera.center
    model.reset_view()
    assert camera.zoom == zoom
    assert camera.center == center


def test_viewer_controls_act_on_video_viewer_only(viewer, panel, add_video):
    add_video(_FakeVideo(), "a")
    add_video(_FakeVideo(), "b")
    video_viewer = panel._video_viewer
    buttons = video_viewer.buttons
    controls = video_viewer.controls
    assert panel.isAncestorOf(controls)
    # napari's 242 px `min-width` on its button bars and layer list, and its
    # spin-box width floor, are overridden so the controls fit the sidebar
    # (issue #183 pattern).
    for widget in [*controls.findChildren(QWidget), controls]:
        if widget.layout() is not None:
            widget.layout().activate()
    assert controls.minimumSizeHint().width() < 242
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
    assert not panel._no_videos_label.isHidden()
    assert controls.isHidden()


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
    assert [e.name for e in panel._videos] == ["Video: b"]
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
    make_napari_viewer, sample_voxeldata_3dt, monkeypatch, tmp_path, capsys
):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication, QDockWidget

    from confusius._napari._widget import ConfUSIusWidget

    def diag(label: str) -> None:
        # TODO: temporary CI diagnostic of the macOS window width; remove.
        screen = window.screen()
        central = window.centralWidget()
        left = [
            (dock.width(), dock.minimumSizeHint().width())
            for dock in window.findChildren(QDockWidget)
            if window.dockWidgetArea(dock) == Qt.DockWidgetArea.LeftDockWidgetArea
            and not dock.isHidden()
        ]
        video = getattr(panel._video_viewer, "dock", None)
        with capsys.disabled():
            print(
                f"\nVIDEO-DIAG {label}: platform={QApplication.platformName()} "
                f"screen={screen.geometry().getRect() if screen else None} "
                f"avail={screen.availableGeometry().getRect() if screen else None} "
                f"window={window.width()}x{window.height()} "
                f"window_min={window.minimumSizeHint().width()} "
                f"visible={window.isVisible()} left(width,min)={left} "
                f"central={central.width()} central_min="
                f"{central.minimumSizeHint().width()} panel={panel_dock.width()} "
                f"video={video.width() if video is not None else None}"
            )

    viewer = make_napari_viewer()
    plot_napari(
        sample_voxeldata_3dt, viewer=viewer, show_colorbar=False, show_scale_bar=False
    )
    widget = ConfUSIusWidget(viewer)
    panel_dock = viewer.window.add_dock_widget(widget, name="ConfUSIus", area="right")
    panel = widget.findChild(VideoPanel)

    window = panel_dock.parentWidget()
    window.resize(1600, 900)

    def relayout() -> None:
        window.layout().activate()
        QApplication.processEvents()

    def assert_video_takes_canvas_space() -> None:
        # The video gets the panel's width, or all the canvas can spare when
        # the window is narrower than asked (macOS CI keeps it near 1244 px).
        central = window.centralWidget()
        assert video_dock.width() == pytest.approx(panel_width, abs=4) or (
            central.width() <= central.minimumSizeHint().width()
        )

    relayout()
    panel_width = panel_dock.width()
    diag("before video")

    path = tmp_path / "clip.mp4"
    path.touch()
    monkeypatch.setattr(panel_module, "VideoReaderNP", lambda p, **kw: _FakeVideo())
    panel._path_edit.setText(str(path))
    panel._load_from_path()
    video_viewer = panel._video_viewer
    assert video_viewer is not None
    video_dock = video_viewer.dock

    relayout()
    diag("after video")
    window.resize(2400, 900)
    relayout()
    diag("asked 2400")
    window.resize(1600, 900)
    relayout()
    try:
        assert (
            window.dockWidgetArea(video_dock) == Qt.DockWidgetArea.RightDockWidgetArea
        )
        assert not panel_dock.isHidden()
        assert video_dock.x() < panel_dock.x()
        assert video_dock.y() == panel_dock.y()
        # Full window height, not capped at the initial size hint.
        assert video_dock.height() >= 0.85 * window.height()
        # The video takes its space from the canvas, not from the panel.
        assert panel_dock.width() == pytest.approx(panel_width, abs=4)
        assert_video_takes_canvas_space()

        # Removing the last video hides the dock and gives its space back to
        # the canvas, not to the panel; re-adding shows it sized as before.
        _delete_video(panel, "Video: clip")
        relayout()
        assert video_dock.isHidden()
        assert panel_dock.width() == pytest.approx(panel_width, abs=4)
        panel._load_from_path()
        relayout()
        assert not video_dock.isHidden()
        assert video_dock.x() < panel_dock.x()
        assert panel_dock.width() == pytest.approx(panel_width, abs=4)
        assert_video_takes_canvas_space()
    finally:
        video_viewer.close()
    relayout()
    assert not panel_dock.isHidden()
    assert panel_dock.width() == pytest.approx(panel_width, abs=4)


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
    panel._path_edit.setText(str(path))
    panel._load_from_path()
    assert "Video" in viewer.window.dock_widgets
    assert "reset_view" in viewer.__dict__

    # What the panel dock's close button does.
    viewer.window.remove_dock_widget(panel_dock)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qtbot.waitUntil(lambda: "Video" not in viewer.window.dock_widgets)
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
