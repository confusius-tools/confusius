"""Dedicated napari viewer for synchronized video playback.

Videos live in their own `ViewerModel` + `QtViewer` shown beside the main canvas
(the pattern from napari's ``multiple_viewer_widget`` gallery example), so they
never share dims, scale, translate, grid or axis order with the fUSI data. The
only coupling to the main viewer is time: the main viewer's world time along
its time axis drives the video viewer's time slider and vice versa.
"""

from __future__ import annotations

import contextlib
import time
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

# napari has no public toast API: this is the toast it shows its own
# notifications with.
from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari.components import ViewerModel
from napari.qt import QtViewer, QtViewerButtons, get_stylesheet
from napari.settings import get_settings
from napari.utils.notifications import NotificationSeverity
from napari.utils.theme import get_theme, template
from qtpy.QtCore import QCoreApplication, QEvent, Qt, QTimer
from qtpy.QtWidgets import (
    QDockWidget,
    QMenu,
    QSpinBox,
    QSplitter,
    QSplitterHandle,
    QVBoxLayout,
    QWidget,
)

from confusius._napari._qt import find_main_window
from confusius._napari._timeaxis import find_time_dim_index
from confusius.timing import convert_time_units, get_representative_time_step

if TYPE_CHECKING:
    from collections.abc import Callable

    import napari
    import napari.layers
    from napari._qt.widgets.qt_dims import QtDims
    from napari._qt.widgets.qt_dims_slider import QtPlayButton
    from napari_video.napari_video import VideoReaderNP
    from qtpy.QtGui import QMouseEvent


_IRREGULAR_TIME_WARNING = (
    "Reference layer time axis is not regularly sampled; "
    "video synchronization uses a linear approximation and may drift."
)
"""Warning shown when video synchronization uses approximate recording times."""

_HANDLE_STYLE = """
_CanvasSplitter::handle {
  padding: 2px;
  border: none;
  background-color: {{ background }};
  image: url("theme_{{ id }}:/vertical_separator.svg");
}
_CanvasSplitter::handle:hover {
  background: {{ foreground }};
}
"""
"""napari's `QMainWindow::separator` style, for the handle between the two canvases.

The selector names the splitter's own class: a `QSplitter` one would also match
each `QtViewer`, which is a `QSplitter` too. The handle's width is set on the
splitter instead: a `width` here alone makes Qt report a handle width of -1 and lay
the canvases over the handle.
"""


class _VideoArray:
    """Lazy `(T, H, W[, C])` array over a `VideoReaderNP` with frame skipping.

    Provides the `shape`, `dtype`, and `__getitem__` interface that napari
    requires for frame-on-demand display.  Logical frame `t` maps to video
    frame `t * step`.

    Parameters
    ----------
    video : VideoReaderNP
        The opened video reader.
    step : int, default: 1
        Show every *step*-th frame (temporal subsampling).
    """

    def __init__(self, video: VideoReaderNP, *, step: int = 1) -> None:
        self.video = video
        self._step = step
        # Reader metadata avoids seeking back to frame zero on every step change.
        self._frame_shape = video.frame_shape
        n_frames = len(range(0, video.number_of_frames, step))
        self.shape = (n_frames, *self._frame_shape)
        self.dtype = video.dtype
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))

    def __len__(self) -> int:
        return self.shape[0]

    def _fetch(self, t: int) -> np.ndarray:
        """Decode logical frame `t`.

        Parameters
        ----------
        t : int
            Logical frame index.

        Returns
        -------
        numpy.ndarray
            The decoded `(H, W[, C])` frame.
        """
        return np.ascontiguousarray(self.video[t * self._step])

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        time_idx, spatial_idx = key[0], key[1:]

        if isinstance(time_idx, (int, np.integer)):
            return self._fetch(int(time_idx))[spatial_idx]

        if isinstance(time_idx, slice):
            # An unbounded time slice means napari put time among the
            # displayed dims (e.g. the user rolled the video viewer's axes).
            # Decoding every frame would freeze the UI, so return one frame
            # in the expected `(1, ...)` shape instead.  Legitimate thick
            # slicing always arrives as a bounded `slice(low, high)`.
            if time_idx == slice(None):
                return self._fetch(0)[np.newaxis][(slice(None), *spatial_idx)]
            start, stop, step = time_idx.indices(self.shape[0])
            frames = [self._fetch(t) for t in range(start, stop, step)]
            if not frames:
                return np.empty((0, *self._frame_shape), dtype=self.dtype)[
                    (slice(None), *spatial_idx)
                ]
            return np.stack(frames)[(slice(None), *spatial_idx)]

        raise IndexError(f"Unsupported time index type: {type(time_idx)}")


class _VideoToast(NapariQtNotification):
    """napari toast that fits the width of the video canvas it sits on.

    napari's toast is at least 400 px wide, wider than a narrow video viewer.
    napari also places it again on each resize of its canvas, which the
    slide-in animation would undo while a just-shown video canvas takes its
    size.
    """

    def move_to_bottom_right(self, offset: tuple[int, int] = (8, 8)) -> None:
        """Fit the toast to its canvas width and move it to the bottom right.

        Parameters
        ----------
        offset : tuple[int, int], default: (8, 8)
            Gap in pixels between the toast and the right and bottom edges of
            its canvas.
        """
        parent = self.parent()
        if isinstance(parent, QWidget):
            # ponytail: the message keeps napari's 200 px minimum width, so a
            # canvas narrower than about 270 px still clips the toast.
            self.setFixedWidth(min(self.MIN_WIDTH, parent.width() - 2 * offset[0]))
        self.geom_anim.stop()
        super().move_to_bottom_right(offset)


class _CanvasSplitterHandle(QSplitterHandle):
    """Splitter handle that resizes the canvases at most once per event-loop pass.

    A stock handle resizes on every mouse move. Resizing a GL canvas rebuilds its
    framebuffer and repaints it at once, which takes longer than the next mouse
    move takes to arrive: a drag would then work through a backlog of positions
    the pointer has already left. Like Qt's dock separators, this handle keeps
    only the latest move and applies it once the pending events are handled.

    Parameters
    ----------
    orientation : qtpy.QtCore.Qt.Orientation
        Orientation of the splitter.
    splitter : qtpy.QtWidgets.QSplitter
        The splitter owning the handle.
    """

    def __init__(self, orientation: Qt.Orientation, splitter: QSplitter) -> None:
        super().__init__(orientation, splitter)
        self._pending_move: QMouseEvent | None = None

    def mouseMoveEvent(self, a0) -> None:
        """Keep the move for the next event-loop pass instead of resizing now.

        Parameters
        ----------
        a0 : qtpy.QtGui.QMouseEvent
            The Qt mouse move event.
        """
        if self._pending_move is None:
            QTimer.singleShot(0, self._apply_pending_move)
        # Qt deletes the event once this handler returns.
        self._pending_move = a0.clone()

    def mouseReleaseEvent(self, a0) -> None:
        """Apply the last move at once, so the handle ends where it was released.

        Parameters
        ----------
        a0 : qtpy.QtGui.QMouseEvent
            The Qt mouse release event.
        """
        self._apply_pending_move()
        super().mouseReleaseEvent(a0)

    def _apply_pending_move(self) -> None:
        """Resize the canvases to the latest kept move, if any."""
        event, self._pending_move = self._pending_move, None
        if event is not None:
            super().mouseMoveEvent(event)


class _CanvasSplitter(QSplitter):
    """Splitter between the two canvases, with a `_CanvasSplitterHandle`."""

    def createHandle(self) -> QSplitterHandle:
        """Create the handle between the canvases.

        Returns
        -------
        qtpy.QtWidgets.QSplitterHandle
            A handle that resizes at most once per event-loop pass.
        """
        return _CanvasSplitterHandle(self.orientation(), self)


class VideoViewer:
    """A second napari viewer, beside the main canvas, that shows only videos.

    Its canvas shares the window's central area with the main canvas, behind a
    splitter handle. Its layer list and controls live in a separate "video layer
    list" dock, placed right under napari's own layer list.

    Parameters
    ----------
    viewer : napari.Viewer
        The main napari viewer whose time axis drives the video.
    anchor : qtpy.QtWidgets.QWidget, optional
        A widget living inside a dock widget of the main window (typically
        the panel creating this viewer).  The video canvas starts as wide as
        that dock, and closing that dock closes the video viewer too.  If not
        provided, the video canvas starts at the width of its size hint.
    """

    def __init__(self, viewer: napari.Viewer, *, anchor: QWidget | None = None) -> None:
        # napari hands plugin widgets a `PublicOnlyProxy` of the viewer, which
        # also wraps what its methods return: the dock from `add_dock_widget`
        # would then be a proxy that Qt's `splitDockWidget` rejects, and the
        # `reset_view` override below would land on the proxy, not the viewer.
        viewer = getattr(viewer, "__wrapped__", viewer)
        self._main = viewer
        self.model = ViewerModel(title="Video")
        self.model.theme = viewer.theme
        self.qt_viewer = QtViewer(self.model)
        # `QtViewer._qt_open` routes dropped files through `viewer.window`,
        # which a bare `ViewerModel` does not have.
        self.qt_viewer.setAcceptDrops(False)
        # Several videos sit side by side in the video viewer's own grid; the
        # main viewer's grid is never touched.
        self.model.canvas.grid.enabled = True
        self.model.canvas.grid.shape = (1, -1)
        # The video canvas shares the central area with the main canvas, split
        # by a handle. Docks of one dock area share that area's width, so a
        # video dock next to the panel's dock could not be resized alone: the
        # separator on its canvas side resized both docks, and the one between
        # them only traded width from one to the other. Here the handle resizes
        # the video canvas only, and the panel's separator the panel only.
        main_qt_viewer = viewer.window._qt_viewer
        center = main_qt_viewer.parentWidget()
        # Parented before it takes the main canvas: a GL canvas moved to
        # another window loses its context.
        self._splitter = _CanvasSplitter(Qt.Orientation.Horizontal, center)
        self._splitter.setChildrenCollapsible(False)
        # With the 2 px padding of `_HANDLE_STYLE`, as wide as napari's dock
        # separators.
        self._splitter.setHandleWidth(5)
        self._splitter.setGeometry(main_qt_viewer.geometry())
        self._center_layout = center.layout()
        self._center_layout.replaceWidget(main_qt_viewer, self._splitter)
        self._splitter.addWidget(main_qt_viewer)
        self._splitter.addWidget(self.qt_viewer)
        # Window and dock resizes go to the main canvas: the video canvas
        # keeps the width the handle gave it.
        self._splitter.setStretchFactor(0, 1)
        # Shown now rather than on the layout's next pass, so the main canvas
        # does not vanish in between.
        self._splitter.show()
        anchor_dock = anchor
        while anchor_dock is not None and not isinstance(anchor_dock, QDockWidget):
            anchor_dock = anchor_dock.parentWidget()
        if anchor_dock is not None:
            width = anchor_dock.width()
            self._splitter.setSizes([self._splitter.width() - width, width])
            # The anchor's close button deletes its dock (napari's
            # `remove_dock_widget`); the video canvas would then outlive the
            # controls that drive it, with no close button of its own.
            anchor_dock.destroyed.connect(self._on_anchor_destroyed)

        # Controls built from napari's own widgets for this model, with
        # layer buttons above the list and viewer buttons below, like napari.
        # The stock buttons run app-wide actions that resolve to the main
        # viewer, so they are rewired here; the grid button keeps napari's
        # right-click popup, which edits this model's grid.
        self.buttons = QtViewerButtons(self.model)
        for button in (
            self.buttons.consoleButton,
            self.buttons.ndisplayButton,
            self.buttons.rollDimsButton,
            self.buttons.resetViewButton,
        ):
            button.hide()
        transpose = self.buttons.transposeDimsButton
        # The Alt-click filter rotates the main viewer's layers.
        transpose.removeEventFilter(self.buttons)
        transpose.clicked.disconnect()
        transpose.clicked.connect(lambda: self.model.dims.transpose())
        grid = self.buttons.gridViewButton
        grid.clicked.disconnect()
        grid.clicked.connect(self._toggle_grid)

        self.layer_buttons = self.qt_viewer.layerButtons
        for button in (
            self.layer_buttons.newPointsButton,
            self.layer_buttons.newShapesButton,
            self.layer_buttons.newLabelsButton,
        ):
            button.hide()
        delete = self.layer_buttons.deleteButton
        delete.clicked.disconnect()
        delete.clicked.connect(lambda: self.model.layers.remove_selected())

        self.frame_step_spin = QSpinBox()
        self.frame_step_spin.setRange(1, 100)
        self.frame_step_spin.setPrefix("Step: ")
        self.frame_step_spin.setAccessibleName("Frame step")
        self.frame_step_spin.setToolTip(
            "Show every N-th frame. Higher values skip frames for lighter playback."
        )
        self.layer_list = self.qt_viewer.layers
        # napari's layer context menu acts on the main viewer's layers and
        # offers commands a video layer cannot run (duplicate, split RGB,
        # projections decode every frame), so the list's delegate shows a
        # small menu of our own instead, built once: its actions always act on
        # the current selection.
        delegate = self.layer_list.itemDelegate()
        layer_menu = self.build_layer_context_menu()
        delegate.show_context_menu = lambda index, model, pos, parent: (  # type: ignore
            layer_menu.exec(pos)
        )

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(8, 4, 8, 6)
        # Share the existing button bar so its minimum width is not added twice.
        buttons_layout = self.buttons.layout()
        assert buttons_layout is not None
        buttons_layout.addWidget(self.frame_step_spin)
        controls_layout.addWidget(self.layer_buttons)
        controls_layout.addWidget(self.layer_list, stretch=1)
        controls_layout.addWidget(self.buttons)
        # The video layer list lives in its own dock, right under napari's
        # layer list, so videos are listed where layers are normally found
        # instead of in the ConfUSIus sidebar.  Remember that list's width
        # first: adding a dock can widen the shared left column.
        layer_dock = getattr(self._main.window._qt_viewer, "dockLayerList", None)
        self._layer_list_width = (
            layer_dock.width() if isinstance(layer_dock, QDockWidget) else 0
        )
        controls_policy = self.controls.sizePolicy()
        self.controls_dock = viewer.window.add_dock_widget(
            self.controls, name="video layer list", area="left"
        )
        # Like napari's layer dock, undo the plugin dock's Maximum policy.
        self.controls.setSizePolicy(controls_policy)
        self.controls_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        _hide_on_close(self.controls_dock, self.controls_dock.hide)
        self._sync_theme()
        # Size the dock after Qt has polished and laid out its contents.
        QTimer.singleShot(0, self._place_under_layer_list)

        # napari's Home button and Ctrl+R run the `napari:reset_view` action,
        # which calls `viewer.reset_view()` on the main viewer.  Overriding
        # that bound method on the instance makes every such path reset the
        # video viewer too.  pydantic would reject a plain setattr.
        self._main_reset_view = viewer.reset_view
        object.__setattr__(viewer, "reset_view", self._reset_both)

        overlay = self.model.canvas.overlays.text
        overlay.position = "bottom_left"
        overlay.font_size = 14
        overlay.color = "white"
        overlay.opacity = 0.6

        self._syncing = False
        viewer.dims.events.point.connect(self._sync_from_main)
        viewer.events.theme.connect(self._sync_theme)
        viewer.layers.events.inserted.connect(self._warn_if_irregular_time)
        self.model.dims.events.point.connect(self._sync_to_main)
        self.model.dims.events.point.connect(self._update_time_text)
        self.model.layers.events.removing.connect(self._on_layer_removing)
        self.model.layers.events.removed.connect(self._on_layer_removed)

        self._main_dims = viewer.window._qt_viewer.dims
        self._wire_play_buttons()
        viewer.dims.events.ndim.connect(self._wire_play_buttons)
        self.model.dims.events.ndim.connect(self._wire_play_buttons)
        self.frame_step_spin.valueChanged.connect(self.set_frame_step)

    def _place_under_layer_list(self) -> None:
        """Dock the video layer list below napari's main layer list."""
        if not hasattr(self, "controls_dock") or self.controls_dock.isHidden():
            return
        layer_dock = getattr(self._main.window._qt_viewer, "dockLayerList", None)
        window = find_main_window(self.controls_dock)
        if (
            window is None
            or not isinstance(layer_dock, QDockWidget)
            or layer_dock.isFloating()
            or layer_dock is self.controls_dock
        ):
            return
        height = layer_dock.height()
        window.splitDockWidget(layer_dock, self.controls_dock, Qt.Orientation.Vertical)
        self.controls_dock.show()
        # Request less than the minimum and let Qt clamp to the settled layout.
        window.resizeDocks(
            [layer_dock, self.controls_dock],
            [height, 1],
            Qt.Orientation.Vertical,
        )
        width = self._layer_list_width or layer_dock.width()
        if width:
            window.resizeDocks(
                [layer_dock, self.controls_dock],
                [width, width],
                Qt.Orientation.Horizontal,
            )

    # ------------------------------------------------------------------
    # Layers
    # ------------------------------------------------------------------

    def add_video(
        self,
        video: VideoReaderNP,
        *,
        name: str,
    ) -> napari.layers.Image:
        """Add a video image layer using the viewer's current frame step.

        Parameters
        ----------
        video : VideoReaderNP
            The opened video reader. Its frame rate sets the time scale in seconds.
        name : str
            Layer name.

        Returns
        -------
        napari.layers.Image
            The new video layer in the video viewer.
        """
        step = self.frame_step_spin.value()
        data = _VideoArray(video, step=step)
        is_rgb = data.ndim == 4 and data.shape[-1] in (3, 4)
        # Adding a layer resets the video model's point to mid-range; that
        # must not be pushed back into the main viewer.
        self._syncing = True
        try:
            layer = self.model.add_image(
                data,
                name=name,
                rgb=is_rgb,
                scale=(_time_scale(video.frame_rate, step), 1.0, 1.0),
                axis_labels=("time", "y", "x"),
                units=("s", None, None),
                metadata={"fps": video.frame_rate, "time_units": "s"},
            )
        finally:
            self._syncing = False
        assert not isinstance(layer, list)
        if len(self.model.layers) == 1:
            if self.qt_viewer.isHidden():
                self.show()
            self._warn_if_irregular_time()
        self._sync_from_main()
        self._update_time_text()
        # The canvas is laid out on the next event-loop pass; only then does
        # it have its real size, so fitting the view must wait.
        QTimer.singleShot(0, self._reset_view)
        return layer

    def _reset_view(self) -> None:
        """Fit and center the videos in the canvas, unless already closed."""
        if hasattr(self, "model"):
            self.model.reset_view()

    def _reset_both(self, *args, **kwargs) -> None:
        """Reset the main viewer's view, then the video viewer's.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to the main viewer's `reset_view`.
        """
        self._main_reset_view(*args, **kwargs)
        self._reset_view()

    def set_frame_step(self, step: int) -> None:
        """Rebuild video layers with a new frame step, keeping the current time.

        A running playback carries on from that time. Custom playback bounds
        follow the nearest available frames at their original world times.

        Parameters
        ----------
        step : int
            Show every *step*-th frame.
        """
        if self.frame_step_spin.value() != step:
            self.frame_step_spin.setValue(step)
            return
        if not self.model.layers:
            return
        qt_dims = self.qt_viewer.dims
        slider = qt_dims.slider_widgets[0]
        frame_range = slider.frame_range
        origin, _, spacing = self.model.dims.range[0]
        playback_times = tuple(origin + frame * spacing for frame in frame_range)
        thread = qt_dims._animation_thread
        playing = thread.isRunning()
        if playing:
            qt_dims.stop()
            # Queued frames are slider positions, whose time depends on the step:
            # apply them while they still mean what the thread meant.
            QCoreApplication.sendPostedEvents(None, QEvent.Type.MetaCall)

        current_time = float(self.model.dims.point[0])
        shown = [layer for layer in self.model.layers if layer.visible]
        self._syncing = True
        try:
            # Data and scale cannot change together, and the states in between map
            # the current time to frames far away. Layers marked as hidden do not
            # decode them, and nothing paints before the refresh below.
            for layer in shown:
                layer._visible = False
            for layer in self.model.layers:
                video = layer.data.video
                layer.data = _VideoArray(video, step=step)
                layer.scale = (_time_scale(video.frame_rate, step), 1.0, 1.0)  # ty: ignore[invalid-assignment]
            self.model.dims.set_point(0, current_time)
        finally:
            for layer in shown:
                layer._visible = True
            self._syncing = False
        for layer in shown:
            layer.refresh()

        if frame_range != (0, 0):
            origin, _, spacing = self.model.dims.range[0]
            last = self.model.dims.nsteps[0] - 1
            lower, upper = (
                min(last, max(0, round((value - origin) / spacing)))
                for value in playback_times
            )
            # napari needs two distinct bounds; a narrower interval uses the
            # nearest adjacent frames, or the full range for a one-frame video.
            if lower == upper:
                upper = min(last, upper + 1)
                lower = max(0, upper - 1)
            slider.frame_range = (lower, upper)

        if playing:
            # This also updates the thread's position and loop range to the new step.
            thread.set_slider(slider)
            thread.start()
            # The thread clears its stop flag once running. A stop before that is
            # lost and `QtDims.stop` hangs, which fast step changes would hit.
            while thread._waiter.is_set() and thread.isRunning():
                time.sleep(0.001)

    def _on_layer_removing(self, event=None) -> None:
        """Keep removal-induced time clamping from moving the main viewer."""
        self._syncing = True

    def _on_layer_removed(self, event=None) -> None:
        """Resynchronize and refit the remaining videos after one is removed."""
        self._syncing = False
        self._sync_from_main()
        self._update_time_text()
        if self.model.layers:
            self.model.reset_view()
        else:
            self.hide()

    def build_layer_context_menu(self) -> QMenu:
        """Build the right-click menu of the video layer list.

        Returns
        -------
        qtpy.QtWidgets.QMenu
            Menu with the few entries that make sense for video layers.
        """
        layers = self.model.layers

        def toggle_visibility() -> None:
            """Flip the visibility of every selected video."""
            for layer in list(layers.selection):
                layer.visible = not layer.visible

        def show_all() -> None:
            """Show every video."""
            for layer in layers:
                layer.visible = True

        def hide_others() -> None:
            """Show only the selected videos."""
            for layer in layers:
                layer.visible = layer in layers.selection

        menu = QMenu(self.layer_list)
        menu.addAction("Toggle visibility", toggle_visibility)
        menu.addAction("Show all videos", show_all)
        menu.addAction("Hide other videos", hide_others)
        menu.addSeparator()
        menu.addAction("Remove selected videos", layers.remove_selected)
        return menu

    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------

    def _sync_theme(self, event=None) -> None:
        """Follow the main viewer's theme."""
        self.model.theme = self._main.theme
        self.controls.setStyleSheet(get_stylesheet(self._main.theme))
        self._splitter.setStyleSheet(
            template(_HANDLE_STYLE, **get_theme(self._main.theme).to_rgb_dict())
        )

    def notify(
        self, message: str, severity: NotificationSeverity = NotificationSeverity.INFO
    ) -> None:
        """Show `message` as a napari toast on the video canvas.

        napari shows every notification on the main canvas, beside the video
        viewer; messages about videos show on the video canvas instead. Like
        napari's own toasts, messages below napari's GUI notification level are
        not shown.

        Parameters
        ----------
        message : str
            Text to show.
        severity : napari.utils.notifications.NotificationSeverity, default: NotificationSeverity.INFO
            Severity setting the toast's icon.
        """
        if severity < get_settings().application.gui_notification_level:
            return
        _VideoToast(message, severity, parent=self.qt_viewer.canvas.native).show()

    def _dismiss_toasts(self) -> None:
        """Close toasts shown on the video canvas.

        Closing them removes them from napari's notification stack. Without
        this, destroying the canvas leaves a deleted toast in that stack, and
        napari's next focus change raises a `RuntimeError`.
        """
        for toast in self.qt_viewer.canvas.native.findChildren(_VideoToast):
            toast.close()

    def _toggle_grid(self, checked: bool = False) -> None:
        """Toggle this viewer's grid mode.

        Parameters
        ----------
        checked : bool, default: False
            Ignored; the button's checked state follows the grid model.
        """
        self.model.canvas.grid.enabled = not self.model.canvas.grid.enabled

    def _update_time_text(self, event=None) -> None:
        """Show the video viewer's current time in its text overlay."""
        overlay = self.model.canvas.overlays.text
        if not self.model.layers:
            overlay.visible = False
            overlay.text = ""
            return
        overlay.text = f"{self.model.dims.point[0]:.2f} s"
        overlay.visible = True

    def _warn_if_irregular_time(self, event=None) -> None:
        """Warn when videos follow a recording with irregular time samples.

        Parameters
        ----------
        event : napari.utils.events.Event, optional
            Main-viewer layer insertion event. Only inserting the recording that
            drives synchronization triggers a warning.
        """
        if not self.model.layers:
            return
        reference = next(
            (layer for layer in self._main.layers if "time" in layer.axis_labels), None
        )
        if reference is None or (event is not None and event.value is not reference):
            return
        data = reference.metadata.get("xarray")
        if data is not None and "time" in data.coords:
            _, approximate = get_representative_time_step(data)
            if approximate:
                self.notify(_IRREGULAR_TIME_WARNING, NotificationSeverity.WARNING)

    def _wire_play_buttons(self, event=None) -> None:
        """Route the play buttons of both viewers through `_on_play_clicked`.

        Parameters
        ----------
        event : napari.utils.events.Event, optional
            Dims `ndim` event: napari rebuilds a viewer's sliders on each one.
        """
        for qt_dims in (self._main_dims, self.qt_viewer.dims):
            for slider in qt_dims.slider_widgets:
                button = slider.play_button
                button._on_click = partial(self._on_play_clicked, qt_dims, button)

    def _on_play_clicked(self, qt_dims: QtDims, button: QtPlayButton) -> None:
        """Toggle a play button, keeping one time playback across both viewers.

        A time animation in either viewer already moves both, so a second one
        would only fight it: pressing play on a time slider while the other
        viewer plays time stops that playback instead. Other axes of the main
        viewer do not move the video and play independently.

        Parameters
        ----------
        qt_dims : napari._qt.widgets.qt_dims.QtDims
            The sliders `button` belongs to.
        button : napari._qt.widgets.qt_dims_slider.QtPlayButton
            The pressed play button.
        """
        if button.property("playing") == "True":
            qt_dims.stop()
            return
        main_time = find_time_dim_index(self._main)
        if qt_dims is self._main_dims:
            own_time, other, other_time = main_time, self.qt_viewer.dims, 0
        else:
            own_time, other, other_time = 0, self._main_dims, main_time
        if (
            button.axis == own_time
            and other.is_playing
            and other._animation_thread.axis == other_time
        ):
            other.stop()
            return
        button.play_requested.emit(button.axis)

    def _sync_from_main(self, event=None) -> None:
        """Move the video slider to the main viewer's current world time.

        A hidden video canvas skips this, so scrubbing does not decode frames
        nobody sees; `show_viewer` catches up when the canvas is shown again.
        """
        time_idx = find_time_dim_index(self._main)
        if (
            self._syncing
            or time_idx is None
            or not self.model.layers
            or self.qt_viewer.isHidden()
        ):
            return
        self._syncing = True
        try:
            units = self._main.dims.units
            video_time = float(
                convert_time_units(
                    self._main.dims.point[time_idx],
                    str(units[time_idx]) if units else None,
                )
            )
            self.model.dims.set_point(0, video_time)
        finally:
            self._syncing = False

    def _sync_to_main(self, event=None) -> None:
        """Move the main viewer's time slider to the video's current world time."""
        time_idx = find_time_dim_index(self._main)
        if self._syncing or time_idx is None or not self.model.layers:
            return
        self._syncing = True
        try:
            units = self._main.dims.units
            main_time = float(
                convert_time_units(
                    self.model.dims.point[0],
                    "s",
                    str(units[time_idx]) if units else None,
                )
            )
            self._main.dims.set_point(time_idx, main_time)
            # When the main step changes, napari's time slider snaps the point
            # to that step. Setting it again keeps the video's exact time (the
            # step is unchanged, so the slider leaves it alone): the main viewer
            # still shows its nearest frame, and time cursors follow the video.
            # Without a snap the point already equals `main_time`, so this
            # emits no second point event.
            self._main.dims.set_point(time_idx, main_time)
        finally:
            self._syncing = False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def hide(self) -> None:
        """Hide the canvas and the layer list; the main canvas takes the freed width.

        The viewer is kept alive rather than closed: creating another GL
        canvas in the window after one existed crashes the process on some
        GL stacks (Mesa software rendering under WSLg, for one), so one
        video viewer is reused for the whole session.
        """
        self.hide_viewer()
        self.controls_dock.hide()

    def hide_viewer(self) -> None:
        """Stop video playback and hide the canvas, widening the main canvas."""
        self._dismiss_toasts()
        self.qt_viewer.dims.stop()
        self.qt_viewer.hide()

    def show(self) -> None:
        """Show the canvas and the layer list again."""
        self.show_viewer()
        self.controls_dock.show()

    def show_viewer(self) -> None:
        """Show the canvas again, at its last width and the main viewer's time."""
        self.qt_viewer.show()
        self._sync_from_main()

    def show_layer_list(self) -> None:
        """Show the video layer list dock again after it was hidden."""
        self.controls_dock.show()

    def _on_anchor_destroyed(self) -> None:
        """Close the video viewer once the dock of its panel is gone."""
        # Closing the whole window destroys the anchor dock too, and the video
        # canvas along with it; one event-loop pass later, only a lone anchor
        # removal leaves the video canvas in a window to close.
        # ponytail: a panel opened again builds a second video viewer, whose GL
        # canvas can crash the GL stacks `hide` describes; keeping one viewer
        # per main window, outside the panel, would avoid it.
        QTimer.singleShot(0, self._close_if_in_window)

    def _close_if_in_window(self) -> None:
        """Close the video viewer, unless it or its window is already gone."""
        if hasattr(self, "model") and find_main_window(self.qt_viewer) is not None:
            self.close()

    def close(self) -> None:
        """Disconnect from the main viewer and remove the canvas and the dock."""
        self._dismiss_toasts()
        object.__delattr__(self._main, "reset_view")
        self._main.dims.events.point.disconnect(self._sync_from_main)
        self._main.events.theme.disconnect(self._sync_theme)
        self._main.layers.events.inserted.disconnect(self._warn_if_irregular_time)
        self.model.dims.events.point.disconnect(self._sync_to_main)
        self._main.dims.events.ndim.disconnect(self._wire_play_buttons)
        self.model.dims.events.ndim.disconnect(self._wire_play_buttons)
        for qt_dims in (self._main_dims, self.qt_viewer.dims):
            for slider in qt_dims.slider_widgets:
                del slider.play_button._on_click
        self._main.window.remove_dock_widget(self.controls_dock)
        # The main canvas takes the central area back for itself.
        self._center_layout.replaceWidget(self._splitter, self._main.window._qt_viewer)
        self.qt_viewer.setParent(None)
        self._splitter.deleteLater()
        self.qt_viewer.close()
        # Safe to delete: napari's action manager drops its tooltip hooks on
        # the buttons' `destroyed` signal.  Unparenting pulls the controls
        # out of their dock.  They are gone already if the window was
        # deleted first.
        with contextlib.suppress(RuntimeError):
            self.controls.setParent(None)
            self.controls.deleteLater()
        # The model's event callbacks hold bound methods of the QtViewer, so
        # dropping both references is what lets the QtViewer be collected.
        del self.qt_viewer, self.model, self._splitter, self._center_layout
        del self.controls, self.controls_dock


def _hide_on_close(dock: QDockWidget, hide: Callable[[], None]) -> None:
    """Make a dock's title-bar buttons run `hide` instead of deleting it.

    The video viewer keeps one GL canvas for the whole session: destroying it
    and building another can crash some GL stacks, so the close button must
    not remove the dock. napari's title bar calls `destroyOnClose` from its close
    button and `close` from its hide button.

    Parameters
    ----------
    dock : qtpy.QtWidgets.QDockWidget
        Dock whose title-bar buttons should hide it.
    hide : collections.abc.Callable
        Called instead of the dock's `destroyOnClose` and `close`.
    """
    object.__setattr__(dock, "destroyOnClose", hide)
    object.__setattr__(dock, "close", hide)


def _time_scale(fps: float, step: int) -> float:
    """Return the seconds spanned by one logical frame.

    Parameters
    ----------
    fps : float
        Video frame rate.
    step : int
        Frame step.

    Returns
    -------
    float
        `step / fps`, or `1.0` when the frame rate is unknown.
    """
    return step / fps if fps > 0 else 1.0
