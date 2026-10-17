"""Dedicated napari viewer for synchronized video playback.

Videos live in their own `ViewerModel` + `QtViewer` docked in the main window
(the pattern from napari's ``multiple_viewer_widget`` gallery example), so they
never share dims, scale, translate, grid or axis order with the fUSI data. The
only coupling to the main viewer is time: the main viewer's world time along
its time axis drives the video viewer's time slider and vice versa.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from napari.components import ViewerModel
from napari.qt import QtViewer, QtViewerButtons, get_stylesheet
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QWIDGETSIZE_MAX,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from confusius._napari._qt import find_main_window
from confusius._napari._timeaxis import find_time_dim_index

if TYPE_CHECKING:
    import napari
    import napari.layers
    from napari_video.napari_video import VideoReaderNP


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
        self._video = video
        self._step = step
        frame = np.ascontiguousarray(video[0])
        self._frame_shape = frame.shape
        n_frames = len(range(0, video.number_of_frames, step))
        self.shape = (n_frames, *frame.shape)
        self.dtype = frame.dtype
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
        return np.ascontiguousarray(self._video[t * self._step])

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


class VideoViewer:
    """A second napari viewer, docked in the main window, that shows only videos.

    Parameters
    ----------
    viewer : napari.Viewer
        The main napari viewer whose time axis drives the video.
    anchor : qtpy.QtWidgets.QWidget, optional
        A widget living inside a dock widget of the main window (typically
        the panel creating this viewer).  The video dock is placed between
        the main canvas and that dock.  If not provided, or if that dock is
        floating, the video dock is simply added to the right dock area.
    """

    def __init__(self, viewer: napari.Viewer, *, anchor: QWidget | None = None) -> None:
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
        self.dock = viewer.window.add_dock_widget(
            self.qt_viewer, name="Video", area="right"
        )
        # No close button: the dock lives exactly as long as videos are
        # loaded.  No floating: a detached vispy canvas needs
        # `AA_ShareOpenGLContexts`, which must be set before the QApplication
        # exists and a plugin cannot guarantee.
        self.dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        # napari 0.9 caps a plugin dock at its initial size hint (explicit max
        # height on the dock, Maximum policy on its content); lift both so the
        # video canvas spans the full window height like the main one.
        for widget in (self.qt_viewer, self.dock):
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            widget.setMaximumHeight(QWIDGETSIZE_MAX)
        self._anchor_dock: QDockWidget | None = None
        self._place_left_of(anchor)

        # Panel-side controls built from napari's own widgets for this model,
        # laid out like napari's layer panel: a row with the transpose and
        # grid buttons, the frame step and the delete button, then the layer
        # list.  The stock buttons run app-wide actions that resolve to the
        # main viewer, so they are rewired here; the grid button keeps
        # napari's right-click popup, which edits this model's grid.
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
        self.frame_step_spin.setToolTip(
            "Show every N-th frame. Higher values skip frames for lighter playback."
        )
        self.layer_list = self.qt_viewer.layers
        # napari's layer context menu acts on the main viewer's layers and
        # offers commands a video layer cannot run (duplicate, split RGB,
        # projections decode every frame), so the list's delegate shows a
        # small menu of our own instead.
        delegate = self.layer_list.itemDelegate()
        delegate.show_context_menu = lambda index, model, pos, parent: (  # type: ignore
            self.build_layer_context_menu().exec(pos)
        )

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(4)
        # Frame step on the left, the three buttons ragged right.
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(QLabel("Frame step"))
        row.addWidget(self.frame_step_spin)
        row.addStretch()
        row.addWidget(self.buttons)
        row.addWidget(self.layer_buttons)
        controls_layout.addLayout(row)
        controls_layout.addWidget(self.layer_list, stretch=1)
        self._sync_theme()

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
        self.model.dims.events.point.connect(self._sync_to_main)
        self.model.dims.events.point.connect(self._update_time_text)
        self.model.layers.events.removed.connect(self._update_time_text)
        self.model.layers.events.removed.connect(self._on_layer_removed)

    def _place_left_of(self, anchor: QWidget | None) -> None:
        """Move the video dock between the main canvas and `anchor`'s dock.

        Parameters
        ----------
        anchor : qtpy.QtWidgets.QWidget, optional
            A widget inside the dock to sit next to.
        """
        dock = anchor
        while dock is not None and not isinstance(dock, QDockWidget):
            dock = dock.parentWidget()
        window = find_main_window(self.dock)
        if dock is None or dock is self.dock or window is None or dock.isFloating():
            return
        # `splitDockWidget` puts the second dock after the first, so the video
        # dock is added first and the panel's dock re-inserted to its right.
        self._anchor_dock = dock
        width = dock.width()
        window.removeDockWidget(dock)
        window.splitDockWidget(self.dock, dock, Qt.Orientation.Horizontal)
        dock.show()
        # The split halves the panel's width: give the video that width and
        # the panel its own back, so the canvas is what shrinks.
        window.resizeDocks([self.dock, dock], [width, width], Qt.Orientation.Horizontal)

    # ------------------------------------------------------------------
    # Layers
    # ------------------------------------------------------------------

    def add_video(
        self,
        video: VideoReaderNP,
        *,
        name: str,
        fps: float,
        step: int = 1,
    ) -> napari.layers.Image:
        """Add a video as a plain `(time, y, x[, rgb])` image layer.

        Parameters
        ----------
        video : VideoReaderNP
            The opened video reader.
        name : str
            Layer name.
        fps : float
            Video frame rate; the time scale becomes `step / fps` seconds.
        step : int, default: 1
            Show every *step*-th frame.

        Returns
        -------
        napari.layers.Image
            The new video layer in the video viewer.
        """
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
                scale=(_time_scale(fps, step), 1.0, 1.0),
                axis_labels=("time", "y", "x"),
                units=("s", None, None),
                metadata={"fps": fps, "time_units": "s"},
            )
        finally:
            self._syncing = False
        assert not isinstance(layer, list)
        self._sync_from_main()
        self._update_time_text()
        # The dock is laid out on the next event-loop pass; only then does
        # the canvas have its real size, so fitting the view must wait.
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

    def set_frame_step(
        self, layer: napari.layers.Image, video: VideoReaderNP, fps: float, step: int
    ) -> None:
        """Rebuild `layer` with a new frame step, keeping the current time.

        Parameters
        ----------
        layer : napari.layers.Image
            A layer previously returned by `add_video`.
        video : VideoReaderNP
            The reader backing `layer`.
        fps : float
            Video frame rate.
        step : int
            Show every *step*-th frame.
        """
        current_time = float(self.model.dims.point[0])
        self._syncing = True
        try:
            layer.data = _VideoArray(video, step=step)
            layer.scale = (_time_scale(fps, step), 1.0, 1.0)  # ty: ignore[invalid-assignment]
            self.model.dims.set_point(0, current_time)
        finally:
            self._syncing = False

    def _on_layer_removed(self, event=None) -> None:
        """Refit the remaining videos after one is removed."""
        if self.model.layers:
            self.model.reset_view()

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
        # The controls sit inside the ConfUSIus panel, whose stylesheet would
        # restyle them; napari's own stylesheet restores their look, minus the
        # 242 px `min-width` napari gives these widgets to size its layer dock
        # and the spin-box `min-width` that would make the frame step 132 px.
        self.controls.setStyleSheet(
            get_stylesheet(self._main.theme)
            + "QtLayerButtons, QtViewerButtons, QtLayerList { min-width: 0px; }"
            + "QSpinBox { min-width: 0px; }"
        )

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

    def _sync_from_main(self, event=None) -> None:
        """Move the video slider to the main viewer's current world time."""
        time_idx = find_time_dim_index(self._main)
        if self._syncing or time_idx is None or not self.model.layers:
            return
        self._syncing = True
        try:
            self.model.dims.set_point(0, float(self._main.dims.point[time_idx]))
        finally:
            self._syncing = False

    def _sync_to_main(self, event=None) -> None:
        """Move the main viewer's time slider to the video's current world time."""
        time_idx = find_time_dim_index(self._main)
        if self._syncing or time_idx is None or not self.model.layers:
            return
        self._syncing = True
        try:
            video_time = float(self.model.dims.point[0])
            self._main.dims.set_point(time_idx, video_time)
            # When the main step changes, napari's time slider snaps the point
            # to that step. Setting it again keeps the video's exact time (the
            # step is unchanged, so the slider leaves it alone): the main viewer
            # still shows its nearest frame, and time cursors follow the video.
            self._main.dims.set_point(time_idx, video_time)
        finally:
            self._syncing = False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def hide(self) -> None:
        """Hide the dock and controls, giving the dock's width back to the canvas.

        The viewer is kept alive rather than closed: creating another GL
        canvas in the window after one existed crashes the process on some
        GL stacks (Mesa software rendering under WSLg, for one), so one
        video viewer is reused for the whole session.
        """
        anchor = self._anchor_dock
        anchor_width = anchor.width() if anchor is not None else 0
        window = find_main_window(self.dock)
        self.dock.hide()
        self.controls.hide()
        if anchor is not None and window is not None:
            window.resizeDocks([anchor], [anchor_width], Qt.Orientation.Horizontal)

    def show(self) -> None:
        """Show the dock and controls again, sized as on first opening."""
        anchor = self._anchor_dock
        anchor_width = anchor.width() if anchor is not None else 0
        window = find_main_window(self.dock)
        self.dock.show()
        self.controls.show()
        if anchor is not None and window is not None:
            window.resizeDocks(
                [self.dock, anchor],
                [anchor_width, anchor_width],
                Qt.Orientation.Horizontal,
            )

    def close(self) -> None:
        """Disconnect from the main viewer and remove the dock."""
        object.__delattr__(self._main, "reset_view")
        self._main.dims.events.point.disconnect(self._sync_from_main)
        self._main.events.theme.disconnect(self._sync_theme)
        self.model.dims.events.point.disconnect(self._sync_to_main)
        anchor = self._anchor_dock
        anchor_width = anchor.width() if anchor is not None else 0
        window = find_main_window(self.dock)
        self._main.window.remove_dock_widget(self.dock)
        if anchor is not None and window is not None:
            # Qt hands the removed dock's space to the panel beside it; give
            # it back to the canvas by restoring the panel's width.
            window.resizeDocks([anchor], [anchor_width], Qt.Orientation.Horizontal)
        self.qt_viewer.close()
        # Safe to delete: napari's action manager drops its tooltip hooks on
        # the buttons' `destroyed` signal.  Unparenting pulls the controls
        # out of the panel's layout.
        self.controls.setParent(None)
        self.controls.deleteLater()
        # The model's event callbacks hold bound methods of the QtViewer, so
        # dropping both references is what lets the QtViewer be collected.
        del self.qt_viewer, self.model, self.dock, self.controls


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
