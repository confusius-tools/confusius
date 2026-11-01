"""Video playback controls for the ConfUSIus napari plugin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from napari.utils.notifications import NotificationSeverity, show_error
from napari_video.napari_video import VideoReaderNP
from qtpy.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from confusius._napari._video._video_viewer import VideoViewer

if TYPE_CHECKING:
    import napari
    import napari.layers


@dataclass
class _VideoEntry:
    """Per-video state held by `VideoPanel` for each loaded video."""

    video: VideoReaderNP
    layer: napari.layers.Image


class VideoPanel(QWidget):
    """Widget for reopening a dedicated video viewer and its layer list.

    This widget is embedded in the Data I/O section.  Videos are shown in
    their own napari viewer docked in the main window (see `VideoViewer`),
    created when the first video is added, hidden when the last one is
    removed and shown again on the next add.  Each video is a lazy,
    frame-on-demand image layer in that viewer; several videos sit side by
    side in its grid.  The video viewer's time slider follows the main
    viewer's time and vice versa.

    The video layers and their controls are not shown here: they live in a
    dedicated "video layer list" dock below napari's own layer list (see
    `VideoViewer.controls`).  Videos removed there, by any means, are
    forgotten by this widget.

    Parameters
    ----------
    viewer : napari.Viewer
        The main napari viewer instance.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = viewer
        self._videos: list[_VideoEntry] = []
        self._video_viewer: VideoViewer | None = None
        # Remembered across video viewer instances.
        self._frame_step = 1
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._show_viewer_btn = QPushButton("Show video viewer")
        self._show_viewer_btn.setEnabled(False)
        self._show_viewer_btn.clicked.connect(self._show_video_viewer)
        self._show_layer_list_btn = QPushButton("Show video list")
        self._show_layer_list_btn.setEnabled(False)
        self._show_layer_list_btn.clicked.connect(self._show_layer_list)

        window_row = QHBoxLayout()
        window_row.addWidget(self._show_viewer_btn)
        window_row.addWidget(self._show_layer_list_btn)
        layout.addLayout(window_row)

    def _show_video_viewer(self) -> None:
        """Reopen the video canvas if it was closed."""
        if self._video_viewer is not None:
            self._video_viewer.show_viewer()

    def _show_layer_list(self) -> None:
        """Reopen the video layer list if it was closed."""
        if self._video_viewer is not None:
            self._video_viewer.show_layer_list()

    def _update_window_buttons(self) -> None:
        """Enable reopening hidden video windows only while videos are loaded."""
        video_viewer = self._video_viewer
        enabled = video_viewer is not None and bool(self._videos)
        self._show_viewer_btn.setEnabled(enabled and video_viewer.dock.isHidden())
        self._show_layer_list_btn.setEnabled(
            enabled and video_viewer.controls_dock.isHidden()
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def add_video(self, path: Path) -> None:
        """Open `path` and add it to the video viewer.

        Parameters
        ----------
        path : pathlib.Path
            Video file to open.
        """
        if not path.is_file():
            self._show_error(f"File not found: {path}")
            return

        try:
            video = VideoReaderNP(str(path), remove_leading_singleton=False)
            # Decode one frame now: a file that opens but cannot be decoded
            # (truncated, zero frames) must fail before the video viewer shows.
            video[0]
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Cannot open video: {exc}")
            return

        if self._video_viewer is None:
            self._video_viewer = self._open_video_viewer()
        elif not self._videos:
            self._video_viewer.show()

        layer = self._video_viewer.add_video(
            video, name=f"Video: {path.stem}", step=self._frame_step
        )
        self._videos.append(_VideoEntry(video=video, layer=layer))
        self._update_window_buttons()

    def _show_error(self, message: str) -> None:
        """Show an error on the video viewer, or on the main viewer if it is hidden.

        Parameters
        ----------
        message : str
            Error message to show.
        """
        if self._video_viewer is not None and not self._video_viewer.dock.isHidden():
            self._video_viewer.notify(message, NotificationSeverity.ERROR)
        else:
            show_error(message)

    def _open_video_viewer(self) -> VideoViewer:
        """Create the video viewer.

        Returns
        -------
        VideoViewer
            The new video viewer.
        """
        video_viewer = VideoViewer(self._viewer, anchor=self)
        video_viewer.frame_step_spin.setValue(self._frame_step)
        video_viewer.frame_step_spin.valueChanged.connect(self._on_frame_step_changed)
        video_viewer.dock.visibilityChanged.connect(self._update_window_buttons)
        video_viewer.controls_dock.visibilityChanged.connect(
            self._update_window_buttons
        )
        # Let the viewer resynchronize and refit before the panel hides it.
        video_viewer.model.layers.events.removed.connect(
            self._on_layer_removed, position="last"
        )
        return video_viewer

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def _on_layer_removed(self, event) -> None:
        """Forget a removed video; hide the viewer when none is left.

        Parameters
        ----------
        event : napari.utils.events.Event
            Layer-list removal event whose `value` is the removed layer.
        """
        self._videos = [e for e in self._videos if e.layer is not event.value]
        if not self._videos and self._video_viewer is not None:
            self._video_viewer.hide()
        self._update_window_buttons()

    # ------------------------------------------------------------------
    # Frame step
    # ------------------------------------------------------------------

    def _on_frame_step_changed(self, value: int) -> None:
        """Rebuild every video layer with a new frame step.

        Parameters
        ----------
        value : int
            Show every *value*-th frame.
        """
        self._frame_step = value
        if self._video_viewer is None or not self._videos:
            return
        for entry in self._videos:
            self._video_viewer.set_frame_step(entry.layer, entry.video, value)
