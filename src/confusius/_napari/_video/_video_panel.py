"""Video playback controls for the ConfUSIus napari plugin."""

from __future__ import annotations

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


class VideoPanel(QWidget):
    """Widget for reopening a dedicated video viewer and its layer list.

    This widget is embedded in the Data I/O section. Videos are shown in their own
    napari viewer beside the main canvas (see `VideoViewer`), created when the first
    video is added, hidden when the last one is removed and shown again on the next add.
    Each video is a lazy, frame-on-demand image layer in that viewer; several videos sit
    side by side in its grid. The video viewer's time slider follows the main viewer's
    time and vice versa.

    The video layers and their controls are not shown here: they live in a dedicated
    "video layer list" dock below napari's own layer list (see `VideoViewer.controls`).
    The video viewer owns the layers, readers, and frame-step setting; this widget only
    loads files and toggles its windows.

    Parameters
    ----------
    viewer : napari.Viewer
        The main napari viewer instance.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = viewer
        self._video_viewer: VideoViewer | None = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Create the viewer and layer-list toggle buttons."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._show_viewer_btn = QPushButton("Show video viewer")
        self._show_viewer_btn.setEnabled(False)
        self._show_viewer_btn.clicked.connect(self._toggle_video_viewer)
        self._show_layer_list_btn = QPushButton("Show video list")
        self._show_layer_list_btn.setEnabled(False)
        self._show_layer_list_btn.clicked.connect(self._toggle_layer_list)

        window_row = QHBoxLayout()
        window_row.addWidget(self._show_viewer_btn)
        window_row.addWidget(self._show_layer_list_btn)
        layout.addLayout(window_row)

    def _toggle_video_viewer(self) -> None:
        """Show or hide the video canvas."""
        if self._video_viewer is None:
            return
        if self._video_viewer.qt_viewer.isHidden():
            self._video_viewer.show_viewer()
        else:
            self._video_viewer.hide_viewer()
        self._update_window_buttons()

    def _toggle_layer_list(self) -> None:
        """Show or hide the video layer list."""
        if self._video_viewer is None:
            return
        if self._video_viewer.controls_dock.isHidden():
            self._video_viewer.show_layer_list()
        else:
            self._video_viewer.controls_dock.hide()

    def _update_window_buttons(self) -> None:
        """Update the window toggles for the loaded videos."""
        video_viewer = self._video_viewer
        enabled = video_viewer is not None and bool(video_viewer.model.layers)
        viewer_hidden = not enabled or video_viewer.qt_viewer.isHidden()
        list_hidden = not enabled or video_viewer.controls_dock.isHidden()
        self._show_viewer_btn.setEnabled(enabled)
        self._show_viewer_btn.setText(
            "Show video viewer" if viewer_hidden else "Hide video viewer"
        )
        self._show_layer_list_btn.setEnabled(enabled)
        self._show_layer_list_btn.setText(
            "Show video list" if list_hidden else "Hide video list"
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def add_video(self, path: Path) -> None:
        """Open `path` and add it to the video viewer.

        Missing or unreadable files produce an error notification without adding a layer.

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
            # Decode one frame now: a file that opens but cannot be decoded (truncated,
            # zero frames) must fail before the video viewer shows.
            video[0]
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Cannot open video: {exc}")
            return

        if self._video_viewer is None:
            self._video_viewer = self._open_video_viewer()

        self._video_viewer.add_video(video, name=f"Video: {path.stem}")
        self._update_window_buttons()

    def _show_error(self, message: str) -> None:
        """Show an error on the video viewer, or on the main viewer if it is hidden.

        Parameters
        ----------
        message : str
            Error message to show.
        """
        if (
            self._video_viewer is not None
            and not self._video_viewer.qt_viewer.isHidden()
        ):
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
        video_viewer.controls_dock.visibilityChanged.connect(
            self._update_window_buttons
        )
        # Update the toggles after the viewer handles an empty layer list.
        video_viewer.model.layers.events.removed.connect(
            self._update_window_buttons, position="last"
        )
        return video_viewer
