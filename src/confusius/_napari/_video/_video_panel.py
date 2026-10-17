"""Video loading panel for the ConfUSIus napari plugin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from napari.utils.notifications import NotificationSeverity, show_error
from napari_video.napari_video import VideoReaderNP
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

    path: Path
    video: VideoReaderNP
    fps: float
    name: str
    layer: napari.layers.Image


class VideoPanel(QWidget):
    """Panel for loading videos into a dedicated video viewer.

    Videos are shown in their own napari viewer docked in the main window
    (see `VideoViewer`), created when the first video is added, hidden when
    the last one is removed and shown again on the next add.  Each video is
    a lazy, frame-on-demand image layer in that viewer; several videos sit
    side by side in its grid.  The video viewer's time slider follows the
    main viewer's time and vice versa.

    While videos are loaded, the "Loaded videos" group shows the viewer's
    controls (see `VideoViewer.controls`): napari's layer list for the video
    layers plus transpose, grid, frame step and delete buttons.  Videos
    removed there, by any means, are forgotten by the panel.

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
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Video file selector.
        self._file_group = file_group = QGroupBox("Video file")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(6)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Path to .mp4 / .mov / .avi")
        self._path_edit.textChanged.connect(self._on_path_changed)
        self._path_edit.returnPressed.connect(self._load_from_path)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse)

        path_row = QHBoxLayout()
        path_row.addWidget(self._path_edit)
        path_row.addWidget(browse_btn)
        file_layout.addLayout(path_row)
        layout.addWidget(file_group)

        self._load_btn = QPushButton("Add video")
        self._load_btn.setObjectName("primary_btn")
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._load_from_path)
        layout.addWidget(self._load_btn)

        # Loaded videos: the video viewer's controls while it is open.
        self._videos_group = videos_group = QGroupBox("Loaded videos")
        self._videos_layout = QVBoxLayout(videos_group)
        self._videos_layout.setSpacing(4)
        self._no_videos_label = QLabel("No videos loaded")
        self._no_videos_label.setEnabled(False)
        self._videos_layout.addWidget(self._no_videos_label)
        # Takes the remaining height so the layer list grows like napari's.
        layout.addWidget(videos_group, stretch=1)

    def _on_path_changed(self, text: str) -> None:
        """Enable the load button once a path is entered."""
        self._load_btn.setEnabled(bool(text.strip()))

    # ------------------------------------------------------------------
    # Browse / Load
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        start = self._path_edit.text().strip() or str(Path.home())
        dialog = QFileDialog(self, "Open Video", start)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter(
            "Video files (*.mp4 *.mov *.avi);;All files (*)",
        )
        if dialog.exec() and dialog.selectedFiles():
            self._path_edit.setText(dialog.selectedFiles()[0])
            self._load_from_path()

    def _load_from_path(self) -> None:
        """Validate the path field and call `_add_video`."""
        path_str = self._path_edit.text().strip()
        if not path_str:
            show_error("Select a video file first.")
            return
        self._add_video(Path(path_str))

    def _add_video(self, path: Path) -> None:
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
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Cannot open video: {exc}")
            return

        if self._video_viewer is None:
            self._video_viewer = self._open_video_viewer()
        elif not self._videos:
            self._video_viewer.show()
            self._no_videos_label.hide()

        name = f"Video: {path.stem}"
        layer = self._video_viewer.add_video(
            video, name=name, fps=video.frame_rate, step=self._frame_step
        )
        self._videos.append(
            _VideoEntry(
                path=path, video=video, fps=video.frame_rate, name=name, layer=layer
            )
        )

        n_frames = video.number_of_frames
        n_shown = len(range(0, n_frames, self._frame_step))
        self._video_viewer.notify(
            f"Video '{path.stem}' added ({n_frames} frames, {video.frame_rate:.1f} "
            f"fps, showing {n_shown} at step {self._frame_step})."
        )

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
        """Create the video viewer and show its controls in the panel.

        Returns
        -------
        VideoViewer
            The new video viewer.
        """
        video_viewer = VideoViewer(self._viewer, anchor=self)
        video_viewer.frame_step_spin.setValue(self._frame_step)
        video_viewer.frame_step_spin.valueChanged.connect(self._on_frame_step_changed)
        # After the viewer's own handlers, so it has refit itself before the
        # panel may close it.
        video_viewer.model.layers.events.removed.connect(
            self._on_layer_removed, position="last"
        )
        self._no_videos_label.hide()
        self._videos_layout.addWidget(video_viewer.controls, stretch=1)
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
        if self._videos or self._video_viewer is None:
            return
        self._video_viewer.hide()
        self._no_videos_label.show()

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
            self._video_viewer.set_frame_step(
                entry.layer, entry.video, entry.fps, value
            )

        first = self._videos[0]
        n_shown = len(range(0, first.video.number_of_frames, value))
        self._video_viewer.notify(
            f"Frame step {value}: showing {n_shown} frames "
            f"({first.fps / value:.1f} fps effective)."
        )
