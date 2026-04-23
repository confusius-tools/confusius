"""Video loading panel for the ConfUSIus napari plugin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from napari.utils.notifications import show_error, show_info
from napari_video.napari_video import VideoReaderNP
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari
    import napari.layers


class _VideoArray:
    """Array-like wrapper around `VideoReaderNP` for napari Image layers.

    Provides the ``shape``, ``dtype``, and ``__getitem__`` interface that
    napari requires for lazy, frame-on-demand display.  Handles
    singleton-dimension padding so the video matches the fUSI scan's
    dimensionality.

    The positions of H and W in the shape are controlled by ``h_dim`` and
    ``w_dim``.  When ``h_dim > w_dim`` (H appears after W in the layout),
    the raw ``(H, W)`` frame is transposed before reshaping so that the
    data matches the expected axis order.

    Parameters
    ----------
    video : VideoReaderNP
        The opened video reader.
    dtype : numpy.dtype
        Data type of a decoded frame.
    frame_shape : tuple[int, ...]
        Shape of a single decoded frame --- ``(H, W)`` or ``(H, W, C)``.
    n_pad : int, default: 0
        Number of size-1 dimensions inserted between the time axis and
        the spatial axes.
    step : int, default: 1
        Show every *step*-th frame (temporal subsampling).  Logical
        frame ``t`` maps to physical frame ``t * step``.
    time_dim : int, default: 0
        Position of the time axis in the output shape.
    h_dim : int or None, optional
        Position of the video height axis.  Defaults to ``n_core - 2``
        where ``n_core = 1 + n_pad + 2``.
    w_dim : int or None, optional
        Position of the video width axis.  Defaults to ``n_core - 1``.
    """

    def __init__(
        self,
        video: VideoReaderNP,
        *,
        dtype: np.dtype,
        frame_shape: tuple[int, ...],
        n_pad: int = 0,
        step: int = 1,
        time_dim: int = 0,
        h_dim: int | None = None,
        w_dim: int | None = None,
    ) -> None:
        self._video = video
        self._n_pad = n_pad
        self._step = step

        n_frames = len(range(0, video.number_of_frames, step))
        is_rgb = len(frame_shape) == 3 and frame_shape[2] in (3, 4)
        self._is_rgb = is_rgb

        n_core = 1 + n_pad + 2
        self._n_core = n_core

        # Default dim positions: time first, H and W last.
        self._time_dim = time_dim
        self._h_dim = h_dim if h_dim is not None else n_core - 2
        self._w_dim = w_dim if w_dim is not None else n_core - 1

        # Precompute whether H and W need transposing.
        self._transpose_hw = self._h_dim > self._w_dim

        # Build the core shape by placing n_frames, H, W at their positions.
        core_shape = [1] * n_core
        core_shape[self._time_dim] = n_frames
        core_shape[self._h_dim] = frame_shape[0]
        core_shape[self._w_dim] = frame_shape[1]

        self.shape = tuple(core_shape) + (frame_shape[2:3] if is_rgb else ())
        self.dtype = dtype
        self.ndim = len(self.shape)
        # Precompute the padded frame shape used by _fetch.
        # After optional transpose, the frame is (H', W' [,C]) where H'/W'
        # may be swapped.  We reshape it into the non-time core dims.
        non_time_shape = [s for i, s in enumerate(core_shape) if i != self._time_dim]
        if is_rgb:
            non_time_shape.append(frame_shape[2])
        self._padded_frame_shape = tuple(non_time_shape)

    def __len__(self) -> int:  # noqa: D105
        return self.shape[self._time_dim]

    def __getitem__(self, idx):  # noqa: D105
        if not isinstance(idx, tuple):
            idx = (idx,)

        # Separate the optional RGB / channel index.
        if self._is_rgb and len(idx) > self._n_core:
            core_idx = idx[: self._n_core]
            rgb_idx = idx[self._n_core :]
        else:
            core_idx = idx
            rgb_idx = ()

        # Pad to full core rank.
        core_idx += (slice(None),) * (self._n_core - len(core_idx))

        time_idx = core_idx[self._time_dim]
        # Build spatial index from all non-time dims.
        spatial_idx = (
            tuple(core_idx[i] for i in range(self._n_core) if i != self._time_dim)
            + rgb_idx
        )

        def _fetch(t: int) -> np.ndarray:
            frame = np.ascontiguousarray(self._video[t * self._step])
            # Transpose H and W if W comes before H in the layout.
            if self._transpose_hw:
                frame = np.swapaxes(frame, 0, 1)
            # Reshape into padded non-time dims.
            padded = frame.reshape(self._padded_frame_shape)
            return padded[spatial_idx]

        # --- single frame (integer time index) ---
        if isinstance(time_idx, (int, np.integer)):
            return _fetch(int(time_idx))

        # --- multiple frames (slice time index) ---
        if isinstance(time_idx, slice):
            n_time = self.shape[self._time_dim]

            # Guard: an unbounded slice on the time axis (slice(None)) means
            # napari placed time among the displayed dims.  The dim-order
            # guard will fix this momentarily, but _update_layers already
            # fired and issued this slice *before* our handler ran (core
            # callbacks always execute before plugin callbacks).  Decoding
            # every frame would freeze the UI, so we return a single frame
            # in the expected (1, ...) shape instead.
            #
            # Legitimate multi-frame requests (thick slicing / projections)
            # always arrive as bounded slice(low, high) from the margin
            # calculation, never as slice(None).
            if (
                time_idx.start is None
                and time_idx.stop is None
                and time_idx.step is None
            ):
                return _fetch(0)[np.newaxis]

            start, stop, step = time_idx.indices(n_time)
            frames = [_fetch(t) for t in range(start, stop, step or 1)]
            if not frames:
                return np.empty((0,) + _fetch(0).shape, dtype=self.dtype)
            return np.stack(frames)

        raise IndexError(f"Unsupported time index type: {type(time_idx)}")


# ---------------------------------------------------------------------------
# Per-video state
# ---------------------------------------------------------------------------


@dataclass
class _VideoEntry:
    """Per-video state held by `VideoPanel` for each loaded video.

    Groups the reader, decoded frame metadata, and the napari layer so
    the panel can manage multiple videos as independent entries.
    """

    path: Path
    video: VideoReaderNP
    frame_dtype: np.dtype
    frame_shape: tuple[int, ...]
    video_h: int
    video_w: int
    is_rgb: bool
    fps: float
    name: str
    layer: napari.layers.Image | None = None


# ---------------------------------------------------------------------------
# Panel widget
# ---------------------------------------------------------------------------


class VideoPanel(QWidget):
    """Panel for loading videos as image layers in the napari viewer.

    Multiple videos can be added; each becomes its own napari Image
    layer and its own grid cell.  All videos share a single reference
    fUSI scan (selected at first load) so they align on the same time
    and spatial axes.  Videos are passed to napari as lazy, array-like
    objects backed by ``VideoReaderNP`` (OpenCV frame-on-demand
    decoding).  A thin wrapper (`_VideoArray`) handles singleton
    dimension padding and dimension reordering.

    When the viewer's displayed dimensions change, every video layer
    is rebuilt with a new `_VideoArray` whose shape places H and W at
    the currently displayed dim positions.

    The video layers receive the same ``axis_labels`` as the fUSI scan
    so that napari handles dimension reordering identically for both.
    The time scale is ``frame_step / fps``, shared across all videos.
    Spatial dimensions use a per-video isotropic scale matching the
    fUSI scan height.

    Grid mode is enabled with a single-row shape when the first video
    loads, so reference scan, labels, and each video appear side-by-
    side.  The viewer's previous grid state is restored when the last
    video is removed.

    A dimension-order guard prevents the time axis from ever being
    displayed spatially.

    Parameters
    ----------
    viewer : napari.Viewer
        The main napari viewer instance.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self._viewer = viewer

        # Loaded video entries.  All entries share the same reference
        # layer, axis labels, and time index (established at first load).
        self._videos: list[_VideoEntry] = []

        # Shared reference-layer state (from the first video's ref).
        self._ref_layer: napari.layers.Layer | None = None
        self._axis_labels: tuple[str, ...] = ()
        self._units: list[str | None] = []
        self._n_pad: int = 0
        self._fusi_time_idx: int = 0

        # Dim indices of the currently displayed vertical and horizontal axes.
        self._displayed_dims: tuple[int, int] = (0, 0)

        # Grid-mode state saved before enabling, restored on last unload.
        self._grid_was_enabled: bool = False
        self._grid_shape_was: tuple[int, int] = (-1, -1)

        # Dimension guard.
        self._guarding_order: bool = False

        self._setup_ui()

        # Keep the reference layer combo in sync with the layer list.
        self._viewer.layers.events.inserted.connect(self._refresh_layer_combo)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Reference layer selector.
        self._ref_group = ref_group = QGroupBox("Reference layer")
        ref_layout = QVBoxLayout(ref_group)
        ref_layout.setSpacing(4)
        self._ref_combo = QComboBox()
        self._ref_combo.setPlaceholderText("No layers loaded")
        self._ref_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._ref_combo.currentIndexChanged.connect(self._on_ref_changed)
        # `activated` only fires for user interaction, not programmatic
        # setCurrentIndex calls from `_refresh_layer_combo` -- so
        # switching the reference triggers a rebuild only on real user
        # input, never as a side-effect of layer inserts or removals.
        self._ref_combo.activated.connect(self._on_user_ref_changed)
        ref_layout.addWidget(self._ref_combo)
        layout.addWidget(ref_group)

        # Video file selector.
        self._file_group = file_group = QGroupBox("Video file")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(6)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Path to .mp4 / .mov / .avi")
        self._path_edit.textChanged.connect(lambda _: self._on_ref_changed())
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

        # Loaded videos list.
        self._videos_group = videos_group = QGroupBox("Loaded videos")
        videos_layout = QVBoxLayout(videos_group)
        videos_layout.setSpacing(4)
        self._videos_list = QListWidget()
        self._videos_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._videos_list.setFixedHeight(80)
        # Long video names must not force the panel wider than its dock.
        # Ignored horizontal policy makes the widget accept whatever width
        # the layout gives it, and ElideRight adds an ellipsis to long items.
        self._videos_list.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self._videos_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._videos_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._videos_list.itemSelectionChanged.connect(self._on_video_selection_changed)
        videos_layout.addWidget(self._videos_list)

        self._remove_btn = QPushButton("Remove selected")
        self._remove_btn.setEnabled(False)
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        videos_layout.addWidget(self._remove_btn)
        layout.addWidget(videos_group)

        # Playback controls (enabled after loading).
        self._playback_group = playback_group = QGroupBox("Playback")
        playback_layout = QVBoxLayout(playback_group)
        playback_layout.setSpacing(4)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Frame step:"))
        self._step_spin = QSpinBox()
        self._step_spin.setRange(1, 100)
        self._step_spin.setValue(1)
        self._step_spin.setMaximumWidth(50)
        self._step_spin.setToolTip(
            "Show every N-th frame. Higher values skip frames for lighter playback."
        )
        self._step_spin.setEnabled(False)
        self._step_spin.valueChanged.connect(self._on_frame_step_changed)
        step_row.addWidget(self._step_spin)
        playback_layout.addLayout(step_row)

        layout.addWidget(playback_group)

        layout.addStretch()
        self._refresh_layer_combo()

    # ------------------------------------------------------------------
    # Reference layer helpers
    # ------------------------------------------------------------------

    def _on_layer_removed(self, event=None) -> None:
        """Handle external layer removal.

        Drops any entries whose layer was removed outside the panel,
        refreshes the combo, and restores grid state if the last video
        was removed.
        """
        removed = [
            e
            for e in list(self._videos)
            if e.layer is not None and e.layer not in self._viewer.layers
        ]
        for e in removed:
            self._forget_entry(e)

        if self._ref_layer is not None and self._ref_layer not in self._viewer.layers:
            self._ref_layer = None

        if removed:
            self._refresh_videos_list()
            if not self._videos:
                self._on_last_video_removed()

        self._refresh_layer_combo()

    def _refresh_layer_combo(self, event=None) -> None:
        """Repopulate the reference layer combo from current viewer layers."""
        current = self._ref_combo.currentText()
        self._ref_combo.clear()
        video_layers = {e.layer for e in self._videos if e.layer is not None}
        for layer in self._viewer.layers:
            if layer in video_layers:
                continue
            self._ref_combo.addItem(layer.name)
        idx = self._ref_combo.findText(current)
        if idx >= 0:
            self._ref_combo.setCurrentIndex(idx)
        self._on_ref_changed()

    def _on_ref_changed(self) -> None:
        """Update the load button state from the current ref/path selection."""
        has_path = bool(self._path_edit.text().strip())
        has_ref = self._ref_combo.currentIndex() >= 0
        self._load_btn.setEnabled(has_ref and has_path)

    def _on_user_ref_changed(self, _idx: int | None = None) -> None:
        """Handle user-initiated reference layer changes in the combo.

        If videos are already loaded and the user picks a different
        reference layer with valid xarray metadata, refresh the shared
        state (axis labels, n_pad, time index, ...) and rebuild every
        video layer so they align with the new reference.  If the new
        layer has no coordinate metadata, warn and revert the combo.
        """
        if not self._videos:
            return
        new_ref = self._get_ref_layer()
        if new_ref is None or new_ref is self._ref_layer:
            return
        if new_ref.metadata.get("xarray") is None:
            show_error(
                "Selected layer has no coordinate metadata. Keeping previous reference."
            )
            if self._ref_layer is not None:
                idx = self._ref_combo.findText(self._ref_layer.name)
                if idx >= 0:
                    self._ref_combo.blockSignals(True)
                    try:
                        self._ref_combo.setCurrentIndex(idx)
                    finally:
                        self._ref_combo.blockSignals(False)
            return

        self._update_shared_ref_state(new_ref)
        self._rebuild_all_entries()

    def _update_shared_ref_state(self, ref_layer) -> None:
        """Cache axis labels, padding, time index and units from *ref_layer*."""
        self._ref_layer = ref_layer
        ref_xr = ref_layer.metadata["xarray"]
        self._n_pad = max(ref_layer.ndim - 3, 0)
        self._fusi_time_idx = (
            list(ref_xr.dims).index("time") if "time" in ref_xr.dims else 0
        )
        self._axis_labels = tuple(ref_xr.dims)
        self._units = list(getattr(ref_layer, "units", [None] * ref_layer.ndim))

    def _get_ref_layer(self):
        """Return the layer currently selected in the combo, or None."""
        name = self._ref_combo.currentText()
        if not name:
            return None
        try:
            return self._viewer.layers[name]
        except KeyError:
            return None

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
            self._on_ref_changed()  # Re-evaluate button state.
            if self._load_btn.isEnabled():
                self._load_from_path()

    def _load_from_path(self) -> None:
        """Validate inputs and call ``_add_video``."""
        ref = self._get_ref_layer()
        if ref is None:
            show_error("Select a reference layer first.")
            return
        if ref.metadata.get("xarray") is None:
            show_error(
                "Selected layer has no coordinate metadata. "
                "Load the file using the Data panel or the File menu."
            )
            return
        path_str = self._path_edit.text().strip()
        if not path_str:
            show_error("Select a video file first.")
            return
        self._add_video(Path(path_str), ref)

    def _add_video(self, path: Path, ref_layer) -> None:
        """Add a new video as its own layer in the viewer."""
        if not path.is_file():
            show_error(f"File not found: {path}")
            return

        try:
            video = VideoReaderNP(str(path), remove_leading_singleton=False)
        except Exception as exc:  # noqa: BLE001
            show_error(f"Cannot open video: {exc}")
            return

        # Probe a single frame for shape / dtype / channel info.
        frame = np.ascontiguousarray(video[0])
        is_rgb = frame.ndim == 3 and frame.shape[2] in (3, 4)
        video_h, video_w = frame.shape[0], frame.shape[1]
        n_frames = video.number_of_frames
        fps = video.frame_rate

        # First video establishes the shared reference-layer state and
        # enables grid mode.
        if not self._videos:
            self._update_shared_ref_state(ref_layer)
            order = self._viewer.dims.order
            self._displayed_dims = (order[-2], order[-1])

            # Save and enable grid mode with a single-row layout.
            self._grid_was_enabled = self._viewer.grid.enabled
            self._grid_shape_was = tuple(self._viewer.grid.shape)
            self._viewer.grid.enabled = True
            self._viewer.grid.shape = (1, -1)

            # Connect dim-order guard.
            self._viewer.dims.events.order.connect(self._on_dim_order_changed)
            self._guarding_order = True

        entry = _VideoEntry(
            path=path,
            video=video,
            frame_dtype=frame.dtype,
            frame_shape=frame.shape,
            video_h=video_h,
            video_w=video_w,
            is_rgb=is_rgb,
            fps=fps,
            name=f"Video: {path.stem}",
        )
        self._videos.append(entry)

        # Build the layer for this entry.
        self._rebuild_entry(entry)

        # Update UI.
        self._refresh_videos_list()
        self._step_spin.setEnabled(True)

        frame_step = self._step_spin.value()
        n_shown = len(range(0, n_frames, frame_step))
        show_info(
            f"Video '{path.stem}' added ({n_frames} frames, {fps:.1f} fps, "
            f"showing {n_shown} at step {frame_step})."
        )

    # ------------------------------------------------------------------
    # Layer rebuild
    # ------------------------------------------------------------------

    def _rebuild_entry(self, entry: _VideoEntry) -> None:
        """Create or update the image layer for a single video entry.

        Builds a `_VideoArray` whose H and W are placed at the currently
        displayed vertical and horizontal dim positions.  Scale and
        translate are computed per-entry so the video aligns with the
        fUSI scan.

        When a layer already exists on the entry, updates data, scale
        and translate in-place to avoid the expensive remove/add cycle
        (vispy node destruction, contrast-limit recomputation, event
        storms).
        """
        displayed_v, displayed_h = self._displayed_dims
        frame_step = self._step_spin.value()

        data = _VideoArray(
            entry.video,
            dtype=entry.frame_dtype,
            frame_shape=entry.frame_shape,
            n_pad=self._n_pad,
            step=frame_step,
            time_dim=self._fusi_time_idx,
            h_dim=displayed_v,
            w_dim=displayed_h,
        )

        # Time scale = frame_step / fps.  Each logical frame spans
        # ``frame_step`` physical frames, so consecutive data points are
        # ``frame_step / fps`` seconds apart.
        time_scale = frame_step / entry.fps if entry.fps > 0 else 1.0
        # Isotropic spatial scale (video pixels are square).
        spatial_scale = self._compute_spatial_scale(displayed_v, entry.video_h)
        translate_v = self._compute_axis_center_translate(
            displayed_v, entry.video_h, spatial_scale
        )
        translate_h = self._compute_axis_center_translate(
            displayed_h, entry.video_w, spatial_scale
        )

        ndim = len(self._axis_labels)
        scale = [1.0] * ndim
        translate = [0.0] * ndim
        scale[self._fusi_time_idx] = time_scale
        scale[displayed_v] = spatial_scale
        scale[displayed_h] = spatial_scale
        translate[displayed_v] = translate_v
        translate[displayed_h] = translate_h

        if entry.layer is not None:
            # In-place update: reuse the existing vispy node.
            clims = entry.layer.contrast_limits
            entry.layer.data = data  # type: ignore[invalid-assignment]
            entry.layer.contrast_limits = clims
            entry.layer.scale = tuple(scale)  # type: ignore[invalid-assignment]
            entry.layer.translate = tuple(translate)
            return

        # First creation -- full add_image path.
        self._viewer.layers.events.inserted.disconnect(self._refresh_layer_combo)
        self._viewer.layers.events.removed.disconnect(self._on_layer_removed)

        layer = self._viewer.add_image(
            data,
            name=entry.name,
            rgb=entry.is_rgb,
            scale=tuple(scale),
            translate=tuple(translate),
            axis_labels=self._axis_labels,
            metadata={"fps": entry.fps, "time_units": "s"},
            units=self._units,
            visible=False,
        )
        assert not isinstance(layer, list)
        entry.layer = layer

        self._viewer.layers.events.inserted.connect(self._refresh_layer_combo)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)

        entry.layer.visible = True

    def _rebuild_all_entries(self) -> None:
        """Rebuild every loaded video's layer (for dim-order or frame-step changes)."""
        for entry in self._videos:
            self._rebuild_entry(entry)

    # ------------------------------------------------------------------
    # Remove / cleanup
    # ------------------------------------------------------------------

    def _on_video_selection_changed(self) -> None:
        """Enable remove button based on selection."""
        self._remove_btn.setEnabled(self._videos_list.currentRow() >= 0)

    def _on_remove_clicked(self) -> None:
        row = self._videos_list.currentRow()
        if not (0 <= row < len(self._videos)):
            return
        self._remove_video(self._videos[row])

    def _remove_video(self, entry: _VideoEntry) -> None:
        """Remove a single video entry and its layer.

        Actual entry cleanup happens in `_on_layer_removed` after napari
        fires the layer-removed event, so external removals and panel
        removals share the same path.
        """
        layer = entry.layer
        if layer is not None and layer in self._viewer.layers:
            self._viewer.layers.remove(layer)
            return
        # Layer already gone -- clean up entry state directly.
        self._forget_entry(entry)
        self._refresh_videos_list()
        if not self._videos:
            self._on_last_video_removed()

    def _forget_entry(self, entry: _VideoEntry) -> None:
        """Drop an entry from the list without touching the viewer."""
        try:
            self._videos.remove(entry)
        except ValueError:
            pass

    def _on_last_video_removed(self) -> None:
        """Restore viewer state after the last video is removed."""
        self._disconnect()
        self._viewer.grid.enabled = self._grid_was_enabled
        self._viewer.grid.shape = self._grid_shape_was
        self._ref_layer = None
        self._axis_labels = ()
        self._units = []
        self._n_pad = 0
        self._fusi_time_idx = 0
        self._step_spin.setEnabled(False)

    def _refresh_videos_list(self) -> None:
        """Update the UI list of loaded videos."""
        self._videos_list.clear()
        for entry in self._videos:
            self._videos_list.addItem(entry.name)

    # ------------------------------------------------------------------
    # Frame step
    # ------------------------------------------------------------------

    def _on_frame_step_changed(self, value: int) -> None:
        """Rebuild all video layers with a new frame step.

        The step is encoded in each `_VideoArray` shape (fewer logical
        frames) and the layer's time scale (``value / fps``).  napari
        auto-computes the correct slider range from shape and scale.

        The current world time is saved before the rebuild and restored
        afterward so the slider stays at the closest valid position.
        """
        if not self._videos:
            return

        time_axis = self._fusi_time_idx
        current_time = float(self._viewer.dims.point[time_axis])

        self._rebuild_all_entries()

        self._viewer.dims.set_point(time_axis, current_time)

        first = self._videos[0]
        n_frames = first.video.number_of_frames
        n_shown = len(range(0, n_frames, value))
        show_info(
            f"Frame step {value}: showing {n_shown} frames "
            f"({first.fps / value:.1f} fps effective)."
        )

    # ------------------------------------------------------------------
    # Spatial transform
    # ------------------------------------------------------------------

    def _lookup_coord(self, dim_idx: int) -> np.ndarray | None:
        """Return the reference xarray coordinate for *dim_idx*, or None.

        Returns ``None`` when there is no reference layer, no xarray
        metadata, or the corresponding coordinate does not exist.
        """
        ref = self._ref_layer
        if ref is None:
            return None

        try:
            xr_da = ref.metadata.get("xarray")
        except RuntimeError:
            # Layer's C++ wrapper was deleted.
            self._ref_layer = None
            return None
        if xr_da is None:
            return None

        dim_name = self._axis_labels[dim_idx]
        if dim_name not in xr_da.coords:
            return None

        return np.asarray(xr_da.coords[dim_name], dtype=np.float64)

    def _compute_spatial_scale(self, vertical_dim: int, video_h: int) -> float:
        """Return the isotropic spatial scale for the video.

        The scale maps the video's height to the fUSI scan's extent
        along ``vertical_dim`` and is then applied identically to both
        displayed spatial axes so that video pixels remain square --
        webcam pixels are isotropic and must not be stretched.

        Parameters
        ----------
        vertical_dim : int
            Viewer dim index whose coordinate extent the video height
            should match.
        video_h : int
            Video height in pixels.
        """
        coords = self._lookup_coord(vertical_dim)
        if coords is None or coords.size == 0:
            return 1.0

        y_min, y_max = float(coords.min()), float(coords.max())
        if coords.size > 1:
            y_step = float(np.median(np.diff(coords)))
        else:
            dim_name = self._axis_labels[vertical_dim]
            xr_da = self._ref_layer.metadata["xarray"]  # type: ignore[union-attr]
            y_step = float(xr_da.coords[dim_name].attrs.get("voxdim", 1.0))

        fusi_extent = (y_max - y_min) + abs(y_step)
        return fusi_extent / video_h

    def _compute_axis_center_translate(
        self, dim_idx: int, video_n: int, scale: float
    ) -> float:
        """Return the translation that centers the video on the fUSI.

        The video's centre pixel along ``dim_idx`` (at index
        ``(video_n - 1) / 2``) is placed at the midpoint of the fUSI
        coordinate range, so the video overlays the scan in both
        spatial axes.
        """
        coords = self._lookup_coord(dim_idx)
        if coords is None or coords.size == 0:
            return 0.0

        center = (float(coords.min()) + float(coords.max())) / 2
        return center - scale * (video_n - 1) / 2

    # ------------------------------------------------------------------
    # Dimension order guard
    # ------------------------------------------------------------------

    def _on_dim_order_changed(self, event) -> None:
        """Handle dimension order changes.

        Three cases:

        1. Time is in the displayed dims -- fix the order (move time out).
        2. Displayed dims changed -- rebuild every video layer.
        3. Displayed dims unchanged (only slider reorder) -- no-op.
        """
        order = tuple(event.value)
        ndisplay = self._viewer.dims.ndisplay
        displayed = (order[-ndisplay], order[-1])

        time_dim = self._fusi_time_idx

        # Case 1: time in displayed dims -- fix the order.
        if time_dim in displayed:
            non_time = [d for d in order if d != time_dim]
            fixed_order = (time_dim,) + tuple(non_time)
            self._viewer.dims.events.order.disconnect(self._on_dim_order_changed)
            self._viewer.dims.order = fixed_order
            self._viewer.dims.events.order.connect(self._on_dim_order_changed)
            return

        # Case 3: displayed dims unchanged -- skip rebuild.
        if displayed == self._displayed_dims:
            return

        # Case 2: displayed dims changed -- rebuild every video.
        self._displayed_dims = displayed
        self._rebuild_all_entries()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _disconnect(self) -> None:
        """Disconnect the dim-order guard callback."""
        if self._guarding_order:
            try:
                self._viewer.dims.events.order.disconnect(self._on_dim_order_changed)
            except Exception:  # noqa: BLE001
                pass
            self._guarding_order = False
