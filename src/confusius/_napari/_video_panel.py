"""Video loading panel for the ConfUSIus napari plugin."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from napari.utils.notifications import show_error, show_info
from napari_video.napari_video import VideoReaderNP
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari


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
        time_dim: int = 0,
        h_dim: int | None = None,
        w_dim: int | None = None,
    ) -> None:
        self._video = video
        self._n_pad = n_pad

        n_frames = video.number_of_frames
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
            frame = np.ascontiguousarray(self._video[t])
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
            start, stop, step = time_idx.indices(n_time)
            frames = [_fetch(t) for t in range(start, stop, step or 1)]
            if not frames:
                return np.empty((0,) + _fetch(0).shape, dtype=self.dtype)
            return np.stack(frames)

        raise IndexError(f"Unsupported time index type: {type(time_idx)}")


# ---------------------------------------------------------------------------
# Panel widget
# ---------------------------------------------------------------------------


class VideoPanel(QWidget):
    """Panel for loading a video as an image layer in the napari viewer.

    The video is passed directly to napari as a lazy, array-like object
    backed by ``VideoReaderNP`` (OpenCV frame-on-demand decoding).  A thin
    wrapper (`_VideoArray`) handles singleton dimension padding and
    dimension reordering.

    When the viewer's displayed dimensions change, the video layer is
    removed and re-added with a new `_VideoArray` whose shape places H
    and W at the currently displayed dim positions.

    The video layer receives the same ``axis_labels`` as the fUSI scan so
    that napari handles dimension reordering identically for both layers.
    The time scale is set to ``frame_step / fps`` so that napari's slider
    shows physical seconds.

    The spatial dimensions use a single isotropic scale matching the fUSI
    scan height.  The viewer's grid mode is enabled so the video and fUSI
    scan are displayed side-by-side in separate viewports; grid mode is
    restored to its previous state when the video is unloaded.

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

        # Video state.
        self._ref_layer = None
        self._video_layer = None
        self._video: VideoReaderNP | None = None
        self._frame_dtype: np.dtype | None = None
        self._frame_shape: tuple[int, ...] = ()
        self._video_h: int = 0
        self._video_w: int = 0
        self._n_pad: int = 0
        self._is_rgb: bool = False
        self._fusi_time_idx: int = 0
        self._fps: float = 0.0
        self._axis_labels: tuple[str, ...] = ()
        self._units: list[str | None] = []
        # Dim indices of the currently displayed vertical and horizontal axes.
        self._displayed_dims: tuple[int, int] = (0, 0)
        # Name of the video layer (for identification during rebuild).
        self._video_name: str = ""

        # Grid mode state (saved before enabling, restored on unload).
        self._grid_was_enabled: bool = False

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
        ref_group = QGroupBox("Reference layer")
        ref_layout = QVBoxLayout(ref_group)
        ref_layout.setSpacing(4)
        self._ref_combo = QComboBox()
        self._ref_combo.setPlaceholderText("No layers loaded")
        self._ref_combo.currentIndexChanged.connect(self._on_ref_changed)
        ref_layout.addWidget(self._ref_combo)
        layout.addWidget(ref_group)

        # Video file selector.
        file_group = QGroupBox("Video file")
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

        self._load_btn = QPushButton("Load video")
        self._load_btn.setObjectName("primary_btn")
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._load_from_path)
        layout.addWidget(self._load_btn)

        # Playback controls (enabled after loading).
        playback_group = QGroupBox("Playback")
        playback_layout = QVBoxLayout(playback_group)
        playback_layout.setSpacing(4)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Frame step:"))
        self._step_spin = QSpinBox()
        self._step_spin.setRange(1, 100)
        self._step_spin.setValue(1)
        self._step_spin.setMaximumWidth(60)
        self._step_spin.setToolTip(
            "Show every Nth frame. Higher values skip frames for lighter playback."
        )
        self._step_spin.setEnabled(False)
        self._step_spin.valueChanged.connect(self._on_frame_step_changed)
        step_row.addWidget(self._step_spin)
        playback_layout.addLayout(step_row)

        self._fps_label = QLabel("")
        playback_layout.addWidget(self._fps_label)

        layout.addWidget(playback_group)

        layout.addStretch()
        self._refresh_layer_combo()

    # ------------------------------------------------------------------
    # Reference layer helpers
    # ------------------------------------------------------------------

    def _on_layer_removed(self, event=None) -> None:
        """Handle layer removal: clear stale ref and refresh combo."""
        if self._ref_layer is not None and self._ref_layer not in self._viewer.layers:
            self._ref_layer = None
        self._refresh_layer_combo()

    def _refresh_layer_combo(self, event=None) -> None:
        """Repopulate the reference layer combo from current viewer layers."""
        current = self._ref_combo.currentText()
        self._ref_combo.clear()
        for layer in self._viewer.layers:
            if layer is self._video_layer:
                continue
            self._ref_combo.addItem(layer.name)
        idx = self._ref_combo.findText(current)
        if idx >= 0:
            self._ref_combo.setCurrentIndex(idx)
        self._on_ref_changed()

    def _on_ref_changed(self) -> None:
        """Enable/disable load button based on reference selection."""
        has_ref = self._ref_combo.currentIndex() >= 0
        has_path = bool(self._path_edit.text().strip())
        self._load_btn.setEnabled(has_ref and has_path)

    def _get_ref_layer(self):
        """Return the currently selected layer, or None."""
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
        """Validate inputs and call ``_load``."""
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
        self._load(Path(path_str), ref)

    def _unload(self) -> None:
        """Remove the video layer and restore grid mode."""
        self._disconnect()
        if self._video_layer is not None:
            try:
                self._viewer.layers.remove(self._video_layer)
            except ValueError:
                pass
            self._video_layer = None
        # Restore grid mode to its previous state.
        self._viewer.grid.enabled = self._grid_was_enabled
        self._video = None
        self._ref_layer = None
        self._step_spin.setEnabled(False)
        self._fps_label.setText("")

    def _load(self, path: Path, ref_layer) -> None:
        if not path.is_file():
            show_error(f"File not found: {path}")
            return

        try:
            video = VideoReaderNP(str(path), remove_leading_singleton=False)
        except Exception as exc:  # noqa: BLE001
            show_error(f"Cannot open video: {exc}")
            return

        self._unload()

        # Probe a single frame for shape / dtype / channel info.
        frame = np.ascontiguousarray(video[0])
        is_rgb = frame.ndim == 3 and frame.shape[2] in (3, 4)
        video_h, video_w = frame.shape[0], frame.shape[1]
        n_frames = video.number_of_frames
        fps = video.frame_rate

        # Store state.
        self._ref_layer = ref_layer
        self._video = video
        self._frame_dtype = frame.dtype
        self._frame_shape = frame.shape
        self._video_h = video_h
        self._video_w = video_w
        self._is_rgb = is_rgb
        self._fps = fps

        ref_xr = ref_layer.metadata["xarray"]
        self._n_pad = max(ref_layer.ndim - 3, 0)
        self._fusi_time_idx = (
            list(ref_xr.dims).index("time") if "time" in ref_xr.dims else 0
        )
        self._axis_labels = tuple(ref_xr.dims)
        self._units = list(getattr(ref_layer, "units", [None] * ref_layer.ndim))
        self._video_name = f"Video: {path.stem}"

        # Read current displayed dims from the viewer.
        order = self._viewer.dims.order
        self._displayed_dims = (order[-2], order[-1])

        # Build and add the video layer.
        self._rebuild_video_layer()

        # Enable grid mode for side-by-side display.
        self._grid_was_enabled = self._viewer.grid.enabled
        self._viewer.grid.enabled = True

        # Connect dim-order guard.
        self._viewer.dims.events.order.connect(self._on_dim_order_changed)
        self._guarding_order = True

        # Set frame step range on the time axis.
        frame_step = self._step_spin.value()
        time_axis = self._fusi_time_idx
        self._viewer.dims.set_range(time_axis, (0, n_frames - 1, frame_step))

        # Enable playback controls and show effective FPS.
        self._step_spin.setEnabled(True)
        self._update_fps_label()

        n_shown = len(range(0, n_frames, frame_step))
        show_info(
            f"Video '{path.stem}' loaded ({n_frames} frames, {fps:.1f} fps, "
            f"showing {n_shown} frames at step {frame_step})."
        )

    # ------------------------------------------------------------------
    # Layer rebuild
    # ------------------------------------------------------------------

    def _rebuild_video_layer(self) -> None:
        """Remove old video layer and add a new one for current displayed dims.

        Creates a `_VideoArray` whose H and W are placed at the currently
        displayed vertical and horizontal dim positions.  Scale and
        translate are computed per-dim so the video aligns with the fUSI
        scan.
        """
        displayed_v, displayed_h = self._displayed_dims

        # Build the array wrapper with explicit dim positions.
        data = _VideoArray(
            self._video,
            dtype=self._frame_dtype,
            frame_shape=self._frame_shape,
            n_pad=self._n_pad,
            time_dim=self._fusi_time_idx,
            h_dim=displayed_v,
            w_dim=displayed_h,
        )

        # Compute per-dim scale and translate.
        frame_step = self._step_spin.value()
        time_scale = frame_step / self._fps if self._fps > 0 else 1.0
        spatial_scale, ty = self._compute_spatial_params(displayed_v)

        ndim = len(self._axis_labels)
        scale = [1.0] * ndim
        translate = [0.0] * ndim
        scale[self._fusi_time_idx] = time_scale
        scale[displayed_v] = spatial_scale
        scale[displayed_h] = spatial_scale
        translate[displayed_v] = ty

        # Remove old layer before adding new one.
        if self._video_layer is not None:
            try:
                self._viewer.layers.remove(self._video_layer)
            except ValueError:
                pass

        # Add hidden so the initial refresh doesn't run before dims settle.
        self._viewer.layers.events.inserted.disconnect(self._refresh_layer_combo)
        self._viewer.layers.events.removed.disconnect(self._on_layer_removed)

        self._video_layer = self._viewer.add_image(
            data,
            name=self._video_name,
            rgb=self._is_rgb,
            scale=tuple(scale),
            translate=tuple(translate),
            axis_labels=self._axis_labels,
            metadata={"fps": self._fps, "time_units": "s"},
            units=self._units,
            visible=False,
        )

        self._viewer.layers.events.inserted.connect(self._refresh_layer_combo)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)

        self._video_layer.visible = True

    # ------------------------------------------------------------------
    # Frame step
    # ------------------------------------------------------------------

    def _update_fps_label(self) -> None:
        """Update the effective-FPS label from current state."""
        step = self._step_spin.value()
        if self._fps > 0:
            effective = self._fps / step
            self._fps_label.setText(f"{effective:.1f} fps effective")
        else:
            self._fps_label.setText("")

    def _on_frame_step_changed(self, value: int) -> None:
        """Update the time-axis range for the new frame step.

        Instead of rebuilding the video layer, this simply adjusts the
        napari dims range so the slider steps through every *value*-th
        frame.  The time scale on the layer is updated in-place.
        """
        if self._video is None or self._video_layer is None:
            return

        n_frames = self._video.number_of_frames
        time_axis = self._fusi_time_idx

        # Update the dims slider range/step.
        self._viewer.dims.set_range(time_axis, (0, n_frames - 1, value))

        # Update the time scale in-place on the layer.
        time_scale = value / self._fps if self._fps > 0 else 1.0
        new_scale = list(self._video_layer.scale)
        new_scale[self._fusi_time_idx] = time_scale
        self._video_layer.scale = new_scale

        self._update_fps_label()
        n_shown = len(range(0, n_frames, value))
        show_info(
            f"Frame step {value}: showing {n_shown} frames "
            f"({self._fps / value:.1f} fps effective)."
        )

    # ------------------------------------------------------------------
    # Spatial transform
    # ------------------------------------------------------------------

    def _compute_spatial_params(self, vertical_dim: int) -> tuple[float, float]:
        """Return ``(isotropic_scale, translate_y)``.

        Uses the axis label at *vertical_dim* to look up the corresponding
        coordinate in the reference xarray.  The video is scaled
        isotropically to match that extent and centred on the scan.

        Parameters
        ----------
        vertical_dim : int
            The viewer dim index whose coordinate extent the video height
            should match.

        Returns
        -------
        scale : float
            Isotropic scale factor.
        translate : float
            Translation along the vertical axis to centre the video.
        """
        ref = self._ref_layer
        if ref is None:
            return 1.0, 0.0

        try:
            xr_da = ref.metadata.get("xarray")
        except RuntimeError:
            # Layer's C++ wrapper was deleted.
            self._ref_layer = None
            return 1.0, 0.0
        if xr_da is None:
            return 1.0, 0.0

        # Look up the coordinate name from the axis label at vertical_dim.
        y_dim = self._axis_labels[vertical_dim]

        if y_dim not in xr_da.coords:
            return 1.0, 0.0

        y_coords = np.asarray(xr_da.coords[y_dim], dtype=np.float64)

        y_min, y_max = float(y_coords.min()), float(y_coords.max())

        y_step = (
            float(np.median(np.diff(y_coords)))
            if len(y_coords) > 1
            else float(xr_da.coords[y_dim].attrs.get("voxdim", 1.0))
        )

        video_h = self._video_h
        fusi_extent_y = (y_max - y_min) + abs(y_step)
        scale = fusi_extent_y / video_h

        center_y = (y_min + y_max) / 2
        translate_y = center_y - scale * (video_h - 1) / 2

        return scale, translate_y

    # ------------------------------------------------------------------
    # Dimension order guard
    # ------------------------------------------------------------------

    def _on_dim_order_changed(self, event) -> None:
        """Handle dimension order changes.

        Three cases:

        1. Time is in the displayed dims -- fix the order (move time out).
        2. Displayed dims changed -- rebuild the video layer.
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

        # Case 2: displayed dims changed -- rebuild.
        self._displayed_dims = displayed
        self._rebuild_video_layer()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _disconnect(self) -> None:
        """Disconnect callbacks."""
        if self._guarding_order:
            try:
                self._viewer.dims.events.order.disconnect(self._on_dim_order_changed)
            except Exception:  # noqa: BLE001
                pass
            self._guarding_order = False
