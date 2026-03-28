"""Video loading panel for the ConfUSIus napari plugin."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
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


class _LazyVideoArray:
    """Thin array-protocol wrapper around `VideoReaderNP`.

    `dask.array.from_array` requires *shape*, *dtype* and slice-based
    ``__getitem__``.  `VideoReaderNP` may only support integer frame indexing,
    so this adapter normalises slice and tuple indices.

    Parameters
    ----------
    video : VideoReaderNP
        The lazy video reader.
    dtype : numpy.dtype
        Frame data-type (probed from the first frame).
    """

    def __init__(self, video: VideoReaderNP, dtype: np.dtype) -> None:
        self._video = video
        self._lock = threading.Lock()
        self.shape = video.shape
        self.dtype = dtype
        self.ndim = len(self.shape)

    def __len__(self) -> int:  # noqa: D105
        return self.shape[0]

    def __getitem__(self, idx):  # noqa: D105
        if not isinstance(idx, tuple):
            idx = (idx,)

        time_idx = idx[0]
        rest = idx[1:] if len(idx) > 1 else ()

        # Serialize all reads — FFmpeg decoders are not thread-safe.
        with self._lock:
            # Integer → single frame.
            if isinstance(time_idx, (int, np.integer)):
                frame = np.ascontiguousarray(self._video[int(time_idx)])
                return frame[rest] if rest else frame

            # Slice → stack of frames.
            if isinstance(time_idx, slice):
                start, stop, step = time_idx.indices(self.shape[0])
                frames = np.stack(
                    [
                        np.ascontiguousarray(self._video[i])
                        for i in range(start, stop, step or 1)
                    ]
                )
                return frames[(slice(None),) + rest] if rest else frames

        raise IndexError(f"Unsupported time index type: {type(time_idx)}")


# ---------------------------------------------------------------------------
# Panel widget
# ---------------------------------------------------------------------------


class VideoPanel(QWidget):
    """Panel for loading a video as an image layer in the napari viewer.

    The video is added as a full ``(time, H, W)`` (or ``(time, H, W, C)`` for
    RGB) dask-backed image layer with lazy, per-frame loading.  The time axis
    uses seconds (``scale = 1 / fps``) so napari's built-in slider and play
    button keep both the fUSI scan and the video synchronised in physical time.

    The spatial dimensions use a single isotropic scale matching the fUSI scan
    height.  The viewer's grid mode is enabled so the video and fUSI scan are
    displayed side-by-side in separate viewports; grid mode is restored to its
    previous state when the video is unloaded.

    When the user reorders fUSI dimensions the video layer is dynamically
    transposed so that H and W always remain in the displayed positions.  The
    bounding box (scale / translate) is recomputed to match the new fUSI
    spatial extent.  A guard prevents the time axis from ever being displayed
    spatially.

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
        self._video_da_full: da.Array | None = None  # All frames, padded.
        self._video_da_base: da.Array | None = None  # After frame-step subsampling.
        self._base_scale: tuple[float, ...] = ()
        self._base_translate: tuple[float, ...] = ()
        self._video_h: int = 0
        self._video_w: int = 0
        self._n_pad: int = 0
        self._is_rgb: bool = False
        self._fusi_time_idx: int = 0
        self._fps: float = 0.0

        # Grid mode state (saved before enabling, restored on unload).
        self._grid_was_enabled: bool = False

        # Dimension guard.
        self._last_safe_order: tuple[int, ...] | None = None
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
        self._video_da_full = None
        self._video_da_base = None
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

        # Build a dask array (1 chunk per frame) for lazy loading.
        lazy = _LazyVideoArray(video, frame.dtype)
        video_da = da.from_array(lazy, chunks=(1,) + lazy.shape[1:])

        # Pad with singleton dims to match reference layer dimensionality.
        self._ref_layer = ref_layer
        ref_xr = ref_layer.metadata["xarray"]
        n_pad = max(ref_layer.ndim - 3, 0)
        for _ in range(n_pad):
            video_da = da.expand_dims(video_da, axis=1)

        # Store base state (before any transposition).
        self._video_da_full = video_da
        self._video_h = video_h
        self._video_w = video_w
        self._n_pad = n_pad
        self._is_rgb = is_rgb
        self._fps = fps
        self._fusi_time_idx = (
            list(ref_xr.dims).index("time") if "time" in ref_xr.dims else 0
        )

        # Apply frame step (subsample time axis).
        frame_step = self._step_spin.value()
        self._video_da_base = video_da[::frame_step]

        time_scale = frame_step / fps if fps > 0 else 1.0
        spatial_scale, ty = self._compute_spatial_params(video_h, video_w)
        self._base_scale = (
            (time_scale,) + (1.0,) * n_pad + (spatial_scale, spatial_scale)
        )
        self._base_translate = (0.0,) + (0.0,) * n_pad + (ty, 0.0)

        # Arrange for the current dim order.
        current_order = tuple(self._viewer.dims.order)
        data, scale, translate = self._arrange_for_order(current_order)

        # Block layer-list signals while adding so _refresh_layer_combo
        # does not see the new layer before _video_layer is assigned.
        self._viewer.layers.events.inserted.disconnect(self._refresh_layer_combo)
        self._viewer.layers.events.removed.disconnect(self._on_layer_removed)

        self._video_layer = self._viewer.add_image(
            data,
            name=f"Video: {path.stem}",
            rgb=is_rgb,
            scale=scale,
            translate=translate,
        )

        self._viewer.layers.events.inserted.connect(self._refresh_layer_combo)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)

        # Enable grid mode for side-by-side display.
        self._grid_was_enabled = self._viewer.grid.enabled
        self._viewer.grid.enabled = True

        # Connect dim-order callback.
        self._last_safe_order = current_order
        self._viewer.dims.events.order.connect(self._on_dim_order_changed)
        self._guarding_order = True

        # Enable playback controls and show effective FPS.
        self._step_spin.setEnabled(True)
        self._update_fps_label()

        n_shown = self._video_da_base.shape[0]
        show_info(
            f"Video '{path.stem}' loaded ({n_frames} frames, {fps:.1f} fps, "
            f"showing {n_shown} frames at step {frame_step})."
        )

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
        """Rebuild the video layer with a new frame step."""
        if self._video_da_full is None or self._video_layer is None:
            return

        self._video_da_base = self._video_da_full[::value]

        # Update time scale for the new step.
        time_scale = value / self._fps if self._fps > 0 else 1.0
        base_s = list(self._base_scale)
        base_s[0] = time_scale
        self._base_scale = tuple(base_s)

        # Rebuild the layer.
        current_order = self._last_safe_order or tuple(self._viewer.dims.order)
        data, scale, translate = self._arrange_for_order(current_order)

        name = self._video_layer.name
        self._viewer.layers.events.inserted.disconnect(self._refresh_layer_combo)
        self._viewer.layers.events.removed.disconnect(self._on_layer_removed)
        try:
            self._viewer.layers.remove(self._video_layer)
        except ValueError:
            pass

        self._video_layer = self._viewer.add_image(
            data,
            name=name,
            rgb=self._is_rgb,
            scale=scale,
            translate=translate,
        )
        self._viewer.layers.events.inserted.connect(self._refresh_layer_combo)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)

        self._update_fps_label()
        n_shown = self._video_da_base.shape[0]
        show_info(
            f"Frame step {value}: showing {n_shown} frames ({self._fps / value:.1f} fps effective)."
        )

    # ------------------------------------------------------------------
    # Spatial transform
    # ------------------------------------------------------------------

    def _compute_spatial_params(
        self, video_h: int, video_w: int
    ) -> tuple[float, float]:
        """Return ``(isotropic_scale, translate_y)``.

        Uses the selected reference layer to determine the spatial bounding
        box.  The video is scaled isotropically to match the reference height
        and centred vertically on the scan.
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

        displayed = self._viewer.dims.displayed
        labels = self._viewer.dims.axis_labels
        if len(displayed) < 2 or len(labels) <= max(displayed):
            return 1.0, 0.0

        y_dim = str(labels[displayed[-2]])
        x_dim = str(labels[displayed[-1]])

        if y_dim not in xr_da.coords or x_dim not in xr_da.coords:
            return 1.0, 0.0

        y_coords = np.asarray(xr_da.coords[y_dim], dtype=np.float64)

        y_min, y_max = float(y_coords.min()), float(y_coords.max())

        y_step = (
            float(np.median(np.diff(y_coords)))
            if len(y_coords) > 1
            else float(xr_da.coords[y_dim].attrs.get("voxdim", 1.0))
        )

        fusi_extent_y = (y_max - y_min) + abs(y_step)
        scale = fusi_extent_y / video_h

        center_y = (y_min + y_max) / 2
        translate_y = center_y - scale * (video_h - 1) / 2

        return scale, translate_y

    # ------------------------------------------------------------------
    # Dynamic transposition to keep H, W always displayed
    # ------------------------------------------------------------------

    def _build_permutation(self, order: tuple[int, ...]) -> tuple[int, ...]:
        """Map base video dims so H→vertical and W→horizontal.

        The base dask array has dims ``[time, pad…, H, W (, rgb)]``.  This
        method returns a permutation *p* (for the non-RGB dims) such that
        ``base.transpose(p)`` places:

        * time  at ``self._fusi_time_idx``
        * H     at ``order[-2]``  (vertical on canvas)
        * W     at ``order[-1]``  (horizontal on canvas)
        * pads  everywhere else
        """
        napari_ndim = len(order)
        displayed_v = order[-2]
        displayed_h = order[-1]

        perm: list[int | None] = [None] * napari_ndim
        perm[self._fusi_time_idx] = 0  # time
        perm[displayed_v] = self._n_pad + 1  # H
        perm[displayed_h] = self._n_pad + 2  # W

        pad_iter = iter(range(1, self._n_pad + 1))
        for i in range(napari_ndim):
            if perm[i] is None:
                perm[i] = next(pad_iter)

        return tuple(perm)  # type: ignore[arg-type]

    def _arrange_for_order(
        self, order: tuple[int, ...]
    ) -> tuple[da.Array, tuple[float, ...], tuple[float, ...]]:
        """Transpose the base array and compute scale / translate for *order*.

        Also recomputes the spatial scale so the video bounding box matches
        whichever fUSI dimensions are currently displayed.
        """
        perm = self._build_permutation(order)
        napari_ndim = len(perm)

        # Recompute spatial params for the (possibly new) displayed dims.
        spatial_scale, ty = self._compute_spatial_params(
            self._video_h, self._video_w
        )

        # Refresh the H / W entries in the base vectors.
        base_s = list(self._base_scale)
        base_t = list(self._base_translate)
        base_s[self._n_pad + 1] = spatial_scale
        base_s[self._n_pad + 2] = spatial_scale
        base_t[self._n_pad + 1] = ty
        base_t[self._n_pad + 2] = 0.0
        self._base_scale = tuple(base_s)
        self._base_translate = tuple(base_t)

        # Permute scale / translate.
        new_scale = tuple(base_s[perm[i]] for i in range(napari_ndim))
        new_translate = tuple(base_t[perm[i]] for i in range(napari_ndim))

        # Transpose the dask array (RGB dim stays last).
        full_perm = perm + (napari_ndim,) if self._is_rgb else perm
        data = self._video_da_base.transpose(full_perm)

        return data, new_scale, new_translate

    # ------------------------------------------------------------------
    # Dimension order callback
    # ------------------------------------------------------------------

    def _on_dim_order_changed(self, event) -> None:
        """React to a viewer dimension reorder.

        Time is pinned to position 0 (first slider).  If the user's reorder
        moved time elsewhere, we fix the order by extracting time, letting
        the remaining (spatial) dims keep their new arrangement, and
        re-inserting time at the front.  The video layer is then transposed
        so H / W stay in the displayed positions.
        """
        order = list(event.value)
        time_dim = self._fusi_time_idx

        # Build a corrected order: time always first, spatial dims keep
        # whatever relative order the user chose.
        spatial = [d for d in order if d != time_dim]
        fixed_order = (time_dim, *spatial)

        if fixed_order != tuple(order):
            # Apply the corrected order without re-entering this callback.
            self._viewer.dims.events.order.disconnect(self._on_dim_order_changed)
            self._viewer.dims.order = fixed_order
            self._viewer.dims.events.order.connect(self._on_dim_order_changed)

        self._last_safe_order = fixed_order

        if self._video_layer is None or self._video_da_base is None:
            return

        data, scale, translate = self._arrange_for_order(fixed_order)

        # Remove and re-add the layer instead of hot-swapping data.
        # Reassigning layer.data while napari is still slicing the old
        # dask array can trigger concurrent FFmpeg reads, crashing the
        # decoder.  A clean remove → add avoids the race.
        # Block layer-list signals to avoid redundant _refresh_layer_combo calls.
        name = self._video_layer.name
        self._viewer.layers.events.inserted.disconnect(self._refresh_layer_combo)
        self._viewer.layers.events.removed.disconnect(self._on_layer_removed)
        try:
            self._viewer.layers.remove(self._video_layer)
        except ValueError:
            pass

        self._video_layer = self._viewer.add_image(
            data,
            name=name,
            rgb=self._is_rgb,
            scale=scale,
            translate=translate,
        )
        self._viewer.layers.events.inserted.connect(self._refresh_layer_combo)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)

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
