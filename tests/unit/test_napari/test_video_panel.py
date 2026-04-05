"""Unit tests for VideoPanel pure-logic components.

Only tests code with real computation -- no UI state, no napari event wiring.
See https://napari.org/dev/plugins/testing_and_publishing/test.html
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from confusius._napari._video_panel import VideoPanel, _VideoArray

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeVideo:
    """Minimal VideoReaderNP stub."""

    def __init__(self, n_frames=20, h=48, w=64, rgb=False):
        c = (3,) if rgb else ()
        self.shape = (n_frames, h, w, *c)
        self.number_of_frames = n_frames
        self.frame_rate = 30.0
        self.frame_shape = (h, w, 3) if rgb else (h, w)
        self._frames = np.random.default_rng(0).integers(
            0, 255, size=self.shape, dtype=np.uint8
        )

    def __getitem__(self, idx):
        return self._frames[idx]


def _make_panel(sample_4d_volume):
    """Build a VideoPanel with a mocked viewer (no Qt/GL)."""
    mock_viewer = MagicMock()
    mock_viewer.dims.axis_labels = tuple(sample_4d_volume.dims)

    panel = VideoPanel.__new__(VideoPanel)
    panel._viewer = mock_viewer
    panel._axis_labels = tuple(sample_4d_volume.dims)
    panel._video_h = 48
    panel._displayed_dims = (2, 3)

    mock_layer = MagicMock()
    mock_layer.metadata = {"xarray": sample_4d_volume}
    panel._ref_layer = mock_layer

    return panel


def _make_loaded_panel(sample_4d_volume, initial_order=(0, 1, 2, 3)):
    """Build a VideoPanel with a mocked viewer and a 'loaded' video."""
    mock_viewer = MagicMock()
    mock_viewer.dims.ndisplay = 2
    mock_viewer.dims.order = initial_order
    mock_viewer.dims.axis_labels = tuple(sample_4d_volume.dims)

    panel = VideoPanel.__new__(VideoPanel)
    panel._viewer = mock_viewer

    mock_ref = MagicMock()
    mock_ref.metadata = {"xarray": sample_4d_volume}
    mock_ref.ndim = sample_4d_volume.ndim
    panel._ref_layer = mock_ref

    fv = _FakeVideo(n_frames=10, h=48, w=64, rgb=True)
    panel._video = fv
    panel._frame_dtype = np.uint8
    panel._frame_shape = (48, 64, 3)
    panel._video_h = 48
    panel._video_w = 64
    panel._is_rgb = True
    panel._fps = 30.0
    panel._n_pad = 1
    panel._fusi_time_idx = 0
    panel._axis_labels = tuple(sample_4d_volume.dims)
    panel._units = ["s", "m", "m", "m"]
    panel._displayed_dims = (initial_order[-2], initial_order[-1])
    panel._video_name = "Video: test"
    panel._guarding_order = True

    panel._step_spin = MagicMock()
    panel._step_spin.value.return_value = 1

    panel._video_layer = MagicMock()
    panel._video_layer.name = "Video: test"

    return panel


# ---------------------------------------------------------------------------
# _VideoArray
# ---------------------------------------------------------------------------


class TestVideoArray:
    def test_integer_index(self):
        fv = _FakeVideo(n_frames=5, h=4, w=6)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6))
        np.testing.assert_array_equal(arr[2], fv[2])

    def test_slice_index(self):
        fv = _FakeVideo(n_frames=10, h=4, w=6)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6))
        frames = arr[2:5]
        assert frames.shape == (3, 4, 6)
        np.testing.assert_array_equal(frames[0], fv[2])

    def test_tuple_index_with_spatial_slice(self):
        fv = _FakeVideo(n_frames=5, h=8, w=10)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(8, 10))
        result = arr[1, :4, :5]
        np.testing.assert_array_equal(result, fv[1][:4, :5])

    def test_shape_dtype_len(self):
        fv = _FakeVideo(n_frames=8, h=16, w=32)
        arr = _VideoArray(fv, dtype=np.float32, frame_shape=(16, 32))
        assert arr.shape == (8, 16, 32)
        assert arr.dtype == np.float32
        assert len(arr) == 8
        assert arr.ndim == 3

    def test_unsupported_index_raises(self):
        fv = _FakeVideo(n_frames=3, h=4, w=6)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6))
        with pytest.raises(IndexError, match="Unsupported time index type"):
            arr[[0, 1]]

    def test_dimension_padding(self):
        fv = _FakeVideo(n_frames=5, h=4, w=6)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6), n_pad=1)
        assert arr.shape == (5, 1, 4, 6)
        # Integer index on pad dim should return the frame.
        np.testing.assert_array_equal(arr[2, 0], fv[2])

    def test_rgb_shape(self):
        fv = _FakeVideo(n_frames=5, h=4, w=6, rgb=True)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6, 3))
        assert arr.shape == (5, 4, 6, 3)
        np.testing.assert_array_equal(arr[1], fv[1])

    def test_rgb_with_padding(self):
        fv = _FakeVideo(n_frames=5, h=4, w=6, rgb=True)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6, 3), n_pad=1)
        assert arr.shape == (5, 1, 4, 6, 3)
        np.testing.assert_array_equal(arr[2, 0], fv[2])

    def test_empty_slice(self):
        fv = _FakeVideo(n_frames=5, h=4, w=6)
        arr = _VideoArray(fv, dtype=np.uint8, frame_shape=(4, 6))
        result = arr[3:3]
        assert result.shape == (0, 4, 6)

    def test_explicit_h_before_w(self):
        """VideoArray with H at dim 1, W at dim 3 (e.g. displaying z x)."""
        fv = _FakeVideo(n_frames=5, h=4, w=6)
        arr = _VideoArray(
            fv,
            dtype=np.uint8,
            frame_shape=(4, 6),
            n_pad=1,
            time_dim=0,
            h_dim=1,
            w_dim=3,
        )
        assert arr.shape == (5, 4, 1, 6)
        result = arr[0, :, 0, :]
        assert result.shape == (4, 6)
        np.testing.assert_array_equal(result, fv[0])

    def test_explicit_w_before_h(self):
        """VideoArray with W at dim 1, H at dim 3 (e.g. displaying x z)."""
        fv = _FakeVideo(n_frames=5, h=4, w=6)
        arr = _VideoArray(
            fv,
            dtype=np.uint8,
            frame_shape=(4, 6),
            n_pad=1,
            time_dim=0,
            h_dim=3,
            w_dim=1,
        )
        assert arr.shape == (5, 6, 1, 4)
        result = arr[0, :, 0, :]
        assert result.shape == (6, 4)
        np.testing.assert_array_equal(result, fv[0].T)

    def test_explicit_dims_rgb_with_transpose(self):
        """RGB video with W before H in layout."""
        fv = _FakeVideo(n_frames=5, h=4, w=6, rgb=True)
        arr = _VideoArray(
            fv,
            dtype=np.uint8,
            frame_shape=(4, 6, 3),
            n_pad=1,
            time_dim=0,
            h_dim=3,
            w_dim=1,
        )
        assert arr.shape == (5, 6, 1, 4, 3)
        result = arr[0, :, 0, :]
        assert result.shape == (6, 4, 3)
        np.testing.assert_array_equal(result, np.swapaxes(fv[0], 0, 1))


# ---------------------------------------------------------------------------
# _compute_spatial_params
# ---------------------------------------------------------------------------


class TestComputeSpatialParams:
    def test_returns_defaults_without_ref_layer(self, sample_4d_volume):
        panel = _make_panel(sample_4d_volume)
        panel._ref_layer = None
        assert panel._compute_spatial_params(2) == (1.0, 0.0)

    def test_video_scaled_to_scan_height(self, sample_4d_volume):
        panel = _make_panel(sample_4d_volume)
        # vertical_dim=2 -> "y" coord.
        scale, ty = panel._compute_spatial_params(2)

        y_coords = sample_4d_volume.coords["y"].values.astype(np.float64)
        y_min, y_max = y_coords.min(), y_coords.max()
        y_step = float(np.median(np.diff(y_coords)))
        expected_scale = (y_max - y_min + abs(y_step)) / 48
        assert scale == pytest.approx(expected_scale)

        center_y = (y_min + y_max) / 2
        expected_ty = center_y - scale * (48 - 1) / 2
        assert ty == pytest.approx(expected_ty)

    def test_returns_defaults_when_coords_missing(self, sample_4d_volume):
        panel = _make_panel(sample_4d_volume)
        panel._axis_labels = ("time", "z", "foo", "bar")
        assert panel._compute_spatial_params(2) == (1.0, 0.0)

    def test_spatial_params_z_vertical(self, sample_4d_volume):
        """When z is the vertical axis, scale uses z-coordinates."""
        panel = _make_panel(sample_4d_volume)
        scale, tz = panel._compute_spatial_params(1)  # dim 1 = "z".

        z_coords = sample_4d_volume.coords["z"].values.astype(np.float64)
        z_min, z_max = z_coords.min(), z_coords.max()
        z_step = float(np.median(np.diff(z_coords)))
        expected_scale = (z_max - z_min + abs(z_step)) / 48
        assert scale == pytest.approx(expected_scale)


# ---------------------------------------------------------------------------
# Dimension reorder integration
# ---------------------------------------------------------------------------


class TestDimReorder:
    def test_rebuild_on_displayed_dims_change(self, sample_4d_volume):
        """Changing displayed dims triggers a layer rebuild."""
        panel = _make_loaded_panel(sample_4d_volume, initial_order=(0, 1, 2, 3))
        old_layer = panel._video_layer

        # Simulate order change: display z x instead of y x.
        event = MagicMock()
        event.value = (0, 2, 3, 1)  # displayed: dims 3, 1.
        panel._on_dim_order_changed(event)

        # Old layer should have been removed, new one added.
        panel._viewer.layers.remove.assert_called_with(old_layer)
        panel._viewer.add_image.assert_called_once()
        # New displayed dims should be updated.
        assert panel._displayed_dims == (3, 1)

    def test_no_rebuild_when_displayed_dims_unchanged(self, sample_4d_volume):
        """Reordering sliders without changing displayed dims skips rebuild."""
        panel = _make_loaded_panel(sample_4d_volume, initial_order=(0, 1, 2, 3))

        event = MagicMock()
        event.value = (1, 0, 2, 3)  # same displayed: dims 2, 3.
        panel._on_dim_order_changed(event)

        panel._viewer.layers.remove.assert_not_called()
        panel._viewer.add_image.assert_not_called()

    def test_time_in_display_gets_corrected(self, sample_4d_volume):
        """If time ends up displayed, the order is fixed."""
        panel = _make_loaded_panel(sample_4d_volume, initial_order=(0, 1, 2, 3))

        event = MagicMock()
        event.value = (1, 2, 3, 0)  # time (dim 0) in displayed position.

        panel._on_dim_order_changed(event)

        # Should have disconnected, fixed order, reconnected.
        panel._viewer.dims.events.order.disconnect.assert_called()
        panel._viewer.dims.events.order.connect.assert_called()
        # Time should be moved to front of the corrected order.
        assert panel._viewer.dims.order[0] == 0

    def test_rebuild_creates_correct_video_array_shape(self, sample_4d_volume):
        """After rebuild for z x display, VideoArray has H at dim 3, W at dim 1."""
        panel = _make_loaded_panel(sample_4d_volume, initial_order=(0, 1, 2, 3))

        event = MagicMock()
        event.value = (0, 2, 3, 1)  # displayed: vertical=3, horizontal=1.
        panel._on_dim_order_changed(event)

        # Check the data passed to add_image.
        call_args = panel._viewer.add_image.call_args
        data = call_args[0][0] if call_args[0] else call_args[1]["data"]
        # Shape should be (10, 64, 1, 48, 3): time, W, pad, H, C.
        assert data.shape == (10, 64, 1, 48, 3)
