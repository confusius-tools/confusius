"""Unit tests for VideoPanel pure-logic components.

Only tests code with real computation — no UI state, no napari event wiring.
See https://napari.org/dev/plugins/testing_and_publishing/test.html
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from confusius._napari._video_panel import VideoPanel, _LazyVideoArray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeVideo:
    """Minimal VideoReaderNP stub."""

    def __init__(self, n_frames=20, h=48, w=64, rgb=False):
        c = (3,) if rgb else ()
        self.shape = (n_frames, h, w, *c)
        self._frames = np.random.default_rng(0).integers(
            0, 255, size=self.shape, dtype=np.uint8
        )

    def __getitem__(self, idx):
        return self._frames[idx]


def _make_panel(sample_4d_volume, displayed=(2, 3)):
    """Build a VideoPanel with a mocked viewer (no Qt/GL)."""
    mock_viewer = MagicMock()
    mock_viewer.dims.displayed = displayed
    mock_viewer.dims.axis_labels = tuple(sample_4d_volume.dims)

    panel = VideoPanel.__new__(VideoPanel)
    panel._viewer = mock_viewer

    mock_layer = MagicMock()
    mock_layer.metadata = {"xarray": sample_4d_volume}
    panel._ref_layer = mock_layer

    return panel


# ---------------------------------------------------------------------------
# _LazyVideoArray
# ---------------------------------------------------------------------------


class TestLazyVideoArray:
    def test_integer_index(self):
        fv = _FakeVideo(n_frames=5, h=4, w=6)
        lazy = _LazyVideoArray(fv, np.uint8)
        np.testing.assert_array_equal(lazy[2], fv[2])

    def test_slice_index(self):
        fv = _FakeVideo(n_frames=10, h=4, w=6)
        lazy = _LazyVideoArray(fv, np.uint8)
        frames = lazy[2:5]
        assert frames.shape == (3, 4, 6)
        np.testing.assert_array_equal(frames[0], fv[2])

    def test_tuple_index_with_spatial_slice(self):
        fv = _FakeVideo(n_frames=5, h=8, w=10)
        lazy = _LazyVideoArray(fv, np.uint8)
        result = lazy[1, :4, :5]
        np.testing.assert_array_equal(result, fv[1][:4, :5])

    def test_shape_dtype_len(self):
        fv = _FakeVideo(n_frames=8, h=16, w=32)
        lazy = _LazyVideoArray(fv, np.float32)
        assert lazy.shape == (8, 16, 32)
        assert lazy.dtype == np.float32
        assert len(lazy) == 8
        assert lazy.ndim == 3

    def test_unsupported_index_raises(self):
        fv = _FakeVideo(n_frames=3, h=4, w=6)
        lazy = _LazyVideoArray(fv, np.uint8)
        with pytest.raises(IndexError, match="Unsupported time index type"):
            lazy[[0, 1]]


# ---------------------------------------------------------------------------
# _compute_spatial_params
# ---------------------------------------------------------------------------


class TestComputeSpatialParams:
    def test_returns_defaults_without_ref_layer(self, sample_4d_volume):
        panel = _make_panel(sample_4d_volume)
        panel._ref_layer = None
        assert panel._compute_spatial_params(48, 64) == (1.0, 0.0)

    def test_video_scaled_to_scan_height(self, sample_4d_volume):
        panel = _make_panel(sample_4d_volume)
        scale, ty = panel._compute_spatial_params(48, 64)

        y_coords = sample_4d_volume.coords["y"].values.astype(np.float64)

        # Scale matches fUSI height extent / video height.
        y_min, y_max = y_coords.min(), y_coords.max()
        y_step = float(np.median(np.diff(y_coords)))
        expected_scale = (y_max - y_min + abs(y_step)) / 48
        assert scale == pytest.approx(expected_scale)

        # Translate-y centres video on the scan.
        center_y = (y_min + y_max) / 2
        expected_ty = center_y - scale * (48 - 1) / 2
        assert ty == pytest.approx(expected_ty)

    def test_returns_defaults_when_coords_missing(self, sample_4d_volume):
        panel = _make_panel(sample_4d_volume)
        # Axis labels don't match any coordinate in the xarray.
        panel._viewer.dims.axis_labels = ("time", "z", "foo", "bar")
        assert panel._compute_spatial_params(48, 64) == (1.0, 0.0)


# ---------------------------------------------------------------------------
# _build_permutation
# ---------------------------------------------------------------------------


class TestBuildPermutation:
    """Test the permutation logic that maps video dims to display order.

    Base video array dims: [time(0), pad…(1..n_pad), H(n_pad+1), W(n_pad+2)].
    """

    def _make_minimal_panel(self, fusi_time_idx=0, n_pad=1):
        panel = VideoPanel.__new__(VideoPanel)
        panel._fusi_time_idx = fusi_time_idx
        panel._n_pad = n_pad
        return panel

    def test_identity_order(self):
        # 4D: order (0,1,2,3) → time at 0, pad at 1, H at 2, W at 3.
        panel = self._make_minimal_panel(fusi_time_idx=0, n_pad=1)
        perm = panel._build_permutation((0, 1, 2, 3))
        assert perm == (0, 1, 2, 3)

    def test_spatial_swap(self):
        # 4D: display axes swapped → H and W exchange positions.
        panel = self._make_minimal_panel(fusi_time_idx=0, n_pad=1)
        perm = panel._build_permutation((0, 1, 3, 2))
        # H(=2) goes to position 3, W(=3) goes to position 2.
        assert perm[3] == 2  # H at vertical
        assert perm[2] == 3  # W at horizontal

    def test_time_always_maps_to_fusi_time_idx(self):
        panel = self._make_minimal_panel(fusi_time_idx=0, n_pad=1)
        perm = panel._build_permutation((0, 1, 2, 3))
        assert perm[0] == 0  # time base dim at fusi_time_idx

    def test_permutation_is_valid(self):
        """Every permutation must use each base dim exactly once."""
        panel = self._make_minimal_panel(fusi_time_idx=0, n_pad=1)
        for order in [(0, 1, 2, 3), (0, 1, 3, 2)]:
            perm = panel._build_permutation(order)
            assert sorted(perm) == list(range(4))
