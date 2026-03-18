"""Unit tests for the TimeSeriesPlotter widget.

Pure-logic methods (_time_dim_index, _extract_time_series) are tested with
lightweight mock layers.  Methods that touch the napari viewer (_active_layer,
_on_mouse_move) use the make_napari_viewer fixture.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Layer:
    """Minimal layer stub for tests that do not need a live napari viewer."""

    def __init__(self, data, metadata=None, type_string="image", rgb=False):
        self.data = np.asarray(data)
        self.metadata = metadata or {}
        self._type_string = type_string
        self.rgb = rgb
        self.name = "mock"

    def world_to_data(self, pos):
        # Identity mapping: world == data coordinates in these tests.
        return pos


class _FakeEventShift:
    modifiers = {"Shift"}


class _FakeEventNoShift:
    modifiers = set()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture
def plotter(viewer):
    from confusius._napari._time_series_plotter import TimeSeriesPlotter

    return TimeSeriesPlotter(viewer)


# ---------------------------------------------------------------------------
# _time_dim_index
# ---------------------------------------------------------------------------


class TestTimeDimIndex:
    def test_returns_zero_without_xarray_metadata(self, plotter):
        layer = _Layer(np.zeros((10, 4, 6, 8)))
        assert plotter._time_dim_index(layer) == 0

    def test_returns_zero_when_no_time_dim_in_metadata(self, plotter, sample_3d_volume):
        layer = _Layer(np.zeros((4, 6, 8)), metadata={"xarray": sample_3d_volume})
        assert plotter._time_dim_index(layer) == 0

    def test_returns_zero_when_time_is_first_dim(self, plotter, sample_4d_volume):
        layer = _Layer(np.zeros((10, 4, 6, 8)), metadata={"xarray": sample_4d_volume})
        assert plotter._time_dim_index(layer) == 0

    def test_returns_correct_index_when_time_is_not_first(self, plotter):
        da = xr.DataArray(np.zeros((4, 10, 6, 8)), dims=["z", "time", "y", "x"])
        layer = _Layer(np.zeros((4, 10, 6, 8)), metadata={"xarray": da})
        assert plotter._time_dim_index(layer) == 1


# ---------------------------------------------------------------------------
# _extract_time_series
# ---------------------------------------------------------------------------


class TestExtractTimeSeries:
    def test_extracts_correct_time_series(self, plotter, rng):
        data = rng.random((10, 4, 6, 8))
        layer = _Layer(data)
        ts = plotter._extract_time_series(layer, np.array([0, 1, 2, 3]))
        npt.assert_array_equal(ts, data[:, 1, 2, 3])

    def test_returns_none_for_out_of_bounds_position(self, plotter):
        layer = _Layer(np.zeros((10, 4, 6, 8)))
        assert plotter._extract_time_series(layer, np.array([0, 99, 99, 99])) is None

    def test_respects_non_standard_time_dim(self, plotter, rng):
        # Time at dim 1: shape (z=4, time=10, y=6, x=8).
        data = rng.random((4, 10, 6, 8))
        da = xr.DataArray(data, dims=["z", "time", "y", "x"])
        layer = _Layer(data, metadata={"xarray": da})
        ts = plotter._extract_time_series(layer, np.array([1, 0, 2, 3]))
        npt.assert_array_equal(ts, data[1, :, 2, 3])


# ---------------------------------------------------------------------------
# _active_layer
# ---------------------------------------------------------------------------


class TestActiveLayer:
    def test_returns_none_when_no_layer_selected(self, viewer, plotter):
        assert plotter._active_layer() is None

    def test_returns_none_for_2d_image(self, viewer, plotter):
        viewer.add_image(np.zeros((4, 6)))
        assert plotter._active_layer() is None

    def test_returns_none_for_rgb_layer(self, viewer, plotter):
        viewer.add_image(np.zeros((4, 6, 3), dtype=np.uint8), rgb=True)
        assert plotter._active_layer() is None

    def test_returns_layer_for_valid_image(self, viewer, plotter):
        layer = viewer.add_image(np.zeros((10, 4, 6, 8)))
        assert plotter._active_layer() is layer


# ---------------------------------------------------------------------------
# _on_mouse_move
# ---------------------------------------------------------------------------


class TestOnMouseMove:
    def test_shift_updates_cursor_pos_and_current_layer(self, viewer, plotter):
        layer = viewer.add_image(np.zeros((10, 4, 6, 8)))
        viewer.layers.selection.active = layer
        plotter._on_mouse_move(viewer, _FakeEventShift())
        assert plotter._cursor_pos is not None
        assert plotter._current_layer is layer

    def test_no_shift_does_not_update_cursor_pos(self, viewer, plotter):
        # Adding a layer fires _on_layer_change, which may set _current_layer.
        # Without Shift, _on_mouse_move must not update _cursor_pos.
        viewer.add_image(np.zeros((10, 4, 6, 8)))
        plotter._on_mouse_move(viewer, _FakeEventNoShift())
        assert plotter._cursor_pos is None
