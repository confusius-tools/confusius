"""Unit tests for the SignalPlotter widget.

Pure-logic methods (_xaxis_dim_index, _extract_signals) are tested with
lightweight mock layers.  Methods that touch the napari viewer (_active_layer,
_on_mouse_move) use the make_napari_viewer fixture.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from confusius._napari._export import format_export_value
from confusius._napari._signals._store import SignalStore

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
        self.ndim = self.data.ndim

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
    from confusius._napari._signals._plotter import SignalPlotter

    return SignalPlotter(viewer)


@pytest.fixture
def store():
    return SignalStore()


@pytest.fixture
def plotter_with_store(viewer, store):
    from confusius._napari._signals._plotter import SignalPlotter

    return SignalPlotter(viewer, store=store)


# ---------------------------------------------------------------------------
# _xaxis_dim_index
# ---------------------------------------------------------------------------


class TestXaxisDimIndex:
    def test_returns_zero_without_xarray_metadata(self, plotter):
        layer = _Layer(np.zeros((10, 4, 6, 8)))
        assert plotter._xaxis_dim_index(layer) == 0

    def test_returns_zero_when_no_time_dim_in_metadata(self, plotter, sample_3d_volume):
        layer = _Layer(np.zeros((4, 6, 8)), metadata={"xarray": sample_3d_volume})
        assert plotter._xaxis_dim_index(layer) == 0

    def test_returns_zero_when_time_is_first_dim(self, plotter, sample_4d_volume):
        layer = _Layer(np.zeros((10, 4, 6, 8)), metadata={"xarray": sample_4d_volume})
        assert plotter._xaxis_dim_index(layer) == 0

    def test_returns_correct_index_when_time_is_not_first(self, plotter):
        da = xr.DataArray(np.zeros((4, 10, 6, 8)), dims=["z", "time", "y", "x"])
        layer = _Layer(np.zeros((4, 10, 6, 8)), metadata={"xarray": da})
        assert plotter._xaxis_dim_index(layer) == 1


# ---------------------------------------------------------------------------
# _extract_signals
# ---------------------------------------------------------------------------


class TestExtractSignals:
    def test_extracts_correct_signals(self, plotter, rng):
        data = rng.random((10, 4, 6, 8))
        layer = _Layer(data)
        ts = plotter._extract_signals(layer, np.array([0, 1, 2, 3]))
        npt.assert_array_equal(ts, data[:, 1, 2, 3])

    def test_returns_none_for_out_of_bounds_position(self, plotter):
        layer = _Layer(np.zeros((10, 4, 6, 8)))
        assert plotter._extract_signals(layer, np.array([0, 99, 99, 99])) is None

    def test_respects_non_standard_time_dim(self, plotter, rng):
        # Time at dim 1: shape (z=4, time=10, y=6, x=8).
        data = rng.random((4, 10, 6, 8))
        da = xr.DataArray(data, dims=["z", "time", "y", "x"])
        layer = _Layer(data, metadata={"xarray": da})
        ts = plotter._extract_signals(layer, np.array([1, 0, 2, 3]))
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

    def test_returns_none_for_image_without_xarray(self, viewer, plotter):
        viewer.add_image(np.zeros((10, 4, 6, 8)))
        assert plotter._active_layer() is None

    def test_returns_layer_for_valid_image_with_xarray(
        self, viewer, plotter, sample_4d_volume
    ):
        layer = viewer.add_image(
            sample_4d_volume.values, metadata={"xarray": sample_4d_volume}
        )
        assert plotter._active_layer() is layer


# ---------------------------------------------------------------------------
# _on_mouse_move
# ---------------------------------------------------------------------------


class TestOnMouseMove:
    def test_shift_updates_cursor_pos_and_current_layer(
        self, viewer, plotter, sample_4d_volume
    ):
        layer = viewer.add_image(
            sample_4d_volume.values, metadata={"xarray": sample_4d_volume}
        )
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


# ---------------------------------------------------------------------------
# _get_xaxis_coords
# ---------------------------------------------------------------------------


class TestGetXaxisCoords:
    def test_returns_none_without_xarray_metadata(self, plotter):
        layer = _Layer(np.zeros((10, 4, 6, 8)))
        assert plotter._get_xaxis_coords(layer) is None

    def test_returns_none_when_no_time_coord(self, plotter):
        da = xr.DataArray(np.zeros((4, 6, 8)), dims=["z", "y", "x"])
        layer = _Layer(np.zeros((4, 6, 8)), metadata={"xarray": da})
        assert plotter._get_xaxis_coords(layer) is None

    def test_returns_correct_coords(self, plotter, sample_4d_volume):
        layer = _Layer(np.zeros((10, 4, 6, 8)), metadata={"xarray": sample_4d_volume})
        coords = plotter._get_xaxis_coords(layer)
        npt.assert_array_equal(coords, sample_4d_volume.coords["time"].values)


# ---------------------------------------------------------------------------
# _get_xaxis_label
# ---------------------------------------------------------------------------


class TestGetXaxisLabel:
    def test_returns_index_without_xarray_metadata(self, plotter):
        layer = _Layer(np.zeros((10, 4, 6, 8)))
        assert plotter._get_xaxis_label(layer) == "Index"

    def test_returns_capitalized_dim_when_no_coord(self, plotter):
        da = xr.DataArray(np.zeros((4, 6, 8)), dims=["z", "y", "x"])
        layer = _Layer(np.zeros((4, 6, 8)), metadata={"xarray": da})
        assert plotter._get_xaxis_label(layer) == "Time"

    def test_uses_units_from_time_coord(self, plotter, sample_4d_volume):
        # sample_4d_volume has time attrs={"units": "s"}, no long_name.
        layer = _Layer(np.zeros((10, 4, 6, 8)), metadata={"xarray": sample_4d_volume})
        assert plotter._get_xaxis_label(layer) == "Time (s)"

    def test_uses_long_name_and_units(self, plotter):
        da = xr.DataArray(
            np.zeros((5, 4)),
            dims=["time", "x"],
            coords={
                "time": xr.DataArray(
                    np.arange(5) * 0.5,
                    dims=["time"],
                    attrs={"long_name": "Elapsed time", "units": "s"},
                )
            },
        )
        layer = _Layer(np.zeros((5, 4)), metadata={"xarray": da})
        assert plotter._get_xaxis_label(layer) == "Elapsed time (s)"

    def test_omits_parentheses_when_no_units(self, plotter):
        da = xr.DataArray(
            np.zeros((5, 4)),
            dims=["time", "x"],
            coords={
                "time": xr.DataArray(
                    np.arange(5),
                    dims=["time"],
                    attrs={"long_name": "Frame index"},
                )
            },
        )
        layer = _Layer(np.zeros((5, 4)), metadata={"xarray": da})
        assert plotter._get_xaxis_label(layer) == "Frame index"


# ---------------------------------------------------------------------------
# _world_to_xaxis
# ---------------------------------------------------------------------------


class TestWorldToXaxis:
    """_world_to_xaxis maps a world coordinate to the plot x-axis value."""

    def _setup_plotter(self, plotter, xaxis_coords, *, xaxis_dim="time"):
        """Wire a mock layer with identity world_to_data into the plotter."""
        da = xr.DataArray(
            np.zeros((len(xaxis_coords), 4)),
            dims=[xaxis_dim, "x"],
            coords={xaxis_dim: xaxis_coords},
        )
        layer = _Layer(da.values, metadata={"xarray": da})
        plotter._current_layer = layer
        plotter._xaxis_coords = np.asarray(xaxis_coords)
        plotter._xaxis_dim = xaxis_dim

    def test_returns_world_when_no_current_layer(self, plotter):
        plotter._xaxis_coords = np.array([10.0, 10.5, 11.0])
        plotter._current_layer = None
        assert plotter._world_to_xaxis(3.0) == 3.0

    def test_returns_world_when_no_xaxis_coords(self, plotter):
        plotter._xaxis_coords = None
        plotter._current_layer = _Layer(np.zeros((4, 4)))
        assert plotter._world_to_xaxis(3.0) == 3.0

    def test_returns_world_value_for_numeric_coords(self, plotter):
        # Numeric coordinates return the world value directly (no snapping)
        # so the cursor stays in exact sync with the time overlay.
        self._setup_plotter(plotter, [10.0, 10.5, 11.0, 11.5])
        assert plotter._world_to_xaxis(2.0) == 2.0

    def test_out_of_bounds_falls_back_to_world_value(self, plotter):
        self._setup_plotter(plotter, [10.0, 10.5, 11.0])
        assert plotter._world_to_xaxis(99.0) == 99.0

    def test_returns_string_for_non_numeric_coordinates(self, plotter):
        # String coordinates (e.g., feature names) must be returned as-is so
        # matplotlib places the cursor at the correct categorical position
        # instead of inserting a spurious numeric category.
        self._setup_plotter(
            plotter, ["feat_A", "feat_B", "feat_C"], xaxis_dim="feature"
        )
        assert plotter._world_to_xaxis(1.0) == "feat_B"

    def test_nonuniform_time_returns_world_value(self, plotter):
        """Non-uniform numeric coordinates return the world value directly."""
        self._setup_plotter(plotter, [0.0, 0.5, 2.0, 2.1, 5.0])
        assert plotter._world_to_xaxis(3.0) == 3.0

    def test_non_time_xaxis_dim_returns_world_value(self, plotter):
        """Non-time numeric axes also return the world value directly."""
        lag_coords = [0.0, 0.1, 0.2, 0.3, 0.4]
        self._setup_plotter(plotter, lag_coords, xaxis_dim="lag")
        assert plotter._world_to_xaxis(3.0) == 3.0


# ---------------------------------------------------------------------------
# _flush_cursor — axvline synchronisation
# ---------------------------------------------------------------------------


class TestFlushCursor:
    """The blitted x-axis cursor must track the world coordinate.

    Regression: _flush_cursor used a non-existent ``_cursor_frame`` attribute
    instead of the value computed by ``_world_to_xaxis``, causing the axvline
    to silently fail to update (the AttributeError was swallowed by the
    bare except clause).
    """

    def _add_vline_and_bg(self, plotter, x_initial=0.0):
        """Add a vline and capture background so _flush_cursor can blit."""
        plotter._vline = plotter._axes.axvline(x_initial, color="red", animated=True)
        plotter._canvas.draw()
        plotter._bg = plotter._canvas.copy_from_bbox(plotter._figure.bbox)

    def test_flush_cursor_does_not_raise(self, plotter):
        """_flush_cursor must not raise (regression for missing _cursor_frame)."""
        self._add_vline_and_bg(plotter)
        plotter._cursor_world = 2.0
        # Before the fix this raised AttributeError caught by the except clause,
        # which set _bg to None. After the fix, _bg must remain intact.
        plotter._flush_cursor()
        assert plotter._bg is not None

    def test_flush_cursor_moves_vline_to_world_value(self, plotter):
        """The vline x-position must reflect _world_to_xaxis(cursor_world)."""
        self._add_vline_and_bg(plotter)
        plotter._cursor_world = 3.0
        plotter._flush_cursor()
        expected = plotter._world_to_xaxis(3.0)
        actual = plotter._vline.get_xdata()[0]
        assert actual == pytest.approx(expected)

    def test_flush_cursor_uses_world_value_with_xarray_coords(self, plotter):
        """Even with xarray coords, the vline uses the world value directly
        so it stays in exact sync with the time overlay (no snapping)."""
        coords = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
        da = xr.DataArray(np.zeros((5, 4)), dims=["time", "x"], coords={"time": coords})
        layer = _Layer(da.values, metadata={"xarray": da})
        plotter._current_layer = layer
        plotter._xaxis_coords = coords
        plotter._xaxis_dim = "time"

        self._add_vline_and_bg(plotter)
        plotter._cursor_world = 2.0
        plotter._flush_cursor()
        actual = plotter._vline.get_xdata()[0]
        # World value returned directly, not snapped to xarray coords.
        assert actual == pytest.approx(2.0)

    def test_flush_cursor_falls_back_to_world_for_no_xarray(self, plotter):
        """Without xarray metadata the vline position equals the world value.

        This is the video-layer case: no xarray metadata, so
        ``_world_to_xaxis`` returns the world value unchanged, which is
        ``dims.point[time_idx]`` -- the same value the time overlay uses.
        """
        plotter._xaxis_coords = None
        plotter._current_layer = None
        self._add_vline_and_bg(plotter)
        plotter._cursor_world = 3.5
        plotter._flush_cursor()
        actual = plotter._vline.get_xdata()[0]
        assert actual == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Video layer interaction — plotter must ignore non-xarray layers
# ---------------------------------------------------------------------------


class TestVideoLayerInteraction:
    """When a video layer (no xarray metadata) is selected alongside a fUSI
    scan, the plotter must continue using the fUSI layer for signal extraction,
    cursor mapping, and click-to-navigate.
    """

    @pytest.fixture
    def fusi_layer(self, viewer, sample_4d_volume):
        """Add a real fUSI layer with xarray metadata to the viewer."""
        from confusius.plotting import plot_napari

        _, layer = plot_napari(
            sample_4d_volume,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        return layer

    @pytest.fixture
    def video_layer(self, viewer):
        """Add a real video-like layer (no xarray, just fps metadata).

        Uses a 4D grayscale array so that napari sees the same ndim as the
        fUSI scan.  The critical property is the absence of xarray metadata.
        """
        data = np.zeros((20, 4, 48, 64), dtype=np.uint8)
        return viewer.add_image(
            data,
            name="Video: test",
            scale=(1 / 30.0, 1.0, 1.0, 1.0),
            axis_labels=("time", "z", "y", "x"),
            metadata={"fps": 30.0, "time_units": "s"},
        )

    def test_on_layer_change_retains_current_layer_when_video_selected(
        self, viewer, plotter, fusi_layer, video_layer
    ):
        """Selecting the video layer must not clear _current_layer.

        _active_layer() correctly returns None for the video layer, but
        _on_layer_change must not overwrite _current_layer with None --
        the previous fUSI layer must remain active for signal extraction.
        """
        # Set fUSI as the current layer (previous state).
        plotter._current_layer = fusi_layer

        # Now select the video layer.
        viewer.layers.selection.active = video_layer
        plotter._on_layer_change(None)

        # _current_layer must still be the fUSI layer, not None.
        assert plotter._current_layer is fusi_layer

    def test_on_layer_change_updates_current_layer_for_valid_xarray_layer(
        self, viewer, plotter, fusi_layer
    ):
        """Selecting a valid xarray layer must update _current_layer."""
        plotter._current_layer = None

        viewer.layers.selection.active = fusi_layer
        plotter._on_layer_change(None)

        assert plotter._current_layer is fusi_layer

    def test_cursor_uses_world_value_when_video_selected(
        self, viewer, plotter, fusi_layer, video_layer, sample_4d_volume
    ):
        """The x-axis cursor uses the world value directly (no snapping)
        even when the fUSI layer is retained as _current_layer.  This
        keeps the cursor in exact sync with the time overlay.
        """
        plotter._current_layer = fusi_layer
        plotter._xaxis_coords = np.array(sample_4d_volume.coords["time"].values)
        plotter._xaxis_dim = "time"

        # Select the video layer (doesn't change _current_layer).
        viewer.layers.selection.active = video_layer

        # World value returned directly, not snapped to xarray coords.
        result = plotter._world_to_xaxis(2.0)
        assert result == pytest.approx(2.0)

    def test_click_to_navigate_works_when_video_selected(
        self, plotter, fusi_layer, sample_4d_volume
    ):
        """_x_to_frame must still convert xarray coords to frame indices
        when _xaxis_coords is retained from the fUSI layer.
        """
        plotter._current_layer = fusi_layer
        plotter._xaxis_coords = np.array(sample_4d_volume.coords["time"].values)

        # Click on x=11.0 in the plot -> should map to frame index 2.
        frame = plotter._x_to_frame(11.0)
        assert frame == pytest.approx(2.0)

    def test_flush_cursor_with_fusi_retained_after_video_selection(
        self, viewer, plotter, fusi_layer, video_layer, sample_4d_volume
    ):
        """_flush_cursor must use the world value directly (no snapping)
        when the video layer is selected but _current_layer is the fUSI.
        """
        plotter._current_layer = fusi_layer
        plotter._xaxis_coords = np.array(sample_4d_volume.coords["time"].values)
        plotter._xaxis_dim = "time"

        plotter._vline = plotter._axes.axvline(0, color="red", animated=True)
        plotter._canvas.draw()
        plotter._bg = plotter._canvas.copy_from_bbox(plotter._figure.bbox)

        # World value returned directly, not snapped to xarray coord 11.5.
        plotter._cursor_world = 3.0
        plotter._flush_cursor()

        actual = plotter._vline.get_xdata()[0]
        assert actual == pytest.approx(3.0)
        assert plotter._bg is not None


# ---------------------------------------------------------------------------
# Panel x-axis combo — must not break when video layer is active
# ---------------------------------------------------------------------------


class TestPanelXaxisWithVideoLayer:
    """The SignalPanel x-axis combo must remain stable when the active layer
    changes to a video layer that lacks xarray metadata.

    ``_get_available_xaxis_dims`` must fall back to scanning all layers for
    xarray metadata instead of generating generic ``dim_0`` entries from
    the video layer.
    """

    @pytest.fixture
    def fusi_layer(self, viewer, sample_4d_volume):
        """Add a real fUSI layer with xarray metadata to the viewer."""
        from confusius.plotting import plot_napari

        _, layer = plot_napari(
            sample_4d_volume,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        return layer

    @pytest.fixture
    def video_layer(self, viewer):
        """Add a real video-like layer (no xarray, just fps metadata).

        Uses a 4D grayscale array so that napari sees the same ndim as the
        fUSI scan.  The critical property is the absence of xarray metadata.
        """
        data = np.zeros((20, 4, 48, 64), dtype=np.uint8)
        return viewer.add_image(
            data,
            name="Video: test",
            scale=(1 / 30.0, 1.0, 1.0, 1.0),
            axis_labels=("time", "z", "y", "x"),
            metadata={"fps": 30.0, "time_units": "s"},
        )

    @pytest.fixture
    def signal_panel(self, viewer):
        """Create a real SignalPanel wired to the viewer."""
        from confusius._napari._signals._panel import SignalPanel

        return SignalPanel(viewer)

    def test_returns_time_when_fusi_is_active(self, viewer, signal_panel, fusi_layer):
        """Baseline: x-axis dims include 'time' when fUSI is active."""
        viewer.layers.selection.active = fusi_layer

        dims = signal_panel._get_available_xaxis_dims()
        assert "time" in dims

    def test_falls_back_to_xarray_layers_when_video_is_active(
        self, viewer, signal_panel, fusi_layer, video_layer
    ):
        """When the video layer is active, _get_available_xaxis_dims must
        fall back to the fUSI layer's dims instead of returning generic dim names.
        """
        viewer.layers.selection.active = video_layer

        dims = signal_panel._get_available_xaxis_dims()
        assert "time" in dims
        # Must NOT contain generic dim names from the video layer.
        assert not any(d.startswith("dim_") for d in dims)

    def test_returns_empty_when_no_xarray_layers_exist(
        self, viewer, signal_panel, video_layer
    ):
        """When no layers have xarray metadata, return empty list."""
        viewer.layers.selection.active = video_layer

        dims = signal_panel._get_available_xaxis_dims()
        assert dims == []

    def test_xaxis_dim_index_finds_time_from_fusi_when_video_active(
        self, viewer, signal_panel, fusi_layer, video_layer
    ):
        """_xaxis_dim_index must find the time dimension from the fUSI layer
        even when the video layer is the active selection.
        """
        # Ensure the combo is populated from the fUSI layer before switching.
        viewer.layers.selection.active = fusi_layer
        signal_panel._refresh_xaxis_combo()

        viewer.layers.selection.active = video_layer
        signal_panel._xaxis_combo.setCurrentText("time")

        idx = signal_panel._xaxis_dim_index()
        assert idx == 0  # time is dim 0 in the fUSI layer.

    def test_click_to_navigate_uses_world_coordinate_with_video_layer(
        self, viewer, signal_panel, fusi_layer, video_layer
    ):
        """Regression: clicking at 300s on the plot must navigate to ~300s,
        not 300/fps.

        When a video layer is loaded its time scale (1/fps) becomes the
        smallest scale on the time axis, shrinking ``dims.range.step``.
        The old code set ``current_step`` directly from a data index, so
        the world coordinate was ``data_index * range.step`` instead of the
        intended time.  The fix emits the x-axis plot value (already in
        world coordinates) and uses ``dims.set_point`` which accepts
        world coordinates directly.
        """
        # Ensure the combo is populated from the fUSI layer before switching.
        viewer.layers.selection.active = fusi_layer
        signal_panel._refresh_xaxis_combo()

        viewer.layers.selection.active = video_layer
        signal_panel._xaxis_combo.setCurrentText("time")

        # Simulate the plotter emitting a click at x=12.5 (seconds).
        # The panel calls set_point(0, 12.5) -- NOT current_step.
        signal_panel._on_frame_clicked(12.5)

        # Verify the viewer navigated to the correct world coordinate.
        time_idx = signal_panel._xaxis_dim_index()
        actual_world = float(viewer.dims.point[time_idx])
        assert actual_world == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# _update_plot — xlim regression
# ---------------------------------------------------------------------------


class TestUpdatePlotXlim:
    def test_xlim_recovers_after_leaving_image(self, plotter, rng):
        """Regression: xlim must not stay at matplotlib default [0, 1] after the
        cursor briefly leaves the image and then returns.
        """
        data = rng.random((10, 4, 6, 8))
        layer = _Layer(data)
        plotter._current_layer = layer

        # First valid plot — establishes the correct x range.
        plotter._cursor_pos = np.array([0, 1, 2, 3])
        plotter._update_plot()
        xlim_first = plotter._axes.get_xlim()

        # Cursor leaves the image (out-of-bounds → ts is None).
        plotter._cursor_pos = np.array([0, 99, 99, 99])
        plotter._update_plot()

        # Cursor returns to the same valid position.
        plotter._cursor_pos = np.array([0, 1, 2, 3])
        plotter._update_plot()
        xlim_recovered = plotter._axes.get_xlim()

        # The recovered xlim must match the original, not be stuck at [0, 1].
        npt.assert_allclose(xlim_recovered, xlim_first)


# ---------------------------------------------------------------------------
# TSV export
# ---------------------------------------------------------------------------


class TestTsvExport:
    def test_writes_time_first_column_for_current_plot(
        self, plotter, tmp_path, sample_4d_volume
    ):
        data = np.asarray(sample_4d_volume)
        layer = _Layer(data, metadata={"xarray": sample_4d_volume})
        layer.name = "signal"

        plotter._current_layer = layer
        plotter._cursor_pos = np.array([0, 1, 2, 3])
        plotter._update_plot()

        out_path = tmp_path / "timeseries.tsv"
        plotter._write_current_plot_delimited(out_path, delimiter="\t")

        rows = [line.split("\t") for line in out_path.read_text().splitlines()]
        assert rows[0] == ["time", "signal"]
        assert [row[0] for row in rows[1:]] == [
            format_export_value(value)
            for value in sample_4d_volume.coords["time"].values
        ]
        assert [row[1] for row in rows[1:]] == [
            format_export_value(value) for value in data[:, 1, 2, 3]
        ]
        assert plotter._export_button.isEnabled()

    def test_writes_csv_when_requested(self, plotter, tmp_path):
        from confusius._napari._export import ExportSignal

        plotter._set_export_signals(
            [ExportSignal("signal", np.array([0.0, 1.0]), np.array([10.0, 11.5]))]
        )

        out_path = tmp_path / "timeseries.csv"
        plotter._write_current_plot_delimited(out_path, delimiter=",")

        rows = [line.split(",") for line in out_path.read_text().splitlines()]
        assert rows == [["time", "signal"], ["0", "10"], ["1", "11.5"]]

    def test_aligns_multiple_signals_and_deduplicates_headers(self, plotter, tmp_path):
        from confusius._napari._export import ExportSignal

        plotter._set_export_signals(
            [
                ExportSignal("series", np.array([0.0, 1.0]), np.array([10.0, 11.0])),
                ExportSignal("series", np.array([1.0, 2.0]), np.array([20.0, 21.0])),
            ]
        )

        out_path = tmp_path / "timeseries.tsv"
        plotter._write_current_plot_delimited(out_path, delimiter="\t")

        rows = [line.split("\t") for line in out_path.read_text().splitlines()]
        assert rows == [
            ["time", "series", "series_2"],
            ["0", "10", ""],
            ["1", "11", "20"],
            ["2", "", "21"],
        ]

    def test_invalid_plot_clears_export_state(self, plotter):
        from confusius._napari._export import ExportSignal

        plotter._set_export_signals(
            [ExportSignal("series", np.array([0.0, 1.0]), np.array([10.0, 11.0]))]
        )

        layer = _Layer(np.zeros((10, 4, 6, 8)))
        plotter._current_layer = layer
        plotter._cursor_pos = np.array([0, 99, 99, 99])
        plotter._update_plot()

        assert plotter._export_signals == []
        assert not plotter._export_button.isEnabled()


class TestImportedSignalIntegration:
    def test_overlays_imported_signals_with_live_mouse_plot(
        self, plotter_with_store, store, tmp_path
    ):
        path = tmp_path / "imported.csv"
        path.write_text("time,external\n0,20\n1,21\n2,22\n")
        store.import_file(path)

        layer = _Layer(np.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2))
        layer.name = "live"
        plotter_with_store._current_layer = layer
        plotter_with_store._cursor_pos = np.array([0, 1, 1, 1])

        plotter_with_store._update_plot()

        out_path = tmp_path / "combined.tsv"
        plotter_with_store._write_current_plot_delimited(out_path, delimiter="\t")
        rows = [line.split("\t") for line in out_path.read_text().splitlines()]
        assert rows == [
            ["time", "live", "external"],
            ["0", "7", "20"],
            ["1", "15", "21"],
            ["2", "23", "22"],
        ]

    def test_plots_imported_signals_without_live_data(
        self, plotter_with_store, store, tmp_path
    ):
        path = tmp_path / "imported.tsv"
        path.write_text("time\tbaseline\n0\t1.0\n0\t2.0\n1\t3.0\n")
        store.import_file(path)

        plotter_with_store._refresh_plot()

        out_path = tmp_path / "imported_only.tsv"
        plotter_with_store._write_current_plot_delimited(out_path, delimiter="\t")
        rows = [line.split("\t") for line in out_path.read_text().splitlines()]
        assert rows == [
            ["time", "baseline"],
            ["0", "1"],
            ["0", "2"],
            ["1", "3"],
        ]

    def test_hidden_imported_signal_is_excluded_from_export(
        self, plotter_with_store, store, tmp_path
    ):
        path = tmp_path / "imported.csv"
        path.write_text("time,external\n0,20\n1,21\n2,22\n")
        imported = store.import_file(path)
        store.set_signal_visible(imported[0].id, False)

        layer = _Layer(np.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2))
        layer.name = "live"
        plotter_with_store._current_layer = layer
        plotter_with_store._cursor_pos = np.array([0, 1, 1, 1])

        plotter_with_store._update_plot()

        out_path = tmp_path / "combined.tsv"
        plotter_with_store._write_current_plot_delimited(out_path, delimiter="\t")
        rows = [line.split("\t") for line in out_path.read_text().splitlines()]
        assert rows == [
            ["time", "live"],
            ["0", "7"],
            ["1", "15"],
            ["2", "23"],
        ]

    def test_imported_signal_color_is_used_for_overlay(
        self, plotter_with_store, store, tmp_path
    ):
        path = tmp_path / "imported.csv"
        path.write_text("time,external\n0,20\n1,21\n2,22\n")
        imported = store.import_file(path)
        store.set_signal_color(imported[0].id, "#123456")

        plotter_with_store._refresh_plot()

        line = plotter_with_store._axes.lines[0]
        assert line.get_label() == "external"
        assert line.get_color() == "#123456"

    def test_xlim_defaults_to_largest_plotted_signal_extent(
        self, plotter_with_store, store, tmp_path
    ):
        path = tmp_path / "imported.csv"
        path.write_text("time,external\n-2,20\n0,21\n4,22\n")
        store.import_file(path)

        layer = _Layer(np.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2))
        layer.name = "live"
        plotter_with_store._current_layer = layer
        plotter_with_store._cursor_pos = np.array([0, 1, 1, 1])

        plotter_with_store._update_plot()

        xlim = plotter_with_store._axes.get_xlim()
        assert xlim[0] == pytest.approx(-2.0)
        assert xlim[1] == pytest.approx(4.0)

    def test_importing_longer_signal_expands_existing_auto_xlim(
        self, plotter_with_store, store, tmp_path
    ):
        layer = _Layer(np.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2))
        layer.name = "live"
        plotter_with_store._current_layer = layer
        plotter_with_store._cursor_pos = np.array([0, 1, 1, 1])
        plotter_with_store._update_plot()

        xlim_before = plotter_with_store._axes.get_xlim()
        assert xlim_before[0] == pytest.approx(0.0)
        assert xlim_before[1] == pytest.approx(2.0)

        path = tmp_path / "imported.csv"
        path.write_text("time,external\n-2,20\n0,21\n4,22\n")
        store.import_file(path)

        xlim_after = plotter_with_store._axes.get_xlim()
        assert xlim_after[0] == pytest.approx(-2.0)
        assert xlim_after[1] == pytest.approx(4.0)

    def test_user_zoom_is_retained_across_mouse_moves(
        self, plotter_with_store, store, tmp_path
    ):
        path = tmp_path / "imported.csv"
        path.write_text("time,external\n-2,20\n0,21\n4,22\n")
        store.import_file(path)

        layer = _Layer(np.arange(5 * 3 * 3 * 3).reshape(5, 3, 3, 3))
        layer.name = "live"
        plotter_with_store._current_layer = layer
        plotter_with_store._cursor_pos = np.array([0, 1, 1, 1])
        plotter_with_store._update_plot()

        plotter_with_store._axes.set_xlim((-1.0, 1.0))
        plotter_with_store._cursor_pos = np.array([0, 2, 2, 2])
        plotter_with_store._update_plot()

        xlim = plotter_with_store._axes.get_xlim()
        assert xlim[0] == pytest.approx(-1.0)
        assert xlim[1] == pytest.approx(1.0)

    def test_invalid_cursor_clears_previous_lines(
        self, plotter_with_store, store, tmp_path
    ):
        path = tmp_path / "imported.csv"
        path.write_text("time,external\n0,20\n1,21\n2,22\n")
        store.import_file(path)

        layer = _Layer(np.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2))
        layer.name = "live"
        plotter_with_store._current_layer = layer
        plotter_with_store._cursor_pos = np.array([0, 1, 1, 1])
        plotter_with_store._update_plot()

        plotter_with_store._signals_store.clear()
        plotter_with_store._cursor_pos = np.array([0, 99, 99, 99])
        plotter_with_store._update_plot()

        assert len(plotter_with_store._axes.lines) == 0
        assert plotter_with_store._axes.get_title() == ""

    def test_hiding_imported_signal_resets_xlim_to_remaining_visible_signals(
        self, plotter_with_store, store, tmp_path
    ):
        path = tmp_path / "imported.csv"
        path.write_text("time,external\n-2,20\n0,21\n4,22\n")
        imported = store.import_file(path)

        layer = _Layer(np.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2))
        layer.name = "live"
        plotter_with_store._current_layer = layer
        plotter_with_store._cursor_pos = np.array([0, 1, 1, 1])
        plotter_with_store._update_plot()

        store.set_signal_visible(imported[0].id, False)

        xlim = plotter_with_store._axes.get_xlim()
        assert xlim[0] == pytest.approx(0.0)
        assert xlim[1] == pytest.approx(2.0)

    def test_zscore_of_constant_signal_gives_all_zeros(
        self, plotter_with_store, store, tmp_path
    ):
        # A signal with all-identical values has std=0; z-scoring should yield
        # all-zeros (ts - mean) rather than NaN.
        path = tmp_path / "constant.csv"
        path.write_text("time,flat\n0,5.0\n1,5.0\n2,5.0\n")
        store.import_file(path)

        plotter_with_store.set_zscore(True)
        result = plotter_with_store._render_imported_only()

        assert result is True
        assert len(plotter_with_store._axes.lines) == 1
        y_data = plotter_with_store._axes.lines[0].get_ydata()
        npt.assert_array_equal(y_data, np.zeros(3))

    def test_zscore_with_partial_nan_signal(self, plotter_with_store, store, tmp_path):
        # A signal with some NaN values (e.g. alternating missing values as in
        # a TSV with partial columns). Non-NaN values should be z-scored; NaN
        # values should remain NaN and the plot should not crash.
        path = tmp_path / "partial_nan.csv"
        path.write_text("time,signal\n0,1.0\n1,\n2,3.0\n3,\n4,5.0\n")
        store.import_file(path)

        plotter_with_store.set_zscore(True)
        result = plotter_with_store._render_imported_only()

        assert result is True
        assert len(plotter_with_store._axes.lines) == 1
        y_data = plotter_with_store._axes.lines[0].get_ydata()
        # NaN positions must remain NaN.
        assert np.isnan(y_data[1])
        assert np.isnan(y_data[3])
        # Non-NaN positions must not be NaN.
        assert not np.any(np.isnan(y_data[[0, 2, 4]]))

    def test_close_event_disconnects_store_signals(
        self, plotter_with_store, store, tmp_path, monkeypatch
    ):
        # Import a first signal so the plotter is in a known state.
        path = tmp_path / "first.csv"
        path.write_text("time,a\n0,1\n1,2\n")
        store.import_file(path)

        # Track calls to _refresh_plot after the plotter is closed.
        call_count = {"n": 0}
        original_refresh = plotter_with_store._refresh_plot

        def _counting_refresh():
            call_count["n"] += 1
            original_refresh()

        monkeypatch.setattr(plotter_with_store, "_refresh_plot", _counting_refresh)

        # Close the plotter — this should disconnect the store signals.
        plotter_with_store.close()

        # Import another signal; _refresh_plot must NOT be called.
        path2 = tmp_path / "second.csv"
        path2.write_text("time,b\n0,10\n1,20\n")
        store.import_file(path2)

        assert call_count["n"] == 0
