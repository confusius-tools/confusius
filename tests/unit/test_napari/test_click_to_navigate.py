"""Tests for the click-to-navigate feature.

Left-clicking on a matplotlib plot (signals or QC) should navigate the napari
viewer to the corresponding time slice.  Pure-logic helpers (_x_to_frame) are
tested with lightweight stubs; viewer integration is tested with
make_napari_viewer following the napari plugin testing guidelines.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeMouseEvent:
    """Minimal matplotlib MouseEvent stub."""

    def __init__(self, inaxes, xdata, button=1):
        self.inaxes = inaxes
        self.xdata = xdata
        self.button = button


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
def qc_widget(viewer):
    from confusius._napari._qc._plots import QCPlotsWidget

    return QCPlotsWidget(viewer)


# ---------------------------------------------------------------------------
# SignalPlotter._x_to_frame
# ---------------------------------------------------------------------------


class TestXToFrame:
    def test_returns_x_val_when_no_coords(self, plotter):
        plotter._xaxis_coords = None
        assert plotter._x_to_frame(5.0) == 5.0

    def test_maps_time_value_to_nearest_frame_index(self, plotter):
        plotter._xaxis_coords = np.array([10.0, 10.5, 11.0, 11.5])
        assert plotter._x_to_frame(10.8) == 2.0  # closest to 11.0
        assert plotter._x_to_frame(10.0) == 0.0

    def test_non_numeric_coords_fall_back_to_x_val(self, plotter):
        plotter._xaxis_coords = np.array(["feat_A", "feat_B", "feat_C"])
        assert plotter._x_to_frame(1.0) == 1.0


# ---------------------------------------------------------------------------
# SignalPlotter._on_click
# ---------------------------------------------------------------------------


class TestSignalPlotterOnClick:
    def test_emits_frame_clicked_on_left_click(self, plotter, signal_spy):
        spy = signal_spy()
        plotter.frame_clicked.connect(spy)
        plotter._xaxis_coords = np.array([10.0, 10.5, 11.0])

        event = _FakeMouseEvent(inaxes=plotter._axes, xdata=10.5)
        plotter._on_click(event)

        assert spy.count == 1

    def test_ignores_right_click(self, plotter, signal_spy):
        spy = signal_spy()
        plotter.frame_clicked.connect(spy)

        event = _FakeMouseEvent(inaxes=plotter._axes, xdata=5.0, button=3)
        plotter._on_click(event)

        assert spy.count == 0

    def test_ignores_click_outside_axes(self, plotter, signal_spy):
        spy = signal_spy()
        plotter.frame_clicked.connect(spy)

        event = _FakeMouseEvent(inaxes=None, xdata=5.0)
        plotter._on_click(event)

        assert spy.count == 0

    def test_ignores_click_during_zoom(self, plotter, signal_spy):
        spy = signal_spy()
        plotter.frame_clicked.connect(spy)
        plotter._toolbar.zoom()  # Activate zoom mode.

        event = _FakeMouseEvent(inaxes=plotter._axes, xdata=5.0)
        plotter._on_click(event)

        assert spy.count == 0


# ---------------------------------------------------------------------------
# QCPlotsWidget click handlers
# ---------------------------------------------------------------------------


class TestQCPlotsOnClick:
    def test_dvars_click_emits_time_clicked(self, qc_widget, signal_spy):
        dvars = xr.DataArray(
            np.array([0.2, 0.4, 0.6]),
            dims=["time"],
            coords={"time": xr.DataArray(np.array([1.0, 1.5, 2.0]), dims=["time"])},
        )
        qc_widget.update_dvars(dvars)

        spy = signal_spy()
        qc_widget.time_clicked.connect(spy)

        event = _FakeMouseEvent(inaxes=qc_widget._dvars_ax, xdata=1.5)
        qc_widget._on_dvars_click(event)

        assert spy.count == 1

    def test_carpet_click_emits_time_clicked(self, qc_widget, signal_spy):
        time_coords = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        carpet_data = {
            "signals": xr.DataArray(
                np.random.default_rng(0).random((20, 5)),
                dims=["space", "time"],
                coords={"time": time_coords},
            ),
            "vmin": 0.0,
            "vmax": 1.0,
            "xlabel": "Time (s)",
            "time_coord": time_coords,
        }
        qc_widget.update_carpet(carpet_data)

        spy = signal_spy()
        qc_widget.time_clicked.connect(spy)

        event = _FakeMouseEvent(inaxes=qc_widget._carpet_ax, xdata=1.0)
        qc_widget._on_carpet_click(event)

        assert spy.count == 1

    def test_dvars_click_ignored_during_pan(self, qc_widget, signal_spy):
        dvars = xr.DataArray(np.array([0.2, 0.4]), dims=["time"])
        qc_widget.update_dvars(dvars)

        spy = signal_spy()
        qc_widget.time_clicked.connect(spy)
        qc_widget._dvars_toolbar.pan()  # Activate pan mode.

        event = _FakeMouseEvent(inaxes=qc_widget._dvars_ax, xdata=0.5)
        qc_widget._on_dvars_click(event)

        assert spy.count == 0
