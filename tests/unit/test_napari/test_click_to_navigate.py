"""Tests for the click-to-navigate feature.

Left-clicking on a matplotlib plot (signals or QC) should navigate the napari
viewer to the corresponding time slice. Click events use lightweight stubs;
widgets use make_napari_viewer following the napari plugin testing guidelines.
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
# SignalPlotter._on_click
# ---------------------------------------------------------------------------


class TestSignalPlotterOnClick:
    def test_emits_frame_clicked_on_left_click(self, plotter, signal_spy):
        spy = signal_spy()
        plotter.frame_clicked.connect(spy)
        plotter._axis_coordinates = np.array([10.0, 10.5, 11.0])

        event = _FakeMouseEvent(inaxes=plotter._axes, xdata=10.5)
        plotter._on_click(event)

        assert spy.count == 1

    @pytest.mark.parametrize(
        "coordinates",
        [None, np.array([10.0, 10.5, 11.0]), np.array(["A", "B", "C"])],
        ids=["no-coordinates", "numeric", "categorical-without-reference"],
    )
    def test_emits_xaxis_value_not_data_index(self, plotter, coordinates):
        """The signal must emit the x-axis plot coordinate (world value),
        not a nearest-frame index.

        Regression: the old code emitted _x_to_frame(xdata), which is a
        data index.  When a video layer changed dims.range.step, the panel
        incorrectly converted this index again, causing a click at 300s to
        navigate to 300/fps instead.
        """
        received = []
        plotter.frame_clicked.connect(lambda v: received.append(v))
        plotter._axis_coordinates = coordinates

        # Click at x=10.5 on the plot — must emit 10.5 (world), not 1 (index).
        event = _FakeMouseEvent(inaxes=plotter._axes, xdata=10.5)
        plotter._on_click(event)

        assert len(received) == 1
        assert received[0] == pytest.approx(10.5)

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

    def test_carpet_click_ignored_during_pan(self, qc_widget, signal_spy):
        time_coords = np.array([0.0, 0.5, 1.0])
        carpet_data = {
            "signals": xr.DataArray(
                np.random.default_rng(0).random((4, 3)),
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
        qc_widget._carpet_toolbar.pan()  # Activate pan mode.

        event = _FakeMouseEvent(inaxes=qc_widget._carpet_ax, xdata=0.5)
        qc_widget._on_carpet_click(event)

        assert spy.count == 0
