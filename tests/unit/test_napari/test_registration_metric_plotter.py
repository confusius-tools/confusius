"""Unit tests for the bottom-dock registration metric plotter."""

from __future__ import annotations

import pytest
from qtpy.QtCore import Qt


@pytest.fixture
def registration_metric_plotter(make_napari_viewer):
    from confusius._napari._registration._metric_plotter import (
        RegistrationMetricPlotter,
    )

    viewer = make_napari_viewer()
    return RegistrationMetricPlotter(viewer)


class TestRegistrationMetricPlotterBuffer:
    """Pure-logic: add_metric / reset / metric_values."""

    def test_empty_after_construction(self, registration_metric_plotter) -> None:
        assert registration_metric_plotter.metric_values == []

    def test_add_metric_appends_value(self, registration_metric_plotter) -> None:
        registration_metric_plotter.add_metric(0.5)
        registration_metric_plotter.add_metric(0.25)
        registration_metric_plotter.add_metric(0.1)
        # The QTimer is single-shot at 16ms; force a render so the line
        # state is finalised before we read it.
        registration_metric_plotter._render()  # type: ignore[attr-defined]
        assert registration_metric_plotter.metric_values == [0.5, 0.25, 0.1]

    def test_metric_values_returns_a_copy(
        self, registration_metric_plotter
    ) -> None:
        registration_metric_plotter.add_metric(1.0)
        snapshot = registration_metric_plotter.metric_values
        snapshot.append(99.0)
        # Mutating the snapshot must not affect the internal buffer.
        assert registration_metric_plotter.metric_values == [1.0]

    def test_reset_clears_buffer(self, registration_metric_plotter) -> None:
        registration_metric_plotter.add_metric(0.5)
        registration_metric_plotter.add_metric(0.25)
        registration_metric_plotter.reset()
        assert registration_metric_plotter.metric_values == []

    def test_reset_after_data_keeps_axes_valid(
        self, registration_metric_plotter
    ) -> None:
        registration_metric_plotter.add_metric(0.5)
        registration_metric_plotter.reset()
        registration_metric_plotter._render()  # type: ignore[attr-defined]
        # After reset + render, the line data is empty but the axes are
        # still configured.
        line = registration_metric_plotter._metric_line  # type: ignore[attr-defined]
        assert list(line.get_xdata()) == []
        assert list(line.get_ydata()) == []


class TestRegistrationMetricPlotterThrottling:
    """The redraw timer coalesces rapid `add_metric` calls."""

    def test_single_timer_per_burst(
        self, registration_metric_plotter, qtbot
    ) -> None:
        # Burst a series of values without yielding to the event loop; the
        # timer should be active but only one render should fire when the
        # loop runs.
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            registration_metric_plotter.add_metric(v)
        # The buffer holds every value; the canvas will be redrawn once the
        # timer fires.
        assert registration_metric_plotter.metric_values == [0.1, 0.2, 0.3, 0.4, 0.5]
        timer = registration_metric_plotter._redraw_timer  # type: ignore[attr-defined]
        assert timer.isSingleShot()
        assert timer.interval() == 16

    def test_render_after_timer_fire(
        self, registration_metric_plotter, qtbot
    ) -> None:
        registration_metric_plotter.add_metric(0.5)
        # Wait for the throttled redraw to fire.
        with qtbot.waitSignal(
            registration_metric_plotter._redraw_timer.timeout,  # type: ignore[attr-defined]
            timeout=2000,
        ):
            pass
        line = registration_metric_plotter._metric_line  # type: ignore[attr-defined]
        npt_import = pytest.importorskip("numpy")
        npt_import.testing.assert_array_equal(
            npt_import.asarray(line.get_xdata()), npt_import.asarray([1])
        )
        npt_import.testing.assert_array_equal(
            npt_import.asarray(line.get_ydata()), npt_import.asarray([0.5])
        )


class TestRegistrationMetricPlotterLayout:
    """Construction and theme integration."""

    def test_widget_has_minimum_height(self, registration_metric_plotter) -> None:
        assert registration_metric_plotter.minimumHeight() >= 100

    def test_size_hint(self, registration_metric_plotter) -> None:
        hint = registration_metric_plotter.sizeHint()
        assert hint.width() >= 400
        assert hint.height() >= 150

    def test_metric_line_created(self, registration_metric_plotter) -> None:
        assert registration_metric_plotter._metric_line is not None  # type: ignore[attr-defined]
        assert registration_metric_plotter._axes.get_xlabel() == "Iteration"  # type: ignore[attr-defined]
        assert registration_metric_plotter._axes.get_ylabel() == "Metric value"  # type: ignore[attr-defined]