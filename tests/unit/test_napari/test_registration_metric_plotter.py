"""Unit tests for the bottom-dock registration metric plotter."""

from __future__ import annotations

import pytest


@pytest.fixture
def registration_metric_plotter(make_napari_viewer_proxy):
    from confusius._napari._registration._metric_plotter import (
        RegistrationMetricPlotter,
    )

    viewer = make_napari_viewer_proxy()
    return RegistrationMetricPlotter(viewer)


class TestRegistrationMetricPlotterBuffer:
    """Pure-logic: add_metric / reset / metric_values."""

    def test_empty_after_construction(self, registration_metric_plotter) -> None:
        assert registration_metric_plotter.metric_values == []

    def test_add_metric_appends_value(self, registration_metric_plotter) -> None:
        registration_metric_plotter.add_metric(0.5)
        registration_metric_plotter.add_metric(0.25)
        registration_metric_plotter.add_metric(0.1)
        assert registration_metric_plotter.metric_values == [0.5, 0.25, 0.1]

    def test_metric_values_returns_a_copy(
        self, registration_metric_plotter
    ) -> None:
        registration_metric_plotter.add_metric(1.0)
        snapshot = registration_metric_plotter.metric_values
        snapshot.append(99.0)
        assert registration_metric_plotter.metric_values == [1.0]

    def test_reset_clears_buffer(self, registration_metric_plotter) -> None:
        registration_metric_plotter.add_metric(0.5)
        registration_metric_plotter.add_metric(0.25)
        registration_metric_plotter.reset()
        assert registration_metric_plotter.metric_values == []

    def test_add_metric_after_reset_starts_new_run(
        self, registration_metric_plotter
    ) -> None:
        registration_metric_plotter.add_metric(0.5)
        registration_metric_plotter.add_metric(0.25)
        registration_metric_plotter.reset()
        registration_metric_plotter.add_metric(0.1)
        assert registration_metric_plotter.metric_values == [0.1]
