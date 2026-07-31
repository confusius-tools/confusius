"""Unit tests for the napari volumewise diagnostics plotter."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from confusius._napari._registration._volumewise_diagnostics_plotter import (
    VolumewiseRegistrationDiagnosticsPlotter,
)
from confusius.registration import RegistrationDiagnostics


@pytest.fixture
def reference_2d() -> xr.DataArray:
    """Return a small spatial reference DataArray."""
    return xr.DataArray(
        np.zeros((3, 4), dtype=np.float32),
        dims=("y", "x"),
        coords={
            "y": xr.DataArray(np.arange(3), dims=("y",), attrs={"units": "mm"}),
            "x": xr.DataArray(np.arange(4), dims=("x",), attrs={"units": "mm"}),
        },
    )


def _diagnostics(value: float, n_iterations: int) -> RegistrationDiagnostics:
    """Return minimal completed-frame diagnostics."""
    return RegistrationDiagnostics(
        metric="correlation",
        metric_values=np.asarray([value]),
        final_metric_value=value,
        n_iterations=n_iterations,
        stop_condition="done",
        status="completed",
    )


class TestVolumewiseRegistrationDiagnosticsPlotter:
    """Tests for the floating volumewise diagnostics widget."""

    def test_add_frame_updates_lines(
        self, make_napari_viewer_proxy, qtbot, reference_2d
    ):
        """Completed-frame diagnostics update the plotted buffers."""
        viewer = make_napari_viewer_proxy()
        plotter = VolumewiseRegistrationDiagnosticsPlotter(
            viewer,
            n_frames=2,
            reference=reference_2d,
            time_coords=np.asarray([0.0, 0.3]),
            time_units="s",
            redraw_every=1,
        )
        qtbot.addWidget(plotter)

        affine0 = np.eye(3)
        affine1 = np.eye(3)
        affine1[0, 2] = 1.0
        plotter.add_frame(0, affine0, _diagnostics(-1.0, 4))
        plotter.add_frame(1, affine1, _diagnostics(-0.5, 6))
        plotter._render()

        np.testing.assert_allclose(plotter._metric_line.get_ydata(), [-1.0, -0.5])
        np.testing.assert_allclose(plotter._iteration_line.get_ydata(), [4.0, 6.0])
        assert plotter._optimizer_ax.get_xlabel() == "Time (s)"

    def test_reset_clears_lines(self, make_napari_viewer_proxy, qtbot, reference_2d):
        """Reset returns the plot to an empty state."""
        viewer = make_napari_viewer_proxy()
        plotter = VolumewiseRegistrationDiagnosticsPlotter(
            viewer,
            n_frames=1,
            reference=reference_2d,
            redraw_every=1,
        )
        qtbot.addWidget(plotter)

        plotter.add_frame(0, np.eye(3), _diagnostics(-1.0, 4))
        plotter.reset()

        assert plotter._metric_line.get_xdata().size == 0
        assert plotter._iteration_line.get_xdata().size == 0
