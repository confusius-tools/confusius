"""Unit tests for the napari volumewise diagnostics plotter."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from confusius._napari._registration._volumewise_diagnostics_plotter import (
    VolumewiseRegistrationDiagnosticsPlotter,
)
from confusius.registration import RegistrationDiagnostics
from confusius.xarray import create_voxeldata


@pytest.fixture
def reference() -> xr.DataArray:
    """Return a small singleton-k VoxelData reference with unit spacing."""
    return create_voxeldata(
        np.zeros((1, 3, 4), dtype=np.float32),
        dims=("k", "j", "i"),
        spacing=(1.0, 1.0, 1.0),
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
        self, make_napari_viewer_proxy, qtbot, reference
    ):
        """Completed-frame diagnostics update the plotted buffers."""
        viewer = make_napari_viewer_proxy()
        plotter = VolumewiseRegistrationDiagnosticsPlotter(
            viewer,
            n_frames=2,
            reference=reference,
            time_coords=np.asarray([0.0, 0.3]),
            time_units="s",
            redraw_every=1,
        )
        qtbot.addWidget(plotter)

        affine0 = np.eye(4)
        affine1 = np.eye(4)
        affine1[0, 3] = 1.0
        plotter.add_frame(0, affine0, _diagnostics(-1.0, 4))
        plotter.add_frame(1, affine1, _diagnostics(-0.5, 6))
        plotter._render()

        np.testing.assert_allclose(plotter._metric_line.get_ydata(), [-1.0, -0.5])
        np.testing.assert_allclose(plotter._iteration_line.get_ydata(), [4.0, 6.0])
        assert plotter._optimizer_ax.get_xlabel() == "Time (s)"

    def test_reset_clears_lines(self, make_napari_viewer_proxy, qtbot, reference):
        """Reset returns the plot to an empty state."""
        viewer = make_napari_viewer_proxy()
        plotter = VolumewiseRegistrationDiagnosticsPlotter(
            viewer,
            n_frames=1,
            reference=reference,
            redraw_every=1,
        )
        qtbot.addWidget(plotter)

        plotter.add_frame(0, np.eye(4), _diagnostics(-1.0, 4))
        plotter.reset()

        assert plotter._metric_line.get_xdata().size == 0
        assert plotter._iteration_line.get_xdata().size == 0
