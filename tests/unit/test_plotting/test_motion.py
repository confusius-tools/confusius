"""Unit tests for motion diagnostics plotting."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from confusius.plotting import plot_motion_diagnostics


class TestPlotMotionDiagnostics:
    """Tests for `plot_motion_diagnostics`."""

    def test_rejects_dataframe_without_supported_columns(self):
        """A motion table needs at least one supported diagnostics column."""
        motion_df = pd.DataFrame(index=pd.Index([0, 1], name="frame"))

        with pytest.raises(ValueError, match="supported diagnostics columns"):
            plot_motion_diagnostics(motion_df)

    def test_optimizer_iterations_only_uses_single_axis(self):
        """Optimizer-only tables still produce a single panel without twinned axes."""
        motion_df = pd.DataFrame(
            {"n_iterations": [10, 12]},
            index=pd.Index([0, 1], name="frame"),
        )

        fig, axes = plot_motion_diagnostics(motion_df)

        assert len(axes) == 1
        assert axes[0].get_ylabel() == "Iterations"
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_effective_2d_motion_dataframe(self):
        """Effective 2D motion tables plot rotation, translation, and optimizer panels."""
        motion_df = pd.DataFrame(
            {
                "rotation": [0.0, 0.1],
                "trans_x": [0.0, 1.0],
                "trans_y": [0.0, 2.0],
                "mean_fd": [0.5, 0.0],
                "max_fd": [1.0, 0.0],
                "final_metric_value": [-1.0, -0.9],
                "n_iterations": [10, 12],
            },
            index=pd.Index([0.0, 0.5], name="time"),
        )

        fig, axes = plot_motion_diagnostics(motion_df)

        assert len(axes) == 4
        assert axes[0].get_ylabel() == "Rotation (deg)"
        assert axes[1].get_ylabel() == "Translation (mm)"
        assert axes[2].get_ylabel() == "Displacement (mm)"
        assert axes[-1].get_xlabel() == "Time (s)"
        plt.close(fig)

    def test_true_3d_motion_dataframe(self):
        """True 3D motion tables plot all standard panels."""
        motion_df = pd.DataFrame(
            {
                "rot_x": [0.0, 0.1],
                "rot_y": [0.0, 0.2],
                "rot_z": [0.0, 0.3],
                "trans_x": [0.0, 1.0],
                "trans_y": [0.0, 2.0],
                "trans_z": [0.0, 3.0],
                "mean_fd": [0.5, 0.0],
                "max_fd": [1.0, 0.0],
                "rms_fd": [0.7, 0.0],
            },
            index=pd.Index([0, 1], name="frame"),
        )

        fig, axes = plot_motion_diagnostics(motion_df)

        assert len(axes) == 3
        assert axes[-1].get_xlabel() == "Frame"
        plt.close(fig)
