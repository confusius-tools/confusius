"""Tests for confusius.plotting.matrix.

See conftest.py for the matplotlib_pyplot fixture setup.
"""

import numpy as np
import pytest
import xarray as xr

from confusius.plotting import plot_matrix


def _correlation_matrix(size: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.corrcoef(rng.standard_normal((size, size * 3)))


class TestPlotMatrixValidation:
    """Input validation for plot_matrix."""

    def test_non_square_matrix_raises(self, matplotlib_pyplot):
        """A non-square matrix raises ValueError."""
        with pytest.raises(ValueError, match="square"):
            plot_matrix(np.zeros((3, 4)))

    def test_labels_length_mismatch_raises(self, matplotlib_pyplot):
        """labels with the wrong length raises ValueError."""
        with pytest.raises(ValueError, match="Length of labels"):
            plot_matrix(_correlation_matrix(4), labels=["a", "b"])

    def test_groups_length_mismatch_raises(self, matplotlib_pyplot):
        """groups with the wrong length raises ValueError."""
        with pytest.raises(ValueError, match="Length of groups"):
            plot_matrix(_correlation_matrix(4), groups=["a", "b"])

    def test_incomplete_group_colors_raises(self, matplotlib_pyplot):
        """group_colors missing a group value raises ValueError."""
        with pytest.raises(ValueError, match="group_colors is missing"):
            plot_matrix(
                _correlation_matrix(4),
                groups=["a", "a", "b", "b"],
                group_colors={"a": "red"},
            )


class TestPlotMatrixBehaviour:
    """Non-visual behaviour of plot_matrix."""

    def test_labels_false_forces_no_ticks(self, matplotlib_pyplot):
        """labels=False disables ticks even with a labeled DataArray input."""
        matrix = xr.DataArray(
            _correlation_matrix(4),
            dims=["region", "region2"],
            coords={"region": ["a", "b", "c", "d"]},
        )
        _, ax = plot_matrix(matrix, labels=False)
        assert ax.get_xticks().size == 0

    def test_dataarray_default_labels_from_coordinate(self, matplotlib_pyplot):
        """Coordinate values on a DataArray's first dim become default labels."""
        matrix = xr.DataArray(
            _correlation_matrix(3),
            dims=["region", "region2"],
            coords={"region": ["a", "b", "c"]},
        )
        _, ax = plot_matrix(matrix)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]

    def test_symmetric_default_vmax(self, matplotlib_pyplot):
        """With no explicit vmin/vmax, the colormap is centered on zero."""
        matrix = np.array([[0.0, -0.8], [-0.8, 0.0]])
        _, ax = plot_matrix(matrix)
        image = ax.images[0]
        assert image.norm.vmin == pytest.approx(-0.8)
        assert image.norm.vmax == pytest.approx(0.8)

    def test_fontsize_scales_text_elements(self, matplotlib_pyplot):
        """plot_matrix scales title, tick, and colorbar text from fontsize."""
        fig, ax = plot_matrix(
            _correlation_matrix(4),
            labels=["a", "b", "c", "d"],
            title="Matrix",
            fontsize=18,
        )
        assert ax.title.get_fontsize() == pytest.approx(18)
        assert ax.get_xticklabels()[0].get_fontsize() == pytest.approx(15.3)

        cbar_axes = [axis for axis in fig.axes if axis is not ax]
        assert len(cbar_axes) == 1
        assert cbar_axes[0].get_yticklabels()[0].get_fontsize() == pytest.approx(15.3)

    def test_groups_adds_side_axes(self, matplotlib_pyplot):
        """Passing groups appends the row/column group-color strip axes."""
        matrix = _correlation_matrix(6)
        groups = ["cortex"] * 3 + ["thalamus"] * 3
        fig, ax = plot_matrix(matrix, groups=groups, show_colorbar=False)
        assert len(fig.axes) == 3

    def test_no_groups_no_side_axes(self, matplotlib_pyplot):
        """Without groups, no extra axes are added beyond the colorbar."""
        fig, ax = plot_matrix(_correlation_matrix(6), show_colorbar=False)
        assert len(fig.axes) == 1


class TestPlotMatrixVisualRegression:
    """Visual regression tests for plot_matrix using pytest-mpl."""

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_matrix_default(self, matplotlib_pyplot):
        """Baseline test for default plot_matrix appearance."""
        fig, _ = plot_matrix(
            _correlation_matrix(6), labels=[f"r{i}" for i in range(6)]
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_matrix_lower_triangle_with_grid(self, matplotlib_pyplot):
        """Baseline test for lower-triangle display with a grid."""
        fig, _ = plot_matrix(
            _correlation_matrix(6),
            labels=[f"r{i}" for i in range(6)],
            tri="lower",
            grid="gray",
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_matrix_groups(self, matplotlib_pyplot):
        """Baseline test for the group-color-strip annotation."""
        groups = ["cortex"] * 3 + ["thalamus"] * 3
        fig, _ = plot_matrix(
            _correlation_matrix(6),
            labels=[f"r{i}" for i in range(6)],
            groups=groups,
            tri="diag",
        )
        return fig
