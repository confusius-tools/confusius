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

    def test_dataarray_without_coordinate_has_no_default_labels(
        self, matplotlib_pyplot
    ):
        """A DataArray with no coordinate on its first dim falls back to no labels."""
        matrix = xr.DataArray(_correlation_matrix(3), dims=["region", "region2"])
        _, ax = plot_matrix(matrix)
        assert ax.get_xticks().size == 0

    def test_empty_labels_list_is_treated_as_no_labels(self, matplotlib_pyplot):
        """An empty labels list disables ticks, same as labels=False."""
        _, ax = plot_matrix(_correlation_matrix(3), labels=[])
        assert ax.get_xticks().size == 0

    def test_blank_labels_with_groups_do_not_crash(self, matplotlib_pyplot):
        """groups + all-blank labels: the y tick labels have no rendered extent.

        Regression case for _y_tick_label_width_inches, which must fall back to a
        default pad when get_tightbbox returns None (blank labels have no bbox).
        """
        groups = ["cortex"] * 2 + ["thalamus"] * 2
        fig, ax = plot_matrix(_correlation_matrix(4), labels=["", "", "", ""], groups=groups)
        assert len(fig.axes) > 1

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

    def test_labels_survive_alongside_groups(self, matplotlib_pyplot):
        """Row/column labels stay visible when groups are also passed.

        Regression test: the group-strip axes share their perpendicular axis
        (sharex/sharey) with the main axes so that they line up with its rows and
        columns; clearing that shared axis's ticks previously wiped out the main
        axes' own tick labels too (see _draw_group_bar).
        """
        labels = ["a", "b", "c", "d", "e", "f"]
        groups = ["cortex"] * 3 + ["thalamus"] * 3
        _, ax = plot_matrix(_correlation_matrix(6), labels=labels, groups=groups)
        assert [t.get_text() for t in ax.get_xticklabels()] == labels
        assert [t.get_text() for t in ax.get_yticklabels()] == labels

    def test_long_labels_do_not_overlap_group_strip(self, matplotlib_pyplot):
        """The row group-strip pad scales with label length (regression).

        A fixed pad only fits labels up to some length: past that, the row strip's
        opaque rectangles visually cover the y tick labels even though the label text
        objects themselves are set correctly (see _y_tick_label_width_inches).
        """
        labels = [f"very_long_region_name_{i}" for i in range(6)]
        groups = ["cortex"] * 3 + ["thalamus"] * 3
        fig, ax = plot_matrix(_correlation_matrix(6), labels=labels, groups=groups)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_bbox = ax.yaxis.get_tightbbox(renderer)
        row_ax = min(
            (axis for axis in fig.axes if axis is not ax),
            key=lambda axis: axis.get_position().x0,
        )
        row_bbox = row_ax.get_window_extent(renderer)
        assert row_bbox.x1 <= label_bbox.x0

    def test_show_group_labels_false_omits_annotations(self, matplotlib_pyplot):
        """show_group_labels=False draws the color strips without text annotations."""
        groups = ["cortex"] * 3 + ["thalamus"] * 3
        fig, ax = plot_matrix(
            _correlation_matrix(6),
            groups=groups,
            show_group_labels=False,
            show_colorbar=False,
        )
        group_axes = [axis for axis in fig.axes if axis is not ax]
        assert all(len(axis.texts) == 0 for axis in group_axes)

    def test_reuses_provided_axes(self, matplotlib_pyplot):
        """plot_matrix draws on a caller-provided Axes instead of creating a figure."""
        fig, ax = matplotlib_pyplot.subplots()
        returned_figure, returned_ax = plot_matrix(_correlation_matrix(4), ax=ax)
        assert returned_ax is ax
        assert returned_figure is fig

    def test_cbar_label_is_set(self, matplotlib_pyplot):
        """cbar_label sets the colorbar's axis label."""
        fig, ax = plot_matrix(_correlation_matrix(4), cbar_label="correlation")
        cbar_axes = [axis for axis in fig.axes if axis is not ax]
        assert cbar_axes[0].yaxis.label.get_text() == "correlation"

    @pytest.mark.parametrize(
        "tri", ["full", "lower", "diag_lower", "diag_upper", "upper"]
    )
    def test_grid_draws_lines_for_every_tri_mode(self, matplotlib_pyplot, tri):
        """grid draws separator lines regardless of the triangle mode."""
        _, ax = plot_matrix(_correlation_matrix(4), tri=tri, grid="gray")
        assert len(ax.lines) > 0

    @pytest.mark.parametrize(
        ("tri", "masked_upper"),
        [("lower", True), ("diag_lower", True), ("diag_upper", False), ("upper", False)],
    )
    def test_tri_masks_expected_side(self, matplotlib_pyplot, tri, masked_upper):
        """lower/diag_lower mask the strict upper triangle and vice versa."""
        _, ax = plot_matrix(_correlation_matrix(4), tri=tri)
        image_mask = ax.images[0].get_array().mask
        assert bool(image_mask[0, 1]) == masked_upper
        assert bool(image_mask[1, 0]) == (not masked_upper)

    @pytest.mark.parametrize("tri", ["lower", "upper"])
    def test_tri_strict_excludes_diagonal(self, matplotlib_pyplot, tri):
        """lower/upper mask the diagonal; diag_lower/diag_upper keep it."""
        _, ax = plot_matrix(_correlation_matrix(4), tri=tri)
        assert bool(ax.images[0].get_array().mask[0, 0])


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
            tri="diag_lower",
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_matrix_lower_upper_overlay(self, matplotlib_pyplot):
        """Baseline test for overlaying two matrices via tri='diag_lower'/'upper'."""
        labels = [f"r{i}" for i in range(6)]
        rng = np.random.default_rng(1)
        pvalues = rng.uniform(size=(6, 6))
        fig, ax = plot_matrix(
            _correlation_matrix(6), labels=labels, tri="diag_lower", cbar_label="corr"
        )
        plot_matrix(
            pvalues,
            labels=labels,
            tri="upper",
            ax=ax,
            cmap="viridis",
            vmin=0,
            vmax=1,
            show_colorbar=False,
        )
        return fig
