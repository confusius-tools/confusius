"""Tests for confusius.plotting.matrix.

See conftest.py for the matplotlib_pyplot fixture setup.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.figure import Figure

from confusius.plotting import plot_contrast_matrix, plot_design_matrix, plot_matrix


def _design_matrix(
    n_columns: int = 4,
    n_volumes: int = 20,
    names: list[str] | None = None,
    volume_times: np.ndarray | None = None,
) -> pd.DataFrame:
    columns = names if names is not None else [f"reg_{i}" for i in range(n_columns)]
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.standard_normal((n_volumes, len(columns))),
        columns=columns,
        index=volume_times,
    )


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

    def test_unknown_triangle_raises(self, matplotlib_pyplot):
        """An unsupported triangle mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown triangle mode"):
            plot_matrix(_correlation_matrix(4), triangle="bogus")  # ty: ignore[invalid-argument-type]


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
        fig, ax = plot_matrix(
            _correlation_matrix(4), labels=["", "", "", ""], groups=groups
        )
        assert len(fig.axes) > 1

    def test_diverging_matrix_gets_symmetric_range(self, matplotlib_pyplot):
        """A matrix with both signs gets a symmetric [-m, m] range and coolwarm."""
        matrix = np.array([[1.0, -0.8], [-0.8, 1.0]])
        _, ax = plot_matrix(matrix)
        image = ax.collections[0]
        assert image.norm.vmin == pytest.approx(-1.0)
        assert image.norm.vmax == pytest.approx(1.0)
        assert image.cmap.name == "coolwarm"

    def test_non_negative_matrix_gets_sequential_range(self, matplotlib_pyplot):
        """A non-negative matrix gets a [0, vmax] range and a sequential cmap."""
        matrix = np.array([[0.2, 0.8], [0.8, 0.2]])
        _, ax = plot_matrix(matrix)
        image = ax.collections[0]
        assert image.norm.vmin == pytest.approx(0.0)
        assert image.norm.vmax == pytest.approx(0.8)
        assert image.cmap.name == "viridis"

    def test_non_positive_matrix_gets_sequential_range(self, matplotlib_pyplot):
        """A non-positive matrix gets a [vmin, 0] range and a reversed sequential cmap."""
        matrix = np.array([[-0.2, -0.8], [-0.8, -0.2]])
        _, ax = plot_matrix(matrix)
        image = ax.collections[0]
        assert image.norm.vmin == pytest.approx(-0.8)
        assert image.norm.vmax == pytest.approx(0.0)
        assert image.cmap.name == "viridis_r"

    def test_auto_range_false_uses_raw_bounds(self, matplotlib_pyplot):
        """auto_range=False uses the raw min/max with no zero-anchoring."""
        matrix = np.array([[0.2, 0.8], [0.8, 0.2]])
        _, ax = plot_matrix(matrix, auto_range=False)
        image = ax.collections[0]
        assert image.norm.vmin == pytest.approx(0.2)
        assert image.norm.vmax == pytest.approx(0.8)
        assert image.cmap.name == "coolwarm"

    def test_explicit_cmap_overrides_auto_range(self, matplotlib_pyplot):
        """An explicit cmap is used as-is regardless of auto_range."""
        matrix = np.array([[0.2, 0.8], [0.8, 0.2]])
        _, ax = plot_matrix(matrix, cmap="magma")
        assert ax.collections[0].cmap.name == "magma"

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

        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig.canvas.draw()
        assert isinstance(fig.canvas, FigureCanvasAgg)
        renderer = fig.canvas.get_renderer()
        label_bbox = ax.yaxis.get_tightbbox(renderer)
        assert label_bbox is not None
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
        _, ax = plot_matrix(_correlation_matrix(4), triangle=tri, grid="gray")
        assert len(ax.lines) > 0

    def test_grid_true_uses_foreground_color(self, matplotlib_pyplot):
        """grid=True draws the grid in the resolved foreground color."""
        _, ax_light = plot_matrix(_correlation_matrix(4), grid=True)
        assert ax_light.lines
        assert all(line.get_color() == "black" for line in ax_light.lines)
        _, ax_dark = plot_matrix(_correlation_matrix(4), grid=True, bg_color="black")
        assert all(line.get_color() == "white" for line in ax_dark.lines)

    def test_group_pad_works_on_non_agg_backend(self, matplotlib_pyplot):
        """Reserving room for the group strip must not depend on Agg-only APIs.

        Measuring the y tick label width via `canvas.get_renderer()` crashes on
        non-Agg backends (e.g. pdf/svg), which do not define that method; the group
        strip pad path must survive there.
        """
        original_backend = matplotlib_pyplot.get_backend()
        matplotlib_pyplot.switch_backend("pdf")
        try:
            fig, _ = plot_matrix(
                _correlation_matrix(4),
                labels=["a", "b", "c", "d"],
                groups=["x", "x", "y", "y"],
            )
            assert len(fig.axes) > 1
        finally:
            matplotlib_pyplot.switch_backend(original_backend)

    @pytest.mark.parametrize(
        ("tri", "masked_upper"),
        [
            ("lower", True),
            ("diag_lower", True),
            ("diag_upper", False),
            ("upper", False),
        ],
    )
    def test_tri_masks_expected_side(self, matplotlib_pyplot, tri, masked_upper):
        """lower/diag_lower mask the strict upper triangle and vice versa."""
        _, ax = plot_matrix(_correlation_matrix(4), triangle=tri)
        image_mask = np.asarray(
            np.ma.masked_array(ax.collections[0].get_array()).mask,
            dtype=bool,
        )
        assert bool(image_mask[0, 1]) == masked_upper
        assert bool(image_mask[1, 0]) == (not masked_upper)

    @pytest.mark.parametrize("tri", ["lower", "upper"])
    def test_tri_strict_excludes_diagonal(self, matplotlib_pyplot, tri):
        """lower/upper mask the diagonal; diag_lower/diag_upper keep it."""
        _, ax = plot_matrix(_correlation_matrix(4), triangle=tri)
        image_mask = np.asarray(
            np.ma.masked_array(ax.collections[0].get_array()).mask,
            dtype=bool,
        )
        assert bool(image_mask[0, 0])


class TestPlotMatrixVisualRegression:
    """Visual regression tests for plot_matrix using pytest-mpl."""

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_matrix_default(self, matplotlib_pyplot):
        """Baseline test for default plot_matrix appearance."""
        fig, _ = plot_matrix(_correlation_matrix(6), labels=[f"r{i}" for i in range(6)])
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
            triangle="lower",
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
            triangle="diag_lower",
        )
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_matrix_lower_upper_overlay(self, matplotlib_pyplot):
        """Baseline test for overlaying two matrices via triangle='diag_lower'/'upper'."""
        labels = [f"r{i}" for i in range(6)]
        rng = np.random.default_rng(1)
        pvalues = rng.uniform(size=(6, 6))
        fig, ax = plot_matrix(
            _correlation_matrix(6),
            labels=labels,
            triangle="diag_lower",
            cbar_label="corr",
        )
        plot_matrix(
            pvalues,
            labels=labels,
            triangle="upper",
            ax=ax,
            cmap="viridis",
            vmin=0,
            vmax=1,
            show_colorbar=False,
        )
        return fig


class TestPlotDesignMatrix:
    """Behaviour of plot_design_matrix."""

    def test_non_2d_input_raises(self, matplotlib_pyplot):
        """A design matrix that is not 2D raises ValueError."""
        with pytest.raises(ValueError, match="2D design matrix"):
            plot_design_matrix(np.zeros(5))

    def test_dataframe_columns_become_xticklabels(self, matplotlib_pyplot):
        """A DataFrame's column names label the regressor ticks, in order."""
        names = ["active", "drift", "constant"]
        _, ax = plot_design_matrix(_design_matrix(names=names))
        assert [t.get_text() for t in ax.get_xticklabels()] == names

    def test_ndarray_input_gets_one_tick_per_column(self, matplotlib_pyplot):
        """A bare array (no column names) still gets one x-tick per regressor."""
        _, ax = plot_design_matrix(np.zeros((20, 5)))
        assert ax.get_xticks().size == 5

    def test_width_scales_with_column_count(self, matplotlib_pyplot):
        """The auto-sized figure width grows with the number of regressors."""
        narrow, _ = plot_design_matrix(_design_matrix(n_columns=12), column_width=1.0)
        assert isinstance(narrow, Figure)
        # 12 regressors * 1.0 in each == 12 in.
        assert narrow.get_figwidth() == pytest.approx(12.0)

    def test_width_floored_for_few_columns(self, matplotlib_pyplot):
        """A few-regressor design is floored at 4 inches, not a sliver."""
        fig, _ = plot_design_matrix(_design_matrix(n_columns=2), column_width=1.0)
        assert isinstance(fig, Figure)
        assert fig.get_figwidth() == pytest.approx(4.0)

    def test_time_index_labels_y_axis_with_seconds(self, matplotlib_pyplot):
        """index_yaxis=True relabels the y-axis with the DataFrame's acquisition times."""
        times = np.arange(20) * 2.0  # 2 s TR.
        _, ax = plot_design_matrix(_design_matrix(volume_times=times), index_yaxis=True)
        assert ax.get_ylabel() == "Time (s)"
        # Row positions map to acquisition time, including auto-placed half-row ticks.
        assert ax.yaxis.get_major_formatter()(3, None) == "6"
        assert ax.yaxis.get_major_formatter()(2.5, None) == "5"

    def test_rangeindex_falls_back_to_volume_index(self, matplotlib_pyplot):
        """A trivial RangeIndex has no time index, so index_yaxis=True still shows volumes."""
        _, ax = plot_design_matrix(_design_matrix(), index_yaxis=True)
        assert ax.get_ylabel() == "Volume"

    def test_index_yaxis_false_forces_volume_axis(self, matplotlib_pyplot):
        """index_yaxis=False shows the volume index even when a time index exists."""
        times = np.arange(20) * 2.0
        _, ax = plot_design_matrix(
            _design_matrix(volume_times=times), index_yaxis=False
        )
        assert ax.get_ylabel() == "Volume"

    def test_yaxis_label_overrides_default(self, matplotlib_pyplot):
        """yaxis_label replaces the automatic "Time (s)"/"Volume" label."""
        times = np.arange(20) * 2.0
        _, ax = plot_design_matrix(
            _design_matrix(volume_times=times), index_yaxis=True, yaxis_label="Frame"
        )
        assert ax.get_ylabel() == "Frame"

    def test_reuses_provided_axes_and_sets_title(self, matplotlib_pyplot):
        """Draws on a caller-provided Axes and sets the title on it."""
        fig, ax = matplotlib_pyplot.subplots()
        returned_fig, returned_ax = plot_design_matrix(
            _design_matrix(), title="design", ax=ax
        )
        assert returned_ax is ax
        assert returned_fig is fig
        assert ax.get_title() == "design"


class TestPlotContrastMatrix:
    """Behaviour of plot_contrast_matrix."""

    def test_string_expression_resolves_against_columns(self, matplotlib_pyplot):
        """A string contrast resolves to weights over the DataFrame's columns."""
        design = _design_matrix(names=["stim_A", "stim_B", "drift", "constant"])
        _, ax = plot_contrast_matrix("stim_A - stim_B", design)
        np.testing.assert_array_equal(ax.images[0].get_array(), [[1.0, -1.0, 0.0, 0.0]])
        assert [t.get_text() for t in ax.get_xticklabels()] == list(design.columns)

    def test_gray_cmap_with_symmetric_range_around_zero(self, matplotlib_pyplot):
        """Weights use a gray cmap on a symmetric [-max|w|, max|w|] range."""
        _, ax = plot_contrast_matrix(np.array([2.0, -1.0, 0.0, 0.0]), _design_matrix())
        image = ax.images[0]
        assert image.cmap.name == "gray"
        assert image.norm.vmin == pytest.approx(-2.0)
        assert image.norm.vmax == pytest.approx(2.0)

    def test_f_contrast_matrix_gets_one_row_each(self, matplotlib_pyplot):
        """A 2D F-contrast draws one heatmap row per contrast, with a row axis."""
        f_contrast = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
        _, ax = plot_contrast_matrix(f_contrast, _design_matrix())
        array = ax.images[0].get_array()
        assert array is not None
        assert array.shape == (2, 4)
        assert ax.get_yticks().size == 2

    def test_short_vector_is_zero_padded_to_design_width(self, matplotlib_pyplot):
        """A contrast narrower than the design is zero-padded on the right."""
        _, ax = plot_contrast_matrix(np.array([1.0, -1.0]), _design_matrix(n_columns=4))
        np.testing.assert_array_equal(ax.images[0].get_array(), [[1.0, -1.0, 0.0, 0.0]])

    def test_string_contrast_with_bare_array_raises(self, matplotlib_pyplot):
        """A string contrast needs named columns; a bare array design raises."""
        with pytest.raises(ValueError, match="named columns"):
            plot_contrast_matrix("a - b", np.zeros((20, 4)))

    def test_colorbar_sits_beside_the_strip(self, matplotlib_pyplot):
        """show_colorbar places the colorbar next to the strip, not on top of it."""
        fig, ax = plot_contrast_matrix(
            np.array([1.0, -1.0, 0.0, 0.0]), _design_matrix(), show_colorbar=True
        )
        cbar_ax = next(axis for axis in fig.axes if axis is not ax)
        fig.canvas.draw()  # constrained layout finalizes positions only on draw.
        assert cbar_ax.get_position().x0 >= ax.get_position().x1

    def test_reuses_provided_axes_and_sets_title(self, matplotlib_pyplot):
        """Draws on a caller-provided Axes and sets the title on it."""
        fig, ax = matplotlib_pyplot.subplots()
        returned_fig, returned_ax = plot_contrast_matrix(
            np.array([1.0, -1.0, 0.0, 0.0]), _design_matrix(), title="A vs B", ax=ax
        )
        assert returned_ax is ax
        assert returned_fig is fig
        assert ax.get_title() == "A vs B"

    def test_all_zero_contrast_uses_symmetric_unit_range(self, matplotlib_pyplot):
        """An all-zero contrast falls back to a symmetric [-1, 1] range, not vmin==vmax."""
        _, ax = plot_contrast_matrix(np.zeros(4), _design_matrix())
        image = ax.images[0]
        assert image.norm.vmin == pytest.approx(-1.0)
        assert image.norm.vmax == pytest.approx(1.0)
