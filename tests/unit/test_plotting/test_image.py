"""Tests for image plotting functions.

These tests use real matplotlib with the Agg backend (non-interactive).
See conftest.py for the matplotlib_pyplot fixture setup.
"""

import numpy as np
import pytest
import xarray as xr

from confusius.plotting import VolumePlotter, plot_volume


class TestPlotVolume:
    """Tests for plot_volume function."""

    def test_invalid_slice_mode_raises(self, sample_3d_volume):
        """plot_volume raises ValueError for a slice_mode not in data.dims."""
        with pytest.raises(ValueError, match="slice_mode"):
            plot_volume(sample_3d_volume, slice_mode="t", slice_coords=[0.0])

    def test_non_3d_data_raises(self):
        """plot_volume raises ValueError for 4D data with no unitary dimensions."""
        data = xr.DataArray(
            np.zeros((5, 8, 10, 12)),
            dims=["time", "z", "y", "x"],
        )
        with pytest.raises(ValueError, match="3D"):
            plot_volume(data, slice_mode="z")

    def test_complex_data_converted_to_magnitude(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        """plot_volume converts complex-valued data to magnitude before plotting."""
        complex_data = sample_3d_volume * (1 + 1j)
        z_coord = complex_data.coords["z"].values[0]
        plotter = plot_volume(complex_data, slice_mode="z", slice_coords=[z_coord])

        plotted_values = plotter.axes[0, 0].collections[0].get_array().data
        assert np.all(plotted_values >= 0)

    @pytest.mark.parametrize("threshold_mode", ["lower", "upper"])
    def test_threshold_masks_correctly(
        self, sample_3d_volume, matplotlib_pyplot, threshold_mode
    ):
        """plot_volume masks data correctly based on threshold_mode.

        For 'lower': masks |data| < threshold.
        For 'upper': masks |data| > threshold.
        """
        threshold = 0.5
        z_coord = sample_3d_volume.coords["z"].values[0]
        plotter = plot_volume(
            sample_3d_volume,
            slice_mode="z",
            slice_coords=[z_coord],
            threshold=threshold,
            threshold_mode=threshold_mode,
        )

        ax = plotter.axes[0, 0]
        plotted_data = ax.collections[0].get_array()
        original_slice = sample_3d_volume.sel(z=z_coord, method="nearest").values

        abs_data = np.abs(original_slice)
        if threshold_mode == "lower":
            expected_mask = abs_data < threshold
        else:
            expected_mask = abs_data > threshold

        np.testing.assert_array_equal(plotted_data.mask, expected_mask)

    def test_explicit_vmin_vmax(self, sample_3d_volume, matplotlib_pyplot):
        """plot_volume passes explicit vmin and vmax to pcolormesh."""
        z_coord = sample_3d_volume.coords["z"].values[0]
        plotter = plot_volume(
            sample_3d_volume,
            slice_mode="z",
            slice_coords=[z_coord],
            vmin=-3.0,
            vmax=3.0,
        )

        collection = plotter.axes[0, 0].collections[0]
        assert collection.norm.vmin == pytest.approx(-3.0)
        assert collection.norm.vmax == pytest.approx(3.0)

    def test_colorbar_added_by_default(self, sample_3d_volume, matplotlib_pyplot):
        """plot_volume adds a colorbar when show_colorbar=True (default)."""
        z_coord = sample_3d_volume.coords["z"].values[0]
        plotter = plot_volume(sample_3d_volume, slice_mode="z", slice_coords=[z_coord])

        plot_axes = set(plotter.axes.ravel())
        extra_axes = [ax for ax in plotter.figure.axes if ax not in plot_axes]
        assert len(extra_axes) == 1

    def test_no_colorbar_when_disabled(self, sample_3d_volume, matplotlib_pyplot):
        """plot_volume skips colorbar when show_colorbar=False."""
        z_coord = sample_3d_volume.coords["z"].values[0]
        plotter = plot_volume(
            sample_3d_volume,
            slice_mode="z",
            slice_coords=[z_coord],
            show_colorbar=False,
        )

        plot_axes = set(plotter.axes.ravel())
        extra_axes = [ax for ax in plotter.figure.axes if ax not in plot_axes]
        assert len(extra_axes) == 0

    def test_cbar_label_is_set(self, sample_3d_volume, matplotlib_pyplot):
        """plot_volume sets the colorbar label when cbar_label is provided."""
        z_coord = sample_3d_volume.coords["z"].values[0]
        plotter = plot_volume(
            sample_3d_volume,
            slice_mode="z",
            slice_coords=[z_coord],
            cbar_label="Power (dB)",
        )

        plot_axes = set(plotter.axes.ravel())
        extra_axes = [ax for ax in plotter.figure.axes if ax not in plot_axes]
        assert len(extra_axes) == 1
        assert extra_axes[0].get_ylabel() == "Power (dB)"

    def test_existing_axes_used(self, sample_3d_volume, matplotlib_pyplot):
        """plot_volume uses provided axes without creating new ones."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 1, squeeze=False)
        z_coord = sample_3d_volume.coords["z"].values[0]

        plotter = plot_volume(
            sample_3d_volume, slice_mode="z", slice_coords=[z_coord], axes=axes
        )

        assert plotter.axes is axes
        assert plotter.figure is fig

    def test_axes_count_mismatch_raises(self, sample_3d_volume, matplotlib_pyplot):
        """plot_volume raises ValueError when axes count doesn't match slices."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 1, squeeze=False)
        z_coords = sample_3d_volume.coords["z"].values[:3].tolist()

        with pytest.raises(ValueError, match="must match number of axes"):
            plot_volume(
                sample_3d_volume, slice_mode="z", slice_coords=z_coords, axes=axes
            )

    def test_unused_axes_hidden(self, sample_3d_volume, matplotlib_pyplot):
        """plot_volume hides axes beyond the number of slices."""
        z_coords = sample_3d_volume.coords["z"].values[:2].tolist()
        plotter = plot_volume(
            sample_3d_volume, slice_mode="z", slice_coords=z_coords, nrows=2, ncols=2
        )

        for ax in plotter.axes.ravel()[2:]:
            assert not ax.get_visible()

    def test_axis_limits_match_data_edges(self, sample_3d_volume, matplotlib_pyplot):
        """Axes limits exactly equal data edges — no matplotlib auto-margin."""
        z_coord = sample_3d_volume.coords["z"].values[0]
        plotter = plot_volume(sample_3d_volume, slice_mode="z", slice_coords=[z_coord])
        ax = plotter.axes[0, 0]

        x_centers = sample_3d_volume.coords["x"].values.astype(float)
        y_centers = sample_3d_volume.coords["y"].values.astype(float)
        dx = x_centers[1] - x_centers[0]
        dy = y_centers[1] - y_centers[0]

        assert ax.get_xlim() == pytest.approx(
            (x_centers[0] - dx / 2, x_centers[-1] + dx / 2)
        )
        # Upper origin: ylim is (y_max_edge, y_min_edge).
        assert ax.get_ylim() == pytest.approx(
            (y_centers[-1] + dy / 2, y_centers[0] - dy / 2)
        )

    def test_no_coordinate_arrays(self, matplotlib_pyplot):
        """plot_volume uses pixel-index edges when coordinate arrays are absent."""
        data = xr.DataArray(np.random.rand(3, 4, 5), dims=["z", "y", "x"])
        plotter = plot_volume(
            data, slice_mode="z", slice_coords=[0], show_colorbar=False
        )
        ax = plotter.axes[0, 0]

        assert ax.get_xlim() == pytest.approx((0.0, 5.0))
        assert ax.get_ylim() == pytest.approx((4.0, 0.0))

    def test_origin_lower_flips_ylim(self, sample_3d_volume, matplotlib_pyplot):
        """plot_volume with origin='lower' places y-origin at bottom."""
        z_coord = sample_3d_volume.coords["z"].values[0]
        plotter = plot_volume(
            sample_3d_volume,
            slice_mode="z",
            slice_coords=[z_coord],
            origin="lower",
            show_colorbar=False,
        )
        ax = plotter.axes[0, 0]
        y_centers = sample_3d_volume.coords["y"].values.astype(float)
        dy = y_centers[1] - y_centers[0]
        assert ax.get_ylim() == pytest.approx(
            (y_centers[0] - dy / 2, y_centers[-1] + dy / 2)
        )

    def test_existing_figure_used(self, sample_3d_volume, matplotlib_pyplot):
        """plot_volume uses the provided figure to create new axes inside it."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        z_coord = sample_3d_volume.coords["z"].values[0]
        plotter = plot_volume(
            sample_3d_volume,
            slice_mode="z",
            slice_coords=[z_coord],
            figure=fig,
            show_colorbar=False,
        )
        assert plotter.figure is fig
        assert plotter.axes is not None

    def test_4d_with_unitary_dim_squeezed(self, sample_3d_volume, matplotlib_pyplot):
        """plot_volume squeezes a unitary dimension before plotting."""
        data_4d = sample_3d_volume.expand_dims("time")
        z_coord = sample_3d_volume.coords["z"].values[0]
        plotter = plot_volume(
            data_4d, slice_mode="z", slice_coords=[z_coord], show_colorbar=False
        )
        assert plotter.axes is not None


class TestCentersToEdges:
    """Tests for _centers_to_edges helper function."""

    def test_single_element(self):
        """_centers_to_edges handles single-element array."""
        from confusius.plotting.image import _centers_to_edges

        centers = np.array([5.0])
        edges = _centers_to_edges(centers)
        np.testing.assert_array_almost_equal(edges, [4.5, 5.5])

    def test_uniform_spacing(self):
        """_centers_to_edges with uniform spacing."""
        from confusius.plotting.image import _centers_to_edges

        centers = np.array([0.0, 1.0, 2.0, 3.0])
        edges = _centers_to_edges(centers)
        np.testing.assert_array_almost_equal(edges, [-0.5, 0.5, 1.5, 2.5, 3.5])

    def test_non_uniform_spacing(self):
        """_centers_to_edges with non-uniform spacing."""
        from confusius.plotting.image import _centers_to_edges

        centers = np.array([0.0, 1.0, 3.0, 6.0])  # Spacing: 1, 2, 3
        edges = _centers_to_edges(centers)
        # Interior edges are midpoints
        expected = np.array([-0.5, 0.5, 2.0, 4.5, 7.5])
        np.testing.assert_array_almost_equal(edges, expected)


class TestVolumePlotterAddVolume:
    """Tests for VolumePlotter.add_volume method."""

    def test_overlay_lands_on_correct_axes(self, sample_3d_volume, matplotlib_pyplot):
        """add_volume overlays only on axes whose coordinates match."""
        plotter = plot_volume(sample_3d_volume, slice_mode="z", show_colorbar=False)

        subset = sample_3d_volume.sel(z=sample_3d_volume.coords["z"].values[:2])
        plotter.add_volume(subset, cmap="hot", alpha=0.5, show_colorbar=False)

        axes_flat = plotter.axes.ravel()
        assert len(axes_flat[0].collections) == 2
        assert len(axes_flat[1].collections) == 2
        assert len(axes_flat[2].collections) == 1
        assert len(axes_flat[3].collections) == 1

    def test_add_volume_warns_on_missing_coords(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        """add_volume warns when some coordinates don't match."""
        plotter = plot_volume(
            sample_3d_volume,
            slice_mode="z",
            slice_coords=[sample_3d_volume.coords["z"].values[2]],
        )

        with pytest.warns(UserWarning, match="Could not find matching axes"):
            plotter.add_volume(
                sample_3d_volume.sel(z=[0.0, 0.1, 0.3], method="nearest"),
                cmap="viridis",
            )


class TestVolumePlotterUtilities:
    """Tests for VolumePlotter utility methods."""

    def test_savefig_creates_file(self, sample_3d_volume, matplotlib_pyplot, tmp_path):
        """savefig creates a non-empty file."""
        plotter = plot_volume(sample_3d_volume, slice_mode="z")
        output_file = tmp_path / "test_output.png"
        plotter.savefig(str(output_file))

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_savefig_before_figure_raises(self, tmp_path):
        """savefig raises RuntimeError when called before any plot."""
        plotter = VolumePlotter()
        with pytest.raises(RuntimeError):
            plotter.savefig(str(tmp_path / "output.png"))

    def test_close_figure(self, sample_3d_volume, matplotlib_pyplot):
        """close releases the figure and resets state."""
        import matplotlib.pyplot as plt

        plotter = plot_volume(sample_3d_volume, slice_mode="z")
        fig_num = plotter.figure.number

        plotter.close()

        assert plotter.figure is None
        assert plotter.axes is None
        assert fig_num not in plt.get_fignums()

    def test_close_is_idempotent(self, sample_3d_volume, matplotlib_pyplot):
        """close can be called multiple times without error."""
        plotter = plot_volume(sample_3d_volume, slice_mode="z")
        plotter.close()
        plotter.close()

        assert plotter.figure is None


# Image comparison tests with pytest-mpl
# These generate baseline images for visual regression testing


def _create_deterministic_volume():
    """Create deterministic 3D volume for visual regression tests.

    Uses fixed seed to ensure reproducible baseline images.
    """
    rng = np.random.default_rng(42)
    shape = (4, 6, 8)
    data = rng.random(shape)
    return xr.DataArray(
        data,
        dims=["z", "y", "x"],
        coords={
            "z": xr.DataArray(
                np.arange(4) * 0.1,
                dims=["z"],
                attrs={"units": "mm"},
            ),
            "y": xr.DataArray(
                np.arange(6) * 0.05,
                dims=["y"],
                attrs={"units": "mm"},
            ),
            "x": xr.DataArray(
                np.arange(8) * 0.05,
                dims=["x"],
                attrs={"units": "mm"},
            ),
        },
        attrs={
            "long_name": "Intensity",
            "units": "a.u.",
        },
    )


class TestPlotVolumeVisualRegression:
    """Visual regression tests using pytest-mpl.

    These tests generate baseline images that can be used to detect
    visual regressions in the plotting code.

    To generate/update baselines:
        pytest --mpl-generate-path=tests/unit/test_plotting/baseline
    """

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_volume_default(self, matplotlib_pyplot):
        """Baseline test for default plot_volume appearance (black background)."""
        volume = _create_deterministic_volume()
        plotter = plot_volume(volume, slice_mode="z")
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_volume_white_bg(self, matplotlib_pyplot):
        """Baseline test for white background."""
        volume = _create_deterministic_volume()
        plotter = plot_volume(volume, slice_mode="z", black_bg=False)
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_volume_single_slice(self, matplotlib_pyplot):
        """Baseline test for single slice."""
        volume = _create_deterministic_volume()
        z_coord = volume.coords["z"].values[0]
        plotter = plot_volume(volume, slice_mode="z", slice_coords=[z_coord])
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_volume_custom_grid(self, matplotlib_pyplot):
        """Baseline test for custom grid layout (1 row, 4 columns)."""
        volume = _create_deterministic_volume()
        plotter = plot_volume(volume, slice_mode="z", nrows=1, ncols=4)
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_volume_overlay(self, matplotlib_pyplot):
        """Baseline test for overlaying two volumes with transparency."""
        volume = _create_deterministic_volume()
        plotter = plot_volume(volume, slice_mode="z")

        subset_coords = volume.coords["z"].values[[0, 3]].tolist()
        subset_data = volume.sel(z=subset_coords)
        plotter.add_volume(subset_data, cmap="hot", alpha=0.5)

        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_volume_threshold(self, matplotlib_pyplot):
        """Baseline test for thresholding visualization."""
        volume = _create_deterministic_volume()
        plotter = plot_volume(
            volume,
            slice_mode="z",
            threshold=0.5,
            threshold_mode="lower",
        )
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_volume_no_colorbar(self, matplotlib_pyplot):
        """Baseline test without colorbar."""
        volume = _create_deterministic_volume()
        plotter = plot_volume(volume, slice_mode="z", show_colorbar=False)
        return plotter.figure
