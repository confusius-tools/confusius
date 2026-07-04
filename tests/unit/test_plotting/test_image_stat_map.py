"""Tests for the top-level plot_stat_map helper."""

import numpy as np
import pytest
import xarray as xr

from confusius.plotting import VolumePlotter, plot_stat_map


def _signed_stat_map(template: xr.DataArray) -> xr.DataArray:
    """Return a stat map on `template`'s grid with known signed values.

    Values range linearly from -10 to 10 (exact endpoints), so the default `vmin`/
    `vmax` (the data's actual min/max) are deterministic.
    """
    values = np.linspace(-10.0, 10.0, template.size).reshape(template.shape)
    return xr.DataArray(
        values,
        name="t_stat",
        dims=template.dims,
        coords={d: template.coords[d] for d in template.dims if d in template.coords},
    )


def _nonneg_stat_map(template: xr.DataArray) -> xr.DataArray:
    """Return a stat map on `template`'s grid with non-negative values (e.g. R²)."""
    values = np.linspace(0.0, 10.0, template.size).reshape(template.shape)
    return xr.DataArray(
        values,
        name="r2",
        dims=template.dims,
        coords={d: template.coords[d] for d in template.dims if d in template.coords},
    )


def _create_deterministic_bg_and_stat_map() -> tuple[xr.DataArray, xr.DataArray]:
    """Deterministic (bg_volume, stat_map) pair for visual regression baselines."""
    rng = np.random.default_rng(42)
    shape = (4, 6, 8)
    coords = {
        "z": xr.DataArray(np.arange(4) * 0.1, dims=["z"], attrs={"units": "mm"}),
        "y": xr.DataArray(np.arange(6) * 0.05, dims=["y"], attrs={"units": "mm"}),
        "x": xr.DataArray(np.arange(8) * 0.05, dims=["x"], attrs={"units": "mm"}),
    }
    bg_volume = xr.DataArray(
        rng.random(shape), dims=["z", "y", "x"], coords=coords, name="power_doppler"
    )
    stat_map = xr.DataArray(
        np.linspace(-10.0, 10.0, np.prod(shape)).reshape(shape),
        dims=["z", "y", "x"],
        coords=coords,
        name="t_stat",
    )
    return bg_volume, stat_map


class TestPlotStatMapVisualRegression:
    """Visual regression tests using pytest-mpl.

    These tests generate baseline images that can be used to detect visual
    regressions in the plotting code.

    To generate/update baselines:
        pytest --mpl-generate-path=tests/unit/test_plotting/baseline
    """

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_stat_map_default(self, matplotlib_pyplot):
        """Baseline test for default plot_stat_map appearance (gray bg + coolwarm overlay)."""
        bg_volume, stat_map = _create_deterministic_bg_and_stat_map()
        plotter = plot_stat_map(stat_map, bg_volume=bg_volume, slice_mode="z")
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_stat_map_threshold(self, matplotlib_pyplot):
        """Baseline test for thresholding subthreshold voxels transparent."""
        bg_volume, stat_map = _create_deterministic_bg_and_stat_map()
        plotter = plot_stat_map(
            stat_map,
            bg_volume=bg_volume,
            slice_mode="z",
            threshold=5.0,
            threshold_mode="lower",
        )
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_stat_map_alpha_blend(self, matplotlib_pyplot):
        """Baseline test for blending the overlay with the background via alpha."""
        bg_volume, stat_map = _create_deterministic_bg_and_stat_map()
        plotter = plot_stat_map(
            stat_map, bg_volume=bg_volume, slice_mode="z", alpha=0.5
        )
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_stat_map_non_diverging(self, matplotlib_pyplot):
        """Baseline test for a non-diverging statistic (auto-detected: sequential + viridis)."""
        bg_volume, stat_map = _create_deterministic_bg_and_stat_map()
        plotter = plot_stat_map(np.abs(stat_map), bg_volume=bg_volume, slice_mode="z")
        return plotter.figure


class TestPlotStatMap:
    def test_returns_volume_plotter_with_one_panel_per_slice(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        stat_map = _signed_stat_map(sample_3d_volume)
        plotter = plot_stat_map(stat_map, bg_volume=sample_3d_volume, slice_mode="z")
        assert isinstance(plotter, VolumePlotter)
        rendered = [ax for ax in plotter.axes.ravel() if ax.collections]
        assert len(rendered) == sample_3d_volume.sizes["z"]
        # Background + overlay were both drawn on every panel.
        assert all(len(ax.collections) == 2 for ax in rendered)

    def test_forwards_slice_mode(self, sample_3d_volume, matplotlib_pyplot):
        stat_map = _signed_stat_map(sample_3d_volume)
        plotter = plot_stat_map(stat_map, bg_volume=sample_3d_volume, slice_mode="y")
        assert plotter.slice_mode == "y"
        rendered = [ax for ax in plotter.axes.ravel() if ax.collections]
        assert len(rendered) == sample_3d_volume.sizes["y"]

    def test_overlay_is_fully_opaque_by_default(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        stat_map = _signed_stat_map(sample_3d_volume)
        plotter = plot_stat_map(stat_map, bg_volume=sample_3d_volume, slice_mode="z")
        overlay = plotter.axes.ravel()[0].collections[-1]
        assert overlay.get_alpha() == 1.0

    def test_explicit_alpha_blends_overlay_with_background(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        stat_map = _signed_stat_map(sample_3d_volume)
        plotter = plot_stat_map(
            stat_map, bg_volume=sample_3d_volume, slice_mode="z", alpha=0.5
        )
        overlay = plotter.axes.ravel()[0].collections[-1]
        assert overlay.get_alpha() == 0.5

    def test_overlay_uses_coolwarm_by_default_for_signed_data(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        stat_map = _signed_stat_map(sample_3d_volume)
        plotter = plot_stat_map(stat_map, bg_volume=sample_3d_volume, slice_mode="z")
        overlay = plotter.axes.ravel()[0].collections[-1]
        assert overlay.cmap.name.startswith("coolwarm")

    def test_default_bounds_are_symmetric_min_max_for_signed_data(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        stat_map = _signed_stat_map(sample_3d_volume)
        plotter = plot_stat_map(stat_map, bg_volume=sample_3d_volume, slice_mode="z")
        norm = plotter.axes.ravel()[0].collections[-1].norm
        assert norm.vmax == 10.0
        assert norm.vmin == -10.0

    def test_explicit_vmin_and_vmax_cap_the_symmetric_range(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        stat_map = _signed_stat_map(sample_3d_volume)
        plotter = plot_stat_map(
            stat_map, bg_volume=sample_3d_volume, slice_mode="z", vmin=-5.0, vmax=5.0
        )
        norm = plotter.axes.ravel()[0].collections[-1].norm
        assert norm.vmax == 5.0
        assert norm.vmin == -5.0

    def test_vmax_alone_does_not_cap_the_range_when_data_min_is_larger(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        """vmin defaults to the data's actual min when not given, so a lone vmax
        smaller than |data min| does not shrink the symmetric range."""
        stat_map = _signed_stat_map(sample_3d_volume)  # min=-10, max=10
        plotter = plot_stat_map(
            stat_map, bg_volume=sample_3d_volume, slice_mode="z", vmax=5.0
        )
        norm = plotter.axes.ravel()[0].collections[-1].norm
        assert norm.vmax == 10.0
        assert norm.vmin == -10.0

    def test_auto_range_uses_sequential_range_and_viridis_for_nonneg_data(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        stat_map = _nonneg_stat_map(sample_3d_volume)  # min=0, max=10
        plotter = plot_stat_map(stat_map, bg_volume=sample_3d_volume, slice_mode="z")
        overlay = plotter.axes.ravel()[0].collections[-1]
        assert overlay.cmap.name.startswith("viridis")
        assert overlay.norm.vmin == 0.0
        assert overlay.norm.vmax == 10.0

    def test_auto_range_uses_sequential_range_and_viridis_r_for_nonpos_data(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        stat_map = -_nonneg_stat_map(sample_3d_volume)  # min=-10, max=0
        plotter = plot_stat_map(stat_map, bg_volume=sample_3d_volume, slice_mode="z")
        overlay = plotter.axes.ravel()[0].collections[-1]
        assert overlay.cmap.name.startswith("viridis_r")
        assert overlay.norm.vmin == -10.0
        assert overlay.norm.vmax == 0.0

    def test_auto_range_false_disables_zero_anchoring(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        stat_map = _signed_stat_map(sample_3d_volume)
        plotter = plot_stat_map(
            stat_map,
            bg_volume=sample_3d_volume,
            slice_mode="z",
            vmin=-2.0,
            vmax=5.0,
            auto_range=False,
        )
        overlay = plotter.axes.ravel()[0].collections[-1]
        assert overlay.cmap.name.startswith("coolwarm")
        assert overlay.norm.vmin == -2.0
        assert overlay.norm.vmax == 5.0

    def test_explicit_cmap_is_used_as_is_regardless_of_sign(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        stat_map = _nonneg_stat_map(sample_3d_volume)
        plotter = plot_stat_map(
            stat_map, bg_volume=sample_3d_volume, slice_mode="z", cmap="hot"
        )
        overlay = plotter.axes.ravel()[0].collections[-1]
        assert overlay.cmap.name.startswith("hot")

    def test_threshold_masks_overlay(self, sample_3d_volume, matplotlib_pyplot):
        stat_map = _signed_stat_map(sample_3d_volume)
        plotter = plot_stat_map(
            stat_map,
            bg_volume=sample_3d_volume,
            slice_mode="z",
            threshold=9.0,
            threshold_mode="lower",
        )
        overlay = plotter.axes.ravel()[0].collections[-1]
        arr = overlay.get_array()
        assert np.ma.is_masked(arr)

    def test_bg_kwargs_forwarded_to_background_layer(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        stat_map = _signed_stat_map(sample_3d_volume)
        plotter = plot_stat_map(
            stat_map,
            bg_volume=sample_3d_volume,
            slice_mode="z",
            bg_kwargs={"cmap": "hot", "vmin": 0.0, "vmax": 1.0},
        )
        background = plotter.axes.ravel()[0].collections[0]
        assert background.cmap.name.startswith("hot")
        assert background.norm.vmin == 0.0
        assert background.norm.vmax == 1.0

    def test_dataarray_alpha_validated_against_stat_map_not_bg_volume(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        """alpha (a DataArray) only needs to match stat_map; bg_volume may differ."""
        stat_map = _signed_stat_map(sample_3d_volume)
        alpha = xr.ones_like(stat_map) * 0.5
        # bg_volume deliberately has a coarser, non-matching x grid.
        bg_volume = sample_3d_volume.isel(x=slice(0, 4))

        plotter = plot_stat_map(
            stat_map, bg_volume=bg_volume, slice_mode="z", alpha=alpha
        )
        overlay = plotter.axes.ravel()[0].collections[-1]
        assert overlay.get_alpha() == pytest.approx(0.5)

    def test_without_background_plots_stat_map_alone(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        stat_map = _signed_stat_map(sample_3d_volume)
        plotter = plot_stat_map(stat_map, slice_mode="z")
        rendered = [ax for ax in plotter.axes.ravel() if ax.collections]
        assert len(rendered) == stat_map.sizes["z"]
        assert all(len(ax.collections) == 1 for ax in rendered)
        norm = rendered[0].collections[0].norm
        assert norm.vmax == 10.0
        assert norm.vmin == -10.0


class TestStatMapAccessor:
    """Tests for the `data.fusi.plot.stat_map()` accessor wrapper."""

    def test_accessor_forwards_to_plot_stat_map(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        import confusius  # noqa: F401 - register accessor.

        stat_map = _signed_stat_map(sample_3d_volume)
        plotter = stat_map.fusi.plot.stat_map(
            bg_volume=sample_3d_volume, slice_mode="z"
        )
        assert isinstance(plotter, VolumePlotter)
        rendered = [ax for ax in plotter.axes.ravel() if ax.collections]
        assert len(rendered) == sample_3d_volume.sizes["z"]

    def test_accessor_without_background(self, sample_3d_volume, matplotlib_pyplot):
        import confusius  # noqa: F401 - register accessor.

        stat_map = _signed_stat_map(sample_3d_volume)
        plotter = stat_map.fusi.plot.stat_map(slice_mode="z")
        rendered = [ax for ax in plotter.axes.ravel() if ax.collections]
        assert len(rendered) == stat_map.sizes["z"]
        assert all(len(ax.collections) == 1 for ax in rendered)
