"""Tests for image plotting functions.

These tests use real matplotlib with the Agg backend (non-interactive).
See conftest.py for the matplotlib_pyplot fixture setup.
"""

import warnings

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from confusius._utils.geometry import (
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
    get_voxel_to_world_units,
)
from confusius.plotting import (
    VolumePlotter,
    plot_carpet,
    plot_contours,
    plot_volume,
)
from confusius.xarray import create_voxeldata

_VOXEL_DIM_BY_WORLD_NAME = {"z": "k", "y": "j", "x": "i"}


def _world_coord_1d(da: xr.DataArray, name: str) -> np.ndarray:
    """Return a world coordinate's 1D values, reducing other axis-aligned dims."""
    coord = da.coords[name]
    dim = name if name in coord.dims else _VOXEL_DIM_BY_WORLD_NAME[name]
    if coord.dims == (dim,):
        return coord.values
    others = {d: 0 for d in coord.dims if d != dim}
    return coord.isel(others).values


def _axes(plotter):
    assert plotter.axes is not None
    return plotter.axes


def _figure(plotter):
    assert plotter.figure is not None
    return plotter.figure


@pytest.fixture
def make_region_voxeldata(sample_voxeldata_3d):
    """Build region-stacked VoxelData on the shared plotting fixture grid."""

    def _make(regions=("a", "b"), values=None, *, name="r"):
        template = sample_voxeldata_3d.isel(k=[0]).expand_dims(region=list(regions))
        if values is None:
            values = np.zeros(tuple(template.sizes[dim] for dim in template.dims))
        return template.copy(data=np.asarray(values), deep=True).rename(name)

    return _make


class TestPlotVolume:
    """Tests for plot_volume function."""

    def test_invalid_slice_mode_raises(self, sample_voxeldata_3d):
        """plot_volume raises ValueError for a slice_mode not in data.dims."""
        with pytest.raises(ValueError, match="slice_mode"):
            plot_volume(sample_voxeldata_3d, slice_mode="t", slice_coords=[0.0])

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("bad_arg", ["vmin", "vmax"])
    def test_nonfinite_vmin_vmax_raises(
        self, sample_voxeldata_3d, matplotlib_pyplot, bad_arg, bad_value
    ):
        """plot_volume raises a clear ValueError for non-finite vmin/vmax.

        Regression test for #258: non-finite bounds used to produce an empty
        colormap color list and crash deep inside
        `matplotlib.colors.LinearSegmentedColormap.from_list` with an opaque
        `IndexError`.
        """
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]
        with pytest.raises(ValueError, match="finite"):
            plot_volume(
                sample_voxeldata_3d,
                slice_mode="z",
                slice_coords=[z_coord],
                **{bad_arg: bad_value},
            )

    def test_slice_mode_pose_facets_over_poses(self, matplotlib_pyplot):
        """plot_volume facets over `pose` with world-coordinate axis labels.

        Regression: `pose`-containing data was never recognized as "plottable
        voxel-to-world" data, so slice_mode="pose" happened to work by accident
        (the validation/labeling gate silently no-opped) but produced plain
        voxel-index axis labels instead of world-coordinate ones.
        """
        npose = 3
        affine = np.stack(
            [
                np.diag([0.2, 0.1, 0.05, 1.0]),
                np.array(
                    [
                        [0.2, 0.0, 0.0, 1.0],
                        [0.0, 0.1, 0.0, 0.0],
                        [0.0, 0.0, 0.05, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [0.2, 0.0, 0.0, 2.0],
                        [0.0, 0.1, 0.0, 0.0],
                        [0.0, 0.0, 0.05, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ]
        )
        data = create_voxeldata(
            np.random.default_rng(0).random((npose, 1, 6, 8)),
            dims=("pose", "k", "j", "i"),
            pose=np.arange(npose),
            voxel_to_world=affine,
        )

        plotter = plot_volume(data.isel(k=0), slice_mode="pose", show_colorbar=False)

        axes = _axes(plotter).ravel()
        assert sum(len(ax.collections) for ax in axes) == npose
        assert "mm" in axes[0].get_xlabel()
        assert "mm" in axes[0].get_ylabel()

    def test_spatial_slice_mode_squeezes_singleton_pose_dependent_affine(
        self, matplotlib_pyplot
    ):
        """A single-pose `pose` dim with a technically stacked affine is squeezed
        away before spatial slicing, not rejected.

        Regression: `stack_poses`-style construction always produces a per-pose
        `(npose, 4, 4)` affine, even for a single real pose -- a `(1, 4, 4)`
        affine is still "pose-dependent" by shape (`affine.ndim == 3`), even
        though there is only one pose and no genuine ambiguity to resolve.
        Spatial `slice_mode` must not reject this squeeze-friendly case.
        """
        affine = np.diag([0.2, 0.1, 0.05, 1.0])[np.newaxis]
        data = create_voxeldata(
            np.random.default_rng(0).random((1, 2, 3, 4)),
            dims=("pose", "k", "j", "i"),
            pose=[0],
            voxel_to_world=affine,
        )

        plotter = plot_volume(data, slice_mode="z", show_colorbar=False)

        axes = _axes(plotter).ravel()
        assert sum(len(ax.collections) for ax in axes) > 0

    def test_slice_mode_pose_uses_each_poses_own_world_position(
        self, matplotlib_pyplot
    ):
        """Each pose panel is positioned using its own world coordinates.

        Regression: `_prepare_slice_inputs` used to materialize the whole
        pose-stacked array up front whenever it happened to report
        axis-aligned (each pose individually diagonal, even with a different
        origin per pose) -- materializing collapses every non-spatial dim,
        including `pose`, to its first index when building the new `z`/`y`/`x`
        dim-coordinates, silently mislabeling every pose with pose 0's world
        position.
        """
        affine = np.stack(
            [
                np.diag([0.2, 0.1, 0.05, 1.0]),
                np.array(
                    [
                        [0.2, 0.0, 0.0, 100.0],
                        [0.0, 0.1, 0.0, 100.0],
                        [0.0, 0.0, 0.05, 100.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ]
        )
        data = create_voxeldata(
            np.random.default_rng(0).random((2, 1, 6, 8)),
            dims=("pose", "k", "j", "i"),
            pose=[0, 1],
            voxel_to_world=affine,
        )

        plotter = plot_volume(data.isel(k=0), slice_mode="pose", show_colorbar=False)

        axes = _axes(plotter).ravel()
        assert axes[0].get_xlim()[0] < 10.0
        assert axes[1].get_xlim()[0] > 90.0

    def test_non_spatial_facet_labels_extra_dim_axis_without_units(
        self, matplotlib_pyplot
    ):
        """Faceting over a non-spatial dim labels a non-world display axis
        plainly, without "mm", when that axis remains one of the panel's own
        display axes.

        Regression: a non-world axis (e.g. a plain extra facet dim) isn't a
        world-space axis (it has no `units`), but `_build_axis_label` labeled
        *any* axis "(mm)" whenever the array carried plottable voxel-to-world
        geometry at all, regardless of which dim was actually being labeled.
        """
        data = (
            create_voxeldata(
                np.random.default_rng(0).random((3, 1, 6, 8)),
                dims=("channel", "k", "j", "i"),
                spacing=(0.2, 0.1, 0.05),
                origin=(0.0, 0.0, 0.0),
            )
            .isel(k=0, i=0)
            .expand_dims(region=["r0"])
        )

        plotter = plot_volume(
            data, slice_mode="region", slice_coords=["r0"], show_colorbar=False
        )

        axes = _axes(plotter).ravel()
        assert "mm" in axes[0].get_xlabel()
        assert axes[0].get_ylabel() == "channel"

    def test_non_3d_data_raises(self, sample_voxeldata_3dt):
        """plot_volume raises ValueError for 4D data with no unitary dimensions."""
        data = sample_voxeldata_3dt
        with pytest.raises(ValueError, match="3D"):
            plot_volume(data, slice_mode="z")

    def test_voxel_to_world_2d_data_is_rejected(self, matplotlib_pyplot):
        """VolumePlotter requires full VoxelData geometry."""
        data = xr.DataArray(
            np.arange(3 * 4, dtype=float).reshape(3, 4),
            dims=["j", "i"],
            coords={"j": [0, 1, 2], "i": [0, 1, 2, 3]},
        )
        with pytest.raises(ValueError, match="must have all native voxel dims"):
            data = attach_voxel_to_world_index(
                data,
                np.array([[0.3, 0.0, 20.0], [0.0, 0.25, 30.0], [0.0, 0.0, 1.0]]),
            )
            plot_volume(data, slice_mode="y")

    def test_complex_data_converted_to_magnitude(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """plot_volume converts complex-valued data to magnitude before plotting."""
        complex_data = sample_voxeldata_3d * (1 + 1j)
        z_coord = _world_coord_1d(complex_data, "z")[0]
        with pytest.warns(UserWarning, match="Complex-valued data"):
            plotter = plot_volume(complex_data, slice_mode="z", slice_coords=[z_coord])

        plotted_values = _axes(plotter)[0, 0].collections[0].get_array().data
        assert np.all(plotted_values >= 0)

    @pytest.mark.parametrize("threshold_mode", ["lower", "upper"])
    def test_threshold_masks_correctly(
        self, sample_voxeldata_3d, matplotlib_pyplot, threshold_mode
    ):
        """plot_volume masks data correctly based on threshold_mode.

        For 'lower': masks |data| < threshold.
        For 'upper': masks |data| > threshold.
        """
        threshold = 0.5
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]
        plotter = plot_volume(
            sample_voxeldata_3d,
            slice_mode="z",
            slice_coords=[z_coord],
            threshold=threshold,
            threshold_mode=threshold_mode,
        )

        ax = _axes(plotter)[0, 0]
        plotted_data = ax.collections[0].get_array()
        original_slice = sample_voxeldata_3d.sel(z=z_coord, method="nearest").values

        abs_data = np.abs(original_slice)
        if threshold_mode == "lower":
            expected_mask = abs_data < threshold
        else:
            expected_mask = abs_data > threshold

        np.testing.assert_array_equal(plotted_data.mask, expected_mask)

    def test_threshold_gray_band_applied_with_attrs_norm(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """Gray band is present in the cmap even when norm comes from data.attrs."""
        from matplotlib.colors import Normalize

        data = sample_voxeldata_3d.copy()
        data.attrs["norm"] = Normalize(vmin=-2.0, vmax=2.0)
        z_coord = _world_coord_1d(data, "z")[0]
        plotter = plot_volume(
            data, slice_mode="z", slice_coords=[z_coord], threshold=0.5
        )
        # norm(0) = 0.5, which is inside [-0.5, 0.5] — must map to gray.
        r, g, b, _ = _axes(plotter)[0, 0].collections[0].cmap(0.5)
        assert r == pytest.approx(g, abs=1e-2)
        assert g == pytest.approx(b, abs=1e-2)

    def test_threshold_gray_band_uses_norm_not_linear_arithmetic(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """Gray-band boundaries are placed at norm(±threshold), not linearly."""
        from matplotlib.colors import TwoSlopeNorm

        # norm(1.0) ≈ 0.667; linear formula gives 0.5 — check position 0.55 is gray.
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=3.0)
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]
        plotter = plot_volume(
            sample_voxeldata_3d,
            slice_mode="z",
            slice_coords=[z_coord],
            norm=norm,
            threshold=1.0,
            threshold_mode="lower",
            show_colorbar=False,
        )
        # Position 0.55 is between the wrong linear boundary (0.5) and the correct
        # norm boundary (≈0.667), so it must map to gray.
        r, g, b, _ = _axes(plotter)[0, 0].collections[0].cmap(0.55)
        assert r == pytest.approx(g, abs=1e-2)
        assert g == pytest.approx(b, abs=1e-2)

    def test_explicit_vmin_vmax(self, sample_voxeldata_3d, matplotlib_pyplot):
        """plot_volume passes explicit vmin and vmax to pcolormesh."""
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]
        plotter = plot_volume(
            sample_voxeldata_3d,
            slice_mode="z",
            slice_coords=[z_coord],
            vmin=-3.0,
            vmax=3.0,
        )

        collection = _axes(plotter)[0, 0].collections[0]
        assert collection.norm.vmin == pytest.approx(-3.0)
        assert collection.norm.vmax == pytest.approx(3.0)

    def test_default_alpha_preserves_cmap_alpha(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """Default alpha (None) lets a colormap's own alpha channel through."""
        from matplotlib.colors import ListedColormap

        # Semi-transparent colormap; the old alpha=1.0 default would erase it.
        cmap = ListedColormap([(1.0, 0.0, 0.0, 0.3), (0.0, 0.0, 1.0, 0.7)])
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]
        plotter = plot_volume(
            sample_voxeldata_3d,
            slice_mode="z",
            slice_coords=[z_coord],
            cmap=cmap,
            show_colorbar=False,
        )

        quadmesh = _axes(plotter)[0, 0].collections[0]
        quadmesh.update_scalarmappable()
        alphas = np.unique(quadmesh.get_facecolor()[:, 3])
        assert alphas.size > 0
        assert np.all(np.isclose(alphas[:, None], [0.3, 0.7]).any(axis=1))

    def test_colorbar_added_by_default(self, sample_voxeldata_3d, matplotlib_pyplot):
        """plot_volume adds a colorbar when show_colorbar=True (default)."""
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]
        plotter = plot_volume(
            sample_voxeldata_3d, slice_mode="z", slice_coords=[z_coord]
        )

        plot_axes = set(_axes(plotter).ravel())
        extra_axes = [ax for ax in _figure(plotter).axes if ax not in plot_axes]
        assert len(extra_axes) == 1

    def test_no_colorbar_when_disabled(self, sample_voxeldata_3d, matplotlib_pyplot):
        """plot_volume skips colorbar when show_colorbar=False."""
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]
        plotter = plot_volume(
            sample_voxeldata_3d,
            slice_mode="z",
            slice_coords=[z_coord],
            show_colorbar=False,
        )

        plot_axes = set(_axes(plotter).ravel())
        extra_axes = [ax for ax in _figure(plotter).axes if ax not in plot_axes]
        assert len(extra_axes) == 0

    def test_cbar_label_is_set(self, sample_voxeldata_3d, matplotlib_pyplot):
        """plot_volume sets the colorbar label when cbar_label is provided."""
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]
        plotter = plot_volume(
            sample_voxeldata_3d,
            slice_mode="z",
            slice_coords=[z_coord],
            cbar_label="Power (dB)",
        )

        plot_axes = set(_axes(plotter).ravel())
        extra_axes = [ax for ax in _figure(plotter).axes if ax not in plot_axes]
        assert len(extra_axes) == 1
        assert extra_axes[0].get_ylabel() == "Power (dB)"

    def test_fontsize_scales_volume_text_elements(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """plot_volume scales title, label, tick, and colorbar text from fontsize."""
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]
        plotter = plot_volume(
            sample_voxeldata_3d,
            slice_mode="z",
            slice_coords=[z_coord],
            fontsize=20,
            cbar_label="Power (dB)",
        )

        ax = _axes(plotter)[0, 0]
        assert ax.title.get_fontsize() == pytest.approx(20)
        assert ax.xaxis.label.get_fontsize() == pytest.approx(18)
        assert ax.yaxis.label.get_fontsize() == pytest.approx(18)
        assert ax.get_xticklabels()[0].get_fontsize() == pytest.approx(17)

        plot_axes = set(_axes(plotter).ravel())
        cbar_axes = [ax for ax in _figure(plotter).axes if ax not in plot_axes]
        assert len(cbar_axes) == 1
        assert cbar_axes[0].yaxis.label.get_fontsize() == pytest.approx(18)
        assert cbar_axes[0].get_yticklabels()[0].get_fontsize() == pytest.approx(17)

    def test_existing_axes_used(self, sample_voxeldata_3d, matplotlib_pyplot):
        """plot_volume uses provided axes without creating new ones."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 1, squeeze=False)
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]

        plotter = plot_volume(
            sample_voxeldata_3d, slice_mode="z", slice_coords=[z_coord], axes=axes
        )

        assert plotter.axes is axes
        assert plotter.figure is fig

    def test_single_axes_object_accepted(self, sample_voxeldata_3d, matplotlib_pyplot):
        """plot_volume accepts a bare Axes object, not only an ndarray of Axes.

        Regression test for issue #66: previously raised
        AttributeError: 'Axes' object has no attribute 'flat'.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]

        plotter = plot_volume(
            sample_voxeldata_3d,
            slice_mode="z",
            slice_coords=[z_coord],
            axes=ax,
            show_colorbar=False,
        )

        assert plotter.figure is fig
        assert len(ax.collections) == 1

    def test_invalid_axes_array_raises(self, matplotlib_pyplot):
        """VolumePlotter raises TypeError when an axes array contains non-Axes."""
        with pytest.raises(TypeError, match="matplotlib.axes.Axes"):
            VolumePlotter(slice_mode="z", axes=np.array([[object()]]))

    def test_axes_count_mismatch_raises(self, sample_voxeldata_3d, matplotlib_pyplot):
        """plot_volume raises ValueError when axes count doesn't match slices."""
        import matplotlib.pyplot as plt

        _fig, axes = plt.subplots(1, 1, squeeze=False)
        z_coords = _world_coord_1d(sample_voxeldata_3d, "z")[:3].tolist()

        with pytest.raises(ValueError, match="must match number of axes"):
            plot_volume(
                sample_voxeldata_3d, slice_mode="z", slice_coords=z_coords, axes=axes
            )

    def test_unused_axes_hidden(self, sample_voxeldata_3d, matplotlib_pyplot):
        """plot_volume hides axes beyond the number of slices."""
        z_coords = _world_coord_1d(sample_voxeldata_3d, "z")[:2].tolist()
        plotter = plot_volume(
            sample_voxeldata_3d, slice_mode="z", slice_coords=z_coords, nrows=2, ncols=2
        )

        for ax in _axes(plotter).ravel()[2:]:
            assert not ax.get_visible()

    def test_axis_limits_match_data_edges(self, sample_voxeldata_3d, matplotlib_pyplot):
        """Axes limits exactly equal data edges — no matplotlib auto-margin."""
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]
        plotter = plot_volume(
            sample_voxeldata_3d, slice_mode="z", slice_coords=[z_coord]
        )
        ax = _axes(plotter)[0, 0]

        x_centers = _world_coord_1d(sample_voxeldata_3d, "x").astype(float)
        y_centers = _world_coord_1d(sample_voxeldata_3d, "y").astype(float)
        dx = x_centers[1] - x_centers[0]
        dy = y_centers[1] - y_centers[0]

        assert ax.get_xlim() == pytest.approx(
            (x_centers[0] - dx / 2, x_centers[-1] + dx / 2)
        )
        # Upper origin: ylim is (y_max_edge, y_min_edge).
        assert ax.get_ylim() == pytest.approx(
            (y_centers[-1] + dy / 2, y_centers[0] - dy / 2)
        )

    def test_fixture_voxel_coordinates_define_edges(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """plot_volume derives pixel edges from VoxelData world coordinates."""
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]
        plotter = plot_volume(
            sample_voxeldata_3d,
            slice_mode="z",
            slice_coords=[z_coord],
            show_colorbar=False,
        )
        ax = _axes(plotter)[0, 0]
        x_centers = _world_coord_1d(sample_voxeldata_3d, "x").astype(float)
        y_centers = _world_coord_1d(sample_voxeldata_3d, "y").astype(float)
        dx = x_centers[1] - x_centers[0]
        dy = y_centers[1] - y_centers[0]

        assert ax.get_xlim() == pytest.approx(
            (x_centers[0] - dx / 2, x_centers[-1] + dx / 2)
        )
        assert ax.get_ylim() == pytest.approx(
            (y_centers[-1] + dy / 2, y_centers[0] - dy / 2)
        )

    def test_yincrease_true_places_origin_at_bottom(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """plot_volume with yincrease=True places y-origin at bottom."""
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]
        plotter = plot_volume(
            sample_voxeldata_3d,
            slice_mode="z",
            slice_coords=[z_coord],
            yincrease=True,
            show_colorbar=False,
        )
        ax = _axes(plotter)[0, 0]
        y_centers = _world_coord_1d(sample_voxeldata_3d, "y").astype(float)
        dy = y_centers[1] - y_centers[0]
        assert ax.get_ylim() == pytest.approx(
            (y_centers[0] - dy / 2, y_centers[-1] + dy / 2)
        )

    def test_existing_figure_used(self, sample_voxeldata_3d, matplotlib_pyplot):
        """plot_volume uses the provided figure to create new axes inside it."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]
        plotter = plot_volume(
            sample_voxeldata_3d,
            slice_mode="z",
            slice_coords=[z_coord],
            figure=fig,
            show_colorbar=False,
        )
        assert plotter.figure is fig
        assert plotter.axes is not None

    def test_4d_with_unitary_dim_squeezed(
        self, sample_voxeldata_3dt, matplotlib_pyplot
    ):
        """plot_volume squeezes unitary dimensions except slice_mode."""
        data_4d = sample_voxeldata_3dt.isel(time=[0])
        z_coord = _world_coord_1d(data_4d, "z")[0]
        plotter = plot_volume(
            data_4d, slice_mode="z", slice_coords=[z_coord], show_colorbar=False
        )
        assert plotter.axes is not None

    def test_bool_dtype_does_not_raise(self, sample_voxeldata_3d, matplotlib_pyplot):
        """plot_volume handles boolean dtype data without raising TypeError.

        np.percentile on bool arrays fails with a TypeError because numpy does
        not support subtraction on bool dtype during linear interpolation.
        Casting to float before computing percentiles fixes this.
        """
        bool_data = sample_voxeldata_3d > sample_voxeldata_3d.mean()
        z_coord = _world_coord_1d(sample_voxeldata_3d, "z")[0]
        # Should not raise TypeError: numpy boolean subtract.
        plotter = plot_volume(
            bool_data, slice_mode="z", slice_coords=[z_coord], show_colorbar=False
        )
        assert plotter.axes is not None

    def test_unitary_slice_mode_preserved(self, sample_voxeldata_3d, matplotlib_pyplot):
        """plot_volume preserves slice_mode dimension even when unitary."""
        data = sample_voxeldata_3d.isel(k=[0])
        # Should plot single slice without error
        plotter = plot_volume(data, slice_mode="z", show_colorbar=False)
        assert _axes(plotter).shape == (1, 1)
        # Verify the slice was plotted
        assert len(_axes(plotter)[0, 0].collections) == 1

    def test_scalar_slice_mode_from_selection(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """plot_volume accepts a scalar slice_mode coordinate (issue #295).

        Selecting a single index (isel(z=1)) drops z to a scalar coordinate; it
        should plot like the size-1 z dimension from isel(z=[1]).
        """
        plotter = plot_volume(
            sample_voxeldata_3d.isel(k=[1]), slice_mode="z", show_colorbar=False
        )
        assert _axes(plotter).shape == (1, 1)
        assert len(_axes(plotter)[0, 0].collections) == 1

    def test_non_monotonic_voxel_coords_are_rejected(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """VolumePlotter rejects inputs outside the VoxelData model."""
        data = sample_voxeldata_3d.copy().isel(j=[2, 0, 1], i=[3, 1, 2, 0])

        with pytest.raises(ValueError, match="must be strictly monotonic"):
            plot_volume(data, slice_mode="z", show_colorbar=False)

    def test_voxel_to_world_volume_resamples_for_world_slice_mode(
        self, matplotlib_pyplot, sample_voxeldata_3d_oblique
    ):
        """World z-slicing resamples oblique data onto the axis-aligned world grid.

        `slice_mode`'s own world axis is regularized for cross-volume position
        matching; the two in-plane axes are also forced onto the global frame
        (each keeping its own native per-axis spacing), so display always shows
        proper world axis labels (`"y"`/`"x"`) regardless of the source data's
        in-plane orientation.
        """
        data = sample_voxeldata_3d_oblique
        z_coord = float(np.asarray(_world_coord_1d(data, "z"), dtype=float).mean())

        plotter = plot_volume(
            data,
            slice_mode="z",
            slice_coords=[z_coord],
            show_colorbar=False,
        )

        ax = _axes(plotter)[0, 0]
        quadmesh = ax.collections[0]
        assert quadmesh.get_coordinates().ndim == 3
        assert ax.get_xlabel() == "x (mm)"
        assert ax.get_ylabel() == "y (mm)"

    def test_voxel_to_world_world_resampling_preserves_per_axis_spacing(
        self, sample_voxeldata_3d_oblique
    ):
        """Each of the 3 output axes keeps its own native per-axis spacing.

        All 3 axes are forced onto the global (world-aligned) frame, but the two
        in-plane axes are not resampled to match one another or the slice axis --
        each keeps the spacing `compute_shared_slice_axis_grid_geometry` derives
        for it individually.
        """
        from confusius.plotting.image import compute_shared_slice_axis_grid_geometry

        data = sample_voxeldata_3d_oblique
        _, expected_spacing, _, _ = compute_shared_slice_axis_grid_geometry(data, "z")

        result, slice_spacing = VolumePlotter(slice_mode="z")._prepare_slice_inputs(
            data
        )

        assert result.dims == ("z", "y", "x")
        z_spacing = float(np.diff(_world_coord_1d(result, "z"))[0])
        assert z_spacing == pytest.approx(expected_spacing["k"])
        assert slice_spacing == pytest.approx(expected_spacing["k"])
        y_spacing = float(np.diff(_world_coord_1d(result, "y"))[0])
        x_spacing = float(np.diff(_world_coord_1d(result, "x"))[0])
        assert y_spacing == pytest.approx(expected_spacing["j"], abs=1e-6)
        assert x_spacing == pytest.approx(expected_spacing["i"], abs=1e-6)

    def test_axis_aligned_voxel_to_world_world_slice_promotes_world_dims(self):
        """Axis-aligned geometry uses world dims directly for world slicing."""
        data = xr.DataArray(
            np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
            dims=["k", "j", "i"],
            coords={"k": [0, 1], "j": [0, 1, 2], "i": [0, 1, 2, 3]},
        )
        data = attach_voxel_to_world_index(data, np.diag([0.4, 0.3, 0.25, 1.0]))

        result, _ = VolumePlotter(slice_mode="z")._prepare_slice_inputs(data)

        assert result.dims == ("z", "y", "x")
        assert "voxel_to_world" not in result.attrs


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


class TestPlottingUtilsVoxelToWorldHelpers:
    """Tests for shared voxel-to-world display helpers in `plotting._utils`."""

    def test_resample_derives_spacing_from_affine_not_materialized_array_axis(self):
        """Resampled spacing matches the voxel-to-world affine, not a naive diff
        along the materialized (k, j, i)-shaped world-coordinate array's last axis.

        `z` and `y` here depend only on the `k`/`j` voxel axes, never on the
        trailing `i` axis -- so a spacing computation that differenced the
        materialized (k, j, i)-shaped world-coordinate array along its last axis
        (the default `numpy.diff` behaviour) would see zero variation and divide
        by zero downstream. Assert the exact spacing values (matching
        `.fusi.spacing`), not just that resampling doesn't crash.
        """
        from confusius._utils.plotting import resample_to_axis_aligned_world_grid

        data = xr.DataArray(
            np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
            dims=["k", "j", "i"],
            coords={"k": [0, 1], "j": [0, 1, 2], "i": [0, 1, 2, 3]},
        )
        voxel_to_world = np.array(
            [
                [0.4, 0.1, 0.0, 10.0],
                [0.0, 0.3, 0.0, 20.0],
                [0.0, 0.0, 0.25, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        data = attach_voxel_to_world_index(data, voxel_to_world)

        result = resample_to_axis_aligned_world_grid(data)

        # The shear keeps `z` constant along the trailing `i` axis, so a naive
        # spacing calculation over the materialized `z` array's last axis would see
        # zero spacing. The affine's scale factors are hand-computable here because
        # its linear block is already upper triangular.
        assert result.fusi.spacing["k"] == pytest.approx(0.4)
        assert result.fusi.spacing["j"] == pytest.approx(0.3)
        assert result.fusi.spacing["i"] == pytest.approx(0.25)

    def test_resample_raises_when_dominant_voxel_dim_is_irregular(self):
        """Resampling onto an axis-aligned world grid raises `ValueError` when the
        voxel dimension dominating a world axis has irregularly spaced coordinates,
        since no single spacing value can represent it on the output grid.
        """
        from confusius._utils.plotting import resample_to_axis_aligned_world_grid

        data = xr.DataArray(
            np.arange(3 * 3 * 4, dtype=float).reshape(3, 3, 4),
            dims=["k", "j", "i"],
            # `k` is irregularly spaced (0, 1, 3) and dominates world axis `z`.
            coords={"k": [0, 1, 3], "j": [0, 1, 2], "i": [0, 1, 2, 3]},
        )
        # Sheared (oblique) affine so the fast axis-aligned path is not taken.
        data = attach_voxel_to_world_index(
            data,
            np.array(
                [
                    [0.4, 0.0, 0.1, 10.0],
                    [0.1, 0.3, 0.0, 20.0],
                    [0.0, 0.05, 0.25, 30.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

        with pytest.raises(ValueError, match="no well-defined spacing"):
            resample_to_axis_aligned_world_grid(data)

class TestVolumePlotterAddVolume:
    """Tests for VolumePlotter.add_volume method."""

    def test_overlay_lands_on_correct_axes(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """add_volume overlays only on axes whose coordinates match."""
        plotter = plot_volume(sample_voxeldata_3d, slice_mode="z", show_colorbar=False)

        subset = sample_voxeldata_3d.sel(
            z=_world_coord_1d(sample_voxeldata_3d, "z")[:2]
        )
        plotter.add_volume(subset, cmap="hot", alpha=0.5, show_colorbar=False)

        axes_flat = _axes(plotter).ravel()
        assert len(axes_flat[0].collections) == 2
        assert len(axes_flat[1].collections) == 2
        assert len(axes_flat[2].collections) == 1
        assert len(axes_flat[3].collections) == 1

    def test_add_volume_warns_on_missing_coords(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """add_volume warns when some coordinates don't match."""
        plotter = plot_volume(
            sample_voxeldata_3d,
            slice_mode="z",
            slice_coords=[_world_coord_1d(sample_voxeldata_3d, "z")[2]],
        )

        z_vals = _world_coord_1d(sample_voxeldata_3d, "z")
        with pytest.warns(UserWarning, match="Could not find matching axes"):
            plotter.add_volume(
                sample_voxeldata_3d.sel(z=z_vals[[0, 1, 3]], method="nearest"),
                cmap="viridis",
            )

    def test_far_off_slice_coord_warns_and_skips_instead_of_mislabeling(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """A `slice_coords` value far from any real data warns and is skipped.

        Regression: `_extract_slices`'s nearest-neighbour `.sel` had no distance
        limit, so requesting a coordinate nowhere near any real slice (e.g. `z=0`
        against a single-slice volume whose real z is `1.0`) silently returned
        that one slice, mislabeled with whatever coordinate was requested.
        """
        single_slice = sample_voxeldata_3d.isel(k=[0])
        real_z = _world_coord_1d(single_slice, "z")[0]
        far_off_z = real_z - 10.0

        with pytest.warns(UserWarning, match="No slice found"):
            plotter = plot_volume(
                single_slice,
                slice_mode="z",
                slice_coords=[far_off_z],
                show_colorbar=False,
            )
        assert plotter.axes is None

    def test_non_spatial_slice_mode_never_nearest_matches(self, matplotlib_pyplot):
        """A non-spatial `slice_mode` (e.g. `pose`) never nearest-matches.

        Regression: numeric non-spatial coordinates (e.g. an integer `pose` id)
        were matched by nearest-neighbour like spatial ones, so requesting pose 2
        when only poses 0/1 exist silently returned pose 1's data mislabeled as
        pose 2 -- nearness has no physical meaning for a discrete facet.
        """
        data = create_voxeldata(
            np.random.default_rng(0).random((2, 1, 6, 8)),
            dims=("pose", "k", "j", "i"),
            pose=[0, 1],
            voxel_to_world=np.broadcast_to(
                np.diag([0.2, 0.1, 0.05, 1.0]), (2, 4, 4)
            ).copy(),
        )

        with pytest.warns(UserWarning, match="No slice found"):
            plotter = plot_volume(
                data.isel(k=0),
                slice_mode="pose",
                slice_coords=[2],
                show_colorbar=False,
            )
        assert plotter.axes is None

    def test_axis_aligned_volumes_with_different_spacing_overlay_without_warning(
        self, matplotlib_pyplot
    ):
        """Two axis-aligned volumes at different native z spacing still overlay.

        Regression: cross-volume matching used a fixed `1e-6` tolerance meant only
        for floating-point noise, so two axis-aligned volumes with genuinely
        different (but comparable, physically overlapping) native z resolutions
        produced spurious "Could not find matching axes" warnings even though
        their slice positions were physically close.
        """
        # z positions: coarse = 0.0, 0.2, 0.4; fine = 0.0, 0.18, 0.36 -- each fine
        # position is within half of fine's own spacing (0.09) of the coarse
        # position it should overlay onto.
        coarse = create_voxeldata(
            np.random.default_rng(0).random((3, 6, 8)),
            dims=("k", "j", "i"),
            spacing=(0.2, 0.1, 0.05),
            origin=(0.0, 0.0, 0.0),
        )
        fine = create_voxeldata(
            np.random.default_rng(1).random((3, 6, 8)),
            dims=("k", "j", "i"),
            spacing=(0.18, 0.1, 0.05),
            origin=(0.0, 0.0, 0.0),
        )

        plotter = plot_volume(coarse, slice_mode="z", show_colorbar=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            plotter.add_volume(fine, cmap="viridis", show_colorbar=False)

        axes_flat = _axes(plotter).ravel()
        # 3 panels x 2 collections (coarse + fine overlay) each; the 2x2 grid
        # auto-sized for 3 panels leaves one unused axis with no collections.
        assert sum(len(ax.collections) for ax in axes_flat) == 6

    def test_world_slice_mode_projects_differently_rotated_volumes_consistently(
        self, matplotlib_pyplot
    ):
        """World display resamples onto the global frame, not a per-volume one.

        Two volumes sharing the same slice axis (z) but with different in-plane
        rotations must land at their own true, differing world positions when
        both are displayed with slice_mode="z" -- not at identical display
        coordinates, which is what a per-volume-derived basis would silently
        produce, hiding the rotation.
        """
        straight = create_voxeldata(
            np.arange(3 * 5 * 5.0).reshape(3, 5, 5),
            dims=("k", "j", "i"),
            voxel_to_world=np.diag([0.4, 1.0, 1.0, 1.0]),
        )
        theta = np.deg2rad(45)
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        rotated_affine = np.eye(4)
        rotated_affine[0, 0] = 0.4
        rotated_affine[1:3, 1:3] = rotation
        rotated = create_voxeldata(
            np.arange(3 * 5 * 5.0).reshape(3, 5, 5),
            dims=("k", "j", "i"),
            voxel_to_world=rotated_affine,
        )

        plotter_straight = VolumePlotter(slice_mode="z").add_volume(
            straight,
            match_coordinates=False,
            show_colorbar=False,
            slice_coords=[0.4],
        )
        plotter_rotated = VolumePlotter(slice_mode="z").add_volume(
            rotated,
            match_coordinates=False,
            show_colorbar=False,
            slice_coords=[0.4],
        )

        coords_straight = _axes(plotter_straight)[0, 0].collections[0].get_coordinates()
        coords_rotated = _axes(plotter_rotated)[0, 0].collections[0].get_coordinates()

        # A per-volume-local basis would make both identical, each showing
        # "upright" in its own frame. Resampling onto the fixed global frame
        # instead reveals the true relative rotation: the rotated volume's world
        # footprint is genuinely wider/different (even its output grid shape
        # differs, from the larger axis-aligned bounding box), not a copy of the
        # straight one's.
        assert coords_straight.shape != coords_rotated.shape or not np.allclose(
            coords_straight, coords_rotated
        )
        assert _axes(plotter_straight)[0, 0].get_xlabel() == "x (mm)"
        assert _axes(plotter_rotated)[0, 0].get_xlabel() == "x (mm)"

    def test_oblique_data_resamples_to_rectangular_cells(self, matplotlib_pyplot):
        """Display always resamples oblique in-plane geometry onto rectangular cells."""
        theta = np.deg2rad(30)
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        affine = np.eye(4)
        affine[0, 0] = 0.4
        affine[1:3, 1:3] = rotation
        data = create_voxeldata(
            np.arange(3 * 8 * 8.0).reshape(3, 8, 8),
            dims=("k", "j", "i"),
            voxel_to_world=affine,
        )

        plotter = VolumePlotter(slice_mode="z").add_volume(
            data,
            match_coordinates=False,
            show_colorbar=False,
            slice_coords=[0.4],
        )

        coords = _axes(plotter)[0, 0].collections[0].get_coordinates()
        x = coords[..., 0]
        assert np.allclose(x, x[0:1, :])
        assert _axes(plotter)[0, 0].get_xlabel() == "x (mm)"

    def test_world_slice_mode_default_slice_coords_handles_single_native_slice(
        self, matplotlib_pyplot
    ):
        """Default slice_coords works for an oblique single-k-slice overlay volume.

        A spatial `slice_mode` overlay volume that starts as a single native slice
        (`k` size 1, e.g. a single-plane fUSI recording) gets resampled onto the
        first volume's shared slice-axis grid, landing on several `z` positions --
        the default `slice_coords is None` path must list that resampled `z`
        coordinate's actual values, not the pre-resample single native position.
        """
        fixed = create_voxeldata(
            np.arange(5 * 10 * 10.0).reshape(5, 10, 10),
            dims=("k", "j", "i"),
            voxel_to_world=np.diag([0.4, 0.1, 0.1, 1.0]),
        )
        theta = np.deg2rad(10)
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        moving_affine = np.eye(4)
        moving_affine[0, 0] = 0.4
        moving_affine[1:3, 1:3] = rotation
        moving = create_voxeldata(
            np.arange(1 * 10 * 10.0).reshape(1, 10, 10),
            dims=("k", "j", "i"),
            voxel_to_world=moving_affine,
        )

        plotter = fixed.fusi.plot.volume(show_colorbar=False)
        plotter.add_volume(moving, cmap="viridis", alpha=0.5, show_colorbar=False)

        # `moving`'s single native slice gets resampled onto `fixed`'s entire
        # shared slice-axis grid, so every one of `fixed`'s panels gets an overlay
        # (filled with `resample_fill_value` outside `moving`'s own field of view)
        # -- the point here is that this doesn't crash, not the exact overlay
        # extent.
        n_with_overlay = sum(len(ax.collections) == 2 for ax in plotter.axes.ravel())
        assert n_with_overlay == fixed.sizes["k"]

    def test_world_slice_mode_single_native_slice_on_both_sides(
        self, matplotlib_pyplot
    ):
        """Two single-k-slice oblique volumes overlay correctly (both k=1).

        Regression: a genuinely size-1 `self.slice_mode` dim (a single-plane fUSI
        recording, or a volume resampled onto another single-plane volume's
        size-1 SliceAxisGrid) must survive `_prepare_slice_inputs`'s squeeze step
        rather than being dropped as if incidental, which would leave only 2 real
        dims and raise "Data must be 3D".
        """
        fixed = create_voxeldata(
            np.arange(1 * 10 * 10.0).reshape(1, 10, 10),
            dims=("k", "j", "i"),
            voxel_to_world=np.diag([0.4, 0.1, 0.1, 1.0]),
        )
        theta = np.deg2rad(10)
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        moving_affine = np.eye(4)
        moving_affine[0, 0] = 0.4
        moving_affine[1:3, 1:3] = rotation
        moving = create_voxeldata(
            np.arange(1 * 10 * 10.0).reshape(1, 10, 10),
            dims=("k", "j", "i"),
            voxel_to_world=moving_affine,
        )

        plotter = fixed.fusi.plot.volume(show_colorbar=False)
        plotter.add_volume(moving, cmap="viridis", alpha=0.5, show_colorbar=False)

        assert plotter.axes.size == 1
        assert len(plotter.axes[0, 0].collections) == 2

    def test_voxel_to_world_world_overlay_reuses_first_display_grid(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """World-coordinate overlays resample onto the first plotted world grid."""
        overlay = attach_voxel_to_world_index(
            sample_voxeldata_3d.copy().assign_coords(
                k=np.arange(sample_voxeldata_3d.sizes["k"]),
                j=np.arange(sample_voxeldata_3d.sizes["j"]),
                i=np.arange(sample_voxeldata_3d.sizes["i"]),
            ),
            np.array(
                [
                    [0.9, -0.1, 0.0, 0.2],
                    [0.1, 0.9, 0.0, -0.1],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            units=get_voxel_to_world_units(sample_voxeldata_3d),
        )
        z_coords = list(_world_coord_1d(sample_voxeldata_3d, "z")[:2])

        plotter = plot_volume(
            sample_voxeldata_3d,
            slice_mode="z",
            slice_coords=z_coords,
            show_colorbar=False,
        )
        plotter.add_volume(
            overlay,
            slice_coords=z_coords,
            cmap="hot",
            alpha=0.5,
            show_colorbar=False,
        )

        axes_flat = _axes(plotter).ravel()
        assert len(axes_flat[0].collections) == 2
        assert len(axes_flat[1].collections) == 2

    def test_dataarray_alpha_applies_independently_per_slice(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """A DataArray alpha gives each z-slice its own opacity, unlike a bare array."""
        alpha = xr.zeros_like(sample_voxeldata_3d)
        alpha[0] = 0.25
        alpha[1] = 0.75

        plotter = VolumePlotter(slice_mode="z").add_volume(
            sample_voxeldata_3d, match_coordinates=False, alpha=alpha
        )

        axes_flat = _axes(plotter).ravel()
        npt.assert_allclose(axes_flat[0].collections[0].get_alpha(), 0.25)
        npt.assert_allclose(axes_flat[1].collections[0].get_alpha(), 0.75)

    def test_dataarray_alpha_accepts_descending_coordinate_volume(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """A DataArray alpha sharing data's own grid must be accepted even when a
        display coordinate is monotonic-decreasing.

        `add_volume` sorts `data` ascending internally (via `sort_coords_for_plot`);
        `alpha` must be sorted the same way rather than rejected for carrying data's
        genuine (descending) coordinates.
        """
        descending = sample_voxeldata_3d.isel(j=slice(None, None, -1))
        alpha = xr.zeros_like(descending)
        alpha[0] = 0.25
        alpha[1] = 0.75

        plotter = plot_volume(descending, alpha=alpha)

        axes_flat = _axes(plotter).ravel()
        npt.assert_allclose(axes_flat[0].collections[0].get_alpha(), 0.25)
        npt.assert_allclose(axes_flat[1].collections[0].get_alpha(), 0.75)

    def test_numpy_array_alpha_rejected(self, sample_voxeldata_3d, matplotlib_pyplot):
        """A bare per-voxel array is rejected; opacity arrays must be DataArrays.

        A numpy array carries no coordinates, so it cannot be validated or aligned
        against `data`; only a scalar or a `xarray.DataArray` is accepted.
        """
        alpha = np.full(sample_voxeldata_3d.shape, 0.5)
        with pytest.raises(TypeError, match="DataArray"):
            VolumePlotter(slice_mode="z").add_volume(
                sample_voxeldata_3d, match_coordinates=False, alpha=alpha
            )

    def test_dataarray_alpha_size_mismatch_raises(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """Same dims as data, but a differently-sized one, is rejected explicitly."""
        alpha = sample_voxeldata_3d.isel(i=slice(0, 4))
        with pytest.raises(ValueError, match="size along 'x'"):
            VolumePlotter(slice_mode="z").add_volume(
                sample_voxeldata_3d, match_coordinates=False, alpha=alpha
            )

    def test_dataarray_alpha_dim_mismatch_raises(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """Renaming a native voxel dim away from `k`/`j`/`i` is rejected outright.

        Voxel dims are always exactly `k`/`j`/`i`, so a real VoxelToWorldIndex-backed
        `alpha` can never legitimately end up with a differently-named voxel
        dimension; the rejection now happens at `rename` time (geometry layer)
        rather than in `add_volume`'s own dims-equality check.
        """
        with pytest.raises(ValueError, match="must exactly cover"):
            alpha = sample_voxeldata_3d.rename(i="w")
            VolumePlotter(slice_mode="z").add_volume(
                sample_voxeldata_3d, match_coordinates=False, alpha=alpha
            )

    def test_dataarray_alpha_coordinate_mismatch_raises(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        affine = get_voxel_to_world_affine(sample_voxeldata_3d).copy()
        affine[2, 3] += 1.0
        alpha = attach_voxel_to_world_index(sample_voxeldata_3d.copy(), affine)
        with pytest.raises(ValueError, match="does not match"):
            VolumePlotter(slice_mode="z").add_volume(
                sample_voxeldata_3d, match_coordinates=False, alpha=alpha
            )


class TestNonNumericSliceMode:
    """Tests for slicing along a non-numeric coordinate (e.g. region labels)."""

    def test_string_coord_slice_mode_selects_exact_match(
        self, make_region_voxeldata, matplotlib_pyplot
    ):
        """plot_volume selects the slice matching a string coordinate exactly.

        Regression test for a `TypeError` previously raised by the nearest-neighbour
        lookup (`.sel(..., method="nearest")`) on non-numeric coordinates, and a
        `ValueError` previously raised by the `.3g`-formatted slice title.
        """
        data = make_region_voxeldata(
            values=np.arange(2 * 1 * 6 * 8).reshape(2, 1, 6, 8)
        )
        plotter = plot_volume(
            data, slice_mode="region", slice_coords=["b"], show_colorbar=False
        )

        ax = _axes(plotter)[0, 0]
        np.testing.assert_array_equal(
            ax.collections[0].get_array().data, data.sel(region="b").isel(k=0).values
        )
        assert ax.get_title() == "region = b"

    def test_string_coord_mismatch_warns_with_label(
        self, make_region_voxeldata, matplotlib_pyplot
    ):
        """add_volume reports unmatched non-numeric coordinates by their label."""
        data = make_region_voxeldata()
        plotter = plot_volume(
            data, slice_mode="region", slice_coords=["a"], show_colorbar=False
        )
        other = data.assign_coords(region=["a", "c"]).sel(region=["c"])

        with pytest.warns(UserWarning, match="region=c"):
            plotter.add_volume(other, show_colorbar=False)

    def test_non_numeric_slice_coords_without_coordinate_array_raises(
        self, make_region_voxeldata, matplotlib_pyplot
    ):
        """A non-numeric slice_coords entry is rejected when slice_mode is coordless."""
        data = make_region_voxeldata().drop_vars("region")
        with pytest.raises(ValueError, match="must be numeric positional indices"):
            plot_volume(data, slice_mode="region", slice_coords=["b"])

    def test_region_panel_order_matches_input_not_alphabetical(
        self, make_region_voxeldata, matplotlib_pyplot
    ):
        """Regression test: panels follow the given region order, unsorted.

        `_prepare_slice_inputs` used to sort every dim (including `slice_mode`)
        for pcolormesh geometry, which silently reordered non-alphabetical
        `region` coordinates and desynced them from externally-tracked labels.
        Only the two display dims should be sorted.
        """
        regions = ["SSp-bfd", "RSP", "HIP", "VPM"]
        data = make_region_voxeldata(
            regions=regions, values=np.arange(4 * 1 * 6 * 8).reshape(4, 1, 6, 8)
        )
        plotter = plot_volume(data, slice_mode="region", show_colorbar=False)

        titles = [ax.get_title() for ax in _axes(plotter).ravel()]
        assert titles == [f"region = {region}" for region in regions]


class TestTranspose:
    """Tests for `transpose` and world-space display with a non-spatial `slice_mode`."""

    def test_non_spatial_slice_mode_renders_on_world_dims(
        self, make_region_voxeldata, matplotlib_pyplot
    ):
        """A non-spatial `slice_mode` always displays in world space: panels are
        exposed on world `y`/`x` dims, same as `slice_mode="z"`/`"y"`/`"x"`."""
        data = make_region_voxeldata(
            values=np.arange(2 * 1 * 6 * 8).reshape(2, 1, 6, 8)
        )
        plotter = plot_volume(
            data, slice_mode="region", slice_coords=["a"], show_colorbar=False
        )

        ax = _axes(plotter)[0, 0]
        assert ax.get_xlabel() == "x (mm)"
        assert ax.get_ylabel() == "y (mm)"

    def test_oblique_non_flat_panel_raises(self, sample_voxeldata_3d_oblique):
        """A single-slice panel that is oblique to all 3 world axes has no
        well-defined 2D display once resampled onto an axis-aligned world grid
        (its extent spreads across more than one voxel along every world axis), so
        world display must raise instead of guessing a thickness -- and must do so
        before running the (wasted) actual interpolation."""
        data = sample_voxeldata_3d_oblique.isel(k=[0]).expand_dims(region=["a", "b"])
        with pytest.raises(ValueError, match="would not collapse to a 2D plane"):
            plot_volume(
                data, slice_mode="region", slice_coords=["a"], show_colorbar=False
            )

    def test_dask_backed_data_and_mask_each_computed_once(self, matplotlib_pyplot):
        """A dask-backed data/mask array must be computed exactly once through the
        plotting pipeline, not once per `.values` touch (whole-array materialize,
        per-panel isel, pcolormesh draw, ...).

        Regression: `_prepare_slice_inputs` never called `.compute()`, so a
        not-yet-materialized dask array was silently recomputed from scratch on
        every touch -- for an expensive upstream pipeline (e.g. SeedBasedMaps'
        confound regression + correlation), this multiplied real cost by however
        many times incidental code paths happened to touch `.values`.
        """
        import dask.array as dask_array
        from dask import delayed

        calls = {"data": 0, "mask": 0}
        shape = (2, 1, 6, 8)

        def _make(key):
            @delayed
            def _gen():
                calls[key] += 1
                return np.arange(np.prod(shape)).reshape(shape).astype(float)

            return dask_array.from_delayed(_gen(), shape=shape, dtype=float)

        affine = np.diag([0.2, 0.1, 0.1, 1.0])
        data = create_voxeldata(
            _make("data"), dims=("region", "k", "j", "i"), voxel_to_world=affine
        ).assign_coords(region=["a", "b"])
        mask = create_voxeldata(
            _make("mask"), dims=("region", "k", "j", "i"), voxel_to_world=affine
        ).assign_coords(region=["a", "b"])

        plotter = plot_volume(data, slice_mode="region", show_colorbar=False)
        plotter.add_contours(mask)

        assert calls["data"] == 1
        assert calls["mask"] == 1

    def test_transpose_swaps_row_and_column_display_dims(
        self, make_region_voxeldata, matplotlib_pyplot
    ):
        """`transpose=True` swaps which display dim is row vs column, and the plotted
        pixel array is reoriented to match (not just the axis labels)."""
        values = np.arange(2 * 1 * 6 * 8).reshape(2, 1, 6, 8)
        data = make_region_voxeldata(values=values)
        plotter = plot_volume(
            data,
            slice_mode="region",
            transpose=True,
            slice_coords=["a"],
            show_colorbar=False,
        )

        ax = _axes(plotter)[0, 0]
        assert ax.get_xlabel() == "y (mm)"
        assert ax.get_ylabel() == "x (mm)"
        expected = data.sel(region="a").isel(k=0).transpose("i", "j").values
        npt.assert_array_equal(ax.collections[0].get_array().data, expected)


class TestVolumePlotterUtilities:
    """Tests for VolumePlotter utility methods."""

    def test_savefig_creates_file(
        self, sample_voxeldata_3d, matplotlib_pyplot, tmp_path
    ):
        """savefig creates a non-empty file."""
        plotter = plot_volume(sample_voxeldata_3d, slice_mode="z")
        output_file = tmp_path / "test_output.png"
        plotter.savefig(str(output_file))

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_savefig_before_figure_raises(self, tmp_path):
        """savefig raises RuntimeError when called before any plot."""
        plotter = VolumePlotter()
        with pytest.raises(RuntimeError):
            plotter.savefig(str(tmp_path / "output.png"))

    def test_close_figure(self, sample_voxeldata_3d, matplotlib_pyplot):
        """close releases the figure and resets state."""
        import matplotlib.pyplot as plt

        plotter = plot_volume(sample_voxeldata_3d, slice_mode="z")
        fig_num = _figure(plotter).number

        plotter.close()

        assert plotter.figure is None
        assert plotter.axes is None
        assert fig_num not in plt.get_fignums()

    def test_close_is_idempotent(self, sample_voxeldata_3d, matplotlib_pyplot):
        """close can be called multiple times without error."""
        plotter = plot_volume(sample_voxeldata_3d, slice_mode="z")
        plotter.close()
        plotter.close()

        assert plotter.figure is None


def _mask_voxeldata(data: np.ndarray) -> xr.DataArray:
    """Wrap a raw `(k, j, i)` int label array as a VoxelData mask."""
    return create_voxeldata(
        data,
        dims=("k", "j", "i"),
        spacing=(1.0, 0.5, 0.5),
        origin=(0.0, 0.0, 0.0),
    )


class TestPlotContours:
    """Tests for the plot_contours function."""

    def test_invalid_slice_mode_raises(self):
        """plot_contours raises ValueError when slice_mode is not in mask dims."""
        mask = _mask_voxeldata(np.zeros((2, 4, 4), dtype=int))
        with pytest.raises(ValueError, match="slice_mode"):
            plot_contours(mask, slice_mode="t")

    def test_non_3d_mask_raises(self, sample_voxeldata_3dt):
        """plot_contours raises ValueError for a mask with an unreduced time dim."""
        mask = sample_voxeldata_3dt.copy(
            data=np.zeros(sample_voxeldata_3dt.shape, dtype=sample_voxeldata_3dt.dtype)
        )
        with pytest.raises(ValueError, match="3D"):
            plot_contours(mask, slice_mode="z")

    def test_single_axes_object_accepted(self, matplotlib_pyplot):
        """plot_contours accepts a bare Axes object, not only an ndarray of Axes.

        Regression test for issue #66: previously raised
        AttributeError: 'Axes' object has no attribute 'flat'.
        """
        import matplotlib.pyplot as plt

        mask = _mask_voxeldata(
            np.array([[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]])
        )
        fig, ax = plt.subplots()

        plotter = plot_contours(mask, slice_mode="z", axes=ax)

        assert plotter.figure is fig

    def test_axes_count_mismatch_raises(self, matplotlib_pyplot):
        """plot_contours raises ValueError when axes count doesn't match slices."""
        import matplotlib.pyplot as plt

        mask = _mask_voxeldata(np.ones((3, 4, 4), dtype=int))
        _fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="must match number of axes"):
            plot_contours(mask, slice_mode="z", axes=ax)

    def test_all_zero_mask_returns_without_figure(self, matplotlib_pyplot):
        """plot_contours returns early without creating a figure for all-zero mask."""
        mask = _mask_voxeldata(np.zeros((2, 4, 4), dtype=int))
        plotter = plot_contours(mask, slice_mode="z")
        assert plotter.figure is None

    def test_fontsize_scales_contour_text_elements(self, matplotlib_pyplot):
        """plot_contours scales title, label, and tick text from fontsize."""
        mask = _mask_voxeldata(
            np.array([[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]])
        )
        plotter = plot_contours(mask, slice_mode="z", fontsize=16)
        ax = _axes(plotter)[0, 0]

        assert ax.title.get_fontsize() == pytest.approx(16)
        assert ax.xaxis.label.get_fontsize() == pytest.approx(14.4)
        assert ax.yaxis.label.get_fontsize() == pytest.approx(14.4)
        assert ax.get_xticklabels()[0].get_fontsize() == pytest.approx(13.6)

    def test_scalar_slice_mode_from_selection(self, matplotlib_pyplot):
        """plot_contours accepts a scalar slice_mode coordinate (issue #295).

        Selecting a single index (sel(z=0.0)) drops z to a scalar coordinate; it
        should plot like the size-1 z dimension it was selected from.
        """
        mask = _mask_voxeldata(
            np.array([[[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]])
        )
        plotter = plot_contours(mask.sel(z=0.0), slice_mode="z")
        assert _axes(plotter).shape == (1, 1)


class TestVolumePlotterAddContours:
    """Tests for VolumePlotter.add_contours method."""

    def _make_mask(self, sample_voxeldata_3d, z_indices):
        """Create a mask with label 1 in a small region for the given z indices."""
        mask_data = np.zeros(
            (
                len(z_indices),
                sample_voxeldata_3d.sizes["j"],
                sample_voxeldata_3d.sizes["i"],
            ),
            dtype=int,
        )
        mask_data[:, 1:3, 1:3] = 1
        return sample_voxeldata_3d.isel(k=z_indices).copy(data=mask_data)

    def test_contours_only_on_matching_axes(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """add_contours draws lines only on axes whose z coord matches the mask."""
        plotter = plot_volume(sample_voxeldata_3d, slice_mode="z", show_colorbar=False)
        mask = self._make_mask(sample_voxeldata_3d, [0, 1])
        plotter.add_contours(mask, colors="red")

        axes_flat = _axes(plotter).ravel()
        assert len(axes_flat[0].lines) > 0
        assert len(axes_flat[1].lines) > 0
        assert len(axes_flat[2].lines) == 0
        assert len(axes_flat[3].lines) == 0

    def test_add_contours_restores_mask_with_scalar_fixed_k(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """add_contours restores a mask whose k dim was fixed away by scalar isel.

        `ensure_voxeldata` canonicalizes `mask` the same way it does the plotter's
        own `data`, so a mask coming from e.g. `full_mask.isel(k=0)` still draws on
        the single z-slice it came from instead of raising.
        """
        plotter = plot_volume(sample_voxeldata_3d, slice_mode="z", show_colorbar=False)
        single_slice = sample_voxeldata_3d.isel(k=0)
        mask_data = np.zeros(single_slice.shape, dtype=int)
        mask_data[1:3, 1:3] = 1
        mask = single_slice.copy(data=mask_data)

        plotter.add_contours(mask, colors="red")

        axes_flat = _axes(plotter).ravel()
        assert len(axes_flat[0].lines) > 0
        assert all(len(ax.lines) == 0 for ax in axes_flat[1:])

    def test_add_contours_matches_extra_facet_dim_to_slice_mode(
        self, make_region_voxeldata, matplotlib_pyplot
    ):
        """A mask can carry an extra facet dim (e.g. "region") matching slice_mode.

        Regression test: seed-based connectivity maps facet volume panels by an
        extra "region" dim (`plot_volume(data, slice_mode="region")`) and overlay
        each seed's own contour via `add_contours(seed_masks.rename(mask="region"))`
        -- one contour per matching region panel.
        """
        data = make_region_voxeldata(regions=("a", "b", "c"))
        plotter = plot_volume(data, slice_mode="region", show_colorbar=False)

        mask_data = np.zeros(data.shape, dtype=int)
        mask_data[0, 0, 1:3, 1:3] = 1
        mask_data[1, 0, 1:3, 1:3] = 2
        mask_data[2, 0, 1:3, 1:3] = 3
        seed_masks = data.copy(data=mask_data)

        plotter.add_contours(seed_masks)

        axes_flat = _axes(plotter).ravel()
        assert [len(ax.lines) for ax in axes_flat[:3]] == [1, 1, 1]

    def test_world_resample_preserves_each_panels_own_resolution(self):
        """Design: design/world-mode-resample-scoping.md, Design A.

        Each pose-faceted panel displayed in world space must keep its own
        native resolution, never get forced onto another panel's grid via a
        shared reference. `_resample_pose_slices_to_world_grid`'s per-panel loop is
        now only exercised for `slice_mode="pose"` -- any other non-spatial
        `slice_mode` (e.g. `"region"`) is resampled once for the whole array in
        `_prepare_slice_inputs`, which structurally can't share a reference grid
        across panels (there's only one array).

        Regression: `self._world_grid_reference` used to get set from the
        first panel processed on a plotter and reused for every later one,
        silently downsampling a finer-resolution overlay (e.g. atlas
        annotations) to a coarser background's resolution -- exactly the
        `plot_stat_map(..., slice_mode="pose") + add_contours(seed_masks)`
        scenario, where seed_masks (25 um) got downsampled to bg_volume's
        (100 um) resolution.
        """
        coarse = create_voxeldata(
            np.zeros((1, 30, 42)),
            dims=("k", "j", "i"),
            voxel_to_world=np.diag([0.2, 0.1, 0.1, 1.0]),
        )
        fine = create_voxeldata(
            np.zeros((1, 90, 126)),
            dims=("k", "j", "i"),
            voxel_to_world=np.diag([0.2, 0.1 / 3, 0.1 / 3, 1.0]),
        )
        plotter = VolumePlotter(slice_mode="pose")

        resampled_coarse = plotter._resample_pose_slices_to_world_grid([coarse])[0]
        resampled_fine = plotter._resample_pose_slices_to_world_grid([fine])[0]

        assert resampled_coarse.shape == (30, 42)
        assert resampled_fine.shape == (90, 126)

    def test_add_contours_string_rgb_lookup_keys(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """add_contours must not raise when rgb_lookup keys are strings.

        Masks whose rgb_lookup has string keys (e.g. from a user-drawn seed map)
        should render without error.
        """
        plotter = plot_volume(sample_voxeldata_3d, slice_mode="z", show_colorbar=False)

        mask_data = np.zeros(sample_voxeldata_3d.shape, dtype=int)
        mask_data[:, 1:3, 1:3] = 1
        mask = sample_voxeldata_3d.copy(data=mask_data)
        mask.attrs["rgb_lookup"] = {"1": [255, 0, 0]}
        # Should not raise TypeError about concatenating str and int.
        plotter.add_contours(mask)

    def test_add_contours_warns_on_missing_coords(
        self, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """add_contours warns when mask slice coordinates don't match any axes."""
        plotter = plot_volume(
            sample_voxeldata_3d,
            slice_mode="z",
            slice_coords=[_world_coord_1d(sample_voxeldata_3d, "z")[2]],
            show_colorbar=False,
        )
        # Mask with z coords that don't match the single plotted slice
        mask = self._make_mask(sample_voxeldata_3d, [0, 1])
        with pytest.warns(UserWarning, match="Could not find matching axes"):
            plotter.add_contours(mask, colors="red")


class TestRoiHover:
    """Tests for the ROI hover feature wired into add_volume / add_contours."""

    @staticmethod
    def _fire_motion(ax, xdata, ydata):
        """Synthesise a `motion_notify_event` at axes-data `(xdata, ydata)`."""
        from matplotlib.backend_bases import MouseEvent

        fig = ax.figure
        fig.canvas.draw()
        x_disp, y_disp = ax.transData.transform((xdata, ydata))
        ev = MouseEvent("motion_notify_event", fig.canvas, x_disp, y_disp)
        fig.canvas.callbacks.process("motion_notify_event", ev)
        return ev

    def test_hover_shows_value_and_roi(
        self, sample_roi_labels, sample_voxeldata_3d, matplotlib_pyplot
    ):
        """Cover the three hover paths: label-only, value-only, and overlay.

        Each registered slice contributes its own `<DataArray.name>=<value>`
        segment, so the overlay path produces both segments without either
        shadowing the other.
        """
        labels = sample_roi_labels
        rng = np.random.default_rng(0)
        volume = sample_voxeldata_3d.copy(
            data=rng.normal(size=sample_voxeldata_3d.shape).astype(np.float32)
        ).rename("pd")
        volume.attrs["units"] = "dB"
        roi_labels = {3: "motor", 7: "somatosensory", 42: "visual"}
        x = float(_world_coord_1d(labels, "x")[1])
        y = float(_world_coord_1d(labels, "y")[1])
        z = float(_world_coord_1d(labels, "z")[0])
        sampled_value = float(volume.sel(z=z, y=y, x=x).values)

        # Atlas-only: one segment from the labels slice, no value-line.
        atlas_plotter = plot_volume(
            labels,
            slice_mode="z",
            slice_coords=[z],
            show_colorbar=False,
            roi_labels=roi_labels,
        )
        ax = _axes(atlas_plotter).flat[0]
        self._fire_motion(ax, x, y)
        assert ax.format_coord(x, y) == f"x={x:.3g}, y={y:.3g}; roi_labels=3 (motor)"

        # Background voxel (label=0) drops the labels segment entirely.
        bg_x = float(_world_coord_1d(labels, "x")[-1])
        bg_y = float(_world_coord_1d(labels, "y")[-1])
        self._fire_motion(ax, bg_x, bg_y)
        bg_info = ax.format_coord(bg_x, bg_y)
        assert "annotation" not in bg_info

        # Volume-only: one segment using `data.name` and `units`.
        volume_plotter = plot_volume(
            volume, slice_mode="z", slice_coords=[z], show_colorbar=False
        )
        ax = _axes(volume_plotter).flat[0]
        self._fire_motion(ax, x, y)
        assert (
            ax.format_coord(x, y) == f"x={x:.3g}, y={y:.3g}; pd={sampled_value:.4g} dB"
        )

        # Overlay: both segments in registration order, neither shadowing the other.
        overlay = VolumePlotter(slice_mode="z")
        overlay.add_volume(
            volume, slice_coords=[z], match_coordinates=False, show_colorbar=False
        )
        overlay.add_contours(labels, slice_coords=[z], roi_labels=roi_labels)
        ax = _axes(overlay).flat[0]
        self._fire_motion(ax, x, y)
        assert (
            ax.format_coord(x, y) == f"x={x:.3g}, y={y:.3g}; pd={sampled_value:.4g} dB"
            "; roi_labels=3 (motor)"
        )

    def test_hover_survives_plotter_gc(self, sample_roi_labels, matplotlib_pyplot):
        """Hover stays wired up after the returned plotter is dropped and GC'd.

        Regression test for the fix that anchors active `_HoverManager`
        instances in a module-level set: matplotlib's `CallbackRegistry`
        stores bound methods as `WeakMethod`, so without the strong-ref
        registry the manager would be collected as soon as the
        `VolumePlotter` returned by `plot_volume(...)` went out of scope
        (e.g. `plot_volume(...).show()`), silently disabling hover.
        """
        import gc
        import weakref

        from confusius.plotting._hover import _CONFUSIUS_HOVER_MANAGERS

        labels = sample_roi_labels
        roi_labels = {3: "motor", 7: "somatosensory", 42: "visual"}
        x = float(_world_coord_1d(labels, "x")[1])
        y = float(_world_coord_1d(labels, "y")[1])
        z = float(_world_coord_1d(labels, "z")[0])

        plotter = plot_volume(
            labels,
            slice_mode="z",
            slice_coords=[z],
            show_colorbar=False,
            roi_labels=roi_labels,
        )
        fig = _figure(plotter)
        ax = _axes(plotter).flat[0]
        plotter_ref = weakref.ref(plotter)
        manager_ref = weakref.ref(plotter._hover_manager)
        assert manager_ref() in _CONFUSIUS_HOVER_MANAGERS

        del plotter
        gc.collect()

        # The plotter is gone, but the hover manager must still be alive
        # (anchored by the module-level registry) and still wired to the
        # canvas's motion_notify_event.
        assert plotter_ref() is None
        assert manager_ref() is not None
        assert manager_ref() in _CONFUSIUS_HOVER_MANAGERS

        self._fire_motion(ax, x, y)
        assert ax.format_coord(x, y) == f"x={x:.3g}, y={y:.3g}; roi_labels=3 (motor)"

        # Closing the figure must release the manager from the registry,
        # after which it is free to be garbage collected. The Agg backend
        # used in tests does not dispatch `close_event` from `plt.close`,
        # so emit it explicitly to mirror what interactive backends do.
        from matplotlib.backend_bases import CloseEvent

        fig.canvas.callbacks.process(
            "close_event", CloseEvent("close_event", fig.canvas)
        )
        assert manager_ref() not in _CONFUSIUS_HOVER_MANAGERS

        matplotlib_pyplot.close(fig)
        del ax, fig
        gc.collect()
        assert manager_ref() is None


@pytest.fixture
def reproducible_baseline_voxeldata():
    """Reproducible VoxelData array for plotting baseline tests."""
    rng = np.random.default_rng(42)
    shape = (4, 6, 8)
    data = rng.random(shape)
    return create_voxeldata(
        data,
        dims=("k", "j", "i"),
        spacing=(0.1, 0.05, 0.05),
        origin=(0.0, 0.0, 0.0),
        attrs={"long_name": "Intensity", "units": "a.u."},
    )


@pytest.fixture
def contour_baseline_voxeldata():
    """Small VoxelData grid matching the existing contour baselines."""
    return create_voxeldata(
        np.zeros((2, 4, 4), dtype=float),
        dims=("k", "j", "i"),
        spacing=(1.0, 0.5, 0.5),
        origin=(0.0, 0.0, 0.0),
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
    def test_plot_volume_default(
        self, matplotlib_pyplot, reproducible_baseline_voxeldata
    ):
        """Baseline test for default plot_volume appearance (black background)."""
        volume = reproducible_baseline_voxeldata
        plotter = plot_volume(volume, slice_mode="z")
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_volume_single_slice(
        self, matplotlib_pyplot, reproducible_baseline_voxeldata
    ):
        """Baseline test for single slice."""
        volume = reproducible_baseline_voxeldata
        z_coord = _world_coord_1d(volume, "z")[0]
        plotter = plot_volume(volume, slice_mode="z", slice_coords=[z_coord])
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_volume_custom_grid(
        self, matplotlib_pyplot, reproducible_baseline_voxeldata
    ):
        """Baseline test for custom grid layout (1 row, 4 columns)."""
        volume = reproducible_baseline_voxeldata
        plotter = plot_volume(volume, slice_mode="z", nrows=1, ncols=4)
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_volume_overlay(
        self, matplotlib_pyplot, reproducible_baseline_voxeldata
    ):
        """Baseline test for overlaying two volumes with transparency."""
        volume = reproducible_baseline_voxeldata
        plotter = plot_volume(volume, slice_mode="z")

        subset_coords = _world_coord_1d(volume, "z")[[0, 3]].tolist()
        subset_data = volume.sel(z=subset_coords)
        plotter.add_volume(subset_data, cmap="hot", alpha=0.5)

        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_volume_threshold(
        self, matplotlib_pyplot, reproducible_baseline_voxeldata
    ):
        """Baseline test for thresholding visualization."""
        volume = reproducible_baseline_voxeldata
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
    def test_plot_volume_no_colorbar(
        self, matplotlib_pyplot, reproducible_baseline_voxeldata
    ):
        """Baseline test without colorbar."""
        volume = reproducible_baseline_voxeldata
        plotter = plot_volume(volume, slice_mode="z", show_colorbar=False)
        return plotter.figure

    @pytest.mark.parametrize(
        "bg_color",
        [
            pytest.param("#1a1a2e", id="dark"),  # WCAG luminance < 0.179 → white fg
            pytest.param("white", id="light"),  # WCAG luminance = 1.0 → black fg
        ],
    )
    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_volume_custom_bg_color(
        self, matplotlib_pyplot, bg_color, reproducible_baseline_voxeldata
    ):
        """Baseline test for custom bg_color — WCAG auto-derives white or black fg."""
        volume = reproducible_baseline_voxeldata
        plotter = plot_volume(volume, slice_mode="z", bg_color=bg_color)
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_volume_explicit_fg_color(
        self, matplotlib_pyplot, reproducible_baseline_voxeldata
    ):
        """Baseline test for explicit fg_color override."""
        volume = reproducible_baseline_voxeldata
        plotter = plot_volume(
            volume, slice_mode="z", bg_color="black", fg_color="#aaaaaa"
        )
        return plotter.figure


class TestPlotContoursVisualRegression:
    """Visual regression tests for plot_contours using pytest-mpl."""

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_contours_basic(self, matplotlib_pyplot, contour_baseline_voxeldata):
        """Baseline test for basic plot_contours."""
        mask_data = np.array(
            [
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]],
            ]
        )
        mask = contour_baseline_voxeldata.copy(data=mask_data)
        plotter = plot_contours(mask, slice_mode="z", colors={1: "red", 2: "blue"})
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_contours_white_bg(
        self, matplotlib_pyplot, contour_baseline_voxeldata
    ):
        """Baseline test for plot_contours on a white background."""
        mask_data = np.array(
            [
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]],
            ]
        )
        mask = contour_baseline_voxeldata.copy(data=mask_data)
        plotter = plot_contours(
            mask, slice_mode="z", colors={1: "red", 2: "blue"}, bg_color="white"
        )
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_contours_overlay_on_volume(
        self, matplotlib_pyplot, contour_baseline_voxeldata
    ):
        """Baseline test for add_contours overlay on volume."""
        rng = np.random.default_rng(42)
        volume = contour_baseline_voxeldata.copy(
            data=rng.random(contour_baseline_voxeldata.shape)
        )
        mask_data = np.array(
            [
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]],
            ]
        )
        mask = contour_baseline_voxeldata.copy(data=mask_data)
        plotter = plot_volume(volume, slice_mode="z", show_colorbar=False)
        plotter.add_contours(mask, colors={1: "red", 2: "blue"})
        return plotter.figure


def _create_deterministic_time_series() -> xr.DataArray:
    """Create deterministic reduced `(time, region)` signals for visual tests."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 12))
    return xr.DataArray(
        data,
        dims=("time", "region"),
        coords={
            "time": xr.DataArray(
                np.arange(20) * 0.1,
                dims=("time",),
                attrs={"units": "s"},
            ),
            "region": np.arange(12),
        },
    )


class TestPlotCarpet:
    """Tests for non-visual plot_carpet behaviour."""

    def test_plot_carpet_accepts_voxeldata_input(
        self, sample_voxeldata_3dt, matplotlib_pyplot
    ):
        """plot_carpet extracts signals via the VoxelData grid-aware path.

        Unlike an already-extracted signals array, a genuine (time, k, j, i)
        VoxelData array carries a `VoxelToWorldIndex`, so `select_masked_features`
        must dispatch to `extract_with_mask` rather than the plain-flatten path.
        """
        fig, ax = plot_carpet(sample_voxeldata_3dt, standardize=False)

        assert fig is not None
        assert ax.get_ylabel() == "Voxels"

    def test_plot_carpet_accepts_reduced_space_dim(self, matplotlib_pyplot):
        """plot_carpet accepts already-reduced `(time, space)` signals."""
        data = _create_deterministic_time_series().rename(region="space")

        fig, ax = plot_carpet(data, standardize=False)

        assert fig is not None
        assert ax.get_ylabel() == "Voxels"

    def test_fontsize_scales_carpet_text_elements(self, matplotlib_pyplot):
        """plot_carpet scales title, label, tick, and colorbar text from fontsize."""
        data = _create_deterministic_time_series()
        fig, ax = plot_carpet(
            data,
            standardize=False,
            title="Carpet",
            fontsize=18,
        )

        assert ax.title.get_fontsize() == pytest.approx(18)
        assert ax.xaxis.label.get_fontsize() == pytest.approx(16.2)
        assert ax.yaxis.label.get_fontsize() == pytest.approx(16.2)
        assert ax.get_xticklabels()[0].get_fontsize() == pytest.approx(15.3)

        cbar_axes = [axis for axis in fig.axes if axis is not ax]
        assert len(cbar_axes) == 1
        assert cbar_axes[0].get_yticklabels()[0].get_fontsize() == pytest.approx(15.3)


class TestPlotCarpetVisualRegression:
    """Visual regression tests for plot_carpet using pytest-mpl."""

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_carpet_default(self, matplotlib_pyplot):
        """Baseline test for default plot_carpet appearance (white background)."""
        data = _create_deterministic_time_series()
        fig, _ = plot_carpet(data, standardize=False)
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_carpet_dark_bg(self, matplotlib_pyplot):
        """Baseline test for plot_carpet with dark background."""
        data = _create_deterministic_time_series()
        fig, _ = plot_carpet(data, standardize=False, bg_color="black")
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_plot_carpet_explicit_fg_color(self, matplotlib_pyplot):
        """Baseline test for plot_carpet with explicit fg_color."""
        data = _create_deterministic_time_series()
        fig, _ = plot_carpet(
            data, standardize=False, bg_color="black", fg_color="#aaaaaa"
        )
        return fig
