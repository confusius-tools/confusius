"""Tests for VolumePlotter.add_composite."""

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from confusius.plotting import VolumePlotter, plot_composite


def _axes(plotter):
    assert plotter.axes is not None
    return plotter.axes


def _shifted_volume(template: xr.DataArray, shift: float = 0.07) -> xr.DataArray:
    """Return a second volume on the same grid as `template` with shifted values."""
    return xr.DataArray(
        np.roll(template.values, shift=1, axis=-1) + shift,
        name="moving",
        dims=template.dims,
        coords={d: template.coords[d] for d in template.dims if d in template.coords},
        attrs=dict(template.attrs),
    )


class TestAddCompositeChannels:
    """Verify the red/cyan channel mapping."""

    def test_red_channel_tracks_data1_cyan_tracks_data2(
        self, sample_3d_volume: xr.DataArray, matplotlib_pyplot
    ):
        data1 = sample_3d_volume
        data2 = _shifted_volume(data1)

        plotter = VolumePlotter(slice_mode="z").add_composite(
            data1, data2, resample=False, normalize_strategy="per_slice"
        )

        slice1 = data1.isel(z=0).values.astype(float)
        slice2 = data2.isel(z=0).values.astype(float)

        def _norm(arr):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

        rgb = _axes(plotter)[0, 0].collections[0].get_array()
        assert rgb.shape == (data1.sizes["y"], data1.sizes["x"], 3)
        npt.assert_allclose(rgb[..., 0], _norm(slice1), atol=1e-6)
        npt.assert_allclose(rgb[..., 1], _norm(slice2), atol=1e-6)
        npt.assert_allclose(rgb[..., 2], _norm(slice2), atol=1e-6)


class TestAddCompositeResample:
    """Verify the resample=True grid-alignment behaviour."""

    def test_resamples_data2_onto_data1_grid(self, sample_3d_volume, matplotlib_pyplot):
        data1 = sample_3d_volume
        data2 = xr.DataArray(
            np.linspace(0, 1, 3 * 4 * 5).reshape(3, 4, 5),
            dims=["z", "y", "x"],
            coords={
                "z": np.array([1.05, 1.35, 1.65]),
                "y": np.array([2.0, 2.2, 2.4, 2.6]),
                "x": np.array([3.0, 3.1, 3.2, 3.3, 3.4]),
            },
        )

        plotter = VolumePlotter(slice_mode="z").add_composite(
            data1, data2, resample=True
        )

        # All panels should be rendered at data1's (y, x) shape.
        rgb0 = _axes(plotter)[0, 0].collections[0].get_array()
        assert rgb0.shape == (data1.sizes["y"], data1.sizes["x"], 3)
        # And we should get one panel per data1 z-slice.
        rendered = [ax for ax in _axes(plotter).ravel() if ax.collections]
        assert len(rendered) == data1.sizes["z"]

    def test_resample_false_shape_mismatch_raises(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        data2 = sample_3d_volume.isel(x=slice(0, 4))
        with pytest.raises(ValueError, match="share shape"):
            VolumePlotter(slice_mode="z").add_composite(
                sample_3d_volume, data2, resample=False
            )

    def test_resample_false_coord_mismatch_raises_by_default(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        data2 = sample_3d_volume.assign_coords(
            x=sample_3d_volume.coords["x"].values + 0.5
        )
        with pytest.raises(ValueError, match="does not match"):
            VolumePlotter(slice_mode="z").add_composite(
                sample_3d_volume, data2, resample=False
            )

    def test_resample_false_tolerant_drift_replaces_with_data1_coords(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        # Tiny floating-point drift in data2's coords (well below the default
        # atol) should pass validation, and data2's coords should be replaced
        # with data1's so downstream slicing sees a single coordinate frame.
        x_coords = sample_3d_volume.coords["x"].values.astype(float)
        drift = 1e-11
        data2 = sample_3d_volume.assign_coords(x=x_coords + drift)

        plotter = VolumePlotter(slice_mode="z").add_composite(
            sample_3d_volume, data2, resample=False
        )

        # The rendered xlim should sit on data1's coordinate centre.
        x_min, x_max = _axes(plotter).ravel()[0].get_xlim()
        xlim_mid = 0.5 * (x_min + x_max)
        data1_mid = 0.5 * (float(x_coords.min()) + float(x_coords.max()))
        assert abs(xlim_mid - data1_mid) < 1e-6

    def test_resample_false_widened_tol_accepts_large_drift(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        # A larger coord shift fails by default but passes when atol is widened
        # past the offset. data2's coords are replaced with data1's, so the
        # rendered axes sit on data1's grid.
        x_coords = sample_3d_volume.coords["x"].values.astype(float)
        x_shift = 0.5
        data2 = sample_3d_volume.assign_coords(x=x_coords + x_shift)

        plotter = VolumePlotter(slice_mode="z").add_composite(
            sample_3d_volume, data2, resample=False, atol=1.0
        )

        x_min, x_max = _axes(plotter).ravel()[0].get_xlim()
        xlim_mid = 0.5 * (x_min + x_max)
        data1_mid = 0.5 * (float(x_coords.min()) + float(x_coords.max()))
        data2_mid = data1_mid + x_shift
        assert abs(xlim_mid - data1_mid) < abs(xlim_mid - data2_mid)


class TestAddCompositeResampleKwargs:
    """Verify resample_kwargs fill behaviour."""

    @pytest.fixture
    def narrow_data2(self, sample_3d_volume):
        """data2 that covers only a sub-region of data1's grid."""
        # Use data1's central x-slice only, forcing out-of-FOV voxels on resample.
        x = sample_3d_volume.coords["x"].values
        x_sub = x[len(x) // 4 : 3 * len(x) // 4]
        data2 = xr.DataArray(
            np.full(
                (sample_3d_volume.sizes["z"], sample_3d_volume.sizes["y"], len(x_sub)),
                fill_value=5.0,
            ),
            dims=["z", "y", "x"],
            coords={
                "z": sample_3d_volume.coords["z"].values,
                "y": sample_3d_volume.coords["y"].values,
                "x": x_sub,
            },
        )
        return sample_3d_volume, data2

    def test_default_fill_is_data2_min(self, narrow_data2, matplotlib_pyplot):
        """Out-of-FOV cyan voxels default to data2.min(), not 0."""
        data1, data2 = narrow_data2
        plotter = VolumePlotter(slice_mode="z").add_composite(
            data1, data2, resample=True, normalize_strategy="shared"
        )
        # data2 is constant 5.0, so data2.min() == 5.0. Out-of-FOV voxels should
        # be filled with 5.0, meaning the cyan channel is uniform (not bright).
        # Under shared normalisation the range spans [data1.min(), 5.0]; the
        # fill value 5.0 normalises to 1.0, so no cyan voxel should exceed the
        # cyan contribution from in-FOV data2 voxels.
        rgb0 = _axes(plotter)[0, 0].collections[0].get_array()
        # Cyan channel (G == B). All voxels in data2 have the same value (5.0),
        # so after normalisation the cyan channel should be uniform at 1.0.
        npt.assert_allclose(rgb0[..., 1], 1.0, atol=1e-5)

    def test_explicit_fill_value_overrides_default(self, narrow_data2, matplotlib_pyplot):
        """Explicit fill_value in resample_kwargs is respected."""
        data1, data2 = narrow_data2
        # Passing fill_value=0.0 explicitly should fill out-of-FOV with 0.0.
        plotter = VolumePlotter(slice_mode="z").add_composite(
            data1, data2, resample=True, normalize_strategy="shared",
            resample_kwargs={"fill_value": 0.0},
        )
        rgb0 = _axes(plotter)[0, 0].collections[0].get_array()
        # range is [min(data1), 5.0]. fill=0.0 normalises to something < in-FOV.
        # At least some out-of-FOV cyan voxels should be < 1.0 (since fill < max).
        assert float(rgb0[..., 1].min()) < 1.0


class TestAddCompositeNormalize:
    """Verify the per_volume, per_slice, and shared normalisation modes."""

    @pytest.fixture
    def lopsided_pair(self, sample_3d_volume):
        """Pair where the last z-slice is much brighter than the rest."""
        data1 = sample_3d_volume.copy()
        boosted = data1.values.copy()
        boosted[:-1] *= 0.05
        boosted[-1] += 10.0
        data1 = xr.DataArray(
            boosted, dims=data1.dims, coords=data1.coords, attrs=data1.attrs
        )
        return data1, data1.copy()

    def test_per_volume_preserves_dim_slices(self, lopsided_pair, matplotlib_pyplot):
        data1, data2 = lopsided_pair
        plotter = VolumePlotter(slice_mode="z").add_composite(
            data1, data2, resample=False, normalize_strategy="per_volume"
        )

        # The dim slices share the same volume-wide normalisation, so their max
        # in the red channel should stay well below 1.0.
        dim_max = max(
            float(_axes(plotter).ravel()[i].collections[0].get_array()[..., 0].max())
            for i in range(data1.sizes["z"] - 1)
        )
        bright_max = float(
            _axes(plotter).ravel()[data1.sizes["z"] - 1]
            .collections[0]
            .get_array()[..., 0]
            .max()
        )
        assert dim_max < 0.2
        assert bright_max == pytest.approx(1.0, abs=1e-6)

    def test_per_slice_saturates_every_slice(self, lopsided_pair, matplotlib_pyplot):
        data1, data2 = lopsided_pair
        plotter = VolumePlotter(slice_mode="z").add_composite(
            data1, data2, resample=False, normalize_strategy="per_slice"
        )
        for i in range(data1.sizes["z"]):
            rgb = _axes(plotter).ravel()[i].collections[0].get_array()
            assert float(rgb[..., 0].max()) == pytest.approx(1.0, abs=1e-6)

    def test_shared_uses_one_range(self, sample_3d_volume, matplotlib_pyplot):
        # data1 covers [0, 1]; data2 covers [0, 4]. With shared normalisation
        # both volumes share the same [0, 4] denominator, so data1 should max
        # out near 0.25 in the red channel and data2 near 1.0 in green/blue.
        data1 = xr.DataArray(
            np.linspace(0, 1, 4 * 6 * 8).reshape(4, 6, 8),
            dims=sample_3d_volume.dims,
            coords=sample_3d_volume.coords,
        )
        data2 = xr.DataArray(
            np.linspace(0, 4, 4 * 6 * 8).reshape(4, 6, 8),
            dims=sample_3d_volume.dims,
            coords=sample_3d_volume.coords,
        )
        plotter = VolumePlotter(slice_mode="z").add_composite(
            data1, data2, resample=False, normalize_strategy="shared"
        )
        red_max = max(
            float(_axes(plotter).ravel()[i].collections[0].get_array()[..., 0].max())
            for i in range(data1.sizes["z"])
        )
        cyan_max = max(
            float(_axes(plotter).ravel()[i].collections[0].get_array()[..., 1].max())
            for i in range(data1.sizes["z"])
        )
        assert red_max == pytest.approx(0.25, abs=1e-6)
        assert cyan_max == pytest.approx(1.0, abs=1e-6)


class TestAddCompositeValidation:
    """Validation guards on add_composite inputs."""

    def test_rejects_time_dim(self, sample_3dt_volume, matplotlib_pyplot):
        spatial = sample_3dt_volume.isel(time=0).drop_vars("time")
        with pytest.raises(ValueError, match="time"):
            VolumePlotter(slice_mode="z").add_composite(sample_3dt_volume, spatial)

    def test_requires_slice_mode_dim(self, sample_3d_volume, matplotlib_pyplot):
        with pytest.raises(ValueError, match="slice_mode"):
            VolumePlotter(slice_mode="t").add_composite(
                sample_3d_volume, sample_3d_volume
            )

    def test_invalid_normalize_raises(self, sample_3d_volume, matplotlib_pyplot):
        with pytest.raises(ValueError, match="normalization strategy"):
            VolumePlotter(slice_mode="z").add_composite(
                sample_3d_volume,
                sample_3d_volume,
                resample=False,
                normalize_strategy="foo",  # ty: ignore[invalid-argument-type]
            )


def _create_deterministic_composite_pair():
    """Deterministic 3D volume pair designed to differentiate normalisation modes.

    `data1` carries a per-slice intensity ramp so dim slices stay dim under
    `per_volume` but saturate under `per_slice`. `data2` covers an absolute
    range several times wider than `data1`'s, so `shared` compresses `data1`
    while `per_volume` does not.
    """
    rng = np.random.default_rng(42)
    shape = (4, 6, 8)
    coords = {
        "z": xr.DataArray(np.arange(4) * 0.1, dims=["z"], attrs={"units": "mm"}),
        "y": xr.DataArray(np.arange(6) * 0.05, dims=["y"], attrs={"units": "mm"}),
        "x": xr.DataArray(np.arange(8) * 0.05, dims=["x"], attrs={"units": "mm"}),
    }

    # Per-slice intensity scaling: max(data1) = 1.0 only on slice 2; the other
    # slices peak at 0.1, 0.3, and 0.05 respectively.
    base1 = rng.random(shape)
    per_slice_scale = np.array([0.1, 0.3, 1.0, 0.05])
    data1 = xr.DataArray(
        base1 * per_slice_scale[:, None, None],
        dims=["z", "y", "x"],
        coords=coords,
        name="fixed",
    )

    # data2 spans roughly [0, 4] — about 4x data1's full-volume range — so the
    # shared scale compresses data1 noticeably while data2 stays bright.
    base2 = rng.random(shape)
    data2 = xr.DataArray(
        base2 * 4.0,
        dims=["z", "y", "x"],
        coords=coords,
        name="moving",
    )
    return data1, data2


class TestPlotComposite:
    """Tests for the top-level plot_composite helper."""

    def test_returns_volume_plotter_with_one_panel_per_slice(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        plotter = plot_composite(sample_3d_volume, sample_3d_volume, resample=False)
        assert isinstance(plotter, VolumePlotter)
        rendered = [ax for ax in _axes(plotter).ravel() if ax.collections]
        assert len(rendered) == sample_3d_volume.sizes["z"]

    def test_forwards_slice_mode_to_volume_plotter(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        plotter = plot_composite(
            sample_3d_volume, sample_3d_volume, resample=False, slice_mode="y"
        )
        assert plotter.slice_mode == "y"
        rendered = [ax for ax in _axes(plotter).ravel() if ax.collections]
        assert len(rendered) == sample_3d_volume.sizes["y"]

    def test_rejects_time_dim(self, sample_3dt_volume, matplotlib_pyplot):
        with pytest.raises(ValueError, match="time"):
            plot_composite(sample_3dt_volume, sample_3dt_volume)


class TestAddCompositeVisualRegression:
    """Visual regression tests for VolumePlotter.add_composite."""

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_add_composite_per_volume_normalize(self, matplotlib_pyplot):
        """Baseline test for per-volume normalisation (the default)."""
        data1, data2 = _create_deterministic_composite_pair()
        plotter = VolumePlotter(slice_mode="z").add_composite(
            data1, data2, resample=False, normalize_strategy="per_volume"
        )
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_add_composite_per_slice_normalize(self, matplotlib_pyplot):
        """Baseline test for per-slice normalisation."""
        data1, data2 = _create_deterministic_composite_pair()
        plotter = VolumePlotter(slice_mode="z").add_composite(
            data1, data2, resample=False, normalize_strategy="per_slice"
        )
        return plotter.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline",
        tolerance=0,
        savefig_kwargs={"facecolor": "auto"},
    )
    def test_add_composite_shared_normalize(self, matplotlib_pyplot):
        """Baseline test for shared (joint) normalisation."""
        data1, data2 = _create_deterministic_composite_pair()
        plotter = VolumePlotter(slice_mode="z").add_composite(
            data1, data2, resample=False, normalize_strategy="shared"
        )
        return plotter.figure


class TestCompositeAccessor:
    """Tests for the `data.fusi.plot.composite()` accessor wrapper."""

    def test_accessor_forwards_to_plot_composite(
        self, sample_3d_volume, matplotlib_pyplot
    ):
        import confusius  # noqa: F401 - register accessor.

        plotter = sample_3d_volume.fusi.plot.composite(sample_3d_volume, resample=False)
        assert isinstance(plotter, VolumePlotter)
        rendered = [ax for ax in _axes(plotter).ravel() if ax.collections]
        assert len(rendered) == sample_3d_volume.sizes["z"]

    def test_accessor_shared_normalize(self, sample_3d_volume, matplotlib_pyplot):
        import confusius  # noqa: F401 - register accessor.

        plotter = sample_3d_volume.fusi.plot.composite(
            sample_3d_volume, resample=False, normalize_strategy="shared"
        )
        # data1 == data2, so red and cyan channels must be pointwise equal.
        rgb = _axes(plotter).ravel()[0].collections[0].get_array()
        npt.assert_allclose(rgb[..., 0], rgb[..., 1])
