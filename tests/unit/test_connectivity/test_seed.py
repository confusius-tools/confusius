"""Tests for confusius.connectivity.SeedBasedMaps"""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from scipy.stats import pearsonr

from confusius.connectivity import SeedBasedMaps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data_2d(rng):
    """(time, z, x) DataArray with time coordinate — small for fast tests."""
    n_time, nz, nx = 50, 6, 8
    values = rng.standard_normal((n_time, nz, nx))
    return xr.DataArray(
        values,
        dims=["time", "z", "x"],
        coords={
            "time": np.arange(n_time) * 0.1,
            "z": np.arange(nz) * 0.1,
            "x": np.arange(nx) * 0.05,
        },
    )


@pytest.fixture
def flat_labels(data_2d):
    """Flat integer label map: two non-overlapping seed regions."""
    labels = xr.DataArray(
        np.zeros((6, 8), dtype=int),
        dims=["z", "x"],
        coords={
            "z": data_2d.coords["z"],
            "x": data_2d.coords["x"],
        },
    )
    labels[:2, :] = 1  # Region 1.
    labels[2:4, :] = 2  # Region 2.
    return labels


@pytest.fixture
def single_region_labels(data_2d):
    """Flat integer label map with exactly one non-zero label."""
    labels = xr.DataArray(
        np.zeros((6, 8), dtype=int),
        dims=["z", "x"],
        coords={
            "z": data_2d.coords["z"],
            "x": data_2d.coords["x"],
        },
    )
    labels[:2, :] = 1
    return labels


@pytest.fixture
def stacked_labels(data_2d):
    """Stacked (masks, z, x) label map with two overlapping regions."""
    layer1 = np.zeros((6, 8), dtype=int)
    layer1[:3, :] = 1

    layer2 = np.zeros((6, 8), dtype=int)
    layer2[2:5, :] = 2  # Overlaps with layer1 at z=2.

    stacked = np.stack([layer1, layer2], axis=0)
    return xr.DataArray(
        stacked,
        dims=["mask", "z", "x"],
        coords={
            "mask": ["roi_a", "roi_b"],
            "z": data_2d.coords["z"],
            "x": data_2d.coords["x"],
        },
    )


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for input validation in SeedBasedMaps.fit."""

    def test_no_time_dim_raises(self):
        """fit raises ValueError when data has no time dimension."""
        # Use coordinate-free data and seeds so spatial coord validation
        # does not fire before the time-dimension check.
        data = xr.DataArray(np.ones((6, 8)), dims=["z", "x"])
        seeds = xr.DataArray(np.ones((6, 8), dtype=int), dims=["z", "x"])
        mapper = SeedBasedMaps(seed_masks=seeds)
        with pytest.raises(ValueError, match="time"):
            mapper.fit(data)

    def test_single_timepoint_raises(self):
        """fit raises ValueError when data has only one timepoint."""
        # Use coordinate-free data and seeds so spatial coord validation
        # does not fire before the timepoint count check.
        data = xr.DataArray(np.ones((1, 6, 8)), dims=["time", "z", "x"])
        seeds = xr.DataArray(np.ones((6, 8), dtype=int), dims=["z", "x"])
        mapper = SeedBasedMaps(seed_masks=seeds)
        with pytest.raises(ValueError, match="timepoint"):
            mapper.fit(data)

    def test_non_integer_seed_raises(self, data_2d):
        """fit raises TypeError when seeds has float dtype."""
        seeds = xr.DataArray(np.zeros((6, 8)), dims=["z", "x"])
        mapper = SeedBasedMaps(seed_masks=seeds)
        with pytest.raises(TypeError, match="integer dtype"):
            mapper.fit(data_2d)

    def test_seed_dim_mismatch_raises(self, data_2d):
        """fit raises ValueError when seeds dims don't match data spatial dims."""
        seeds = xr.DataArray(np.ones((5,), dtype=int), dims=["w"])
        mapper = SeedBasedMaps(seed_masks=seeds)
        with pytest.raises(ValueError, match="missing spatial dimensions"):
            mapper.fit(data_2d)

    def test_no_seed_raises(self, data_2d):
        """fit raises ValueError when neither seed_masks nor seed_signals is given."""
        mapper = SeedBasedMaps()
        with pytest.raises(ValueError, match="neither"):
            mapper.fit(data_2d)

    def test_both_seeds_raises(self, data_2d, flat_labels):
        """fit raises ValueError when both seed_masks and seed_signals are given."""
        seed_sig = xr.DataArray(
            np.ones(data_2d.sizes["time"]),
            dims=["time"],
            coords={"time": data_2d.coords["time"]},
        )
        mapper = SeedBasedMaps(seed_masks=flat_labels, seed_signals=seed_sig)
        with pytest.raises(ValueError, match="both"):
            mapper.fit(data_2d)


# ---------------------------------------------------------------------------
# seed_signals-specific validation tests
# ---------------------------------------------------------------------------


class TestSeedSignalsValidation:
    """Tests for the extra validation applied when seed_signals is provided."""

    def test_no_time_dim_raises(self, data_2d):
        """fit raises ValueError when seed_signals has no time dimension."""
        signal = xr.DataArray(np.ones(data_2d.sizes["time"]), dims=["nottime"])
        mapper = SeedBasedMaps(seed_signals=signal)
        with pytest.raises(ValueError, match="time"):
            mapper.fit(data_2d)

    def test_single_timepoint_raises(self, data_2d):
        """fit raises ValueError when seed_signals has only one timepoint."""
        signal = xr.DataArray(np.ones(1), dims=["time"])
        mapper = SeedBasedMaps(seed_signals=signal)
        with pytest.raises(ValueError, match="timepoint"):
            mapper.fit(data_2d)

    def test_unexpected_dims_raises(self, data_2d):
        """fit raises ValueError when seed_signals has non-time/regions dimensions."""
        signal = xr.DataArray(
            np.ones((data_2d.sizes["time"], 2, 3)),
            dims=["time", "z", "x"],
        )
        mapper = SeedBasedMaps(seed_signals=signal)
        with pytest.raises(ValueError, match="unexpected dimensions"):
            mapper.fit(data_2d)

    def test_time_size_mismatch_raises(self, data_2d):
        """fit raises ValueError when seed_signals has a different number of timepoints
        than data."""
        wrong_n = data_2d.sizes["time"] + 5
        signal = xr.DataArray(np.ones(wrong_n), dims=["time"])
        mapper = SeedBasedMaps(seed_signals=signal)
        with pytest.raises(ValueError, match="timepoints"):
            mapper.fit(data_2d)

    def test_time_coord_mismatch_raises(self, data_2d):
        """fit raises ValueError when seed_signals time coordinates differ from data."""
        n = data_2d.sizes["time"]
        signal = xr.DataArray(
            np.ones(n),
            dims=["time"],
            # Shift time coords by a large amount so they cannot match.
            coords={"time": data_2d.coords["time"].values + 999.0},
        )
        mapper = SeedBasedMaps(seed_signals=signal)
        with pytest.raises(ValueError, match="time coordinates"):
            mapper.fit(data_2d)

    def test_time_coord_small_drift_is_accepted(self, data_2d):
        """fit accepts small numeric drift in seed signal time coordinates."""
        n = data_2d.sizes["time"]
        signal = xr.DataArray(
            np.ones(n),
            dims=["time"],
            coords={"time": data_2d.coords["time"].values + 1e-10},
        )

        SeedBasedMaps(seed_signals=signal).fit(data_2d)

    def test_no_time_coord_on_either_skips_coord_check(self, data_2d):
        """fit succeeds when neither seed_signals nor data has a time coordinate."""
        data_no_coord = xr.DataArray(
            data_2d.values,
            dims=data_2d.dims,
            # Omit time coordinate entirely.
            coords={k: v for k, v in data_2d.coords.items() if k != "time"},
        )
        n = data_no_coord.sizes["time"]
        signal = xr.DataArray(np.ones(n), dims=["time"])
        # Should not raise — no time coords to compare.
        SeedBasedMaps(seed_signals=signal).fit(data_no_coord)

    def test_time_coord_only_on_data_skips_coord_check(self, data_2d):
        """fit succeeds when only data has a time coordinate (no coord on signal)."""
        n = data_2d.sizes["time"]
        signal = xr.DataArray(np.ones(n), dims=["time"])
        # Should not raise — seed_signals has no time coord to compare against.
        SeedBasedMaps(seed_signals=signal).fit(data_2d)

    def test_dask_time_chunked_seed_signals_raises(self, data_2d):
        """fit raises ValueError when seed_signals is chunked along the time dimension."""
        n = data_2d.sizes["time"]
        # Build an xarray signal then chunk it along time so it becomes Dask-backed
        # with multiple time chunks.
        signal = xr.DataArray(np.ones(n), dims=["time"]).chunk({"time": 10})
        mapper = SeedBasedMaps(seed_signals=signal)
        with pytest.raises(ValueError, match="chunked"):
            mapper.fit(data_2d)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


class TestCorrectness:
    """Tests that correlation values match a scipy reference."""

    def test_pearson_r_matches_scipy(self, data_2d, single_region_labels):
        """Pearson r values match scipy.stats.pearsonr; single region is squeezed out."""
        mapper = SeedBasedMaps(seed_masks=single_region_labels).fit(data_2d)

        # Squeeze behavior: regions dim must be absent for a single region.
        assert mapper.maps_.dims == ("z", "x")

        seed_signal = mapper.seed_signals_.values.squeeze()  # (time,)
        map_values = mapper.maps_.values  # (z, x)

        for zi in range(data_2d.sizes["z"]):
            for xi in range(data_2d.sizes["x"]):
                voxel = data_2d.values[:, zi, xi]
                r, _ = pearsonr(seed_signal, voxel)
                np.testing.assert_allclose(map_values[zi, xi], r, rtol=1e-6)  # type: ignore[arg-type]

    def test_stacked_labels_correctness(self, data_2d, stacked_labels):
        """Pearson r maps for stacked labels match scipy reference for each region,
        and string region coords are propagated correctly."""
        mapper = SeedBasedMaps(seed_masks=stacked_labels).fit(data_2d)

        np.testing.assert_array_equal(
            mapper.maps_.coords["region"].values, ["roi_a", "roi_b"]
        )

        for region in ["roi_a", "roi_b"]:
            seed = mapper.seed_signals_.sel(region=region).values
            map_vals = mapper.maps_.sel(region=region).values
            for zi in range(data_2d.sizes["z"]):
                for xi in range(data_2d.sizes["x"]):
                    voxel = data_2d.values[:, zi, xi]
                    r, _ = pearsonr(seed, voxel)
                    np.testing.assert_allclose(
                        map_vals[zi, xi],
                        r,  # type: ignore[arg-type]
                        rtol=1e-6,
                    )

    def test_seed_voxels_have_high_self_correlation(self, flat_labels):
        """Seed region voxels correlate strongly with their own region signal.

        We inject a deterministic signal into all seed-region voxels so that
        their average (the seed signal) is that same deterministic signal.
        Random noise outside the seed region should not affect this.
        """
        rng = np.random.default_rng(42)
        n_time, nz, nx = 50, 6, 8
        values = rng.standard_normal((n_time, nz, nx))

        # Overwrite region-1 voxels (z=0,1) with a pure deterministic signal.
        signal = np.sin(np.linspace(0, 4 * np.pi, n_time))
        values[:, :2, :] = signal[:, None, None]

        data = xr.DataArray(
            values,
            dims=["time", "z", "x"],
            coords={
                "time": np.arange(n_time) * 0.1,
                "z": flat_labels.coords["z"],
                "x": flat_labels.coords["x"],
            },
        )
        mapper = SeedBasedMaps(seed_masks=flat_labels).fit(data)

        # Region 1 occupies z=0,1.  Those voxels are identical to the seed
        # mean, so r must be exactly 1.
        seed_1_voxels = mapper.maps_.sel(region=1).values[:2, :]  # (z, x) slice
        np.testing.assert_array_less(0.99, seed_1_voxels)

    def test_constant_voxel_yields_zero_correlation(self, flat_labels):
        """A constant voxel (zero variance) returns r=0, not NaN."""
        rng = np.random.default_rng(1)
        n_time, nz, nx = 50, 6, 8
        values = rng.standard_normal((n_time, nz, nx))
        # Make one voxel constant.
        values[:, 3, 3] = 5.0
        data = xr.DataArray(
            values,
            dims=["time", "z", "x"],
            coords={
                "time": np.arange(n_time) * 0.1,
                "z": flat_labels.coords["z"],
                "x": flat_labels.coords["x"],
            },
        )
        mapper = SeedBasedMaps(seed_masks=flat_labels).fit(data)

        # The constant voxel should have r=0 for both regions.
        r_constant = mapper.maps_.isel(z=3, x=3).values
        np.testing.assert_array_equal(r_constant, 0.0)


# ---------------------------------------------------------------------------
# seed_signals tests
# ---------------------------------------------------------------------------


class TestSeedSignals:
    """Tests for the seed_signals (pre-computed signal) code path."""

    def test_seed_signals_1d_matches_mask_based(self, data_2d, single_region_labels):
        """1-D seed_signals produces the same map as the equivalent seed_masks path."""
        mapper_mask = SeedBasedMaps(seed_masks=single_region_labels).fit(data_2d)

        # Use the extracted signal from the mask-based fit as the pre-computed signal.
        signal_1d = mapper_mask.seed_signals_.squeeze("region")
        mapper_sig = SeedBasedMaps(seed_signals=signal_1d).fit(data_2d)

        # Both should produce a squeezed (z, x) map.
        assert mapper_sig.maps_.dims == ("z", "x")
        np.testing.assert_allclose(
            mapper_sig.maps_.values, mapper_mask.maps_.values, rtol=1e-6
        )

    def test_seed_signals_2d_matches_mask_based(self, data_2d, flat_labels):
        """2-D (time, regions) seed_signals produces the same maps as seed_masks."""
        mapper_mask = SeedBasedMaps(seed_masks=flat_labels).fit(data_2d)

        mapper_sig = SeedBasedMaps(seed_signals=mapper_mask.seed_signals_).fit(data_2d)

        np.testing.assert_allclose(
            mapper_sig.maps_.values, mapper_mask.maps_.values, rtol=1e-6
        )

    def test_seed_signals_regions_dim_squeezed(self, data_2d):
        """A single-column (time, regions) signal squeezes the regions dim in maps_."""
        rng = np.random.default_rng(7)
        signal = xr.DataArray(
            rng.standard_normal((data_2d.sizes["time"], 1)),
            dims=["time", "region"],
            coords={"time": data_2d.coords["time"]},
        )
        mapper = SeedBasedMaps(seed_signals=signal).fit(data_2d)

        assert mapper.maps_.dims == ("z", "x")

    def test_seed_signals_skips_extraction_when_cleaning(
        self, data_2d, flat_labels, rng
    ):
        """With seed_signals, clean_kwargs is applied to data but not to the signal."""
        # Build a pre-computed signal from the uncleaned data.
        signal = SeedBasedMaps(seed_masks=flat_labels).fit(data_2d).seed_signals_

        # The signal-based path with cleaning should differ from the mask-based path
        # with cleaning, because cleaning is NOT re-applied to the pre-computed signal.
        mapper_mask = SeedBasedMaps(
            seed_masks=flat_labels, clean_kwargs={"detrend_order": 1}
        ).fit(data_2d)
        mapper_sig = SeedBasedMaps(
            seed_signals=signal, clean_kwargs={"detrend_order": 1}
        ).fit(data_2d)

        assert not np.allclose(mapper_mask.maps_.values, mapper_sig.maps_.values)


# ---------------------------------------------------------------------------
# Cleaning integration tests
# ---------------------------------------------------------------------------


class TestCleaning:
    """Tests for signal cleaning integration."""

    def test_clean_kwargs_applied(self, data_2d, flat_labels):
        """fit with clean_kwargs produces different maps than without."""
        mapper_raw = SeedBasedMaps(seed_masks=flat_labels).fit(data_2d)
        mapper_clean = SeedBasedMaps(
            seed_masks=flat_labels,
            clean_kwargs={"detrend_order": 1},
        ).fit(data_2d)

        assert not np.allclose(mapper_raw.maps_.values, mapper_clean.maps_.values)


# ---------------------------------------------------------------------------
# Dask / laziness tests
# ---------------------------------------------------------------------------


class TestDask:
    """Tests for Dask-backed inputs."""

    def test_dask_input_produces_lazy_map(self, data_2d, flat_labels):
        """maps_ is a lazy Dask array when input is Dask-backed."""
        data_dask = data_2d.chunk({"time": -1, "z": 3, "x": 4})
        mapper = SeedBasedMaps(seed_masks=flat_labels).fit(data_dask)

        assert isinstance(mapper.maps_.data, da.Array)

    def test_dask_result_matches_eager(self, data_2d, flat_labels):
        """Dask and eager computations give identical correlation maps."""
        mapper_eager = SeedBasedMaps(seed_masks=flat_labels).fit(data_2d)

        data_dask = data_2d.chunk({"time": -1, "z": 3, "x": 4})
        mapper_dask = SeedBasedMaps(seed_masks=flat_labels).fit(data_dask)

        np.testing.assert_allclose(
            mapper_dask.maps_.values, mapper_eager.maps_.values, rtol=1e-6
        )


# ---------------------------------------------------------------------------
# sklearn interface tests
# ---------------------------------------------------------------------------


class TestSklearnInterface:
    """Tests for sklearn BaseEstimator compatibility."""

    def test_get_params(self, flat_labels):
        """get_params returns all constructor parameters."""
        mapper = SeedBasedMaps(seed_masks=flat_labels, labels_reduction="median")
        params = mapper.get_params()

        assert "seed_masks" in params
        assert "seed_signals" in params
        assert "labels_reduction" in params
        assert "clean_kwargs" in params
        assert params["labels_reduction"] == "median"

    def test_set_params(self, flat_labels):
        """set_params correctly updates parameters."""
        mapper = SeedBasedMaps(seed_masks=flat_labels)
        mapper.set_params(labels_reduction="sum")

        assert mapper.labels_reduction == "sum"

    def test_is_fitted_false_before_fit(self, flat_labels):
        """__sklearn_is_fitted__ returns False before fit."""
        from sklearn.utils.validation import check_is_fitted

        mapper = SeedBasedMaps(seed_masks=flat_labels)
        with pytest.raises(Exception):
            check_is_fitted(mapper)

    def test_is_fitted_true_after_fit(self, data_2d, flat_labels):
        """__sklearn_is_fitted__ returns True after fit."""
        from sklearn.utils.validation import check_is_fitted

        mapper = SeedBasedMaps(seed_masks=flat_labels).fit(data_2d)
        check_is_fitted(mapper)  # Should not raise.

    def test_fit_returns_self(self, data_2d, flat_labels):
        """fit returns the estimator itself."""
        mapper = SeedBasedMaps(seed_masks=flat_labels)
        result = mapper.fit(data_2d)

        assert result is mapper


# ---------------------------------------------------------------------------
# Xarray accessor tests
# ---------------------------------------------------------------------------


class TestAccessor:
    """Tests for the data.fusi.connectivity.seed_map accessor."""

    def test_accessor_maps_match_direct_fit(self, data_2d, flat_labels):
        """Accessor result matches directly fitting SeedBasedMaps."""
        import confusius  # noqa: F401 — registers the accessor

        mapper_direct = SeedBasedMaps(seed_masks=flat_labels).fit(data_2d)
        mapper_accessor = data_2d.fusi.connectivity.seed_map(seed_masks=flat_labels)

        np.testing.assert_allclose(
            mapper_accessor.maps_.values, mapper_direct.maps_.values
        )
