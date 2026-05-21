"""Tests for confusius.glm.first_level."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.glm import FirstLevelModel, make_first_level_design_matrix
from confusius.glm._models import OLSModel
from confusius.glm.first_level import _flatten_spatial


# -----------------------------------------------------------------------------
# FirstLevelModel: fitting
# -----------------------------------------------------------------------------


class TestFirstLevelModelFit:
    """Tests for FirstLevelModel.fit."""

    def test_fit_with_prebuilt_design_matrix_uses_it(
        self, fusi_data, frame_times, events
    ):
        """A pre-built design matrix is used as-is rather than rebuilt."""
        dm = make_first_level_design_matrix(frame_times, events=events)
        model = FirstLevelModel(noise_model="ols")
        model.fit(fusi_data, design_matrices=dm)
        # Auto-built design must match the user-supplied one column-for-column.
        pd.testing.assert_frame_equal(model.design_matrices_[0], dm)

    def test_fit_2d_spatial(self, fusi_data_2d, events):
        """Fitting a `(time, y, x)` array yields a contrast map with the same
        spatial dims and shape."""
        model = FirstLevelModel(noise_model="ols")
        model.fit(fusi_data_2d, events=events)
        z_map = model.compute_contrast("A - B")
        assert z_map.dims == ("y", "x")
        assert z_map.shape == (5, 6)

    def test_minimize_memory_strips_diagnostic_fields(self, fusi_data, events):
        """minimize_memory=True drops Y/whitened_Y/whitened_residuals/model post-fit.

        Contrast-relevant fields (theta, cov, dispersion, df_residuals) must
        survive so contrasts still work; diagnostic accessors raise.
        """
        model = FirstLevelModel(noise_model="ols", minimize_memory=True)
        model.fit(fusi_data, events=events)
        results = model.results_[0]
        assert results.Y is None
        assert results.whitened_Y is None
        assert results.whitened_residuals is None
        assert results.model is None
        assert results.theta is not None
        # Contrast still works after stripping.
        z_map = model.compute_contrast("A - B")
        assert np.all(np.isfinite(z_map.values))
        # Diagnostic accessors raise an informative error.
        with pytest.raises(RuntimeError, match="minimize_memory"):
            _ = results.residuals
        with pytest.raises(RuntimeError, match="minimize_memory"):
            _ = results.predicted
        with pytest.raises(RuntimeError, match="minimize_memory"):
            _ = results.sse

    def test_minimize_memory_false_keeps_fields(self, fusi_data, events):
        """minimize_memory=False keeps the full RegressionResults."""
        model = FirstLevelModel(noise_model="ols", minimize_memory=False)
        model.fit(fusi_data, events=events)
        results = model.results_[0]
        assert results.Y is not None
        assert results.whitened_residuals is not None
        # Diagnostic accessors work.
        assert np.all(np.isfinite(results.residuals))

    def test_fit_with_confounds(self, fusi_data, events, rng):
        confounds = pd.DataFrame(
            {"motion_x": rng.standard_normal(200), "motion_y": rng.standard_normal(200)}
        )
        model = FirstLevelModel(noise_model="ols")
        model.fit(fusi_data, events=events, confounds=confounds)
        dm = model.design_matrices_[0]
        assert "motion_x" in dm.columns
        assert "motion_y" in dm.columns

    def test_sklearn_is_fitted(self, fusi_data, events):
        model = FirstLevelModel(noise_model="ols")
        assert not model.__sklearn_is_fitted__()
        model.fit(fusi_data, events=events)
        assert model.__sklearn_is_fitted__()

    def test_fit_preserves_spatial_coords(self, fusi_data, events):
        model = FirstLevelModel(noise_model="ols")
        model.fit(fusi_data, events=events)
        z_map = model.compute_contrast("A")
        assert_allclose(z_map.coords["z"].values, fusi_data.coords["z"].values)
        assert_allclose(z_map.coords["y"].values, fusi_data.coords["y"].values)
        assert_allclose(z_map.coords["x"].values, fusi_data.coords["x"].values)

    def test_fit_with_mask_sets_outside_voxels_to_zero(self, fusi_data, events):
        """Mask limits fitted voxels and keeps full output geometry."""
        mask = xr.DataArray(
            np.zeros(
                (
                    fusi_data.sizes["z"],
                    fusi_data.sizes["y"],
                    fusi_data.sizes["x"],
                ),
                dtype=bool,
            ),
            dims=["z", "y", "x"],
            coords={
                "z": fusi_data.coords["z"],
                "y": fusi_data.coords["y"],
                "x": fusi_data.coords["x"],
            },
        )
        mask.values[0, :, :] = True

        model = FirstLevelModel(noise_model="ols", mask=mask)
        model.fit(fusi_data, events=events)
        z_map = model.compute_contrast("A")

        outside = z_map.where(~mask, other=np.nan)
        np.testing.assert_array_equal(np.nan_to_num(outside.values), 0.0)

    def test_masked_f_contrast_effect_keeps_contrast_dim(self, fusi_data, events):
        """Masked F-contrast effect maps keep `contrast_dim` and zero-fill outside mask."""
        mask = xr.DataArray(
            np.zeros(
                (
                    fusi_data.sizes["z"],
                    fusi_data.sizes["y"],
                    fusi_data.sizes["x"],
                ),
                dtype=bool,
            ),
            dims=["z", "y", "x"],
            coords={
                "z": fusi_data.coords["z"],
                "y": fusi_data.coords["y"],
                "x": fusi_data.coords["x"],
            },
        )
        mask.values[:, 0, :] = True

        model = FirstLevelModel(noise_model="ols", mask=mask)
        model.fit(fusi_data, events=events)
        dm = model.design_matrices_[0]
        a_idx = list(dm.columns).index("A")
        b_idx = list(dm.columns).index("B")
        c = np.zeros((2, len(dm.columns)))
        c[0, a_idx] = 1.0
        c[1, b_idx] = 1.0

        e_map = model.compute_contrast(c, stat_type="F", output_type="effect")

        assert e_map.dims == ("contrast_dim", "z", "y", "x")
        outside = e_map.where(~mask, other=np.nan)
        np.testing.assert_array_equal(np.nan_to_num(outside.values), 0.0)

    def test_consensus_attrs_propagated_across_runs(self, fusi_data, events):
        """Attributes equal across all runs propagate to the contrast map."""
        run_a = fusi_data.copy()
        run_a.attrs = {"subject_id": "s01", "task": "stim", "session": 1}
        run_b = fusi_data.copy()
        run_b.attrs = {"subject_id": "s01", "task": "stim", "session": 2}
        model = FirstLevelModel(noise_model="ols")
        model.fit([run_a, run_b], events=[events, events])
        z_map = model.compute_contrast("A")
        assert z_map.attrs["subject_id"] == "s01"
        assert z_map.attrs["task"] == "stim"
        # Conflicting key dropped.
        assert "session" not in z_map.attrs
        # Output-specific attrs still set.
        assert z_map.attrs["long_name"] == "zscore"
        assert z_map.attrs["cmap"] == "coolwarm"


# -----------------------------------------------------------------------------
# FirstLevelModel: contrasts
# -----------------------------------------------------------------------------


class TestFirstLevelModelContrast:
    """Tests for FirstLevelModel.compute_contrast."""

    @pytest.fixture(autouse=True)
    def _fitted_model(self, fusi_data, events):
        self.model = FirstLevelModel(noise_model="ols")
        self.model.fit(fusi_data, events=events)

    def test_string_and_array_contrast_agree(self):
        """A string contrast resolves to the same numeric vector applied
        positionally; both must produce identical maps."""
        dm = self.model.design_matrices_[0]
        vec = np.zeros(len(dm.columns))
        vec[list(dm.columns).index("A")] = 1.0
        vec[list(dm.columns).index("B")] = -1.0

        z_string = self.model.compute_contrast("A - B")
        z_array = self.model.compute_contrast(vec)

        assert_allclose(z_string.values, z_array.values, rtol=1e-12)
        assert z_string.dims == ("z", "y", "x")

    def test_output_type_pvalue_in_unit_interval(self):
        p_map = self.model.compute_contrast("A", output_type="pvalue")
        assert np.all((p_map.values >= 0) & (p_map.values <= 1))

    def test_output_type_variance_non_negative(self):
        v_map = self.model.compute_contrast("A", output_type="variance")
        assert np.all(v_map.values >= 0)

    def test_short_contrast_vector_is_zero_padded(self):
        """A 1D contrast shorter than the design is zero-padded; the result
        must equal the manually-padded contrast, not just match in shape."""
        dm = self.model.design_matrices_[0]
        # The two condition columns happen to be A then B at indices 0, 1.
        short = np.array([1.0, -1.0])
        padded = np.zeros(len(dm.columns))
        padded[: short.size] = short

        z_short = self.model.compute_contrast(short).values
        z_padded = self.model.compute_contrast(padded).values

        assert_allclose(z_short, z_padded, rtol=1e-12, atol=1e-12)

    def test_invalid_output_type_raises(self):
        with pytest.raises(ValueError, match="output_type"):
            self.model.compute_contrast("A", output_type="invalid")


class TestFirstLevelModelContrastMultiRun:
    """Test fixed-effects contrast combination across runs."""

    def test_multi_run_effect_is_pooled_average(self, rng, frame_times, events):
        """Multi-run effect_size is the pooled fixed-effects average, not the sum.

        Subjects with more runs would contribute proportionally larger
        effect/variance maps to second-level inputs if `compute_contrast` did
        not divide by `n_runs` after summing.
        """
        data1 = xr.DataArray(
            rng.standard_normal((200, 2, 3, 4)),
            dims=["time", "z", "y", "x"],
            coords={"time": frame_times},
        )
        data2 = xr.DataArray(
            rng.standard_normal((200, 2, 3, 4)),
            dims=["time", "z", "y", "x"],
            coords={"time": frame_times},
        )
        single = FirstLevelModel(noise_model="ols")
        single.fit(data1, events=events)
        e_single = single.compute_contrast("A - B", output_type="effect").values

        multi = FirstLevelModel(noise_model="ols")
        multi.fit([data1, data2], events=[events, events])
        e_multi = multi.compute_contrast("A - B", output_type="effect").values

        # Without averaging the multi-run effect would be roughly twice as
        # large as a single-run effect; with averaging it stays on the same
        # scale (mean of two unbiased estimates).
        assert np.median(np.abs(e_multi)) < 1.5 * np.median(np.abs(e_single))

    def test_multi_run_per_run_confounds_pass_through(
        self, rng, frame_times, events
    ):
        """Per-run confounds passed as a list show up with their values in each
        run's design matrix."""
        data1 = xr.DataArray(
            rng.standard_normal((200, 2, 3, 4)),
            dims=["time", "z", "y", "x"],
            coords={"time": frame_times},
        )
        data2 = xr.DataArray(
            rng.standard_normal((200, 2, 3, 4)),
            dims=["time", "z", "y", "x"],
            coords={"time": frame_times},
        )
        conf1 = pd.DataFrame({"motion": rng.standard_normal(200)})
        conf2 = pd.DataFrame({"motion": rng.standard_normal(200)})
        model = FirstLevelModel(noise_model="ols")
        model.fit([data1, data2], events=[events, events], confounds=[conf1, conf2])

        assert_allclose(
            model.design_matrices_[0]["motion"].to_numpy(),
            conf1["motion"].to_numpy(),
        )
        assert_allclose(
            model.design_matrices_[1]["motion"].to_numpy(),
            conf2["motion"].to_numpy(),
        )


class TestFirstLevelModelFContrast:
    """Test F-contrast path through compute_contrast."""

    @pytest.fixture(autouse=True)
    def _fitted_model(self, fusi_data, events):
        self.model = FirstLevelModel(noise_model="ols")
        self.model.fit(fusi_data, events=events)

    def test_f_contrast_effect_has_contrast_dim_axis(self):
        """F-contrast `output_type="effect"` returns a `(contrast_dim, *spatial)`
        array, not a scalar effect — the components must remain accessible
        for downstream inspection."""
        dm = self.model.design_matrices_[0]
        a_idx = list(dm.columns).index("A")
        b_idx = list(dm.columns).index("B")
        c = np.zeros((2, len(dm.columns)))
        c[0, a_idx] = 1.0
        c[1, b_idx] = 1.0
        e_map = self.model.compute_contrast(c, stat_type="F", output_type="effect")
        assert e_map.dims == ("contrast_dim", "z", "y", "x")
        assert e_map.shape == (2, 2, 3, 4)

    def test_2d_contrast_is_zero_padded(self):
        """A 2D contrast narrower than the design is zero-padded; the result
        must equal the manually-padded contrast, not just match in shape."""
        dm = self.model.design_matrices_[0]
        short = np.array([[1.0, 0.0], [0.0, -1.0]])
        padded = np.zeros((2, len(dm.columns)))
        padded[:, : short.shape[1]] = short

        z_short = self.model.compute_contrast(short, stat_type="F").values
        z_padded = self.model.compute_contrast(padded, stat_type="F").values

        assert_allclose(z_short, z_padded, rtol=1e-12, atol=1e-12)

    def test_f_contrast_matches_underlying_quadratic_form(self):
        """F-contrast statistic matches the proper quadratic form.

        Regression test: an earlier implementation reduced the per-voxel
        contrast covariance to the mean of its diagonal, which only happens
        to be correct for orthogonal designs. A non-axis-aligned contrast on
        non-orthogonal columns exposes the bug.
        """
        dm = self.model.design_matrices_[0]
        a = list(dm.columns).index("A")
        b = list(dm.columns).index("B")
        # Non-axis-aligned 2-row contrast — rows are not orthogonal in design space.
        c = np.zeros((2, len(dm.columns)))
        c[0, a] = 1.0
        c[0, b] = 1.0
        c[1, a] = 1.0
        c[1, b] = -1.0

        stat_map = self.model.compute_contrast(
            c, stat_type="F", output_type="statistic"
        )

        # Independent reference: pull theta/cov/dispersion from the fitted
        # results and compute F = ctheta' · invcov · ctheta / (q · dispersion)
        # voxelwise without going through the contrast wrapper.
        results = self.model.results_[0]
        ctheta = c @ results.theta  # (q, V)
        cov_q = c @ results.cov @ c.T  # (q, q)
        invcov = np.linalg.inv(cov_q)
        f_expected = np.einsum("qv,qp,pv->v", ctheta, invcov, ctheta) / (
            2 * results.dispersion
        )
        assert_allclose(stat_map.values.ravel(), f_expected, rtol=1e-10, atol=1e-12)


# -----------------------------------------------------------------------------
# FirstLevelModel: reference check against manual computation
# -----------------------------------------------------------------------------


class TestFirstLevelModelReference:
    """Verify FirstLevelModel produces the same results as manual low-level usage."""

    def test_matches_manual_ols(self, fusi_data, events, frame_times):
        # Fit via FirstLevelModel.
        model = FirstLevelModel(noise_model="ols", drift_model="cosine")
        model.fit(fusi_data, events=events)
        z_map_auto = model.compute_contrast("A - B")

        # Manual computation.
        dm = make_first_level_design_matrix(
            frame_times, events=events, drift_model="cosine"
        )
        flat, _, _ = _flatten_spatial(fusi_data)
        ols = OLSModel(dm.to_numpy(dtype=np.float64))
        results = ols.fit(flat)

        from confusius.glm._contrasts import Contrast
        from confusius.glm._utils import expression_to_contrast_vector

        cvec = expression_to_contrast_vector("A - B", list(dm.columns))
        t_res = results.compute_t_contrast(cvec)
        contrast = Contrast.from_estimate(
            effect=np.atleast_1d(t_res["effect"]),
            variance=np.atleast_1d(t_res["sd"]) ** 2,
            dof=float(t_res["df_den"]),
            stat_type="t",
        )
        z_manual = contrast.zscore.reshape(2, 3, 4)

        assert_allclose(z_map_auto.values, z_manual, rtol=1e-10)


# -----------------------------------------------------------------------------
# FirstLevelModel: error handling
# -----------------------------------------------------------------------------


class TestFirstLevelModelErrors:
    """Tests for error handling."""

    def test_no_events_no_design_raises(self, fusi_data):
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(ValueError, match="events.*design_matrices"):
            model.fit(fusi_data)

    def test_contrast_before_fit_raises(self):
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(ValueError, match="not fitted"):
            model.compute_contrast("A")

    def test_invalid_noise_model_raises(self, fusi_data, events):
        model = FirstLevelModel(noise_model="invalid")
        with pytest.raises(ValueError, match="noise_model"):
            model.fit(fusi_data, events=events)

    def test_mismatched_run_count_raises(self, fusi_data, events):
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(ValueError, match="events.*runs"):
            model.fit([fusi_data, fusi_data], events=[events])

    def test_contrast_vector_too_long_raises(self, fusi_data, events):
        model = FirstLevelModel(noise_model="ols")
        model.fit(fusi_data, events=events)
        n_cols = len(model.design_matrices_[0].columns)
        with pytest.raises(ValueError, match="exceeds"):
            model.compute_contrast(np.ones(n_cols + 5))

    def test_design_matrix_count_mismatch_raises(self, fusi_data, frame_times, events):
        dm = make_first_level_design_matrix(frame_times, events=events)
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(ValueError, match="design matrices"):
            model.fit([fusi_data, fusi_data], design_matrices=[dm])

    def test_design_matrix_row_mismatch_raises(self, fusi_data, frame_times, events):
        """Pre-built design matrix row count must match the run length."""
        dm = make_first_level_design_matrix(frame_times[:100], events=events)
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(ValueError, match="rows but the run has"):
            model.fit(fusi_data, design_matrices=dm)

    def test_design_matrix_columns_mismatch_raises(self, rng, frame_times, events):
        """Multi-run designs with different column orders are rejected."""
        events_swapped = pd.DataFrame(
            {
                "trial_type": ["B"] * 5 + ["A"] * 5,
                "onset": np.concatenate(
                    [np.arange(5) * 20.0 + 10.0, np.arange(5) * 20.0]
                ),
                "duration": [1.0] * 10,
            }
        )
        data1 = xr.DataArray(
            rng.standard_normal((200, 2, 3, 4)),
            dims=["time", "z", "y", "x"],
            coords={"time": frame_times},
        )
        data2 = xr.DataArray(
            rng.standard_normal((200, 2, 3, 4)),
            dims=["time", "z", "y", "x"],
            coords={"time": frame_times},
        )
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(ValueError, match="design-matrix columns"):
            model.fit([data1, data2], events=[events, events_swapped])

    def test_dropped_spatial_coord_raises(self, rng, frame_times, events):
        """A run that drops a spatial coord present on the reference run is rejected."""
        data1 = xr.DataArray(
            rng.standard_normal((200, 2, 3, 4)),
            dims=["time", "z", "y", "x"],
            coords={
                "time": frame_times,
                "z": np.arange(2) * 0.5,
                "y": np.arange(3) * 0.1,
                "x": np.arange(4) * 0.1,
            },
        )
        data2 = data1.drop_vars("z")
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(ValueError, match=r"Coordinate 'z' is missing from run 1"):
            model.fit([data1, data2], events=[events, events])

    def test_spatial_shape_mismatch_raises(self, rng, frame_times, events):
        """Multi-run fit raises if runs have different spatial shapes."""
        data1 = xr.DataArray(
            rng.standard_normal((200, 2, 3)),
            dims=["time", "z", "x"],
            coords={"time": frame_times},
        )
        data2 = xr.DataArray(
            rng.standard_normal((200, 4, 3)),
            dims=["time", "z", "x"],
            coords={"time": frame_times},
        )
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(ValueError, match="spatial dimensions"):
            model.fit([data1, data2], events=[events, events])

    def test_spatial_axis_order_mismatch_raises(self, rng, frame_times, events):
        """Multi-run fit rejects runs that share spatial sizes but in
        different axis orders.

        `_flatten_spatial` stacks voxels in each run's own dim order, so
        accepting transposed runs would silently mix voxel locations during
        fixed-effects combination.
        """
        data1 = xr.DataArray(
            rng.standard_normal((200, 2, 3, 4)),
            dims=["time", "z", "y", "x"],
            coords={"time": frame_times},
        )
        data2 = data1.transpose("time", "y", "z", "x")
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(ValueError, match="same order"):
            model.fit([data1, data2], events=[events, events])

    def test_spatial_coord_mismatch_raises(self, rng, frame_times, events):
        """Multi-run fit raises if runs have mismatched spatial coordinates."""
        data1 = xr.DataArray(
            rng.standard_normal((200, 2, 3, 4)),
            dims=["time", "z", "y", "x"],
            coords={
                "time": frame_times,
                "z": np.arange(2) * 0.5,
                "y": np.arange(3) * 0.1,
                "x": np.arange(4) * 0.1,
            },
        )
        data2 = data1.assign_coords(z=np.arange(2) * 0.5 + 10.0)
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(
            ValueError,
            match=r"Coordinate 'z' does not match between run 0 and run 1",
        ):
            model.fit([data1, data2], events=[events, events])

    def test_confounds_list_length_mismatch_raises(self, fusi_data, events, rng):
        """Confound list with wrong number of entries raises ValueError."""
        conf = pd.DataFrame({"motion": rng.standard_normal(200)})
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(ValueError, match="confound"):
            model.fit([fusi_data, fusi_data], events=[events, events], confounds=[conf])

    def test_ar_noise_model_non_numeric_order_raises(self, fusi_data, events):
        """noise_model='arfoo' raises ValueError for non-numeric AR order."""
        model = FirstLevelModel(noise_model="arfoo")
        with pytest.raises(ValueError, match="noise_model"):
            model.fit(fusi_data, events=events)

    def test_2d_contrast_too_wide_raises(self, fusi_data, events):
        """2D contrast wider than design columns raises ValueError."""
        model = FirstLevelModel(noise_model="ols")
        model.fit(fusi_data, events=events)
        n_cols = len(model.design_matrices_[0].columns)
        c = np.zeros((2, n_cols + 3))
        with pytest.raises(ValueError, match="exceeds"):
            model.compute_contrast(c, stat_type="F")

    def test_3d_contrast_raises(self, fusi_data, events):
        """3D contrast array raises ValueError."""
        model = FirstLevelModel(noise_model="ols")
        model.fit(fusi_data, events=events)
        with pytest.raises(ValueError, match="string, 1D, or 2D"):
            model.compute_contrast(np.zeros((2, 3, 4)))


# -----------------------------------------------------------------------------
# FirstLevelModel: noise model parsing
# -----------------------------------------------------------------------------


class TestNoiseModelParsing:
    """Tests for noise model string parsing."""

    def test_ols(self):
        model = FirstLevelModel(noise_model="ols")
        assert model._parse_noise_model() == 0

    def test_ar1(self):
        model = FirstLevelModel(noise_model="ar1")
        assert model._parse_noise_model() == 1

    def test_ar2(self):
        model = FirstLevelModel(noise_model="ar2")
        assert model._parse_noise_model() == 2

    def test_case_insensitive(self):
        model = FirstLevelModel(noise_model="AR1")
        assert model._parse_noise_model() == 1

    def test_invalid_raises(self):
        model = FirstLevelModel(noise_model="garch")
        with pytest.raises(ValueError, match="noise_model"):
            model._parse_noise_model()

    def test_ar0_raises(self):
        model = FirstLevelModel(noise_model="ar0")
        with pytest.raises(ValueError, match="AR order must be >= 1"):
            model._parse_noise_model()
