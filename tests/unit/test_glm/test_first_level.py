"""Tests for confusius.glm.first_level."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.glm import FirstLevelModel, make_first_level_design_matrix
from confusius.glm._models import OLSModel
from confusius.spatial import smooth_volume

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

    @pytest.mark.parametrize("show_progress", [True, False])
    def test_show_progress_controls_progress_bar(
        self, fusi_data, events, capsys, show_progress
    ):
        """The progress bar is written only when `show_progress` is set."""
        model = FirstLevelModel(noise_model="ols", show_progress=show_progress)
        model.fit(fusi_data, events=events)

        assert ("Fitting runs" in capsys.readouterr().out) is show_progress

    def test_fit_2d_spatial(self, fusi_data_2d, events):
        """Fitting a singleton-k `(time, k, j, i)` array yields a contrast map with
        the same spatial dims and shape."""
        model = FirstLevelModel(noise_model="ols")
        model.fit(fusi_data_2d, events=events)
        z_map = model.compute_contrast("A - B")
        assert z_map.dims == ("k", "j", "i")
        assert z_map.shape == (1, 5, 6)

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

    # A voxel with zero residual variance divides by zero in `_positive_reciprocal`,
    # which is expected and already handled; this test is about the resulting maps.
    @pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
    @pytest.mark.parametrize("noise_model", ["ols", "ar1"])
    def test_perfectly_fitted_voxels_keep_finite_statistics(
        self, fusi_data, frame_times, events, rng, noise_model
    ):
        """Voxels the design explains exactly must not poison the statistics.

        Resampling a recording onto an atlas grid leaves large constant-in-time
        regions outside the recorded field of view, which the design's constant
        regressor explains completely. Their residual sum of squares is zero to
        within rounding, so recovering it as a difference of two large quantities
        instead of by summing the residuals returns a small negative number whose
        sign is pure rounding, and the negative dispersion turns every downstream map
        into NaN. Half the voxels here are exact linear combinations of the design,
        scaled up so the cancellation is far larger than the true residual.
        """
        design_matrix = make_first_level_design_matrix(frame_times, events=events)
        design = design_matrix.to_numpy()

        fusi_data = fusi_data.copy()
        flat = fusi_data.values.reshape(len(frame_times), -1)
        n_exact = flat.shape[1] // 2
        flat[:, :n_exact] = design @ (
            rng.standard_normal((design.shape[1], n_exact)) * 50.0
        )

        maps = {}
        for minimize_memory in (True, False):
            model = FirstLevelModel(
                noise_model=noise_model,
                minimize_memory=minimize_memory,
                show_progress=False,
            )
            model.fit(fusi_data, design_matrices=design_matrix)
            maps[minimize_memory] = {
                output_type: model.compute_contrast(
                    "A - B", output_type=output_type
                ).values
                for output_type in ("effect", "variance", "zscore")
            }

        for output_type, expected in maps[False].items():
            assert np.all(np.isfinite(maps[True][output_type])), output_type
            assert_allclose(maps[True][output_type], expected, rtol=1e-9)

    def test_fit_with_confounds(self, fusi_data, events, rng):
        confounds = pd.DataFrame(
            {
                "time": fusi_data.coords["time"].values,
                "motion_x": rng.standard_normal(200),
                "motion_y": rng.standard_normal(200),
            }
        )
        model = FirstLevelModel(noise_model="ols")
        model.fit(fusi_data, events=events, confounds=confounds)
        dm = model.design_matrices_[0]
        assert "motion_x" in dm.columns
        assert "motion_y" in dm.columns
        assert_allclose(dm["motion_x"].to_numpy(), confounds["motion_x"].to_numpy())
        assert_allclose(dm["motion_y"].to_numpy(), confounds["motion_y"].to_numpy())

    def test_fit_with_dataarray_confounds(self, fusi_data, events, rng):
        """DataArray confounds match the DataFrame path and are checked against time."""
        values = rng.standard_normal((200, 2))
        names = ["motion_x", "motion_y"]
        frame = pd.DataFrame(values, columns=names)
        frame.insert(0, "time", fusi_data.coords["time"].values)
        confounds = xr.DataArray(
            values,
            dims=["time", "confound"],
            coords={"time": fusi_data.coords["time"], "confound": names},
        )
        expected = FirstLevelModel(noise_model="ols").fit(
            fusi_data, events=events, confounds=frame
        )
        model = FirstLevelModel(noise_model="ols").fit(
            fusi_data, events=events, confounds=confounds
        )
        pd.testing.assert_frame_equal(
            model.design_matrices_[0], expected.design_matrices_[0]
        )

        misaligned = confounds.assign_coords(time=confounds.coords["time"] + 1.0)
        with pytest.raises(ValueError, match="time coordinates do not match"):
            FirstLevelModel(noise_model="ols").fit(
                fusi_data, events=events, confounds=misaligned
            )

    def test_fit_with_1d_dataarray_confounds(self, fusi_data, events, rng):
        """A `(time,)` DataArray becomes one confound column named like NumPy input."""
        values = rng.standard_normal(200)
        confounds = xr.DataArray(
            values, dims=["time"], coords={"time": fusi_data.coords["time"]}
        )
        model = FirstLevelModel(noise_model="ols").fit(
            fusi_data, events=events, confounds=confounds
        )
        assert_allclose(model.design_matrices_[0]["confound_0"].to_numpy(), values)

    def test_fit_rejects_3d_dataarray_confounds(self, fusi_data, events, rng):
        """DataArray confounds must be 1D or 2D."""
        confounds = xr.DataArray(
            rng.standard_normal((200, 2, 2)),
            dims=["time", "a", "b"],
            coords={"time": fusi_data.coords["time"]},
        )
        with pytest.raises(ValueError, match="confounds must be 1D or 2D"):
            FirstLevelModel(noise_model="ols").fit(
                fusi_data, events=events, confounds=confounds
            )

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
        mask = xr.zeros_like(fusi_data.isel(time=0, drop=True), dtype=bool)
        mask.values[0, :, :] = True

        model = FirstLevelModel(noise_model="ols", mask=mask)
        model.fit(fusi_data, events=events)
        z_map = model.compute_contrast("A")

        outside = z_map.where(~mask, other=np.nan)
        np.testing.assert_array_equal(np.nan_to_num(outside.values), 0.0)

    def test_masked_f_contrast_effect_keeps_contrast_dim(self, fusi_data, events):
        """Masked F-contrast effect maps keep `contrast_dim` and zero-fill outside mask."""
        mask = xr.zeros_like(fusi_data.isel(time=0, drop=True), dtype=bool)
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

        assert e_map.dims == ("contrast_dim", "k", "j", "i")
        outside = e_map.where(~mask, other=np.nan)
        np.testing.assert_array_equal(np.nan_to_num(outside.values), 0.0)

    def test_intersect_attrs_propagated_across_runs(self, fusi_data, events):
        """Attributes equal across all runs propagate to the contrast map."""
        run_a = fusi_data.copy()
        run_a.attrs.update({"subject_id": "s01", "task": "stim", "session": 1})
        run_b = fusi_data.copy()
        run_b.attrs.update({"subject_id": "s01", "task": "stim", "session": 2})
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

    def test_smoothing_fwhm_matches_presmoothed_input(self, fusi_data, events):
        """smoothing_fwhm=f equals fitting on smooth_volume(data, f) and changes
        the result relative to no smoothing."""
        smoothed = FirstLevelModel(noise_model="ols", smoothing_fwhm=0.2)
        smoothed.fit(fusi_data, events=events)
        reference = FirstLevelModel(noise_model="ols")
        reference.fit(smooth_volume(fusi_data, 0.2), events=events)
        unsmoothed = FirstLevelModel(noise_model="ols")
        unsmoothed.fit(fusi_data, events=events)

        smoothed_map = smoothed.compute_contrast("A - B").values
        assert_allclose(smoothed_map, reference.compute_contrast("A - B").values)
        # Guard against a silent no-op: smoothing must change the contrast map.
        assert not np.allclose(
            smoothed_map, unsmoothed.compute_contrast("A - B").values
        )


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
        assert z_string.dims == ("k", "j", "i")

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
            self.model.compute_contrast("A", output_type="invalid")  # ty: ignore[invalid-argument-type]


class TestFirstLevelModelContrastMultiRun:
    """Test fixed-effects contrast combination across runs."""

    def test_multi_run_effect_is_pooled_average(
        self, rng, frame_times, events, make_glm_test_dataarray
    ):
        """Multi-run effect_size is the pooled fixed-effects average, not the sum.

        Subjects with more runs would contribute proportionally larger
        effect/variance maps to second-level inputs if `compute_contrast` did
        not divide by `n_runs` after summing.
        """
        data1 = make_glm_test_dataarray(
            rng.standard_normal((200, 2, 3, 4)),
            ("time", "k", "j", "i"),
            time=frame_times,
        )
        data2 = make_glm_test_dataarray(
            rng.standard_normal((200, 2, 3, 4)),
            ("time", "k", "j", "i"),
            time=frame_times,
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
        self, rng, frame_times, events, make_glm_test_dataarray
    ):
        """Per-run confounds passed as a list show up with their values in each
        run's design matrix."""
        data1 = make_glm_test_dataarray(
            rng.standard_normal((200, 2, 3, 4)),
            ("time", "k", "j", "i"),
            time=frame_times,
        )
        data2 = make_glm_test_dataarray(
            rng.standard_normal((200, 2, 3, 4)),
            ("time", "k", "j", "i"),
            time=frame_times,
        )
        conf1 = pd.DataFrame({"time": frame_times, "motion": rng.standard_normal(200)})
        conf2 = pd.DataFrame({"time": frame_times, "motion": rng.standard_normal(200)})
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

    def test_multi_run_single_confounds_table_must_match_every_run(
        self, rng, frame_times, events, make_glm_test_dataarray
    ):
        """A single confounds table is applied to every run, so its times must
        match each run, and the error names the offending run."""
        data1 = make_glm_test_dataarray(
            rng.standard_normal((200, 2, 3, 4)),
            ("time", "k", "j", "i"),
            time=frame_times,
        )
        data2 = make_glm_test_dataarray(
            rng.standard_normal((200, 2, 3, 4)),
            ("time", "k", "j", "i"),
            time=frame_times + 1.0,
        )
        confounds = pd.DataFrame(
            {"time": frame_times, "motion": rng.standard_normal(200)}
        )
        with pytest.raises(
            ValueError, match="Run 1: confounds time coordinates do not match"
        ):
            FirstLevelModel(noise_model="ols").fit(
                [data1, data2], events=[events, events], confounds=confounds
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
        assert e_map.dims == ("contrast_dim", "k", "j", "i")
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
        spatial_dims = tuple(str(d) for d in fusi_data.dims if d != "time")
        flat = fusi_data.stack(space=spatial_dims).transpose("time", "space").values
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

    def test_multipose_matches_per_pose_fit(self, fusi_data_pose, events):
        """Fitting `(time, pose, k, j, i)` data matches fitting each pose slice
        independently and stacking the results.

        Guards the fallback all-True mask and the explicit-mask path both
        correctly carry `pose` through `extract_with_mask`/`unmask`, instead
        of silently collapsing to a single pose or erroring.
        """
        model = FirstLevelModel(noise_model="ols")
        model.fit(fusi_data_pose, events=events)
        z_pose = model.compute_contrast("A - B")
        assert z_pose.dims == ("pose", "k", "j", "i")

        for pose in range(fusi_data_pose.sizes["pose"]):
            single_pose_model = FirstLevelModel(noise_model="ols")
            single_pose_model.fit(fusi_data_pose.isel(pose=pose), events=events)
            z_single = single_pose_model.compute_contrast("A - B")
            assert_allclose(z_pose.isel(pose=pose).values, z_single.values)


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

    def test_design_matrix_columns_mismatch_raises(
        self, rng, frame_times, events, make_glm_test_dataarray
    ):
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
        data1 = make_glm_test_dataarray(
            rng.standard_normal((200, 2, 3, 4)),
            ("time", "k", "j", "i"),
            time=frame_times,
        )
        data2 = make_glm_test_dataarray(
            rng.standard_normal((200, 2, 3, 4)),
            ("time", "k", "j", "i"),
            time=frame_times,
        )
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(ValueError, match="design-matrix columns"):
            model.fit([data1, data2], events=[events, events_swapped])

    def test_dropped_spatial_coord_raises(
        self, rng, frame_times, events, make_glm_test_dataarray
    ):
        """A run missing required spatial coords is rejected during validation."""
        data1 = make_glm_test_dataarray(
            rng.standard_normal((200, 2, 3, 4)),
            ("time", "k", "j", "i"),
            time=frame_times,
        )
        data2 = data1.drop_vars("k")
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(ValueError, match="Missing required coordinate"):
            model.fit([data1, data2], events=[events, events])

    def test_spatial_shape_mismatch_raises(
        self, rng, frame_times, events, make_glm_test_dataarray
    ):
        """Multi-run fit raises if runs have different spatial shapes."""
        data1 = make_glm_test_dataarray(
            rng.standard_normal((200, 2, 3)),
            ("time", "k", "i"),
            time=frame_times,
        )
        data2 = make_glm_test_dataarray(
            rng.standard_normal((200, 4, 3)),
            ("time", "k", "i"),
            time=frame_times,
        )
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(ValueError, match="spatial dimensions"):
            model.fit([data1, data2], events=[events, events])

    def test_transposed_run_aligns_by_label(
        self, rng, frame_times, events, make_glm_test_dataarray
    ):
        """A run with a different (but internally consistent) spatial axis
        order is accepted and produces the same result as an untransposed run.

        `extract_with_mask` stacks voxels by dimension/coordinate name
        (`xr.align` is label-based, not positional), so a per-run axis-order
        permutation doesn't mix up voxel locations during fixed-effects
        combination.
        """
        data1 = make_glm_test_dataarray(
            rng.standard_normal((200, 2, 3, 4)),
            ("time", "k", "j", "i"),
            time=frame_times,
        )
        data2 = data1.transpose("time", "j", "k", "i")

        model_transposed = FirstLevelModel(noise_model="ols")
        model_transposed.fit([data1, data2], events=[events, events])
        z_transposed = model_transposed.compute_contrast("A - B")

        model_ref = FirstLevelModel(noise_model="ols")
        model_ref.fit([data1, data1], events=[events, events])
        z_ref = model_ref.compute_contrast("A - B")

        np.testing.assert_allclose(z_transposed.values, z_ref.values)

    def test_spatial_coord_mismatch_raises(
        self, rng, frame_times, events, make_glm_test_dataarray
    ):
        """Multi-run fit raises if runs have mismatched spatial coordinates."""
        data1 = make_glm_test_dataarray(
            rng.standard_normal((200, 2, 3, 4)),
            ("time", "k", "j", "i"),
            time=frame_times,
        )
        data2 = data1.assign_coords(k=np.arange(2) + 10.0)
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(
            ValueError,
            match=r"Coordinate 'k' does not match between run 0 and run 1",
        ):
            model.fit([data1, data2], events=[events, events])

    def test_confounds_list_length_mismatch_raises(self, fusi_data, events, rng):
        """Confound list with wrong number of entries raises ValueError."""
        conf = pd.DataFrame({"motion": rng.standard_normal(200)})
        model = FirstLevelModel(noise_model="ols")
        with pytest.raises(ValueError, match="confound"):
            model.fit([fusi_data, fusi_data], events=[events, events], confounds=[conf])

    def test_mask_dim_order_is_canonicalized(
        self, frame_times, events, make_glm_test_dataarray
    ):
        """Wrong-order VoxelData masks are canonicalized before fitting."""
        data = make_glm_test_dataarray(
            np.zeros((len(frame_times), 2, 3, 4)),
            ("time", "k", "j", "i"),
            time=frame_times,
        )
        mask = (
            data.isel(time=0, drop=True)
            .transpose("i", "j", "k")
            .copy(
                data=np.ones(
                    (data.sizes["i"], data.sizes["j"], data.sizes["k"]), dtype=bool
                )
            )
        )
        model = FirstLevelModel(noise_model="ols", mask=mask)

        model.fit(data, events=events)

        assert model.mask is not None
        assert model.mask.dims == ("k", "j", "i")

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
# FirstLevelModel: noise model public validation
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("noise_model", ["ar2", "AR1"])
def test_ar_noise_models_fit_and_contrast(noise_model, fusi_data, events):
    """Supported AR noise models work through the public fit/contrast seam."""
    model = FirstLevelModel(noise_model=noise_model)

    model.fit(fusi_data, events=events)
    z_map = model.compute_contrast("A - B")

    assert np.all(np.isfinite(z_map.values))


@pytest.mark.parametrize("noise_model", ["invalid", "arfoo", "garch", "ar0"])
def test_invalid_noise_models_raise_from_fit(noise_model, fusi_data, events):
    """Unsupported noise models raise through public fit validation."""
    model = FirstLevelModel(noise_model=noise_model)

    with pytest.raises(ValueError, match="noise_model|AR order"):
        model.fit(fusi_data, events=events)
