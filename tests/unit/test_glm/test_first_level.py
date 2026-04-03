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
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def frame_times():
    """200 uniformly spaced volume times at 2 Hz (dt=0.5 s)."""
    return np.arange(200) * 0.5


@pytest.fixture
def events():
    """Simple two-condition event table."""
    onsets_a = np.arange(5) * 20.0
    onsets_b = np.arange(5) * 20.0 + 10.0
    return pd.DataFrame(
        {
            "trial_type": ["A"] * 5 + ["B"] * 5,
            "onset": np.concatenate([onsets_a, onsets_b]),
            "duration": [2.0] * 10,
        }
    )


@pytest.fixture
def fusi_data(rng, frame_times):
    """Small (time, z, y, x) DataArray for testing."""
    n_time = len(frame_times)
    return xr.DataArray(
        rng.standard_normal((n_time, 2, 3, 4)),
        dims=["time", "z", "y", "x"],
        coords={
            "time": frame_times,
            "z": np.arange(2) * 0.5,
            "y": np.arange(3) * 0.1,
            "x": np.arange(4) * 0.1,
        },
    )


@pytest.fixture
def fusi_data_2d(rng, frame_times):
    """Small (time, y, x) DataArray (no z dimension)."""
    n_time = len(frame_times)
    return xr.DataArray(
        rng.standard_normal((n_time, 5, 6)),
        dims=["time", "y", "x"],
        coords={
            "time": frame_times,
            "y": np.arange(5) * 0.1,
            "x": np.arange(6) * 0.1,
        },
    )


# -----------------------------------------------------------------------------
# Helper function tests
# -----------------------------------------------------------------------------



class TestFlattenSpatial:
    """Tests for _flatten_spatial."""

    def test_shape_3d(self, fusi_data):
        flat, dims, shape = _flatten_spatial(fusi_data)
        assert flat.shape == (200, 2 * 3 * 4)
        assert dims == ("z", "y", "x")
        assert shape == (2, 3, 4)

    def test_shape_2d(self, fusi_data_2d):
        flat, dims, shape = _flatten_spatial(fusi_data_2d)
        assert flat.shape == (200, 5 * 6)
        assert dims == ("y", "x")
        assert shape == (5, 6)

    def test_handles_transposed_input(self, fusi_data):
        transposed = fusi_data.transpose("x", "y", "z", "time")
        flat, dims, shape = _flatten_spatial(transposed)
        assert flat.shape == (200, 2 * 3 * 4)


# -----------------------------------------------------------------------------
# FirstLevelModel: fitting
# -----------------------------------------------------------------------------


class TestFirstLevelModelFit:
    """Tests for FirstLevelModel.fit."""

    def test_basic_fit_ols(self, fusi_data, events):
        model = FirstLevelModel(noise_model="ols")
        result = model.fit(fusi_data, events=events)
        assert result is model
        assert hasattr(model, "results_")
        assert hasattr(model, "design_matrices_")
        assert len(model.design_matrices_) == 1
        assert len(model.results_) == 1

    def test_basic_fit_ar1(self, fusi_data, events):
        model = FirstLevelModel(noise_model="ar1")
        model.fit(fusi_data, events=events)
        assert len(model.results_) == 1

    def test_fit_with_prebuilt_design_matrix(self, fusi_data, frame_times, events):
        dm = make_first_level_design_matrix(frame_times, events=events)
        model = FirstLevelModel(noise_model="ols")
        model.fit(fusi_data, design_matrices=dm)
        assert len(model.design_matrices_) == 1

    def test_multi_run(self, rng, frame_times, events):
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
        model.fit([data1, data2], events=[events, events])
        assert len(model.results_) == 2
        assert len(model.design_matrices_) == 2

    def test_fit_2d_spatial(self, fusi_data_2d, events):
        model = FirstLevelModel(noise_model="ols")
        model.fit(fusi_data_2d, events=events)
        z_map = model.compute_contrast("A - B")
        assert z_map.dims == ("y", "x")
        assert z_map.shape == (5, 6)

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


# -----------------------------------------------------------------------------
# FirstLevelModel: contrasts
# -----------------------------------------------------------------------------


class TestFirstLevelModelContrast:
    """Tests for FirstLevelModel.compute_contrast."""

    @pytest.fixture(autouse=True)
    def _fitted_model(self, fusi_data, events):
        self.model = FirstLevelModel(noise_model="ols")
        self.model.fit(fusi_data, events=events)

    def test_string_contrast(self):
        z_map = self.model.compute_contrast("A - B")
        assert z_map.dims == ("z", "y", "x")
        assert z_map.shape == (2, 3, 4)
        assert np.all(np.isfinite(z_map.values))

    def test_single_condition_contrast(self):
        z_map = self.model.compute_contrast("A")
        assert z_map.shape == (2, 3, 4)

    def test_array_contrast(self):
        dm = self.model.design_matrices_[0]
        vec = np.zeros(len(dm.columns))
        vec[list(dm.columns).index("A")] = 1.0
        vec[list(dm.columns).index("B")] = -1.0
        z_map = self.model.compute_contrast(vec)
        assert z_map.shape == (2, 3, 4)

    def test_output_type_stat(self):
        t_map = self.model.compute_contrast("A", output_type="stat")
        assert t_map.attrs["long_name"] == "stat"

    def test_output_type_p_value(self):
        p_map = self.model.compute_contrast("A", output_type="p_value")
        assert np.all(p_map.values >= 0)
        assert np.all(p_map.values <= 1)

    def test_output_type_effect_size(self):
        e_map = self.model.compute_contrast("A", output_type="effect_size")
        assert e_map.shape == (2, 3, 4)

    def test_output_type_effect_variance(self):
        v_map = self.model.compute_contrast("A", output_type="effect_variance")
        assert np.all(v_map.values >= 0)

    def test_short_contrast_vector_is_padded(self):
        """A contrast vector shorter than the design is zero-padded."""
        vec = np.array([1.0, -1.0])  # Only covers A and B columns.
        z_map = self.model.compute_contrast(vec)
        assert z_map.shape == (2, 3, 4)

    def test_invalid_output_type_raises(self):
        with pytest.raises(ValueError, match="output_type"):
            self.model.compute_contrast("A", output_type="invalid")


class TestFirstLevelModelContrastMultiRun:
    """Test fixed-effects contrast combination across runs."""

    def test_multi_run_contrast(self, rng, frame_times, events):
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
        model.fit([data1, data2], events=[events, events])
        z_map = model.compute_contrast("A - B")
        assert z_map.shape == (2, 3, 4)
        assert np.all(np.isfinite(z_map.values))

    def test_multi_run_confounds_as_list(self, rng, frame_times, events):
        """Multi-run fit with per-run confounds as a list."""
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
        conf = pd.DataFrame({"motion": rng.standard_normal(200)})
        model = FirstLevelModel(noise_model="ols")
        model.fit([data1, data2], events=[events, events], confounds=[conf, conf])
        assert "motion" in model.design_matrices_[0].columns


class TestFirstLevelModelFContrast:
    """Test F-contrast path through compute_contrast."""

    @pytest.fixture(autouse=True)
    def _fitted_model(self, fusi_data, events):
        self.model = FirstLevelModel(noise_model="ols")
        self.model.fit(fusi_data, events=events)

    def test_f_contrast_2d_array(self):
        """F-contrast with a 2D matrix produces a spatial z-map."""
        dm = self.model.design_matrices_[0]
        a_idx = list(dm.columns).index("A")
        b_idx = list(dm.columns).index("B")
        # 2x K matrix testing A and B jointly.
        c = np.zeros((2, len(dm.columns)))
        c[0, a_idx] = 1.0
        c[1, b_idx] = 1.0
        z_map = self.model.compute_contrast(c, stat_type="F")
        assert z_map.shape == (2, 3, 4)
        assert np.all(np.isfinite(z_map.values))

    def test_f_contrast_effect_size_2d_output(self):
        """F-contrast effect_size output has a contrast_dim dimension."""
        dm = self.model.design_matrices_[0]
        a_idx = list(dm.columns).index("A")
        b_idx = list(dm.columns).index("B")
        c = np.zeros((2, len(dm.columns)))
        c[0, a_idx] = 1.0
        c[1, b_idx] = 1.0
        e_map = self.model.compute_contrast(c, stat_type="F", output_type="effect_size")
        assert e_map.dims[0] == "contrast_dim"
        assert e_map.shape == (2, 2, 3, 4)

    def test_explicit_stat_type_t(self):
        """Passing stat_type='t' explicitly still works for 1D contrasts."""
        z_map = self.model.compute_contrast("A - B", stat_type="t")
        assert z_map.shape == (2, 3, 4)

    def test_2d_contrast_zero_padding(self):
        """2D contrast narrower than design is zero-padded."""
        # A 2xK short contrast, where K < n_design_cols.
        c = np.zeros((2, 2))
        c[0, 0] = 1.0
        c[1, 1] = -1.0
        z_map = self.model.compute_contrast(c, stat_type="F")
        assert z_map.shape == (2, 3, 4)


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
        t_res = results.t_contrast(cvec)
        contrast = Contrast(
            effect=np.atleast_1d(t_res["effect"]),
            variance=np.atleast_1d(t_res["sd"]) ** 2,
            dof=float(t_res["df_den"]),
            stat_type="t",
        )
        z_manual = contrast.z_score().reshape(2, 3, 4)

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
        with pytest.raises(ValueError, match="spatial shape"):
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
        with pytest.raises(ValueError, match="string, 1-D, or 2-D"):
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
