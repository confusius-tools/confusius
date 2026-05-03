"""Tests for error handling and edge cases in GLM modules."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from confusius.glm._contrasts import Contrast
from confusius.glm._design import (
    _compute_sampling_interval,
    _make_drift_regressors,
    _orthogonalize,
    _regressor_names,
    make_first_level_design_matrix,
)
from confusius.glm._hrf_models import _hrf_kernel
from confusius.glm._models import ARModel, OLSModel
from confusius.glm._utils import estimate_ar_coeffs, expression_to_contrast_vector


# -----------------------------------------------------------------------------
# _design.py error tests
# -----------------------------------------------------------------------------


class TestDesignErrors:
    """Tests for error handling in _design module."""

    def test_compute_sampling_interval_non_1d(self):
        """Non-1D volume_times raises ValueError."""
        with pytest.raises(ValueError, match="1D array"):
            _compute_sampling_interval(np.array([[0.0, 0.1], [0.2, 0.3]]))

    def test_compute_sampling_interval_non_increasing(self):
        """Non-increasing volume_times raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            _compute_sampling_interval(np.array([0.1, 0.0, 0.2, 0.3]))

    def test_compute_sampling_interval_single_timepoint(self):
        """Single timepoint raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 timepoints"):
            _compute_sampling_interval(np.array([0.0]))

    def test_compute_sampling_interval_non_uniform(self):
        """Non-uniform spacing raises ValueError."""
        frame_times = np.array([0, 0.1, 0.25, 0.3])
        with pytest.raises(ValueError, match="uniformly spaced"):
            _compute_sampling_interval(frame_times)

    def test_make_drift_regressors_invalid_model(self):
        """Invalid drift model raises ValueError."""
        with pytest.raises(ValueError, match="drift_model"):
            _make_drift_regressors(100, "invalid", 0.01, 1, 0.1)

    def test_make_drift_regressors_high_pass_warning(self):
        """Large cosine high-pass triggers warning."""
        with pytest.warns(UserWarning, match="saturate the design matrix"):
            _make_drift_regressors(10, "cosine", 10.0, 1, 0.1)

    def test_make_drift_regressors_negative_order(self):
        """Negative polynomial drift order raises ValueError."""
        with pytest.raises(ValueError, match="drift_order must be >= 0"):
            _make_drift_regressors(10, "polynomial", 0.01, -1, 0.1)

    def test_make_drift_regressors_single_scan_polynomial(self):
        """Single-scan polynomial drift handles zero tmax."""
        drift, names = _make_drift_regressors(1, "polynomial", 0.01, 1, 0.1)

        assert drift.shape == (1, 2)
        assert names == ["poly_1", "constant"]

    def test_orthogonalize_single_column_returns_input(self):
        """Single-column orthogonalization is a no-op."""
        x = np.arange(5.0)[:, np.newaxis]

        assert_array_equal(_orthogonalize(x), x)

    def test_make_first_level_design_matrix_invalid_hrf(self):
        """Invalid HRF model raises ValueError."""
        frame_times = np.arange(0, 10, 0.1)
        events = pd.DataFrame(
            {
                "trial_type": ["stim"],
                "onset": [1.0],
                "duration": [1.0],
            }
        )

        with pytest.raises(ValueError, match="Unknown hrf_model"):
            make_first_level_design_matrix(frame_times, events, hrf_model="invalid")

    def test_make_first_level_design_matrix_spm_hrf(self):
        """SPM HRF path produces a design matrix."""
        frame_times = np.arange(0, 10, 0.1)
        events = pd.DataFrame(
            {"trial_type": ["stim"], "onset": [1.0], "duration": [1.0]}
        )

        design = make_first_level_design_matrix(frame_times, events, hrf_model="spm")

        assert "stim" in design.columns

    def test_make_first_level_design_matrix_fir_default_delay(self):
        """FIR defaults to a zero-delay regressor when delays are omitted."""
        frame_times = np.arange(0, 10, 0.1)
        events = pd.DataFrame(
            {"trial_type": ["stim"], "onset": [1.0], "duration": [1.0]}
        )

        design = make_first_level_design_matrix(frame_times, events, hrf_model="fir")

        assert "stim_delay_0" in design.columns

    def test_make_first_level_design_matrix_missing_event_column(self):
        """Missing onset or duration columns raise ValueError."""
        frame_times = np.arange(0, 10, 0.1)
        events = pd.DataFrame({"trial_type": ["stim"], "duration": [1.0]})

        with pytest.raises(ValueError, match="no onset column"):
            make_first_level_design_matrix(frame_times, events)

    def test_make_first_level_design_matrix_event_nans(self):
        """NaNs in onset or duration raise ValueError."""
        frame_times = np.arange(0, 10, 0.1)
        events = pd.DataFrame(
            {"trial_type": ["stim"], "onset": [np.nan], "duration": [1.0]}
        )

        with pytest.raises(ValueError, match="must not contain nan values"):
            make_first_level_design_matrix(frame_times, events)

    def test_make_first_level_design_matrix_event_cast_failure(self):
        """Non-numeric onset or duration raises ValueError."""
        frame_times = np.arange(0, 10, 0.1)
        events = pd.DataFrame(
            {"trial_type": ["stim"], "onset": ["oops"], "duration": [1.0]}
        )

        with pytest.raises(ValueError, match="Could not cast onset"):
            make_first_level_design_matrix(frame_times, events)

    def test_make_first_level_design_matrix_negative_duration(self):
        """Negative event durations raise ValueError."""
        frame_times = np.arange(0, 10, 0.1)
        events = pd.DataFrame(
            {"trial_type": ["stim"], "onset": [1.0], "duration": [-1.0]}
        )

        with pytest.raises(ValueError, match="non-negative"):
            make_first_level_design_matrix(frame_times, events)

    def test_make_first_level_design_matrix_unexpected_event_column_warning(self):
        """Unexpected event columns emit a warning."""
        frame_times = np.arange(0, 10, 0.1)
        events = pd.DataFrame(
            {
                "trial_type": ["stim"],
                "onset": [1.0],
                "duration": [1.0],
                "response_time": [0.5],
            }
        )

        with pytest.warns(UserWarning, match="unexpected columns"):
            make_first_level_design_matrix(frame_times, events)

    def test_make_first_level_design_matrix_modulation_nans(self):
        """NaN modulation raises ValueError."""
        frame_times = np.arange(0, 10, 0.1)
        events = pd.DataFrame(
            {
                "trial_type": ["stim"],
                "onset": [1.0],
                "duration": [1.0],
                "modulation": [np.nan],
            }
        )

        with pytest.raises(ValueError, match="modulation"):
            make_first_level_design_matrix(frame_times, events)

    def test_make_first_level_design_matrix_modulation_cast_failure(self):
        """Non-numeric modulation raises ValueError."""
        frame_times = np.arange(0, 10, 0.1)
        events = pd.DataFrame(
            {
                "trial_type": ["stim"],
                "onset": [1.0],
                "duration": [1.0],
                "modulation": ["oops"],
            }
        )

        with pytest.raises(ValueError, match="Could not cast modulation"):
            make_first_level_design_matrix(frame_times, events)

    def test_make_first_level_design_matrix_duplicate_events_warning(self):
        """Duplicate events trigger a warning and are combined."""
        frame_times = np.arange(0, 10, 0.1)
        events = pd.DataFrame(
            {
                "trial_type": ["stim", "stim"],
                "onset": [1.0, 1.0],
                "duration": [1.0, 1.0],
                "modulation": [1.0, 2.0],
            }
        )

        with pytest.warns(UserWarning, match="Duplicated events"):
            make_first_level_design_matrix(frame_times, events, hrf_model=None)

    def test_make_first_level_design_matrix_early_onset_warning(self):
        """Events before the minimum onset threshold trigger a warning."""
        frame_times = np.arange(0, 10, 0.1)
        events = pd.DataFrame(
            {"trial_type": ["stim"], "onset": [-25.0], "duration": [1.0]}
        )

        with pytest.warns(UserWarning, match="earlier than"):
            make_first_level_design_matrix(frame_times, events, min_onset=-24.0)

    def test_make_first_level_design_matrix_confounds_vector(self):
        """1D confounds are accepted and promoted to one column."""
        frame_times = np.arange(0, 10, 0.1)
        confounds = np.linspace(0, 1, len(frame_times))

        design = make_first_level_design_matrix(frame_times, confounds=confounds)

        assert "confound_0" in design.columns

    def test_make_first_level_design_matrix_confounds_bad_ndim(self):
        """Confounds must be 1D or 2D."""
        frame_times = np.arange(0, 10, 0.1)
        confounds = np.zeros((2, 3, 4))

        with pytest.raises(ValueError, match="1D or 2D"):
            make_first_level_design_matrix(frame_times, confounds=confounds)

    def test_make_first_level_design_matrix_confounds_length_mismatch(self):
        """Confound length mismatch raises ValueError."""
        frame_times = np.arange(0, 10, 0.1)
        confounds = np.zeros((len(frame_times) - 1, 2))

        with pytest.raises(ValueError, match="Incorrect specification of confounds"):
            make_first_level_design_matrix(frame_times, confounds=confounds)

    def test_make_first_level_design_matrix_confounds_nan(self):
        """NaNs in confounds raise ValueError."""
        frame_times = np.arange(0, 10, 0.1)
        confounds = np.zeros((len(frame_times), 1))
        confounds[0, 0] = np.nan

        with pytest.raises(ValueError, match="Confounds contain NaN values"):
            make_first_level_design_matrix(frame_times, confounds=confounds)

    def test_make_first_level_design_matrix_confound_names_mismatch(self):
        """Confound name count mismatch raises ValueError."""
        frame_times = np.arange(0, 10, 0.1)
        confounds = np.zeros((len(frame_times), 2))

        with pytest.raises(ValueError, match="Incorrect number of confound names"):
            make_first_level_design_matrix(
                frame_times,
                confounds=confounds,
                confound_names=["only_one"],
            )

    def test_make_first_level_design_matrix_duplicate_names(self):
        """Duplicate regressor names raise ValueError."""
        frame_times = np.arange(0, 10, 0.1)
        confounds = np.zeros((len(frame_times), 1))

        with pytest.raises(ValueError, match="unique names"):
            make_first_level_design_matrix(
                frame_times,
                confounds=confounds,
                confound_names=["constant"],
            )

    def test_make_first_level_design_matrix_events_must_be_dataframe(self):
        """Events must be a DataFrame."""
        frame_times = np.arange(0, 10, 0.1)

        with pytest.raises(TypeError, match="pandas DataFrame"):
            make_first_level_design_matrix(frame_times, events=[1, 2, 3])


# -----------------------------------------------------------------------------
# _contrasts.py error tests
# -----------------------------------------------------------------------------


class TestContrastErrors:
    """Tests for error handling in _contrasts module."""

    def test_contrast_variance_not_1d(self):
        """Variance must be 1D."""
        effect = np.ones(5)
        variance = np.ones((5, 2))

        with pytest.raises(ValueError, match="1D"):
            Contrast.from_estimate(effect, variance)

    def test_contrast_effect_not_1d_or_2d(self):
        """Effect must be 1D or 2D."""
        effect = np.ones((2, 3, 4))
        variance = np.ones(4)

        with pytest.raises(ValueError, match="1D or 2D"):
            Contrast.from_estimate(effect, variance)

    def test_contrast_invalid_stat_type(self):
        """stat_type must be 't' or 'F'."""
        effect = np.ones(5)
        variance = np.ones(5)

        with pytest.raises(ValueError, match="'t' or 'F'"):
            Contrast.from_estimate(effect, variance, stat_type="chi2")

    def test_contrast_add_different_stat_types(self):
        """Cannot add contrasts with different stat types."""
        con1 = Contrast.from_estimate(np.ones(5), np.ones(5), stat_type="t")
        con2 = Contrast.from_estimate(np.ones((2, 5)), np.ones(5), stat_type="F")

        with pytest.raises(ValueError, match="stat types"):
            con1 + con2

    def test_contrast_add_different_dimensions(self):
        """Cannot add contrasts with different dimensions."""
        con1 = Contrast.from_estimate(
            np.ones(5), np.ones(5), dim=1, stat_type="F"
        )
        con2 = Contrast.from_estimate(
            np.ones((2, 5)), np.ones(5), dim=2, stat_type="F"
        )

        with pytest.raises(ValueError, match="dimensions"):
            con1 + con2


# -----------------------------------------------------------------------------
# _models.py error tests
# -----------------------------------------------------------------------------


class TestModelsErrors:
    """Tests for error handling in _models module."""

    def test_ar_model_invalid_rho_shape(self):
        """AR rho must be 2D with shape (order, n_voxels)."""
        design = np.random.randn(10, 3)

        with pytest.raises(ValueError, match="2D"):
            ARModel(design, rho=np.array([0.1, 0.2, 0.3]))

    def test_t_contrast_multi_row_raises(self):
        """t-contrast rejects multi-row contrast matrices."""
        X = np.random.randn(20, 5)
        y = np.random.randn(20, 3)
        model = OLSModel(X)
        results = model.fit(y)

        # Multi-row contrast (should fail for t-test)
        contrast = np.eye(5)[:3]  # 3 x 5

        with pytest.raises(ValueError, match="single row"):
            results.t_contrast(contrast)


# -----------------------------------------------------------------------------
# _utils.py tests
# -----------------------------------------------------------------------------


class TestUtils:
    """Tests for _utils module."""

    def test_regressor_names_fir_default(self):
        """FIR regressor names default to a zero delay."""
        assert _regressor_names("stim", "fir") == ["stim_delay_0"]

    def test_hrf_kernel_spm(self):
        """SPM kernel helper returns a single kernel."""
        kernels = _hrf_kernel("spm", 0.1)

        assert len(kernels) == 1
        assert kernels[0].ndim == 1

    def test_expression_to_contrast_vector_simple(self):
        """Parse simple contrast expression."""
        columns = ["stim_A", "stim_B", "constant"]

        result = expression_to_contrast_vector("stim_A", columns)

        expected = np.array([1.0, 0.0, 0.0])
        assert_array_equal(result, expected)

    def test_expression_to_contrast_vector_difference(self):
        """Parse contrast expression with subtraction."""
        columns = ["stim_A", "stim_B", "constant"]

        result = expression_to_contrast_vector("stim_A - stim_B", columns)

        expected = np.array([1.0, -1.0, 0.0])
        assert_array_equal(result, expected)

    def test_expression_to_contrast_vector_error(self):
        """Invalid expression raises ValueError."""
        columns = ["stim_A", "stim_B"]

        with pytest.raises(ValueError, match="Could not evaluate"):
            expression_to_contrast_vector("invalid_column", columns)

    def test_estimate_ar_coeffs_1d_signal(self):
        """1D signal input returns scalar-shaped coefficients."""
        signal = np.random.default_rng(0).standard_normal(100)
        rho, sigma = estimate_ar_coeffs(signal, order=1)
        assert rho.shape == (1,)

    def test_estimate_ar_coeffs_invalid_ndim(self):
        """3D array input raises ValueError."""
        with pytest.raises(ValueError, match="Expected 1D or 2D"):
            estimate_ar_coeffs(np.zeros((5, 5, 5)))


