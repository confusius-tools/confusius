"""Tests for confusius.glm._design."""

import numpy as np
import pandas as pd
import pytest
import scipy.special as spspecial
from numpy.testing import assert_allclose, assert_array_equal

from confusius.glm._design import (
    _compute_sampling_interval,
    _make_drift_regressors,
    make_first_level_design_matrix,
)
from confusius.glm._hrf_models import (
    claron2021_hrf,
    glover_hrf,
    inverse_gamma_hrf,
    spm_hrf,
)


def _oversampled_time_grid(
    dt: float,
    oversampling: int = 50,
    time_length: float = 32.0,
    onset: float = 0.0,
) -> np.ndarray:
    """Return the oversampled HRF evaluation grid used by the implementation."""
    high_res_dt = dt / oversampling
    time_stamps = np.linspace(
        0,
        time_length,
        np.rint(time_length / high_res_dt).astype(int),
    )
    return time_stamps - onset


def _inverse_gamma_hrf_reference(
    dt: float,
    oversampling: int = 50,
    time_length: float = 32.0,
    alpha: float = 2.5,
    beta: float = 12.7,
    onset: float = 0.0,
) -> np.ndarray:
    """Reference inverse-gamma HRF sampled on the implementation grid."""
    time_stamps = _oversampled_time_grid(
        dt,
        oversampling=oversampling,
        time_length=time_length,
        onset=onset,
    )
    hrf = np.zeros_like(time_stamps)
    positive = time_stamps > 0
    t = time_stamps[positive]
    hrf[positive] = (
        beta**alpha
        / spspecial.gamma(alpha)
        * t ** (-(alpha + 1.0))
        * np.exp(-beta / t)
    )
    hrf /= hrf.sum()
    return hrf

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def frame_times():
    """Volume times for 30 seconds at 10 Hz."""
    return np.arange(0, 30, 0.1)


@pytest.fixture
def basic_events():
    """Simple events DataFrame."""
    return pd.DataFrame(
        {
            "trial_type": ["stim", "stim", "control"],
            "onset": [0.0, 10.0, 20.0],
            "duration": [2.0, 2.0, 2.0],
        }
    )


# -----------------------------------------------------------------------------
# HRF tests
# -----------------------------------------------------------------------------


class TestHRF:
    """Tests for HRF functions."""

    def test_glover_hrf_normalized(self):
        """Glover HRF is normalized to unit mass."""
        hrf = glover_hrf(0.1)

        assert hrf.sum() == pytest.approx(1.0, abs=1e-6)
        assert 0.0 < hrf.max() < 1.0

    def test_glover_hrf_invalid_oversampling(self):
        """Oversampling must be positive."""
        with pytest.raises(ValueError, match="oversampling"):
            glover_hrf(0.1, oversampling=0)

    def test_spm_hrf_normalized(self):
        """SPM HRF is normalized to unit mass."""
        hrf = spm_hrf(0.1)

        assert hrf.sum() == pytest.approx(1.0, abs=1e-6)
        assert 0.0 < hrf.max() < 1.0

    def test_glover_spm_similar(self):
        """Glover and SPM HRFs are similar."""
        dt = 0.1
        glover = glover_hrf(dt)
        spm = spm_hrf(dt)

        # Should be highly correlated
        correlation = np.corrcoef(glover, spm)[0, 1]
        assert correlation > 0.95

    def test_inverse_gamma_hrf_matches_reference(self):
        """Inverse-gamma HRF matches the explicit reference equation."""
        dt = 0.1

        reference = _inverse_gamma_hrf_reference(dt)
        observed = inverse_gamma_hrf(dt)

        assert_allclose(observed, reference)

    def test_inverse_gamma_hrf_matches_reference_with_custom_parameters(self):
        """Inverse-gamma HRF keeps the reference parameterization when overridden."""
        dt = 0.1
        alpha = 3.0
        beta = 9.5
        onset = 0.3

        reference = _inverse_gamma_hrf_reference(
            dt,
            alpha=alpha,
            beta=beta,
            onset=onset,
        )
        observed = inverse_gamma_hrf(dt, alpha=alpha, beta=beta, onset=onset)

        assert_allclose(observed, reference)

    def test_claron2021_hrf_is_inverse_gamma_preset(self):
        """Claron 2021 HRF is a thin preset around the generic inverse-gamma HRF."""
        dt = 0.1

        observed = claron2021_hrf(dt)
        reference = inverse_gamma_hrf(dt, alpha=2.5, beta=12.7)

        assert_allclose(observed, reference)


# -----------------------------------------------------------------------------
# Sampling interval tests
# -----------------------------------------------------------------------------


class TestSamplingInterval:
    """Tests for sampling interval computation."""

    def test_uniform_spacing(self):
        """Compute sampling interval from uniform frame times."""
        frame_times = np.arange(0, 10, 0.1)  # 10 Hz

        dt = _compute_sampling_interval(frame_times)

        assert dt == pytest.approx(0.1)

    def test_non_uniform_spacing_raises(self):
        """Non-uniform frame times raise ValueError."""
        frame_times = np.array([0, 0.1, 0.25, 0.3])  # Irregular

        with pytest.raises(ValueError, match="uniformly spaced"):
            _compute_sampling_interval(frame_times)

    def test_single_timepoint_raises(self):
        """Single timepoint raises ValueError."""
        frame_times = np.array([0.0])

        with pytest.raises(ValueError, match="at least 2 timepoints"):
            _compute_sampling_interval(frame_times)

    def test_two_timepoints(self):
        """Two timepoints works correctly."""
        frame_times = np.array([0.0, 0.1])

        dt = _compute_sampling_interval(frame_times)

        assert dt == pytest.approx(0.1)


# -----------------------------------------------------------------------------
# Drift regressor tests
# -----------------------------------------------------------------------------


class TestDriftRegressors:
    """Tests for drift regressor creation."""

    def test_cosine_drift_basic(self):
        """Cosine drift regressors created correctly."""
        n_scans = 1000
        dt = 0.1
        high_pass = 0.01  # 0.01 Hz = 100s period

        drift_regs, names = _make_drift_regressors(
            n_scans, "cosine", high_pass, drift_order=1, dt=dt
        )

        # Should have ~2*1000*0.1*0.01 = 2 cosine regressors
        assert drift_regs.shape[1] >= 1
        assert len(names) == drift_regs.shape[1]
        assert all("cosine" in name for name in names[:-1])
        assert names[-1] == "constant"

    def test_cosine_drift_orthogonal(self):
        """Cosine drift regressors are orthogonal."""
        n_scans = 200
        dt = 0.1

        drift_regs, _ = _make_drift_regressors(n_scans, "cosine", 0.01, 1, dt)

        if drift_regs.shape[1] > 1:
            # Dot product between different cosines should be ~0
            dot = drift_regs[:, 0] @ drift_regs[:, 1]
            assert abs(dot) < 1e-10

    def test_polynomial_drift(self):
        """Polynomial drift regressors created correctly."""
        n_scans = 100
        drift_regs, names = _make_drift_regressors(
            n_scans, "polynomial", 0.01, drift_order=2, dt=0.1
        )

        # Should have order + 1 = 3 regressors (linear, quadratic, constant)
        assert drift_regs.shape == (100, 3)
        assert names[-1] == "constant"
        assert "poly_1" in names
        assert "poly_2" in names

    def test_polynomial_last_column_constant(self):
        """Polynomial drift last column is constant (1s)."""
        drift_regs, _ = _make_drift_regressors(
            50, "polynomial", 0.01, drift_order=1, dt=0.1
        )

        assert_array_equal(drift_regs[:, -1], np.ones(50))

    def test_no_drift_model(self):
        """drift_model=None returns an intercept only."""
        drift_regs, names = _make_drift_regressors(100, None, 0.01, 1, 0.1)

        assert drift_regs.shape == (100, 1)
        assert names == ["constant"]

    def test_invalid_drift_model(self):
        """Invalid drift model raises ValueError."""
        with pytest.raises(ValueError, match="drift_model"):
            _make_drift_regressors(100, "invalid", 0.01, 1, 0.1)


# -----------------------------------------------------------------------------
# Design matrix tests
# -----------------------------------------------------------------------------


class TestDesignMatrix:
    """Tests for design matrix creation."""

    def test_basic_design_matrix(self, frame_times, basic_events):
        """Basic design matrix creation."""
        design = make_first_level_design_matrix(
            frame_times, basic_events, hrf_model="glover"
        )

        # Should have columns for stim, control, and drift
        assert "stim" in design.columns
        assert "control" in design.columns
        assert design.shape[0] == len(frame_times)

    def test_design_matrix_no_events(self, frame_times):
        """Design matrix with no events (only drift)."""
        design = make_first_level_design_matrix(
            frame_times, events=None, drift_model="cosine"
        )

        assert list(design.columns) == ["constant"]

    def test_design_matrix_with_confounds(self, frame_times, basic_events):
        """Design matrix with confound regressors."""
        confounds = np.random.randn(len(frame_times), 2)

        design = make_first_level_design_matrix(
            frame_times,
            basic_events,
            confounds=confounds,
            confound_names=["motion_x", "motion_y"],
        )

        assert "motion_x" in design.columns
        assert "motion_y" in design.columns

    def test_design_matrix_fir_model(self, frame_times, basic_events):
        """FIR model design matrix."""
        design = make_first_level_design_matrix(
            frame_times,
            basic_events,
            hrf_model="fir",
            fir_delays=[0, 1, 2],
        )

        # Should have stim_delay_0, stim_delay_1, etc.
        assert any("stim_delay_0" in col for col in design.columns)
        assert any("stim_delay_1" in col for col in design.columns)
        assert any("stim_delay_2" in col for col in design.columns)

    def test_design_matrix_no_drift(self, frame_times, basic_events):
        """Design matrix without drift still includes an intercept."""
        design = make_first_level_design_matrix(
            frame_times, basic_events, drift_model=None
        )

        assert set(design.columns) == {"stim", "control", "constant"}

    def test_design_matrix_default_trial_type(self, frame_times):
        """Events without trial_type get default 'dummy' label."""
        events = pd.DataFrame(
            {
                "onset": [0.0, 10.0],
                "duration": [1.0, 1.0],
            }
        )

        with pytest.warns(UserWarning, match="trial_type"):
            design = make_first_level_design_matrix(
                frame_times, events, drift_model=None
            )

        assert "dummy" in design.columns

    def test_design_matrix_preserves_frame_times_index(self, frame_times, basic_events):
        """Design matrix index matches frame_times."""
        design = make_first_level_design_matrix(frame_times, basic_events)

        assert_allclose(design.index.to_numpy(dtype=float), frame_times)

    def test_design_matrix_with_dataframe_confounds(self, frame_times, basic_events):
        """DataFrame confounds keep their column names."""
        confounds = pd.DataFrame(
            {
                "motion_x": np.linspace(0, 1, len(frame_times)),
                "motion_y": np.linspace(1, 0, len(frame_times)),
            }
        )

        design = make_first_level_design_matrix(
            frame_times,
            basic_events,
            confounds=confounds,
        )

        assert "motion_x" in design.columns
        assert "motion_y" in design.columns

    def test_design_matrix_values_shape(self, frame_times, basic_events):
        """Design matrix regressor values have correct shape."""
        design = make_first_level_design_matrix(
            frame_times, basic_events, hrf_model="glover", drift_model=None
        )

        # Should have same number of rows as frame_times
        assert design.shape[0] == len(frame_times)
        # Should have stim and control columns
        assert "stim" in design.columns
        assert "control" in design.columns

    def test_design_matrix_callable_hrf(self, frame_times, basic_events):
        """Custom callable HRF is invoked to build the regressors."""
        design = make_first_level_design_matrix(
            frame_times,
            basic_events,
            hrf_model=claron2021_hrf,
            drift_model=None,
        )

        reference = make_first_level_design_matrix(
            frame_times,
            basic_events,
            hrf_model="claron2021",
            drift_model=None,
        )

        assert_allclose(design["stim"].to_numpy(), reference["stim"].to_numpy())

    def test_design_matrix_hrf_none(self, frame_times, basic_events):
        """Design matrix with hrf_model=None creates stick functions."""
        design = make_first_level_design_matrix(
            frame_times, basic_events, hrf_model=None, drift_model=None
        )

        # Should still create regressors (boxcar/stick functions)
        assert "stim" in design.columns
        assert "control" in design.columns
        # Values should be 0 or 1 (boxcar)
        assert set(np.unique(design["stim"])).issubset({0, 1})


# -----------------------------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_events(self, frame_times):
        """Empty events DataFrame works."""
        events = pd.DataFrame(columns=["trial_type", "onset", "duration"])

        design = make_first_level_design_matrix(frame_times, events, drift_model=None)

        assert list(design.columns) == ["constant"]

    def test_single_event(self, frame_times):
        """Single event in design matrix."""
        events = pd.DataFrame(
            {
                "trial_type": ["stim"],
                "onset": [5.0],
                "duration": [1.0],
            }
        )

        design = make_first_level_design_matrix(frame_times, events, drift_model=None)

        assert "stim" in design.columns
        assert design.shape[1] == 2

    def test_event_at_time_zero(self, frame_times):
        """Event at time 0."""
        events = pd.DataFrame(
            {
                "trial_type": ["stim"],
                "onset": [0.0],
                "duration": [1.0],
            }
        )

        design = make_first_level_design_matrix(frame_times, events, drift_model=None)

        assert np.any(design["stim"][:50] > 0)

    def test_zero_duration_event_with_no_hrf_is_not_persistent(self, frame_times):
        """Zero-duration events stay impulse-like when no HRF is used."""
        events = pd.DataFrame(
            {
                "trial_type": ["stim"],
                "onset": [10.0],
                "duration": [0.0],
            }
        )

        with pytest.warns(UserWarning, match="null duration"):
            design = make_first_level_design_matrix(
                frame_times,
                events,
                hrf_model=None,
                drift_model=None,
            )

        stim = design["stim"].to_numpy()
        assert np.count_nonzero(stim) == 1
        assert stim.sum() == pytest.approx(1.0)

    def test_very_short_event(self, frame_times):
        """Event with very short but non-zero duration."""
        # Use duration >= dt so the event is captured at the sampling resolution
        events = pd.DataFrame(
            {
                "trial_type": ["stim"],
                "onset": [10.0],
                "duration": [0.1],  # One sampling interval
            }
        )

        design = make_first_level_design_matrix(frame_times, events, drift_model=None)

        # Should still create a regressor with some non-zero values
        assert "stim" in design.columns
        # The HRF convolution will spread the brief event over time
        assert np.any(design["stim"] != 0)
