"""Tests for confusius.glm._design."""

import numpy as np
import pandas as pd
import pytest
import scipy.special as spspecial
from numpy.testing import assert_allclose

from confusius.glm._design import make_first_level_design_matrix
from confusius.glm._hrf_models import (
    claron2021_hrf,
    gamma_hrf,
    glover_hrf,
    inverse_gamma_hrf,
    spm_hrf,
    verhoef2025_hrf,
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
        beta**alpha / spspecial.gamma(alpha) * t ** (-(alpha + 1.0)) * np.exp(-beta / t)
    )
    hrf /= hrf.sum()
    return hrf


def _gamma_hrf_reference(
    dt: float,
    oversampling: int = 50,
    time_length: float = 32.0,
    peak_delay: float = 5.0,
    dispersion: float = 1.0,
    onset: float = 0.0,
) -> np.ndarray:
    """Reference positive-only gamma HRF sampled on the implementation grid."""
    time_stamps = _oversampled_time_grid(
        dt,
        oversampling=oversampling,
        time_length=time_length,
        onset=onset,
    )
    hrf = np.zeros_like(time_stamps)
    positive = time_stamps >= 0
    t = time_stamps[positive]
    shape = peak_delay / dispersion + 1.0
    hrf[positive] = (
        t ** (shape - 1.0)
        * np.exp(-t / dispersion)
        / (spspecial.gamma(shape) * dispersion**shape)
    )
    hrf /= hrf.sum()
    return hrf


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def frame_times():
    """Volume times for 30 seconds at 10 Hz.

    Shadows the conftest fixture: design/HRF tests don't fit a model, so the
    short, dense series is faster and produces meaningful HRF coverage.
    """
    return np.arange(0, 30, 0.1)


@pytest.fixture
def basic_events():
    """Simple events DataFrame with two trial types."""
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

    def test_gamma_hrf_matches_reference(self):
        """Positive-only gamma HRF matches the explicit reference equation."""
        dt = 0.1

        reference = _gamma_hrf_reference(dt)
        observed = gamma_hrf(dt)

        assert_allclose(observed, reference)

    def test_gamma_hrf_matches_reference_with_custom_parameters(self):
        """Gamma HRF keeps the reference parameterization when overridden."""
        dt = 0.1
        peak_delay = 3.5
        dispersion = 0.8
        onset = 0.2

        reference = _gamma_hrf_reference(
            dt,
            peak_delay=peak_delay,
            dispersion=dispersion,
            onset=onset,
        )
        observed = gamma_hrf(
            dt,
            peak_delay=peak_delay,
            dispersion=dispersion,
            onset=onset,
        )

        assert_allclose(observed, reference)

    def test_verhoef2025_hrf_is_gamma_preset(self):
        """Verhoef 2025 HRF is a thin preset around the generic gamma HRF."""
        dt = 0.1

        observed = verhoef2025_hrf(dt)
        reference = gamma_hrf(dt, peak_delay=5.0, dispersion=1.0)

        assert_allclose(observed, reference)

    def test_gamma_hrf_invalid_oversampling(self):
        """gamma_hrf rejects non-positive oversampling."""
        with pytest.raises(ValueError, match="oversampling"):
            gamma_hrf(0.1, oversampling=0)

    def test_gamma_hrf_invalid_dispersion(self):
        """gamma_hrf rejects non-positive dispersion (would produce NaN HRF)."""
        with pytest.raises(ValueError, match="dispersion"):
            gamma_hrf(0.1, dispersion=0.0)

    def test_gamma_hrf_invalid_peak_delay(self):
        """gamma_hrf rejects negative peak_delay."""
        with pytest.raises(ValueError, match="peak_delay"):
            gamma_hrf(0.1, peak_delay=-1.0)

    def test_inverse_gamma_hrf_invalid_oversampling(self):
        """inverse_gamma_hrf rejects non-positive oversampling."""
        with pytest.raises(ValueError, match="oversampling"):
            inverse_gamma_hrf(0.1, oversampling=0)


# -----------------------------------------------------------------------------
# Design matrix tests
# -----------------------------------------------------------------------------


class TestDesignMatrix:
    """Tests for design matrix creation."""

    def test_design_matrix_columns_and_index(self, frame_times, basic_events):
        """Design matrix exposes one column per trial type plus drift, and its
        index matches the input frame times."""
        design = make_first_level_design_matrix(
            frame_times, basic_events, hrf_model="glover"
        )

        assert "stim" in design.columns
        assert "control" in design.columns
        assert design.shape[0] == len(frame_times)
        assert_allclose(design.index.to_numpy(dtype=float), frame_times)

    def test_design_matrix_no_events(self, frame_times):
        """Without events and only a drift model, the design reduces to the
        drift regressors plus the constant."""
        design = make_first_level_design_matrix(
            frame_times, events=None, drift_model="cosine"
        )

        assert list(design.columns) == ["constant"]

    def test_design_matrix_array_confounds_pass_through(
        self, frame_times, basic_events
    ):
        """Array confounds keep the user's values column-by-column."""
        confounds = np.column_stack(
            [
                np.linspace(0.0, 1.0, len(frame_times)),
                np.linspace(1.0, 0.0, len(frame_times)),
            ]
        )

        design = make_first_level_design_matrix(
            frame_times,
            basic_events,
            confounds=confounds,
            confound_names=["motion_x", "motion_y"],
        )

        assert_allclose(design["motion_x"].to_numpy(), confounds[:, 0])
        assert_allclose(design["motion_y"].to_numpy(), confounds[:, 1])

    def test_design_matrix_dataframe_confounds_pass_through(
        self, frame_times, basic_events
    ):
        """DataFrame confounds keep both their names and values."""
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

        assert_allclose(design["motion_x"].to_numpy(), confounds["motion_x"].to_numpy())
        assert_allclose(design["motion_y"].to_numpy(), confounds["motion_y"].to_numpy())

    def test_design_matrix_fir_delays_are_time_shifted(self, frame_times, basic_events):
        """FIR regressors with delays 0, 1, 2 are exact one-sample shifts of each
        other (the FIR basis is meant to capture lagged responses)."""
        design = make_first_level_design_matrix(
            frame_times,
            basic_events,
            hrf_model="fir",
            fir_delays=[0, 1, 2],
            drift_model=None,
        )

        d0 = design["stim_delay_0"].to_numpy()
        d1 = design["stim_delay_1"].to_numpy()
        d2 = design["stim_delay_2"].to_numpy()

        # Each delay column is the previous one shifted forward by one frame.
        assert_allclose(d1[1:], d0[:-1], atol=1e-12)
        assert_allclose(d2[1:], d1[:-1], atol=1e-12)

    def test_design_matrix_no_drift(self, frame_times, basic_events):
        """`drift_model=None` keeps only condition columns plus the constant."""
        design = make_first_level_design_matrix(
            frame_times, basic_events, drift_model=None
        )

        assert set(design.columns) == {"stim", "control", "constant"}

    def test_design_matrix_polynomial_drift(self, frame_times, basic_events):
        """Polynomial drift adds `drift_order` columns plus the constant. After
        orthogonalization the constant column is unit-valued by construction."""
        design = make_first_level_design_matrix(
            frame_times,
            basic_events,
            drift_model="polynomial",
            drift_order=2,
        )

        poly_cols = [c for c in design.columns if c.startswith("poly_")]
        assert poly_cols == ["poly_1", "poly_2"]
        assert_allclose(design["constant"].to_numpy(), np.ones(len(frame_times)))

    def test_design_matrix_cosine_drift_orthogonal(self, frame_times, basic_events):
        """Cosine drift regressors emitted by the design matrix are orthogonal,
        which is the property that makes them safe to use as a high-pass basis."""
        design = make_first_level_design_matrix(
            frame_times,
            basic_events,
            drift_model="cosine",
            low_cutoff=0.05,
        )
        cosine_cols = [c for c in design.columns if c.startswith("cosine")]
        assert len(cosine_cols) >= 2  # the test only makes sense with >1 column.

        cosines = design[cosine_cols].to_numpy()
        gram = cosines.T @ cosines
        off_diagonal = gram - np.diag(np.diagonal(gram))
        assert np.all(np.abs(off_diagonal) < 1e-10)

    def test_design_matrix_default_trial_type(self, frame_times):
        """Events without `trial_type` default to a 'event' label and warn."""
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

        assert "event" in design.columns

    def test_design_matrix_callable_hrf_matches_named(self, frame_times, basic_events):
        """A callable HRF produces the same regressors as the equivalent
        named-string HRF — confirming the callable plumbing is wired up."""
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

    def test_design_matrix_hrf_none_is_boxcar(self, frame_times, basic_events):
        """`hrf_model=None` returns boxcar regressors with values in {0, 1}."""
        design = make_first_level_design_matrix(
            frame_times, basic_events, hrf_model=None, drift_model=None
        )

        assert set(np.unique(design["stim"])).issubset({0, 1})
        # Two stim events of duration 2.0 at dt=0.1 should each activate ~20
        # frames → ~40 active frames total.
        assert design["stim"].sum() == pytest.approx(40.0, abs=1.0)


# -----------------------------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_events(self, frame_times):
        """Empty events DataFrame yields a design with only the constant."""
        events = pd.DataFrame(columns=["trial_type", "onset", "duration"])

        design = make_first_level_design_matrix(frame_times, events, drift_model=None)

        assert list(design.columns) == ["constant"]

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

    def test_non_uniform_frame_times_raise(self, basic_events):
        """Non-uniform frame times raise instead of silently using the median."""
        irregular = np.array([0.0, 0.1, 0.25, 0.30, 0.40])
        with pytest.raises(ValueError, match="uniformly spaced"):
            make_first_level_design_matrix(irregular, basic_events)

    def test_invalid_drift_model_raises(self, frame_times, basic_events):
        """Unknown drift_model values raise."""
        with pytest.raises(ValueError, match="drift_model"):
            make_first_level_design_matrix(
                frame_times, basic_events, drift_model="unknown"  # ty: ignore[invalid-argument-type]
            )


# -----------------------------------------------------------------------------
# Public-API error tests
# -----------------------------------------------------------------------------


class TestDesignMatrixInputValidation:
    """Errors and warnings surfaced from `make_first_level_design_matrix` for
    malformed user input."""

    @pytest.fixture
    def stim_event(self):
        return pd.DataFrame({"trial_type": ["stim"], "onset": [1.0], "duration": [1.0]})

    def test_invalid_hrf_model_raises(self, frame_times, stim_event):
        with pytest.raises(ValueError, match="Unknown hrf_model"):
            make_first_level_design_matrix(frame_times, stim_event, hrf_model="invalid")  # ty: ignore[invalid-argument-type]

    def test_invalid_drift_order_raises(self, frame_times, stim_event):
        with pytest.raises(ValueError, match="drift_order must be >= 0"):
            make_first_level_design_matrix(
                frame_times, stim_event, drift_model="polynomial", drift_order=-1
            )

    def test_high_low_cutoff_warns(self, stim_event):
        """A `low_cutoff` close to the Nyquist frequency saturates the cosine
        basis with redundant high-frequency regressors and should warn."""
        # 10 frames at dt=0.1s, low_cutoff at 10 Hz → frequency far above Nyquist.
        frame_times = np.arange(0, 1, 0.1)
        with pytest.warns(UserWarning, match="saturate the design matrix"):
            make_first_level_design_matrix(
                frame_times,
                stim_event,
                drift_model="cosine",
                low_cutoff=10.0,
            )

    def test_events_must_be_dataframe(self, frame_times):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            make_first_level_design_matrix(frame_times, events=[1, 2, 3])  # ty: ignore[invalid-argument-type]

    def test_events_missing_onset_column_raises(self, frame_times):
        events = pd.DataFrame({"trial_type": ["stim"], "duration": [1.0]})
        with pytest.raises(ValueError, match="no onset column"):
            make_first_level_design_matrix(frame_times, events)

    def test_events_nan_onset_raises(self, frame_times):
        events = pd.DataFrame(
            {"trial_type": ["stim"], "onset": [np.nan], "duration": [1.0]}
        )
        with pytest.raises(ValueError, match="must not contain nan values"):
            make_first_level_design_matrix(frame_times, events)

    def test_events_non_numeric_onset_raises(self, frame_times):
        events = pd.DataFrame(
            {"trial_type": ["stim"], "onset": ["oops"], "duration": [1.0]}
        )
        with pytest.raises(ValueError, match="Could not cast onset"):
            make_first_level_design_matrix(frame_times, events)

    def test_events_negative_duration_raises(self, frame_times):
        events = pd.DataFrame(
            {"trial_type": ["stim"], "onset": [1.0], "duration": [-1.0]}
        )
        with pytest.raises(ValueError, match="non-negative"):
            make_first_level_design_matrix(frame_times, events)

    def test_events_unexpected_column_warns(self, frame_times):
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

    def test_events_modulation_nan_raises(self, frame_times):
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

    def test_events_modulation_non_numeric_raises(self, frame_times):
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

    def test_duplicate_events_warn(self, frame_times):
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

    def test_event_before_min_onset_warns(self, frame_times):
        events = pd.DataFrame(
            {"trial_type": ["stim"], "onset": [-25.0], "duration": [1.0]}
        )
        with pytest.warns(UserWarning, match="earlier than"):
            make_first_level_design_matrix(frame_times, events, min_onset=-24.0)

    def test_array_confounds_promoted_to_one_column(self, frame_times):
        """1D array confounds are promoted to a single regressor column."""
        confounds = np.linspace(0, 1, len(frame_times))
        design = make_first_level_design_matrix(frame_times, confounds=confounds)
        assert_allclose(design["confound_0"].to_numpy(), confounds)

    def test_confounds_bad_ndim_raises(self, frame_times):
        with pytest.raises(ValueError, match="1D or 2D"):
            make_first_level_design_matrix(frame_times, confounds=np.zeros((2, 3, 4)))

    def test_confounds_length_mismatch_raises(self, frame_times):
        with pytest.raises(ValueError, match="Incorrect specification of confounds"):
            make_first_level_design_matrix(
                frame_times, confounds=np.zeros((len(frame_times) - 1, 2))
            )

    def test_confounds_with_nan_raises(self, frame_times):
        confounds = np.zeros((len(frame_times), 1))
        confounds[0, 0] = np.nan
        with pytest.raises(ValueError, match="Confounds contain NaN values"):
            make_first_level_design_matrix(frame_times, confounds=confounds)

    def test_confound_names_count_mismatch_raises(self, frame_times):
        confounds = np.zeros((len(frame_times), 2))
        with pytest.raises(ValueError, match="Incorrect number of confound names"):
            make_first_level_design_matrix(
                frame_times,
                confounds=confounds,
                confound_names=["only_one"],
            )

    def test_duplicate_regressor_names_raises(self, frame_times):
        """Confound names colliding with auto-added regressors are rejected."""
        confounds = np.zeros((len(frame_times), 1))
        with pytest.raises(ValueError, match="unique names"):
            make_first_level_design_matrix(
                frame_times,
                confounds=confounds,
                confound_names=["constant"],
            )
