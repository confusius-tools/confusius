"""Tests for DVARS computation."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from scipy import stats as sp_stats

from confusius.qc import compute_dvars


def generate_ar1_signals(
    n_samples: int = 100,
    n_signals: int = 50,
    ar1_coef: float = 0.5,
    rng_seed: int = 42,
) -> xr.DataArray:
    """Generate synthetic AR(1) signals for testing."""
    rng = np.random.default_rng(rng_seed)
    data = np.zeros((n_samples, n_signals))
    data[0] = rng.standard_normal(n_signals)

    for t in range(1, n_samples):
        data[t] = ar1_coef * data[t - 1] + rng.standard_normal(n_signals)

    return xr.DataArray(
        data,
        dims=["time", "space"],
        coords={"time": np.arange(n_samples) * 0.1},
    )


def naive_dvars_nstd(signals: xr.DataArray) -> np.ndarray:
    """Naive reference implementation of non-standardized DVARS."""
    signals_np = signals.values
    signals_diff = np.diff(signals_np, axis=0)
    dvars_nstd = np.sqrt(np.mean(signals_diff**2, axis=1))
    return np.insert(dvars_nstd, 0, dvars_nstd.min())


def naive_ar1_yule_walker(signal: np.ndarray) -> float:
    """Naive reference AR(1) estimation using Yule-Walker."""
    x = signal - signal.mean()
    n = len(x)
    r0 = np.dot(x, x) / n
    r1 = np.dot(x[:-1], x[1:]) / n
    return r1 / r0 if r0 > 1e-15 else 0.0


def naive_standardized_dvars(signals: xr.DataArray) -> np.ndarray:
    """Naive reference implementation of standardized DVARS."""
    signals_np = signals.values
    signals_sd = sp_stats.iqr(signals_np, axis=0, scale="normal", interpolation="lower")
    nonzero_mask = signals_sd > 0
    signals_np = signals_np[:, nonzero_mask]
    signals_sd = signals_sd[nonzero_mask]

    dvars_nstd = naive_dvars_nstd(xr.DataArray(signals_np, dims=["time", "space"]))
    ar1 = np.array(
        [naive_ar1_yule_walker(signals_np[:, i]) for i in range(signals_np.shape[1])]
    )
    diff_sdhat = np.sqrt(2 * (1 - ar1)) * signals_sd

    return dvars_nstd / np.mean(diff_sdhat)


class TestReferenceImplementation:
    """Tests comparing against reference implementations."""

    def test_non_standardized_matches_naive(self):
        """Non-standardized DVARS must match naive reference implementation."""
        signals = generate_ar1_signals(n_samples=50, n_signals=30)

        expected = naive_dvars_nstd(signals)
        result = compute_dvars(signals, standardize=False, normalization_factor=None)

        assert_allclose(result, expected, rtol=1e-10)

    def test_standardized_matches_naive(self):
        """Standardized DVARS must match naive reference implementation."""
        signals = generate_ar1_signals(n_samples=100, n_signals=50, ar1_coef=0.5)

        expected = naive_standardized_dvars(signals)
        result = compute_dvars(
            signals,
            standardize=True,
            normalization_factor=None,
            remove_zero_variance=True,
        )

        assert_allclose(result, expected, rtol=1e-5)

    def test_single_signal_matches_naive(self):
        """Single signal must match naive reference implementation."""
        signals = generate_ar1_signals(n_samples=50, n_signals=1)

        expected_nstd = naive_dvars_nstd(signals)
        expected_stdz = naive_standardized_dvars(signals)

        result_nstd = compute_dvars(
            signals, standardize=False, normalization_factor=None
        )
        result_stdz = compute_dvars(
            signals, standardize=True, normalization_factor=None
        )

        assert_allclose(result_nstd, expected_nstd, rtol=1e-10)
        assert_allclose(result_stdz, expected_stdz, rtol=1e-5)

    def test_normalization_factor_scales_correctly(self):
        """Normalization factor must scale DVARS proportionally for non-standardized."""
        signals = generate_ar1_signals(n_samples=50, n_signals=30)

        # Compute without normalization
        dvars_no_norm = compute_dvars(
            signals, standardize=False, normalization_factor=None
        )

        # Compute with normalization factor 1000
        dvars_norm = compute_dvars(
            signals, standardize=False, normalization_factor=1000
        )

        # Non-standardized DVARS should scale linearly with the normalization factor
        # The scaling factor is (normalization_factor / |median|)
        expected_scale = 1000 / np.abs(np.median(signals))
        assert_allclose(dvars_norm, dvars_no_norm * expected_scale, rtol=1e-10)

    def test_negative_ar1_matches_naive(self):
        """Negative AR(1) coefficient must match naive reference implementation."""
        signals = generate_ar1_signals(n_samples=100, n_signals=50, ar1_coef=-0.3)

        expected_nstd = naive_dvars_nstd(signals)
        expected_stdz = naive_standardized_dvars(signals)

        result_nstd = compute_dvars(
            signals, standardize=False, normalization_factor=None
        )
        result_stdz = compute_dvars(
            signals, standardize=True, normalization_factor=None
        )

        assert_allclose(result_nstd, expected_nstd, rtol=1e-10)
        assert_allclose(result_stdz, expected_stdz, rtol=1e-5)


class TestZeroVarianceHandling:
    """Tests for zero-variance signal handling."""

    def test_zero_variance_removal_is_correct(self):
        """Zero-variance removal must produce same result as excluding those columns."""
        signals = generate_ar1_signals(n_samples=50, n_signals=30)

        # Add zero-variance columns
        zero_variance_data = np.concatenate(
            (signals.values, np.zeros((signals.shape[0], 10))), axis=1
        )
        zero_variance_signals = xr.DataArray(
            zero_variance_data,
            dims=["time", "space"],
            coords={"time": signals.coords["time"]},
        )

        # Compute with zero-variance removal
        result_with_removal = compute_dvars(
            zero_variance_signals,
            standardize=True,
            normalization_factor=None,
            remove_zero_variance=True,
            variance_tolerance=1e-10,
        )

        # Compute on original signals without zero-variance columns
        expected = compute_dvars(
            signals,
            standardize=True,
            normalization_factor=None,
            remove_zero_variance=True,
        )

        assert_allclose(result_with_removal, expected, rtol=1e-10)

    def test_all_zero_variance_raises(self):
        """All-zero variance signals must raise ValueError."""
        data = np.ones((50, 10)) * 1e-15
        signals = xr.DataArray(
            data,
            dims=["time", "space"],
            coords={"time": np.arange(50) * 0.1},
        )

        with pytest.raises(ValueError, match="All signals have variance"):
            compute_dvars(signals, remove_zero_variance=True, variance_tolerance=1e-10)


class TestInputValidation:
    """Tests for input validation."""

    def test_no_time_dimension_raises(self):
        """Input without time dimension must raise ValueError."""
        data = np.random.standard_normal((50, 10))
        signals = xr.DataArray(data, dims=["sample", "space"])

        with pytest.raises(ValueError, match="must have a 'time' dimension"):
            compute_dvars(signals)

    def test_single_timepoint_raises(self):
        """Input with single timepoint must raise ValueError."""
        data = np.random.standard_normal((1, 10))
        signals = xr.DataArray(
            data,
            dims=["time", "space"],
            coords={"time": [0.0]},
        )

        with pytest.raises(ValueError, match="more than 1 timepoint"):
            compute_dvars(signals)

    def test_3d_input_raises(self):
        """3D input (without time dimension) must raise ValueError."""
        data = np.random.standard_normal((10, 10, 10))
        signals = xr.DataArray(data, dims=["z", "y", "x"])

        with pytest.raises(ValueError, match="must have a 'time' dimension"):
            compute_dvars(signals)


class TestNDInput:
    """Tests for N-D (3D+t and beyond) input."""

    def test_nd_input_matches_flattened_2d(self):
        """Any spatial shape must produce same result as flattened 2D input."""
        # Generate 4D AR(1) signals
        rng = np.random.default_rng(42)
        n_time = 50
        shape = (n_time, 4, 6, 8)
        data = np.zeros(shape)
        data[0] = rng.standard_normal(shape[1:])

        ar1_coef = 0.5
        for t in range(1, n_time):
            data[t] = ar1_coef * data[t - 1] + rng.standard_normal(shape[1:])

        signals_4d = xr.DataArray(
            data,
            dims=["time", "z", "y", "x"],
            coords={"time": np.arange(n_time) * 0.1},
        )

        # Flatten to 2D
        signals_2d = signals_4d.stack(space=["z", "y", "x"])

        # Compute DVARS on both
        dvars_nd = compute_dvars(
            signals_4d,
            standardize=True,
            normalization_factor=None,
        )
        dvars_2d = compute_dvars(
            signals_2d,
            standardize=True,
            normalization_factor=None,
        )

        assert_allclose(dvars_nd, dvars_2d, rtol=1e-10)
