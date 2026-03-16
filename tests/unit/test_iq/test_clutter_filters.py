"""Unit tests for clutter filtering functions."""

import numpy as np
import pytest
import scipy.signal as sp_signal
from numpy.testing import assert_allclose, assert_array_equal

from confusius.iq.clutter_filters import (
    clutter_filter_butterworth,
    clutter_filter_sosfiltfilt,
    clutter_filter_svd_from_cumulative_energy,
    clutter_filter_svd_from_energy,
    clutter_filter_svd_from_indices,
)


class TestClutterFilterSvdFromIndices:
    """Tests for clutter_filter_svd_from_indices function."""

    def test_no_cutoffs_returns_unchanged(self, sample_iq_block_4d):
        """Passing no cutoffs returns block unchanged."""
        result = clutter_filter_svd_from_indices(sample_iq_block_4d)
        assert_array_equal(result, sample_iq_block_4d)

    def test_full_range_cutoffs_returns_unchanged(self, sample_iq_block_4d):
        """Cutoffs spanning all components returns block unchanged."""
        time, z, y, x = sample_iq_block_4d.shape
        max_components = min(time, z * y * x)
        result = clutter_filter_svd_from_indices(
            sample_iq_block_4d, low_cutoff=0, high_cutoff=max_components
        )
        assert_array_equal(result, sample_iq_block_4d)

    @pytest.mark.parametrize(
        ("test_case", "kwargs", "expected_error", "match"),
        [
            (
                "non_integer_cutoffs",
                {"low_cutoff": 0.5, "high_cutoff": 5},
                ValueError,
                "must be integers",
            ),
            (
                "low_cutoff_greater_than_high",
                {"low_cutoff": 5, "high_cutoff": 3},
                ValueError,
                "Cutoffs must satisfy",
            ),
            (
                "negative_cutoff",
                {"low_cutoff": -1, "high_cutoff": 5},
                ValueError,
                "Cutoffs must satisfy",
            ),
        ],
    )
    def test_cutoff_validation_errors(
        self, sample_iq_block_4d, test_case, kwargs, expected_error, match
    ):
        """Invalid cutoff values raise appropriate errors."""
        with pytest.raises(expected_error, match=match):
            clutter_filter_svd_from_indices(sample_iq_block_4d, **kwargs)

    def test_non_4d_block_raises(self, sample_iq_block_4d):
        """Non-4D block raises ValueError."""
        with pytest.raises(ValueError, match="must be 4D"):
            clutter_filter_svd_from_indices(sample_iq_block_4d[..., 0])

    def test_high_cutoff_exceeds_max_components_raises(self, sample_iq_block_4d):
        """high_cutoff > max_components raises ValueError."""
        time, z, y, x = sample_iq_block_4d.shape
        max_components = min(time, z * y * x)
        with pytest.raises(ValueError, match="Cutoffs must satisfy"):
            clutter_filter_svd_from_indices(
                sample_iq_block_4d, high_cutoff=max_components + 1
            )

    def test_empty_mask_returns_unchanged(self, sample_iq_block_4d):
        """Empty mask (all False) returns block unchanged."""
        _, z, y, x = sample_iq_block_4d.shape
        empty_mask = np.zeros((z, y, x), dtype=bool)

        result = clutter_filter_svd_from_indices(
            sample_iq_block_4d, mask=empty_mask, low_cutoff=1, high_cutoff=5
        )
        assert_array_equal(result, sample_iq_block_4d)

    def test_mask_shape_mismatch_raises(self, sample_iq_block_4d):
        """Mask with wrong shape raises ValueError."""
        wrong_mask = np.ones((2, 2, 2), dtype=bool)
        with pytest.raises(ValueError, match="doesn't match spatial dimensions"):
            clutter_filter_svd_from_indices(
                sample_iq_block_4d, mask=wrong_mask, low_cutoff=1, high_cutoff=5
            )

    def test_matches_reference_svd_implementation(
        self, sample_iq_block_4d, spatial_mask
    ):
        """Result matches reference implementation using full SVD."""
        low_cutoff, high_cutoff = 2, 6
        time, z, y, x = sample_iq_block_4d.shape

        signals = sample_iq_block_4d.reshape(time, -1)
        masked_signals = signals[:, spatial_mask.ravel()].astype(np.cdouble)

        u, _, _ = np.linalg.svd(masked_signals, full_matrices=False)

        # Clutter vectors are outside [low_cutoff, high_cutoff).
        clutter_vectors = np.concatenate(
            [u[:, :low_cutoff], u[:, high_cutoff:]], axis=1
        )

        filtered_signals = (
            signals - clutter_vectors @ clutter_vectors.conj().T @ signals
        )
        expected = filtered_signals.reshape(sample_iq_block_4d.shape)

        result = clutter_filter_svd_from_indices(
            sample_iq_block_4d,
            mask=spatial_mask,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
        )

        assert_allclose(result, expected, rtol=1e-5, atol=1e-10)


class TestClutterFilterSvdFromEnergy:
    """Tests for clutter_filter_svd_from_energy function."""

    def test_no_cutoffs_returns_unchanged(self, sample_iq_block_4d):
        """No cutoffs (0, inf) returns block unchanged."""
        result = clutter_filter_svd_from_energy(sample_iq_block_4d)
        assert_array_equal(result, sample_iq_block_4d)

    def test_non_4d_block_raises(self, sample_iq_block_4d):
        """Non-4D block raises ValueError."""
        with pytest.raises(ValueError, match="must be 4D"):
            clutter_filter_svd_from_energy(sample_iq_block_4d[..., 0])

    def test_empty_mask_returns_unchanged(self, sample_iq_block_4d):
        """Empty mask returns block unchanged."""
        _, z, y, x = sample_iq_block_4d.shape
        empty_mask = np.zeros((z, y, x), dtype=bool)

        result = clutter_filter_svd_from_energy(
            sample_iq_block_4d, mask=empty_mask, low_cutoff=1e5, high_cutoff=1e10
        )
        assert_array_equal(result, sample_iq_block_4d)

    def test_negative_cutoff_raises(self, sample_iq_block_4d):
        """Negative cutoff raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            clutter_filter_svd_from_energy(sample_iq_block_4d, low_cutoff=-1)

    def test_negative_high_cutoff_raises(self, sample_iq_block_4d):
        """Negative high_cutoff raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            clutter_filter_svd_from_energy(sample_iq_block_4d, high_cutoff=-1)

    def test_low_cutoff_greater_than_high_raises(self, sample_iq_block_4d):
        """low_cutoff >= high_cutoff raises ValueError."""
        with pytest.raises(ValueError, match="must be lower than"):
            clutter_filter_svd_from_energy(
                sample_iq_block_4d, low_cutoff=100, high_cutoff=50
            )

    def test_matches_reference_svd_implementation(
        self, sample_iq_block_4d, spatial_mask
    ):
        """Result matches reference implementation using full SVD."""
        time, z, y, x = sample_iq_block_4d.shape

        signals = sample_iq_block_4d.reshape(time, -1)
        masked_signals = signals[:, spatial_mask.ravel()].astype(np.cdouble)

        u, s, _ = np.linalg.svd(masked_signals, full_matrices=False)

        # Set cutoffs based on actual singular values (energies are s**2 since we're
        # using the SVD and not the eigendecomposition).
        low_cutoff = s[-1] ** 2 + 0.001  # Filter lowest energy.
        high_cutoff = s[0] ** 2 - 0.001  # Filter highest energy.

        energies = s**2
        clutter_mask = np.logical_or(energies < low_cutoff, energies > high_cutoff)
        clutter_vectors = u[:, clutter_mask]
        filtered_signals = (
            signals - clutter_vectors @ clutter_vectors.conj().T @ signals
        )
        expected = filtered_signals.reshape(sample_iq_block_4d.shape)

        result = clutter_filter_svd_from_energy(
            sample_iq_block_4d,
            mask=spatial_mask,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
        )

        assert_allclose(result, expected, rtol=1e-5, atol=1e-10)


class TestClutterFilterSvdFromCumulativeEnergy:
    """Tests for clutter_filter_svd_from_cumulative_energy function."""

    def test_no_cutoffs_returns_unchanged(self, sample_iq_block_4d):
        """No cutoffs returns block unchanged."""
        result = clutter_filter_svd_from_cumulative_energy(sample_iq_block_4d)
        assert_array_equal(result, sample_iq_block_4d)

    def test_non_4d_block_raises(self, sample_iq_block_4d):
        """Non-4D block raises ValueError."""
        with pytest.raises(ValueError, match="must be 4D"):
            clutter_filter_svd_from_cumulative_energy(sample_iq_block_4d[..., 0])

    def test_empty_mask_returns_unchanged(self, sample_iq_block_4d):
        """Empty mask returns block unchanged."""
        _, z, y, x = sample_iq_block_4d.shape
        empty_mask = np.zeros((z, y, x), dtype=bool)

        result = clutter_filter_svd_from_cumulative_energy(
            sample_iq_block_4d, mask=empty_mask, low_cutoff=1e5
        )
        assert_array_equal(result, sample_iq_block_4d)

    def test_negative_cutoff_raises(self, sample_iq_block_4d):
        """Negative cutoff raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            clutter_filter_svd_from_cumulative_energy(sample_iq_block_4d, low_cutoff=-1)

    def test_matches_reference_svd_implementation(
        self, sample_iq_block_4d, spatial_mask
    ):
        """Result matches reference implementation using full SVD."""
        time, z, y, x = sample_iq_block_4d.shape

        signals = sample_iq_block_4d.reshape(time, -1)
        masked_signals = signals[:, spatial_mask.ravel()].astype(np.cdouble)

        u, s, _ = np.linalg.svd(masked_signals, full_matrices=False)

        # SVD returns singular values in descending order; reverse to ascending so that
        # cumsum accumulates from noise to tissue, matching the filter's convention.
        s_asc = s[::-1]
        u_asc = u[:, ::-1]
        cumsum_energy = np.cumsum(s_asc**2)

        low_cutoff = cumsum_energy[-1] * 0.3

        clutter_mask = cumsum_energy < low_cutoff
        clutter_vectors = u_asc[:, clutter_mask]
        filtered_signals = (
            signals - clutter_vectors @ clutter_vectors.conj().T @ signals
        )
        expected = filtered_signals.reshape(sample_iq_block_4d.shape)

        result = clutter_filter_svd_from_cumulative_energy(
            sample_iq_block_4d, mask=spatial_mask, low_cutoff=low_cutoff
        )

        assert_allclose(result, expected, rtol=1e-5, atol=1e-10)


class TestClutterFilterSosfiltfilt:
    """Tests for clutter_filter_sosfiltfilt function."""

    def test_non_4d_block_raises(self, sample_iq_block_4d):
        """Non-4D block raises ValueError."""
        sos = sp_signal.butter(4, 0.1, btype="high", fs=1.0, output="sos")
        with pytest.raises(ValueError, match="must be 4D"):
            clutter_filter_sosfiltfilt(sample_iq_block_4d[..., 0], sos)

    def test_matches_scipy_sosfiltfilt(self, sample_iq_block_4d_long):
        """Result matches direct scipy.signal.sosfiltfilt call."""
        sos = sp_signal.butter(4, 0.1, btype="high", fs=1.0, output="sos")

        result = clutter_filter_sosfiltfilt(sample_iq_block_4d_long, sos)
        expected = sp_signal.sosfiltfilt(sos, sample_iq_block_4d_long, axis=0)

        assert_allclose(result, expected)


class TestClutterFilterButterworth:
    """Tests for clutter_filter_butterworth function."""

    def test_non_4d_block_raises(self, sample_iq_block_4d):
        """Non-4D block raises ValueError."""
        with pytest.raises(ValueError, match="must be 4D"):
            clutter_filter_butterworth(
                sample_iq_block_4d[..., 0], fs=100.0, low_cutoff=5.0
            )

    def test_no_cutoffs_returns_unchanged(self, sample_iq_block_4d_long):
        """No cutoffs returns block unchanged."""
        result = clutter_filter_butterworth(sample_iq_block_4d_long, fs=100.0)
        assert_array_equal(result, sample_iq_block_4d_long)

    def test_high_cutoff_less_than_low_raises(self, sample_iq_block_4d_long):
        """high_cutoff <= low_cutoff raises ValueError."""
        with pytest.raises(ValueError, match="must be greater than"):
            clutter_filter_butterworth(
                sample_iq_block_4d_long, fs=100.0, low_cutoff=20.0, high_cutoff=5.0
            )

    def test_frequency_at_nyquist_raises(self, sample_iq_block_4d_long):
        """Frequency at Nyquist raises ValueError."""
        with pytest.raises(ValueError, match="must be in range"):
            clutter_filter_butterworth(
                sample_iq_block_4d_long, fs=100.0, low_cutoff=50.0
            )

    def test_frequency_at_zero_raises(self, sample_iq_block_4d_long):
        """Frequency at zero raises ValueError."""
        with pytest.raises(ValueError, match="must be in range"):
            clutter_filter_butterworth(
                sample_iq_block_4d_long, fs=100.0, low_cutoff=0.0
            )

    def test_negative_frequency_raises(self, sample_iq_block_4d_long):
        """Negative frequency raises ValueError."""
        with pytest.raises(ValueError, match="must be in range"):
            clutter_filter_butterworth(
                sample_iq_block_4d_long, fs=100.0, low_cutoff=-5.0
            )

    def test_matches_scipy_reference_high_pass(self, sample_iq_block_4d_long):
        """High-pass filter matches direct scipy implementation."""
        fs, low_cutoff, order = 100.0, 5.0, 4

        sos = sp_signal.butter(order, low_cutoff, btype="high", fs=fs, output="sos")
        expected = sp_signal.sosfiltfilt(sos, sample_iq_block_4d_long, axis=0)

        result = clutter_filter_butterworth(
            sample_iq_block_4d_long, fs=fs, low_cutoff=low_cutoff, order=order
        )

        assert_allclose(result, expected)

    def test_matches_scipy_reference_low_pass(self, sample_iq_block_4d_long):
        """Low-pass filter matches direct scipy implementation."""
        fs, high_cutoff, order = 100.0, 20.0, 4

        sos = sp_signal.butter(order, high_cutoff, btype="low", fs=fs, output="sos")
        expected = sp_signal.sosfiltfilt(sos, sample_iq_block_4d_long, axis=0)

        result = clutter_filter_butterworth(
            sample_iq_block_4d_long, fs=fs, high_cutoff=high_cutoff, order=order
        )

        assert_allclose(result, expected)

    def test_matches_scipy_reference_band_pass(self, sample_iq_block_4d_long):
        """Band-pass filter matches direct scipy implementation."""
        fs, low_cutoff, high_cutoff, order = 100.0, 5.0, 20.0, 4

        sos = sp_signal.butter(
            order, [low_cutoff, high_cutoff], btype="band", fs=fs, output="sos"
        )
        expected = sp_signal.sosfiltfilt(sos, sample_iq_block_4d_long, axis=0)

        result = clutter_filter_butterworth(
            sample_iq_block_4d_long,
            fs=fs,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            order=order,
        )

        assert_allclose(result, expected)
