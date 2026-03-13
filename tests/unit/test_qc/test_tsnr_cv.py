"""Tests for tSNR and CV computation."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.qc import compute_cv, compute_tsnr


class TestReferenceImplementation:
    """Tests comparing against naive reference implementations."""

    def test_tsnr_matches_naive(self, sample_timeseries):
        """tSNR must equal voxel-wise mean divided by standard deviation."""
        signals = sample_timeseries()

        expected = signals.mean("time") / signals.std("time")
        result = compute_tsnr(signals)

        assert_allclose(result.values, expected.values, rtol=1e-10)

    def test_cv_matches_naive(self, sample_timeseries):
        """CV must equal voxel-wise standard deviation divided by mean."""
        signals = sample_timeseries()

        expected = signals.std("time") / signals.mean("time")
        result = compute_cv(signals)

        assert_allclose(result.values, expected.values, rtol=1e-10)


class TestEdgeCases:
    """Tests for edge cases with degenerate signals."""

    def test_constant_signal_tsnr_is_inf(self):
        """Constant signal must yield infinite tSNR (zero standard deviation)."""
        data = np.ones((50, 10)) * 5.0
        signals = xr.DataArray(
            data,
            dims=["time", "space"],
            coords={"time": np.arange(50) * 0.1},
        )

        result = compute_tsnr(signals)

        assert np.all(np.isinf(result.values))

    def test_constant_signal_cv_is_zero(self):
        """Constant signal must yield zero CV (zero standard deviation)."""
        data = np.ones((50, 10)) * 5.0
        signals = xr.DataArray(
            data,
            dims=["time", "space"],
            coords={"time": np.arange(50) * 0.1},
        )

        result = compute_cv(signals)

        assert_allclose(result.values, np.zeros(10), atol=1e-15)

    def test_zero_mean_cv_is_inf(self):
        """Zero-mean signal with nonzero variance must yield infinite CV."""
        # Use [1, -1] to guarantee exact zero mean in floating-point arithmetic.
        data = np.vstack([np.ones(10), -np.ones(10)])
        signals = xr.DataArray(
            data,
            dims=["time", "space"],
            coords={"time": np.arange(2) * 0.1},
        )

        result = compute_cv(signals)

        assert np.all(np.isinf(result.values))


class TestInputValidation:
    """Tests for input validation."""

    @pytest.mark.parametrize("func", [compute_tsnr, compute_cv])
    def test_no_time_dimension_raises(self, func):
        """Input without time dimension must raise ValueError."""
        signals = xr.DataArray(
            np.random.standard_normal((50, 10)),
            dims=["sample", "space"],
        )

        with pytest.raises(ValueError, match="must have a 'time' dimension"):
            func(signals)

    @pytest.mark.parametrize("func", [compute_tsnr, compute_cv])
    def test_single_timepoint_raises(self, func):
        """Input with a single timepoint must raise ValueError."""
        signals = xr.DataArray(
            np.random.standard_normal((1, 10)),
            dims=["time", "space"],
            coords={"time": [0.0]},
        )

        with pytest.raises(ValueError, match="more than 1 timepoint"):
            func(signals)
