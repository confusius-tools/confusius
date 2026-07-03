"""Tests for the signal.clean pipeline."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.signal import (
    censor_samples,
    clean,
    filter_butterworth,
    interpolate_samples,
)


def test_clean_no_processing_returns_original(sample_timeseries):
    """Test clean returns input when no steps are requested."""
    signals = sample_timeseries()

    result = clean(
        signals,
        detrend_order=None,
        standardize_method=None,
        low_cutoff=None,
        high_cutoff=None,
    )

    assert_allclose(result.values, signals.values)
    assert result.dims == signals.dims
    assert_allclose(result.coords["time"].values, signals.coords["time"].values)


def test_clean_detrend_and_standardize(sample_timeseries):
    """Test clean detrends and standardizes signals."""
    signals = sample_timeseries()

    result = clean(
        signals,
        detrend_order=1,
        standardize_method="zscore",
        low_cutoff=None,
        high_cutoff=None,
    )

    mean_per_voxel = result.mean(dim="time")
    std_per_voxel = result.std(dim="time", ddof=1)

    assert_allclose(mean_per_voxel.values, 0.0, atol=1e-10)
    assert_allclose(std_per_voxel.values, 1.0, rtol=1e-10)


def test_clean_with_confounds_reduces_correlation(sample_timeseries):
    """Test confound regression reduces correlation with confounds."""
    signals = sample_timeseries(n_time=200, n_voxels=3)
    time = np.arange(signals.sizes["time"]) / 100.0
    confound = np.sin(2 * np.pi * time)
    weights = np.array([2.0, -1.0, 0.5])
    signals = xr.DataArray(
        signals.values + confound[:, None] * weights[None, :],
        dims=signals.dims,
        coords=signals.coords,
    )

    before = np.corrcoef(confound, signals.values[:, 0])[0, 1]

    confounds = xr.DataArray(
        confound, dims=["time"], coords={"time": signals.coords["time"]}
    )
    result = clean(
        signals,
        detrend_order=None,
        standardize_method=None,
        confounds=confounds,
    )

    after = np.corrcoef(confound, result.values[:, 0])[0, 1]

    assert abs(after) < abs(before) * 1e-2


def test_clean_scrub_censors_after_filter(sample_timeseries):
    """Test scrubbing interpolates then censors samples when filtering."""
    signals = sample_timeseries(n_time=100, sampling_rate=100.0)
    mask_values = np.ones(100, dtype=bool)
    mask_values[[10, 25, 60]] = False
    sample_mask = xr.DataArray(
        mask_values, dims=["time"], coords={"time": signals.coords["time"]}
    )

    result = clean(
        signals,
        detrend_order=None,
        standardize_method=None,
        high_cutoff=5.0,
        sample_mask=sample_mask,
    )

    assert result.sizes["time"] == np.sum(mask_values)


def test_clean_censors_first_without_filter_or_detrend(sample_timeseries):
    """Test censoring occurs immediately when no detrend/filter requested."""
    signals = sample_timeseries(n_time=100, sampling_rate=100.0)
    mask_values = np.ones(100, dtype=bool)
    mask_values[[10, 25, 60]] = False
    sample_mask = xr.DataArray(
        mask_values, dims=["time"], coords={"time": signals.coords["time"]}
    )

    result = clean(
        signals,
        detrend_order=None,
        standardize_method=None,
        sample_mask=sample_mask,
    )

    assert result.sizes["time"] == np.sum(mask_values)


def test_clean_filter_low_pass_matches_filter_butterworth(sample_timeseries):
    """Test low_pass matches high_cutoff argument."""
    signals = sample_timeseries(n_time=200, sampling_rate=100.0)

    expected = filter_butterworth(signals, high_cutoff=5.0)
    result = clean(
        signals,
        detrend_order=None,
        standardize_method=None,
        high_cutoff=5.0,
    )

    assert_allclose(result.values, expected.values)


def test_clean_filter_with_boundary_censoring_and_confounds_stays_finite(
    sample_timeseries,
):
    """Test boundary-censored samples do not poison filtered outputs."""
    signals = sample_timeseries(n_time=100, n_voxels=3, sampling_rate=100.0)
    confounds = xr.DataArray(
        np.column_stack(
            [
                np.sin(2 * np.pi * signals.coords["time"].values),
                np.cos(2 * np.pi * signals.coords["time"].values),
            ]
        ),
        dims=["time", "confound"],
        coords={"time": signals.coords["time"], "confound": [0, 1]},
    )
    mask_values = np.ones(signals.sizes["time"], dtype=bool)
    mask_values[[0, -1]] = False
    sample_mask = xr.DataArray(
        mask_values, dims=["time"], coords={"time": signals.coords["time"]}
    )

    result = clean(
        signals,
        detrend_order=None,
        standardize_method=None,
        high_cutoff=5.0,
        confounds=confounds,
        sample_mask=sample_mask,
    )

    assert result.sizes["time"] == np.sum(mask_values)
    assert np.all(np.isfinite(result.values))


def test_clean_interpolate_kwargs_match_manual_pipeline(sample_timeseries):
    """Test interpolate_kwargs are forwarded to pre-scrubbing interpolation."""
    signals = sample_timeseries(n_time=100, sampling_rate=100.0)
    mask_values = np.ones(signals.sizes["time"], dtype=bool)
    mask_values[[0, -1]] = False
    sample_mask = xr.DataArray(
        mask_values, dims=["time"], coords={"time": signals.coords["time"]}
    )

    interpolated = interpolate_samples(
        signals,
        sample_mask,
        fill_value="extrapolate",
    )
    expected = censor_samples(
        filter_butterworth(interpolated, high_cutoff=5.0), sample_mask
    )
    result = clean(
        signals,
        detrend_order=None,
        standardize_method=None,
        high_cutoff=5.0,
        sample_mask=sample_mask,
        interpolate_kwargs={"fill_value": "extrapolate"},
    )

    assert_allclose(result.values, expected.values)


def test_clean_noop_preserves_non_finite_when_ensure_finite_false(sample_timeseries):
    """Test ensure_finite=False leaves non-finite values unchanged."""
    signals = sample_timeseries()
    signals.values[0, 0] = np.nan
    signals.values[1, 1] = np.inf

    result = clean(signals, ensure_finite=False)

    assert np.isnan(result.values[0, 0])
    assert np.isinf(result.values[1, 1])


def test_clean_ensure_finite_matches_manual_interpolation(sample_timeseries):
    """Test ensure_finite=True matches manual time interpolation."""
    signals = sample_timeseries(n_time=200, n_voxels=3, sampling_rate=100.0)
    confounds = xr.DataArray(
        np.column_stack(
            [
                np.sin(2 * np.pi * signals.coords["time"].values),
                np.cos(2 * np.pi * signals.coords["time"].values),
            ]
        ),
        dims=["time", "confound"],
        coords={"time": signals.coords["time"], "confound": [0, 1]},
    )
    signals_with_non_finite = signals.copy()
    signals_with_non_finite.values[0, 0] = np.nan
    signals_with_non_finite.values[1, 1] = np.inf
    signals_with_non_finite.values[-1, 2] = np.nan
    confounds_with_non_finite = confounds.copy()
    confounds_with_non_finite.values[0, 0] = np.nan
    confounds_with_non_finite.values[-1, 1] = np.inf

    expected_signals = signals_with_non_finite.where(
        np.isfinite(signals_with_non_finite)
    )
    expected_signals = expected_signals.interpolate_na(dim="time", method="linear")
    expected_signals = expected_signals.ffill("time").bfill("time")
    expected_confounds = confounds_with_non_finite.where(
        np.isfinite(confounds_with_non_finite)
    )
    expected_confounds = expected_confounds.interpolate_na(dim="time", method="linear")
    expected_confounds = expected_confounds.ffill("time").bfill("time")
    expected = clean(
        expected_signals,
        detrend_order=1,
        standardize_method=None,
        high_cutoff=5.0,
        confounds=expected_confounds,
    )
    result = clean(
        signals_with_non_finite,
        detrend_order=1,
        standardize_method=None,
        high_cutoff=5.0,
        confounds=confounds_with_non_finite,
        ensure_finite=True,
    )

    assert_allclose(result.values, expected.values)


def test_clean_ensure_finite_raises_for_all_non_finite_series():
    """Test ensure_finite=True fails when a whole series has no finite samples."""
    signals = xr.DataArray(
        np.array(
            [
                [np.nan, 0.0],
                [np.nan, 1.0],
                [np.nan, 2.0],
            ]
        ),
        dims=["time", "space"],
        coords={"time": [0.0, 1.0, 2.0]},
    )

    with pytest.raises(
        ValueError,
        match="signals contains a series with no finite values along time",
    ):
        clean(signals, ensure_finite=True)
