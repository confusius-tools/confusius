"""Tests for sample censoring and interpolation functions."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.extract import extract_with_labels
from confusius.signal import censor_samples, interpolate_samples

# ===========================
# Fixtures
# ===========================


@pytest.fixture
def sample_mask_with_gaps(sample_timeseries):
    """Sample mask with several censored gaps."""
    signals = sample_timeseries(n_time=100)
    mask_values = np.ones(100, dtype=bool)
    # Censor frames: single (10), consecutive (25-27), and isolated (60, 85).
    mask_values[[10, 25, 26, 27, 60, 85]] = False
    return xr.DataArray(
        mask_values, dims=["time"], coords={"time": signals.coords["time"]}
    )


@pytest.fixture
def sample_mask_boundary(sample_timeseries):
    """Sample mask with censored boundary frames."""
    signals = sample_timeseries(n_time=100)
    mask_values = np.ones(100, dtype=bool)
    # Censor first and last frames.
    mask_values[[0, 99]] = False
    return xr.DataArray(
        mask_values, dims=["time"], coords={"time": signals.coords["time"]}
    )


# ===========================
# Tests for interpolate_samples
# ===========================


def test_interpolate_preserves_kept_samples(sample_timeseries, sample_mask_with_gaps):
    """Test that kept samples are unchanged after interpolation."""
    signals = sample_timeseries(n_time=100)
    result = interpolate_samples(signals, sample_mask_with_gaps, method="linear")

    mask_bool = sample_mask_with_gaps.values
    # Kept samples should be identical within numerical precision.
    assert_allclose(
        result.values[mask_bool, :],
        signals.values[mask_bool, :],
        rtol=1e-12,
        atol=1e-14,
    )


def test_interpolate_matches_xarray_interp(sample_timeseries, sample_mask_with_gaps):
    """Test interpolation matches xarray.DataArray.interp() directly."""
    signals = sample_timeseries(n_time=100)
    result = interpolate_samples(signals, sample_mask_with_gaps, method="linear")

    # Reference: manual xarray.interp call.
    mask_bool = sample_mask_with_gaps.values
    kept_signals = signals.isel(time=mask_bool)
    expected = kept_signals.interp(time=signals.coords["time"], method="linear")

    assert_allclose(result.values, expected.values, rtol=1e-14)


def test_interpolate_boundary_extrapolation(sample_timeseries, sample_mask_boundary):
    """Test that fill_value='extrapolate' fills boundary values."""
    signals = sample_timeseries(n_time=100)
    result = interpolate_samples(
        signals, sample_mask_boundary, method="linear", fill_value="extrapolate"
    )

    # Boundary samples should NOT be NaN (extrapolated).
    assert not np.any(np.isnan(result.values[0, :]))
    assert not np.any(np.isnan(result.values[99, :]))


def test_interpolate_boundary_no_extrapolation(sample_timeseries, sample_mask_boundary):
    """Test that fill_value=np.nan produces NaN at boundaries."""
    signals = sample_timeseries(n_time=100)
    result = interpolate_samples(
        signals, sample_mask_boundary, method="linear", fill_value=np.nan
    )

    # Boundary samples should be NaN (cannot interpolate without extrapolation).
    assert np.all(np.isnan(result.values[0, :]))
    assert np.all(np.isnan(result.values[99, :]))


def test_interpolate_all_censored_error(sample_timeseries):
    """Test error when all samples are censored."""
    signals = sample_timeseries(n_time=100)
    sample_mask = xr.DataArray(
        np.zeros(100, dtype=bool),
        dims=["time"],
        coords={"time": signals.coords["time"]},
    )

    with pytest.raises(ValueError, match="All samples are censored"):
        interpolate_samples(signals, sample_mask)


def test_interpolate_all_kept_warning(sample_timeseries):
    """Test warning when all samples are kept (no interpolation needed)."""
    signals = sample_timeseries(n_time=100)
    sample_mask = xr.DataArray(
        np.ones(100, dtype=bool),
        dims=["time"],
        coords={"time": signals.coords["time"]},
    )

    with pytest.warns(UserWarning, match="no interpolation was performed"):
        result = interpolate_samples(signals, sample_mask)

    # Should return unchanged data.
    assert_allclose(result.values, signals.values, rtol=1e-14)


def test_interpolate_missing_time_coords(sample_timeseries):
    """Test error when signals have no time coordinates."""
    signals = sample_timeseries(n_time=100).drop_vars("time")
    sample_mask = xr.DataArray(np.ones(100, dtype=bool), dims=["time"])

    with pytest.raises(ValueError, match="must have 'time' coordinates"):
        interpolate_samples(signals, sample_mask)


def test_interpolate_4d_data(sample_4d_volume):
    """Test interpolation on 4D (time, z, y, x) data."""
    n_time = sample_4d_volume.sizes["time"]
    mask_values = np.ones(n_time, dtype=bool)
    mask_values[[2, 5, 8]] = False
    sample_mask = xr.DataArray(
        mask_values,
        dims=["time"],
        coords={"time": sample_4d_volume.coords["time"]},
    )

    result = interpolate_samples(sample_4d_volume, sample_mask, method="linear")

    # Shape and dims preserved.
    assert result.shape == sample_4d_volume.shape
    assert result.dims == sample_4d_volume.dims

    # Kept samples identical within numerical precision.
    assert_allclose(
        result.values[mask_values, ...],
        sample_4d_volume.values[mask_values, ...],
        rtol=1e-12,
        atol=1e-14,
    )


def test_interpolate_accepts_time_match_with_unrelated_scalar_coord(sample_4d_volume):
    """Test time matching ignores unrelated scalar coordinates on signals."""
    mask_data = np.zeros((2, *sample_4d_volume.shape[1:]), dtype=int)
    mask_data[0, 0, :, :] = 1
    mask_data[1, 1, :, :] = 2
    labels = xr.DataArray(
        mask_data,
        dims=["mask", "z", "y", "x"],
        coords={
            "mask": ["VISp", "AUDp"],
            "z": sample_4d_volume.coords["z"],
            "y": sample_4d_volume.coords["y"],
            "x": sample_4d_volume.coords["x"],
        },
    )
    signals = extract_with_labels(sample_4d_volume, labels.isel(mask=0))
    mask_values = np.ones(signals.sizes["time"], dtype=bool)
    mask_values[3] = False
    sample_mask = xr.DataArray(
        mask_values,
        dims=["time"],
        coords={"time": sample_4d_volume.coords["time"]},
    )

    result = interpolate_samples(signals, sample_mask)

    assert result.shape == signals.shape
    assert result.dims == signals.dims


def test_interpolate_accepts_small_time_coordinate_drift(sample_timeseries):
    """Test interpolation accepts small numeric drift in time coordinates."""
    signals = sample_timeseries(n_time=100)
    mask_values = np.ones(100, dtype=bool)
    mask_values[10] = False
    sample_mask = xr.DataArray(
        mask_values,
        dims=["time"],
        coords={"time": signals.coords["time"].values + 1e-10},
    )

    result = interpolate_samples(signals, sample_mask)

    assert result.shape == signals.shape


def test_interpolate_dask(sample_timeseries, sample_mask_with_gaps):
    """Test interpolation works with Dask-backed arrays."""
    signals = sample_timeseries(n_time=100)
    dask_signals = signals.chunk({"time": -1, "space": 10})

    result = interpolate_samples(dask_signals, sample_mask_with_gaps, method="linear")

    # Should still be Dask-backed.
    assert isinstance(result.data, da.Array)

    # Compute and compare with eager version.
    eager_result = interpolate_samples(signals, sample_mask_with_gaps, method="linear")
    assert_allclose(result.compute().values, eager_result.values, rtol=1e-14)


# ===========================
# Tests for censor_samples
# ===========================


def test_censor_removes_correct_samples(sample_timeseries, sample_mask_with_gaps):
    """Test that censoring removes exactly the specified samples."""
    signals = sample_timeseries(n_time=100)
    result = censor_samples(signals, sample_mask_with_gaps)

    # Should remove 6 timepoints.
    mask_bool = sample_mask_with_gaps.values
    expected_n_kept = np.sum(mask_bool)
    assert result.sizes["time"] == expected_n_kept
    assert result.sizes["space"] == signals.sizes["space"]

    # Kept samples should match original.
    expected_data = signals.values[mask_bool, :]
    assert_allclose(result.values, expected_data, rtol=1e-14)

    # Time coordinates should be subsetted.
    expected_times = signals.coords["time"].values[mask_bool]
    assert_allclose(result.coords["time"].values, expected_times, rtol=1e-14)


def test_censor_4d_data(sample_4d_volume):
    """Test censoring on 4D (time, z, y, x) data."""
    n_time = sample_4d_volume.sizes["time"]
    mask_values = np.ones(n_time, dtype=bool)
    mask_values[[2, 5, 8]] = False
    sample_mask = xr.DataArray(
        mask_values,
        dims=["time"],
        coords={"time": sample_4d_volume.coords["time"]},
    )

    result = censor_samples(sample_4d_volume, sample_mask)

    # Shape correct.
    assert result.sizes["time"] == np.sum(mask_values)
    assert result.sizes["z"] == sample_4d_volume.sizes["z"]
    assert result.sizes["y"] == sample_4d_volume.sizes["y"]
    assert result.sizes["x"] == sample_4d_volume.sizes["x"]

    # Data correct.
    expected_data = sample_4d_volume.values[mask_values, ...]
    assert_allclose(result.values, expected_data, rtol=1e-14)


def test_censor_accepts_time_match_with_unrelated_scalar_coord(sample_4d_volume):
    """Test censoring ignores unrelated scalar coordinates on signals."""
    mask_data = np.zeros((2, *sample_4d_volume.shape[1:]), dtype=int)
    mask_data[0, 0, :, :] = 1
    mask_data[1, 1, :, :] = 2
    labels = xr.DataArray(
        mask_data,
        dims=["mask", "z", "y", "x"],
        coords={
            "mask": ["VISp", "AUDp"],
            "z": sample_4d_volume.coords["z"],
            "y": sample_4d_volume.coords["y"],
            "x": sample_4d_volume.coords["x"],
        },
    )
    signals = extract_with_labels(sample_4d_volume, labels.isel(mask=0))
    mask_values = np.ones(signals.sizes["time"], dtype=bool)
    mask_values[[2, 5, 8]] = False
    sample_mask = xr.DataArray(
        mask_values,
        dims=["time"],
        coords={"time": sample_4d_volume.coords["time"]},
    )

    result = censor_samples(signals, sample_mask)

    assert_allclose(result.values, signals.values[mask_values], rtol=1e-14)


def test_censor_accepts_small_time_coordinate_drift(sample_timeseries):
    """Test censoring accepts small numeric drift in time coordinates."""
    signals = sample_timeseries(n_time=100)
    mask_values = np.ones(100, dtype=bool)
    mask_values[[10, 25, 60]] = False
    sample_mask = xr.DataArray(
        mask_values,
        dims=["time"],
        coords={"time": signals.coords["time"].values + 1e-10},
    )

    result = censor_samples(signals, sample_mask)

    assert result.sizes["time"] == np.sum(mask_values)


def test_censor_all_censored_error(sample_timeseries):
    """Test error when all samples are censored."""
    signals = sample_timeseries(n_time=100)
    sample_mask = xr.DataArray(
        np.zeros(100, dtype=bool),
        dims=["time"],
        coords={"time": signals.coords["time"]},
    )

    with pytest.raises(ValueError, match="All samples are censored"):
        censor_samples(signals, sample_mask)


def test_censor_all_kept_warning(sample_timeseries):
    """Test warning when all samples are kept (no censoring)."""
    signals = sample_timeseries(n_time=100)
    sample_mask = xr.DataArray(
        np.ones(100, dtype=bool),
        dims=["time"],
        coords={"time": signals.coords["time"]},
    )

    with pytest.warns(UserWarning, match="no censoring was performed"):
        result = censor_samples(signals, sample_mask)

    # Should return same data.
    assert_allclose(result.values, signals.values, rtol=1e-14)


def test_censor_dask(sample_timeseries, sample_mask_with_gaps):
    """Test censoring works with Dask-backed arrays."""
    signals = sample_timeseries(n_time=100)
    dask_signals = signals.chunk({"time": 20, "space": 10})

    result = censor_samples(dask_signals, sample_mask_with_gaps)

    # Should still be Dask-backed.
    assert isinstance(result.data, da.Array)

    # Compute and compare with eager version.
    eager_result = censor_samples(signals, sample_mask_with_gaps)
    assert_allclose(result.compute().values, eager_result.values, rtol=1e-14)


def test_censor_missing_time_dimension(sample_timeseries):
    """Test error when signals have no time dimension."""
    signals = sample_timeseries(n_time=100).rename({"time": "samples"})
    sample_mask = xr.DataArray(
        np.ones(100, dtype=bool), dims=["time"], coords={"time": np.arange(100) / 100}
    )

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        censor_samples(signals, sample_mask)


# ===========================
# Integration Tests
# ===========================


def test_pre_scrubbing_workflow(sample_timeseries, sample_mask_with_gaps):
    """Test complete pre-scrubbing workflow: interpolate → filter → censor."""
    from confusius.signal import filter_butterworth

    signals = sample_timeseries(n_time=100, sampling_rate=100)

    # 1. Interpolate censored samples.
    interpolated = interpolate_samples(signals, sample_mask_with_gaps, method="linear")

    # 2. Apply temporal filter.
    filtered = filter_butterworth(interpolated, high_cutoff=5.0)

    # 3. Remove censored samples.
    cleaned = censor_samples(filtered, sample_mask_with_gaps)

    # Result should have reduced time dimension.
    expected_n_kept = np.sum(sample_mask_with_gaps.values)
    assert cleaned.sizes["time"] == expected_n_kept
    assert cleaned.sizes["space"] == signals.sizes["space"]


def test_interpolate_and_censor_roundtrip(sample_timeseries, sample_mask_with_gaps):
    """Test that kept samples survive interpolate → censor roundtrip."""
    signals = sample_timeseries(n_time=100)

    # Interpolate then censor.
    interpolated = interpolate_samples(signals, sample_mask_with_gaps, method="linear")
    censored = censor_samples(interpolated, sample_mask_with_gaps)

    # Kept samples should match original (within numerical precision).
    mask_bool = sample_mask_with_gaps.values
    original_kept = signals.values[mask_bool, :]
    assert_allclose(censored.values, original_kept, rtol=1e-12, atol=1e-14)
