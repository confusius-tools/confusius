"""Tests for standardization functions."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.signal import standardize


@pytest.fixture
def random_signals_dask(rng):
    """Random 2D signals (time, voxels) with Dask backend."""
    data = rng.normal(loc=10, scale=2, size=(100, 50))
    return xr.DataArray(
        da.from_array(data, chunks=(50, 25)),
        dims=["time", "voxels"],
        coords={
            "time": np.arange(100) * 0.002,
            "voxels": np.arange(50),
        },
    )


def test_standardize_zscore(sample_timeseries):
    """Test z-score standardization produces mean=0, std=1."""
    random_signals = sample_timeseries(n_time=100, n_voxels=50)
    result = standardize(random_signals, method="zscore")

    # Check shape and coordinates preserved.
    assert result.dims == random_signals.dims
    assert result.shape == random_signals.shape
    assert_allclose(result.coords["time"].values, random_signals.coords["time"].values)
    assert_allclose(
        result.coords["voxels"].values, random_signals.coords["voxels"].values
    )

    # Check mean ≈ 0 and std ≈ 1 per voxel.
    mean_per_voxel = result.mean(dim="time")
    std_per_voxel = result.std(dim="time", ddof=1)

    assert_allclose(mean_per_voxel.values, 0.0, atol=1e-10)
    assert_allclose(std_per_voxel.values, 1.0, rtol=1e-10)


def test_standardize_psc(sample_timeseries):
    """Test percent signal change standardization."""
    random_signals = sample_timeseries(n_time=100, n_voxels=50)
    result = standardize(random_signals, method="psc")

    # Check shape and coordinates preserved.
    assert result.dims == random_signals.dims
    assert result.shape == random_signals.shape

    # Check PSC formula: (x - mean) / |mean| * 100.
    mean = random_signals.mean(dim="time")
    expected = (random_signals - mean) / abs(mean) * 100

    assert_allclose(result.values, expected.values, rtol=1e-10)

    # Check that mean of result ≈ 0 (allow for floating-point precision).
    result_mean = result.mean(dim="time")
    assert_allclose(result_mean.values, 0.0, atol=1e-7)


def test_standardize_invalid_method(sample_timeseries):
    """Test error raised for invalid method."""
    random_signals = sample_timeseries(n_time=100, n_voxels=50)
    with pytest.raises(ValueError, match="method must be"):
        standardize(random_signals, method="invalid")  # type: ignore


def test_standardize_no_time_dimension():
    """Test error raised when signals have no time dimension."""
    signals = xr.DataArray(
        np.random.randn(50, 10, 20),
        dims=["z", "y", "x"],
    )

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        standardize(signals, method="zscore")


def test_standardize_psc_zero_mean():
    """Test NaN-setting for PSC with zero mean voxels."""
    # Create signals with one voxel having zero mean.
    signals = xr.DataArray(
        np.array([[1.0, 1.0], [-1.0, 2.0], [0.0, 3.0]]),
        dims=["time", "voxels"],
    )
    # Voxel 0 has mean = (1 + -1 + 0) / 3 = 0.

    result = standardize(signals, method="psc")

    # Voxel 0 should be set to NaN.
    assert np.all(np.isnan(result.values[:, 0]))
    # Voxel 1 should be computed normally.
    mean_1 = 2.0
    expected_1 = (np.array([1.0, 2.0, 3.0]) - mean_1) / abs(mean_1) * 100
    assert_allclose(result.values[:, 1], expected_1)


def test_standardize_psc_near_zero_mean():
    """Test NaN-setting for PSC with near-zero mean voxels."""
    # Create signals with one voxel having near-zero mean.
    eps = np.finfo(np.float64).eps
    signals = xr.DataArray(
        np.array([[eps / 2, 1], [eps / 2, 2], [eps / 2, 3]]),
        dims=["time", "voxels"],
    )

    result = standardize(signals, method="psc")

    # Voxel 0 should be set to NaN.
    assert np.all(np.isnan(result.values[:, 0]))


def test_standardize_dask_compatibility(random_signals_dask):
    """Test standardization works with Dask-backed arrays."""
    # Should work without computing.
    result = standardize(random_signals_dask, method="zscore")

    # Result should still be Dask-backed.
    assert isinstance(result.data, da.Array)

    # Compute and check values.
    result_computed = result.compute()
    mean_per_voxel = result_computed.mean(dim="time")
    std_per_voxel = result_computed.std(dim="time", ddof=1)

    assert_allclose(mean_per_voxel.values, 0.0, atol=1e-10)
    assert_allclose(std_per_voxel.values, 1.0, rtol=1e-10)


def test_standardize_default_method(sample_timeseries):
    """Test default method is zscore."""
    signals = sample_timeseries()
    result = standardize(signals)

    # Should be same as explicitly passing method='zscore'.
    expected = standardize(signals, method="zscore")
    assert_allclose(result.values, expected.values)


def test_standardize_single_timepoint():
    """Test error raised for single timepoint."""
    signals = xr.DataArray(
        np.array([[1.0, 2.0, 3.0]]),
        dims=["time", "voxels"],
    )

    with pytest.raises(ValueError, match="more than 1 timepoint"):
        standardize(signals, method="zscore")


def test_standardize_zscore_zero_variance():
    """Test NaN-setting for zero-variance voxels."""
    # Create signals where voxel 0 has zero variance.
    signals = xr.DataArray(
        np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]),
        dims=["time", "voxels"],
    )
    # Voxel 0 is constant (zero variance).

    result = standardize(signals, method="zscore")

    # Voxel 0 should be set to NaN.
    assert np.all(np.isnan(result.values[:, 0]))
    # Voxel 1 should be computed normally.
    mean_1 = 2.0
    std_1 = np.std([1.0, 2.0, 3.0], ddof=1)
    expected_1 = (np.array([1.0, 2.0, 3.0]) - mean_1) / std_1
    assert_allclose(result.values[:, 1], expected_1)


def test_standardize_zscore_psc_correlation(sample_timeseries):
    """Test that zscore and psc are perfectly correlated (from nilearn)."""
    signals = sample_timeseries()
    z = standardize(signals, method="zscore")
    psc = standardize(signals, method="psc")

    # For each voxel, correlation between zscore and psc should be 1.
    for i in range(signals.sizes["voxels"]):
        corr = np.corrcoef(z.values[:, i], psc.values[:, i])[0, 1]
        assert_allclose(corr, 1.0, rtol=1e-10)


def test_standardize_4d_imaging_data(sample_4d_volume):
    """Test that standardize works on 4D imaging data (time, z, y, x)."""
    result = standardize(sample_4d_volume, method="zscore")

    # Check shape and dimensions preserved.
    assert result.dims == sample_4d_volume.dims
    assert result.shape == sample_4d_volume.shape

    # Check that each voxel (z, y, x) has mean≈0, std≈1 across time.
    mean_per_voxel = result.mean(dim="time")
    std_per_voxel = result.std(dim="time", ddof=1)

    assert_allclose(mean_per_voxel.values, 0.0, atol=1e-10)
    assert_allclose(std_per_voxel.values, 1.0, rtol=1e-10)

    # Check coordinates preserved.
    assert_allclose(
        result.coords["time"].values, sample_4d_volume.coords["time"].values
    )
    assert_allclose(result.coords["z"].values, sample_4d_volume.coords["z"].values)
    assert_allclose(result.coords["y"].values, sample_4d_volume.coords["y"].values)
    assert_allclose(result.coords["x"].values, sample_4d_volume.coords["x"].values)


def test_standardize_sets_units(sample_timeseries):
    """Test that standardize sets the units attribute on the result."""
    signals = sample_timeseries()

    result_zscore = standardize(signals, method="zscore")
    assert result_zscore.attrs["units"] == "z-score"

    result_psc = standardize(signals, method="psc")
    assert result_psc.attrs["units"] == "%"
