"""Tests for detrending functions."""

import dask.array as da
import numpy as np
import pytest
import scipy.signal
import xarray as xr
from numpy.testing import assert_allclose

from confusius.signal import detrend


def _naive_polynomial_detrend(data, order, axis=0):
    """Naive polynomial detrending for reference testing."""
    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    n_timepoints = data.shape[0]
    time_vals = np.arange(n_timepoints)

    if data.ndim > 1:
        original_shape = data.shape
        data_2d = data.reshape(n_timepoints, -1)
        result = np.zeros_like(data_2d)

        for i in range(data_2d.shape[1]):
            poly_coeffs = np.polyfit(time_vals, data_2d[:, i], order)
            poly_trend = np.polyval(poly_coeffs, time_vals)
            result[:, i] = data_2d[:, i] - poly_trend

        result = result.reshape(original_shape)
    else:
        poly_coeffs = np.polyfit(time_vals, data, order)
        poly_trend = np.polyval(poly_coeffs, time_vals)
        result = data - poly_trend

    if axis != 0:
        result = np.moveaxis(result, 0, axis)

    return result


@pytest.fixture
def signals_with_linear_trend(rng):
    """Signals with linear trend."""
    n_time = 100
    n_voxels = 50

    # Create noise.
    noise = rng.normal(loc=0, scale=0.5, size=(n_time, n_voxels))

    # Add linear trend.
    time = np.arange(n_time)
    trend = time[:, np.newaxis] * 2.0  # Strong linear trend.

    return xr.DataArray(
        trend + noise,
        dims=["time", "voxels"],
        coords={
            "time": np.arange(n_time) * 0.002,
            "voxels": np.arange(n_voxels),
        },
    )


@pytest.fixture
def signals_with_quadratic_trend(rng):
    """Signals with quadratic trend."""
    n_time = 100
    n_voxels = 50

    # Create noise.
    noise = rng.normal(loc=0, scale=0.5, size=(n_time, n_voxels))

    # Add quadratic trend.
    time = np.arange(n_time)
    trend = (time[:, np.newaxis] ** 2) * 0.05  # Quadratic trend.

    return xr.DataArray(
        trend + noise,
        dims=["time", "voxels"],
        coords={
            "time": np.arange(n_time) * 0.002,
            "voxels": np.arange(n_voxels),
        },
    )


@pytest.fixture
def signals_with_cubic_trend(rng):
    """Signals with cubic trend."""
    n_time = 100
    n_voxels = 50

    # Create noise.
    noise = rng.normal(loc=0, scale=0.5, size=(n_time, n_voxels))

    # Add cubic trend.
    time = np.arange(n_time)
    trend = (time[:, np.newaxis] ** 3) * 0.001  # Cubic trend.

    return xr.DataArray(
        trend + noise,
        dims=["time", "voxels"],
        coords={
            "time": np.arange(n_time) * 0.002,
            "voxels": np.arange(n_voxels),
        },
    )


def test_detrend_linear(signals_with_linear_trend):
    """Test linear detrending removes linear trend."""
    result = detrend(signals_with_linear_trend, order=1)

    # Check shape and coordinates preserved.
    assert result.dims == signals_with_linear_trend.dims
    assert result.shape == signals_with_linear_trend.shape
    assert_allclose(
        result.coords["time"].values,
        signals_with_linear_trend.coords["time"].values,
    )

    # After detrending, mean should be near zero.
    mean_per_voxel = result.mean(dim="time")
    assert_allclose(mean_per_voxel.values, 0.0, atol=1e-10)

    # Verify against scipy.signal.detrend.
    scipy_result = scipy.signal.detrend(signals_with_linear_trend.values, axis=0)
    assert_allclose(result.values, scipy_result, rtol=1e-10)


def test_detrend_constant(signals_with_linear_trend):
    """Test constant detrending removes mean."""
    result = detrend(signals_with_linear_trend, order=0)

    # Check shape and coordinates preserved.
    assert result.dims == signals_with_linear_trend.dims
    assert result.shape == signals_with_linear_trend.shape

    # After constant detrending, mean should be near zero.
    mean_per_voxel = result.mean(dim="time")
    assert_allclose(mean_per_voxel.values, 0.0, atol=1e-10)

    # Verify against scipy.signal.detrend.
    scipy_result = scipy.signal.detrend(
        signals_with_linear_trend.values, axis=0, type="constant"
    )
    assert_allclose(result.values, scipy_result, rtol=1e-10)


def test_detrend_polynomial_order1(signals_with_linear_trend):
    """Test polynomial order 1 matches scipy linear detrend."""
    result = detrend(signals_with_linear_trend, order=1)

    scipy_result = scipy.signal.detrend(signals_with_linear_trend.values, axis=0)
    assert_allclose(result.values, scipy_result, rtol=1e-8)


def test_detrend_polynomial_order2(signals_with_quadratic_trend):
    """Test polynomial order 2 removes quadratic trend."""
    result = detrend(signals_with_quadratic_trend, order=2)

    # Check shape and coordinates preserved.
    assert result.dims == signals_with_quadratic_trend.dims
    assert result.shape == signals_with_quadratic_trend.shape

    # After detrending, mean should be near zero.
    mean_per_voxel = result.mean(dim="time")
    assert_allclose(mean_per_voxel.values, 0.0, atol=1e-10)

    # Compare against naive reference implementation.
    naive_result = _naive_polynomial_detrend(
        signals_with_quadratic_trend.values, order=2, axis=0
    )
    # Use relaxed tolerance for numerical precision (matching order 3 test).
    assert_allclose(result.values, naive_result, rtol=1e-8)


def test_detrend_polynomial_order3(signals_with_cubic_trend):
    """Test polynomial order 3 removes cubic trend."""
    result = detrend(signals_with_cubic_trend, order=3)

    # Check shape and coordinates preserved.
    assert result.dims == signals_with_cubic_trend.dims
    assert result.shape == signals_with_cubic_trend.shape

    # After detrending, mean should be near zero.
    mean_per_voxel = result.mean(dim="time")
    assert_allclose(mean_per_voxel.values, 0.0, atol=1e-10)

    # Compare against naive reference implementation.
    naive_result = _naive_polynomial_detrend(
        signals_with_cubic_trend.values, order=3, axis=0
    )
    # Higher-order polynomials have slightly worse numerical precision.
    assert_allclose(result.values, naive_result, rtol=1e-8)


def test_detrend_polynomial_order4(rng):
    """Test polynomial order 4 (quartic) detrending."""
    n_time = 100
    n_voxels = 50

    # Create noise.
    noise = rng.normal(loc=0, scale=0.5, size=(n_time, n_voxels))

    # Add quartic trend.
    time = np.arange(n_time)
    trend = (time[:, np.newaxis] ** 4) * 0.00001  # Quartic trend.

    signals = xr.DataArray(
        trend + noise,
        dims=["time", "voxels"],
        coords={
            "time": np.arange(n_time) * 0.002,
            "voxels": np.arange(n_voxels),
        },
    )

    result = detrend(signals, order=4)

    # Check shape and coordinates preserved.
    assert result.dims == signals.dims
    assert result.shape == signals.shape

    # After detrending, mean should be near zero.
    mean_per_voxel = result.mean(dim="time")
    assert_allclose(mean_per_voxel.values, 0.0, atol=1e-10)

    # Compare against naive reference implementation.
    naive_result = _naive_polynomial_detrend(signals.values, order=4, axis=0)
    # Higher-order polynomials have slightly worse numerical precision.
    assert_allclose(result.values, naive_result, rtol=1e-7)


def test_detrend_no_time_dimension():
    """Test error raised when signals have no time dimension."""
    signals = xr.DataArray(
        np.random.randn(50, 10, 20),
        dims=["z", "y", "x"],
    )

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        detrend(signals, order=1)


def test_detrend_negative_order(signals_with_linear_trend):
    """Test error raised for negative polynomial order."""
    with pytest.raises(ValueError, match="order must be non-negative"):
        detrend(signals_with_linear_trend, order=-1)


def test_detrend_single_timepoint():
    """Test error raised for single timepoint."""
    signals = xr.DataArray(
        np.array([[1.0, 2.0, 3.0]]),
        dims=["time", "voxels"],
    )

    with pytest.raises(ValueError, match="more than 1 timepoint"):
        detrend(signals, order=1)


def test_detrend_3dt_imaging_data(sample_4d_volume):
    """Test that detrend works on 3D+t imaging data (time, z, y, x)."""
    # Add linear trend to the sample 4D volume.
    time = np.arange(sample_4d_volume.sizes["time"])
    trend = time[:, np.newaxis, np.newaxis, np.newaxis] * 1.0
    imaging_3dt = sample_4d_volume + trend

    result = detrend(imaging_3dt, order=1)

    # Check shape and dimensions preserved.
    assert result.dims == imaging_3dt.dims
    assert result.shape == imaging_3dt.shape

    # Check that mean across time is near zero for each voxel.
    mean_per_voxel = result.mean(dim="time")
    assert_allclose(mean_per_voxel.values, 0.0, atol=1e-10)

    # Check coordinates preserved.
    assert_allclose(result.coords["time"].values, imaging_3dt.coords["time"].values)
    assert_allclose(result.coords["z"].values, imaging_3dt.coords["z"].values)


def test_detrend_default_parameters(signals_with_linear_trend):
    """Test default parameters (order=1)."""
    result = detrend(signals_with_linear_trend)

    # Should be same as explicitly passing order=1.
    expected = detrend(signals_with_linear_trend, order=1)
    assert_allclose(result.values, expected.values)


def test_detrend_polynomial_3dt(sample_4d_volume):
    """Test polynomial detrending on 3D+t data."""
    # Add quadratic trend to the sample 4D volume.
    time = np.arange(sample_4d_volume.sizes["time"])
    trend = (time[:, np.newaxis, np.newaxis, np.newaxis] ** 2) * 0.05
    imaging_3dt = sample_4d_volume + trend

    result = detrend(imaging_3dt, order=2)

    assert result.dims == imaging_3dt.dims
    assert result.shape == imaging_3dt.shape

    mean_per_voxel = result.mean(dim="time")
    assert_allclose(mean_per_voxel.values, 0.0, atol=1e-10)

    # Compare against naive reference implementation.
    naive_result = _naive_polynomial_detrend(imaging_3dt.values, order=2, axis=0)
    # Use relaxed tolerance for numerical precision.
    assert_allclose(result.values, naive_result, rtol=1e-8)


def test_detrend_dask_compatibility(rng):
    """Test linear detrending works with Dask-backed arrays."""
    n_time = 100
    n_voxels = 50

    noise = rng.normal(loc=0, scale=0.5, size=(n_time, n_voxels))
    time = np.arange(n_time)
    trend = time[:, np.newaxis] * 2.0
    data = trend + noise

    # Important: Don't chunk along time! Detrending needs full time series.
    signals_dask = xr.DataArray(
        da.from_array(data, chunks=(n_time, 25)),  # Full time, chunked voxels.
        dims=["time", "voxels"],
        coords={
            "time": np.arange(n_time) * 0.002,
            "voxels": np.arange(n_voxels),
        },
    )

    result = detrend(signals_dask, order=1)

    assert isinstance(result.data, da.Array)

    mean_per_voxel = result.mean(dim="time")
    assert_allclose(mean_per_voxel.values, 0.0, atol=1e-10)


def test_detrend_polynomial_dask_compatibility(rng):
    """Test polynomial detrending maintains Dask laziness."""
    n_time = 100
    n_voxels = 50

    noise = rng.normal(loc=0, scale=0.5, size=(n_time, n_voxels))
    time = np.arange(n_time)
    trend = (time[:, np.newaxis] ** 2) * 0.05
    data = trend + noise

    # Important: Don't chunk along time! Detrending needs full time series.
    signals_dask = xr.DataArray(
        da.from_array(data, chunks=(n_time, 25)),  # Full time, chunked voxels.
        dims=["time", "voxels"],
        coords={
            "time": np.arange(n_time) * 0.002,
            "voxels": np.arange(n_voxels),
        },
    )

    result = detrend(signals_dask, order=2)

    assert isinstance(result.data, da.Array)

    mean_per_voxel = result.mean(dim="time")
    assert_allclose(mean_per_voxel.values, 0.0, atol=1e-10)


def test_detrend_raises_on_time_chunking(rng):
    """Test that detrending raises error when time dimension is chunked."""
    n_time = 100
    n_voxels = 50

    data = rng.normal(loc=0, scale=0.5, size=(n_time, n_voxels))

    # Create Dask array chunked along time (WRONG!).
    signals_bad_chunks = xr.DataArray(
        da.from_array(data, chunks=(50, 25)),  # Time is chunked!
        dims=["time", "voxels"],
    )

    # Should raise ValueError.
    with pytest.raises(
        ValueError, match="chunked along the 'time' dimension.*requires the full"
    ):
        detrend(signals_bad_chunks, order=1)
