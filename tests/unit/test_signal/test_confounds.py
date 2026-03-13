"""Tests for confound regression functions."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.signal import regress_confounds


def test_regress_confounds_basic(sample_timeseries):
    """Test basic confound regression removes confound effects."""
    signals = sample_timeseries(n_time=100, n_voxels=50)

    # Create a simple confound (linear trend)
    confound = xr.DataArray(
        np.linspace(-1, 1, 100),
        dims=["time"],
        coords={"time": signals.coords["time"]},
    )

    # Add confound to signals
    signals_with_confound = signals + confound.values[:, np.newaxis] * 2

    # Remove confound
    cleaned = regress_confounds(signals_with_confound, confound)

    # Check shape and coordinates preserved
    assert cleaned.dims == signals.dims
    assert cleaned.shape == signals.shape
    assert_allclose(cleaned.coords["time"].values, signals.coords["time"].values)

    # After regression, the linear trend should be removed
    # (signals should be uncorrelated with confound)
    for i in range(signals.sizes["space"]):
        corr = np.corrcoef(cleaned.values[:, i], confound.values)[0, 1]
        assert abs(corr) < 0.1  # Should be close to 0


def test_regress_confounds_multiple_confounds(sample_timeseries):
    """Test regression with multiple confounds."""
    signals = sample_timeseries(n_time=100, n_voxels=50)

    # Create multiple confounds (without constant to avoid issues)
    time = np.arange(100)
    confounds = xr.DataArray(
        np.column_stack(
            [
                time,  # linear
                time**2,  # quadratic
                np.sin(time * 0.1),  # sinusoidal
            ]
        ),
        dims=["time", "confound"],
        coords={"time": signals.coords["time"]},
    )

    # Add confounds to signals
    coeffs = np.random.randn(3, 50)
    signals_with_confounds = signals + confounds.values @ coeffs

    # Remove confounds
    cleaned = regress_confounds(signals_with_confounds, confounds)

    # Check cleaned signals have no remaining linear dependence on confounds
    for j in range(signals.sizes["space"]):
        coeffs = np.linalg.lstsq(confounds.values, cleaned.values[:, j], rcond=None)[0]
        assert_allclose(coeffs, 0.0, atol=1e-10)


def test_regress_confounds_orthogonalization():
    """Test that regression properly orthogonalizes signals from confounds."""
    n_time = 50
    n_voxels = 10

    confound = xr.DataArray(
        np.sin(np.linspace(0, 4 * np.pi, n_time)),
        dims=["time"],
        coords={"time": np.arange(n_time) * 0.1},
    )

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, size=(n_time, n_voxels))
    signals_data = confound.values[:, np.newaxis] * rng.normal(5, 1, n_voxels) + noise

    signals = xr.DataArray(
        signals_data,
        dims=["time", "space"],
        coords={"time": np.arange(n_time) * 0.1},
    )

    cleaned = regress_confounds(signals, confound)

    for i in range(n_voxels):
        coeff = np.linalg.lstsq(
            confound.values[:, None], cleaned.values[:, i], rcond=None
        )[0]
        assert_allclose(coeff, 0.0, atol=1e-10)


def test_regress_confounds_normalization_preserves_constant():
    """Test that normalization preserves constant confounds."""
    n_time = 50
    n_voxels = 10

    rng = np.random.default_rng(42)
    signals_data = rng.normal(size=(n_time, n_voxels))

    # Include constant confound
    confounds = xr.DataArray(
        np.column_stack(
            [
                np.ones(n_time),  # constant
                np.linspace(-1, 1, n_time),  # linear
            ]
        ),
        dims=["time", "confound"],
        coords={"time": np.arange(n_time) * 0.1},
    )

    signals = xr.DataArray(
        signals_data,
        dims=["time", "space"],
        coords={"time": np.arange(n_time) * 0.1},
    )

    # Should work without error and remove both confounds
    cleaned = regress_confounds(signals, confounds, standardize_confounds=True)

    # Cleaned signals should be orthogonal to both confounds
    for i in range(confounds.shape[1]):
        for j in range(n_voxels):
            dot_product = np.dot(cleaned.values[:, j], confounds.values[:, i])
            assert abs(dot_product) < 1e-10


def test_regress_confounds_rank_deficient():
    """Test handling of rank-deficient confound matrix (collinear confounds)."""
    n_time = 50
    n_voxels = 10

    rng = np.random.default_rng(42)
    signals_data = rng.normal(size=(n_time, n_voxels))

    # Create collinear confounds (second is multiple of first)
    confound1 = np.linspace(-1, 1, n_time)
    confounds = xr.DataArray(
        np.column_stack([confound1, confound1 * 2, confound1 * 3]),
        dims=["time", "confound"],
        coords={"time": np.arange(n_time) * 0.1},
    )

    signals = xr.DataArray(
        signals_data,
        dims=["time", "space"],
        coords={"time": np.arange(n_time) * 0.1},
    )

    # Should not raise error, should handle rank deficiency
    cleaned = regress_confounds(signals, confounds)

    # Result should still be orthogonal to the confound direction
    for i in range(n_voxels):
        dot_product = np.dot(cleaned.values[:, i], confound1)
        assert abs(dot_product) < 1e-10


def test_regress_confounds_invalid_time_dimension(sample_timeseries):
    """Test error when signals have no time dimension."""
    signals = xr.DataArray(
        np.random.randn(50, 10),
        dims=["space", "sample"],
    )
    confounds = xr.DataArray(
        np.random.randn(50, 3),
        dims=["time", "confound"],
        coords={"time": np.arange(50) * 0.1},
    )

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        regress_confounds(signals, confounds)


def test_regress_confounds_mismatched_time(sample_timeseries):
    """Test error when confounds time dimension doesn't match."""
    signals = sample_timeseries(n_time=100, n_voxels=50)
    confounds = xr.DataArray(
        np.random.randn(50, 3),
        dims=["time", "confound"],
        coords={"time": np.arange(50) * 0.1},
    )

    with pytest.raises(ValueError, match="time coordinates do not match"):
        regress_confounds(signals, confounds)


def test_regress_confounds_invalid_type(sample_timeseries):
    """Test error when confounds is not numpy array or xarray."""
    signals = sample_timeseries()

    with pytest.raises(TypeError, match="must be an xarray.DataArray"):
        regress_confounds(signals, "invalid")  # type: ignore[arg-type]


def test_regress_confounds_wrong_dimensions(sample_timeseries):
    """Test error when confounds have wrong number of dimensions."""
    signals = sample_timeseries(n_time=100, n_voxels=50)
    confounds = xr.DataArray(
        np.random.randn(100, 3, 2),
        dims=["time", "confound", "extra"],
        coords={"time": signals.coords["time"]},
    )

    with pytest.raises(ValueError, match="must be 1D or 2D"):
        regress_confounds(signals, confounds)


def test_regress_confounds_single_confound_1d(sample_timeseries):
    """Test with 1D confound array."""
    signals = sample_timeseries(n_time=100, n_voxels=50)
    confound = xr.DataArray(
        np.random.randn(100),
        dims=["time"],
        coords={"time": signals.coords["time"]},
    )

    # Should work and be treated as single confound
    cleaned = regress_confounds(signals, confound)
    assert cleaned.shape == signals.shape


def test_regress_confounds_xarray_confounds(sample_timeseries):
    """Test with xarray DataArray confounds."""
    signals = sample_timeseries(n_time=100, n_voxels=50)

    # Create confounds as xarray DataArray
    confounds = xr.DataArray(
        np.random.randn(100, 3),
        dims=["time", "confound"],
        coords={"time": signals.coords["time"].values},
    )

    # Should work
    cleaned = regress_confounds(signals, confounds)
    assert cleaned.shape == signals.shape


def test_regress_confounds_xarray_time_mismatch(sample_timeseries):
    """Test error when xarray confounds time coordinates mismatch signals."""
    signals = sample_timeseries(n_time=100, n_voxels=50)

    confounds = xr.DataArray(
        np.random.randn(100, 3),
        dims=["time", "confound"],
        coords={"time": signals.coords["time"].values + 1.0},
    )

    with pytest.raises(ValueError, match="time coordinates do not match"):
        regress_confounds(signals, confounds)


def test_regress_confounds_4d_imaging(sample_4d_volume):
    """Test on 4D imaging data (time, z, y, x)."""
    # Create confounds matching time dimension
    n_time = sample_4d_volume.sizes["time"]
    confounds = xr.DataArray(
        np.random.randn(n_time, 6),
        dims=["time", "confound"],
        coords={"time": sample_4d_volume.coords["time"]},
    )

    # Should work on 4D data
    cleaned = regress_confounds(sample_4d_volume, confounds)

    # Check shape preserved
    assert cleaned.dims == sample_4d_volume.dims
    assert cleaned.shape == sample_4d_volume.shape

    # Check coordinates preserved
    for dim in sample_4d_volume.dims:
        assert_allclose(
            cleaned.coords[dim].values,
            sample_4d_volume.coords[dim].values,
        )


def test_regress_confounds_dask_compatibility(sample_timeseries):
    """Test confound regression works with Dask-backed arrays."""
    signals = sample_timeseries(n_time=100, n_voxels=50)

    # Convert to Dask
    dask_data = da.from_array(signals.values, chunks=(100, 25))  # type: ignore[arg-type]
    signals_dask = xr.DataArray(
        dask_data,
        dims=signals.dims,
        coords=signals.coords,
    )

    confounds = xr.DataArray(
        np.random.randn(100, 6),
        dims=["time", "confound"],
        coords={"time": signals.coords["time"]},
    )

    # Should work without computing
    cleaned = regress_confounds(signals_dask, confounds)

    # Result should still be Dask-backed
    assert isinstance(cleaned.data, da.Array)

    # Compute and verify shape
    cleaned_computed = cleaned.compute()
    assert cleaned_computed.shape == signals.shape


def test_regress_confounds_dask_chunked_time(sample_timeseries):
    """Test error when time dimension is chunked."""
    signals = sample_timeseries(n_time=100, n_voxels=50)

    # Chunk along time (which is invalid)
    dask_data = da.from_array(signals.values, chunks=(50, 25))  # type: ignore[arg-type]
    signals_dask = xr.DataArray(
        dask_data,
        dims=signals.dims,
        coords=signals.coords,
    )

    confounds = xr.DataArray(
        np.random.randn(100, 6),
        dims=["time", "confound"],
        coords={"time": signals.coords["time"]},
    )

    with pytest.raises(ValueError, match="chunked along the 'time' dimension"):
        regress_confounds(signals_dask, confounds)


def test_regress_confounds_single_timepoint():
    """Test error raised for single timepoint."""
    signals = xr.DataArray(
        np.random.randn(1, 10),
        dims=["time", "space"],
        coords={"time": [0.0]},
    )
    confounds = xr.DataArray(
        np.random.randn(1, 3),
        dims=["time", "confound"],
        coords={"time": [0.0]},
    )

    with pytest.raises(ValueError, match="more than 1 timepoint"):
        regress_confounds(signals, confounds)


def test_regress_confounds_orthogonal_to_confound():
    """Test that cleaned signals are orthogonal to confounds."""
    n_time = 100
    n_voxels = 10

    rng = np.random.default_rng(42)

    # Random signals
    signals_data = rng.normal(size=(n_time, n_voxels))

    # Random confound
    confound = xr.DataArray(
        np.sin(np.linspace(0, 20 * np.pi, n_time)),
        dims=["time"],
        coords={"time": np.arange(n_time) * 0.1},
    )

    signals = xr.DataArray(
        signals_data,
        dims=["time", "space"],
        coords={"time": np.arange(n_time) * 0.1},
    )

    # Regress
    cleaned = regress_confounds(signals, confound)

    # Cleaned signals should be orthogonal to confound (dot product ≈ 0)
    for i in range(n_voxels):
        dot_product = np.dot(cleaned.values[:, i], confound.values)
        assert abs(dot_product) < 1e-10


def test_regress_confounds_zero_variance_confounds():
    """Test handling of constant (zero-variance) confounds."""
    n_time = 50
    n_voxels = 10

    rng = np.random.default_rng(42)
    signals_data = rng.normal(size=(n_time, n_voxels))

    # Include a constant confound
    confounds = xr.DataArray(
        np.column_stack(
            [
                np.ones(n_time),  # constant
                np.linspace(-1, 1, n_time),  # linear
            ]
        ),
        dims=["time", "confound"],
        coords={"time": np.arange(n_time) * 0.1},
    )

    signals = xr.DataArray(
        signals_data,
        dims=["time", "space"],
        coords={"time": np.arange(n_time) * 0.1},
    )

    # Should handle constant confound without error
    cleaned = regress_confounds(signals, confounds)
    assert cleaned.shape == signals.shape


def test_regress_confounds_reference_implementation(sample_timeseries):
    """Compare against naive OLS implementation without standardization."""
    signals = sample_timeseries(n_time=100, n_voxels=50)
    confounds = xr.DataArray(
        np.random.randn(100, 6),
        dims=["time", "confound"],
        coords={"time": signals.coords["time"]},
    )

    # Our implementation without standardization
    cleaned = regress_confounds(signals, confounds, standardize_confounds=False)

    # Naive OLS: residuals = signals - X @ (X^+ @ signals)
    # where X^+ is pseudoinverse
    X = confounds.values
    signals_2d = signals.values.reshape(signals.sizes["time"], -1)
    coeffs = np.linalg.pinv(X) @ signals_2d
    expected_residuals = signals_2d - X @ coeffs
    expected = expected_residuals.reshape(signals.shape)

    # Results should be very close
    assert_allclose(cleaned.values, expected, rtol=1e-10)
