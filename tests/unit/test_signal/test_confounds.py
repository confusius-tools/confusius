"""Tests for confound regression functions."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.signal import regress_confounds


def test_regress_confounds_basic(make_sample_timeseries):
    """Test basic confound regression removes confound effects."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)

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


def test_regress_confounds_multiple_confounds(make_sample_timeseries):
    """Test regression with multiple confounds."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)

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

    # Check cleaned signals have no remaining linear dependence on centered confounds.
    confounds_centered = confounds.values - confounds.values.mean(axis=0)
    for j in range(signals.sizes["space"]):
        coeffs = np.linalg.lstsq(confounds_centered, cleaned.values[:, j], rcond=None)[
            0
        ]
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


def test_regress_confounds_without_standardization_preserves_constant():
    """Unstandardized confound regression preserves constant confounds."""
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

    # Should work without error and remove both confounds.
    cleaned = regress_confounds(signals, confounds, standardize_confounds=False)

    # Cleaned signals should be orthogonal to both confounds
    for i in range(confounds.shape[1]):
        for j in range(n_voxels):
            dot_product = np.dot(cleaned.values[:, j], confounds.values[:, i])
            assert abs(dot_product) < 1e-10


def test_regress_confounds_standardizes_confounds_by_default():
    """Default confound standardization removes only confound fluctuations."""
    n_time = 20
    time = np.arange(n_time)
    baseline = np.array([10.0, 20.0])
    confound = xr.DataArray(
        100.0 + np.linspace(-1, 1, n_time),
        dims=["time"],
        coords={"time": time},
    )
    signals = xr.DataArray(
        baseline + 2.0 * confound.values[:, np.newaxis],
        dims=["time", "space"],
        coords={"time": time},
    )

    cleaned = regress_confounds(signals, confound)

    assert_allclose(cleaned.mean("time"), signals.mean("time"))
    assert_allclose(cleaned.std("time"), 0.0, atol=1e-10)


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


def test_regress_confounds_invalid_time_dimension(make_sample_timeseries):
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


def test_regress_confounds_mismatched_time(make_sample_timeseries):
    """Test error when confounds time dimension doesn't match."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)
    confounds = xr.DataArray(
        np.random.randn(50, 3),
        dims=["time", "confound"],
        coords={"time": np.arange(50) * 0.1},
    )

    with pytest.raises(ValueError, match="time coordinates do not match"):
        regress_confounds(signals, confounds)


@pytest.mark.parametrize("confounds", ["invalid", [0.0] * 100])
def test_regress_confounds_invalid_type(make_sample_timeseries, confounds):
    """Test error when confounds is not a DataArray, NumPy array, or DataFrame."""
    signals = make_sample_timeseries()

    with pytest.raises(TypeError, match="must be an xarray.DataArray, numpy.ndarray"):
        regress_confounds(signals, confounds)


def test_regress_confounds_dataframe_matches_dataarray(make_sample_timeseries, rng):
    """Test DataFrame confounds with a time column match the DataArray result."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)
    values = rng.standard_normal((100, 2))
    confounds = xr.DataArray(
        values, dims=["time", "confound"], coords={"time": signals.coords["time"]}
    )
    frame = pd.DataFrame(
        {
            "time": signals.coords["time"].values,
            "motion_x": values[:, 0],
            "motion_y": values[:, 1],
        }
    )

    xr.testing.assert_allclose(
        regress_confounds(signals, frame), regress_confounds(signals, confounds)
    )


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"motion": np.zeros(100)}), "must have a 'time' column"),
        (
            pd.DataFrame({"time": np.arange(100) / 100 + 1.0, "motion": np.zeros(100)}),
            "time coordinates do not match",
        ),
        (pd.DataFrame({"time": np.arange(100) / 100}), "at least one column"),
        (
            pd.DataFrame(
                {"time": np.arange(100) / 100, "label": ["a"] * 100, "motion": 0.0}
            ),
            "columns must be numeric",
        ),
        (
            pd.DataFrame(
                np.zeros((100, 3)), columns=["time", "motion", "motion"]
            ).assign(time=np.arange(100) / 100),
            r"duplicate columns: \['motion'\]",
        ),
    ],
)
def test_regress_confounds_invalid_dataframe(make_sample_timeseries, frame, message):
    """Test DataFrame confounds are validated for a matching time column."""
    signals = make_sample_timeseries(n_time=100)

    with pytest.raises(ValueError, match=message):
        regress_confounds(signals, frame)


def test_regress_confounds_dataframe_bool_column_is_spike_regressor(
    make_sample_timeseries,
):
    """Test a boolean DataFrame column regresses like a 0/1 float regressor."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)
    spikes = np.zeros(100, dtype=bool)
    spikes[[3, 40]] = True
    frame = pd.DataFrame({"time": signals.coords["time"].values, "spike": spikes})
    expected = regress_confounds(
        signals,
        xr.DataArray(
            spikes.astype(float), dims=["time"], coords={"time": signals.coords["time"]}
        ),
    )

    xr.testing.assert_allclose(regress_confounds(signals, frame), expected)


def test_regress_confounds_multipose_regresses_every_pose_with_same_confounds(rng):
    """Test `pose` is just another dimension: the joint result matches per-pose runs."""
    times = np.arange(20) / 10.0
    signals = xr.DataArray(
        rng.standard_normal((20, 2, 3)),
        dims=["time", "pose", "space"],
        coords={"time": (("time", "pose"), np.stack([times, times + 0.05], axis=1))},
    )
    confounds = rng.standard_normal((20, 2))

    with pytest.warns(UserWarning) as record:
        result = regress_confounds(signals, confounds)
    messages = [str(warning.message) for warning in record]
    assert any("regressed from every pose" in message for message in messages)
    assert any("pose-dependent, so none are attached" in message for message in messages)

    for pose in range(2):
        with pytest.warns(UserWarning, match="cannot be verified"):
            expected = regress_confounds(signals.isel(pose=pose), confounds)
        xr.testing.assert_allclose(result.isel(pose=pose), expected)


def test_regress_confounds_dataarray_without_time_coordinates_warns(
    make_sample_timeseries, rng
):
    """Test coordinate-less DataArray confounds warn and match the aligned result."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)
    confounds = xr.DataArray(
        rng.standard_normal((100, 2)),
        dims=["time", "confound"],
        coords={"time": signals.coords["time"]},
    )
    expected = regress_confounds(signals, confounds)

    with pytest.warns(UserWarning, match="cannot be verified"):
        result = regress_confounds(signals, confounds.drop_vars("time"))

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("n_confounds", [None, 3])
def test_regress_confounds_numpy_matches_dataarray(
    make_sample_timeseries, rng, n_confounds
):
    """Test NumPy confounds warn and match the aligned DataArray result."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)
    values = rng.standard_normal((100,) if n_confounds is None else (100, n_confounds))
    confounds = xr.DataArray(
        values,
        dims=["time"] if n_confounds is None else ["time", "confound"],
        coords={"time": signals.coords["time"]},
    )
    expected = regress_confounds(signals, confounds)

    with pytest.warns(UserWarning, match="cannot be verified"):
        result = regress_confounds(signals, values)

    xr.testing.assert_allclose(result, expected)


def test_regress_confounds_numpy_rejects_3d(make_sample_timeseries):
    """Test NumPy confounds must be 1D or 2D."""
    signals = make_sample_timeseries(n_time=100)

    with pytest.raises(ValueError, match="confounds must be 1D or 2D"):
        regress_confounds(signals, np.zeros((100, 2, 2)))


def test_regress_confounds_confounds_missing_time_dimension(make_sample_timeseries):
    """Test error when confounds have no time dimension."""
    signals = make_sample_timeseries()
    confounds = xr.DataArray(np.random.randn(10, 3), dims=["sample", "confound"])

    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        regress_confounds(signals, confounds)


def test_regress_confounds_wrong_dimensions(make_sample_timeseries):
    """Test error when confounds have wrong number of dimensions."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)
    confounds = xr.DataArray(
        np.random.randn(100, 3, 2),
        dims=["time", "confound", "extra"],
        coords={"time": signals.coords["time"]},
    )

    with pytest.raises(ValueError, match="must be 1D or 2D"):
        regress_confounds(signals, confounds)


def test_regress_confounds_single_confound_1d(make_sample_timeseries):
    """Test with 1D confound array."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)
    confound = xr.DataArray(
        np.random.randn(100),
        dims=["time"],
        coords={"time": signals.coords["time"]},
    )

    # Should work and be treated as single confound
    cleaned = regress_confounds(signals, confound)
    assert cleaned.shape == signals.shape


def test_regress_confounds_mismatched_time_length_without_coordinates(
    make_sample_timeseries,
):
    """Test shape mismatch path when confounds have no time coordinates."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)
    confounds = xr.DataArray(np.random.randn(50, 3), dims=["time", "confound"])

    with pytest.raises(ValueError, match=r"confounds length \(50\) must match"):
        regress_confounds(signals, confounds)


def test_regress_confounds_xarray_confounds(make_sample_timeseries):
    """Test with xarray DataArray confounds."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)

    # Create confounds as xarray DataArray
    confounds = xr.DataArray(
        np.random.randn(100, 3),
        dims=["time", "confound"],
        coords={"time": signals.coords["time"].values},
    )

    # Should work
    cleaned = regress_confounds(signals, confounds)
    assert cleaned.shape == signals.shape


def test_regress_confounds_accepts_small_time_coordinate_drift(make_sample_timeseries):
    """Small numeric drift in time coordinates is tolerated."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)
    confounds = xr.DataArray(
        np.random.randn(100, 3),
        dims=["time", "confound"],
        coords={"time": signals.coords["time"].values + 1e-10},
    )

    cleaned = regress_confounds(signals, confounds)

    assert cleaned.shape == signals.shape


def test_regress_confounds_xarray_time_mismatch(make_sample_timeseries):
    """Test error when xarray confounds time coordinates mismatch signals."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)

    confounds = xr.DataArray(
        np.random.randn(100, 3),
        dims=["time", "confound"],
        coords={"time": signals.coords["time"].values + 1.0},
    )

    with pytest.raises(ValueError, match="time coordinates do not match"):
        regress_confounds(signals, confounds)


def test_regress_confounds_4d_imaging(sample_voxeldata_3dt):
    """Test on 4D imaging data (time, z, y, x)."""
    # Create confounds matching time dimension
    n_time = sample_voxeldata_3dt.sizes["time"]
    confounds = xr.DataArray(
        np.random.randn(n_time, 6),
        dims=["time", "confound"],
        coords={"time": sample_voxeldata_3dt.coords["time"]},
    )

    # Should work on 4D data
    cleaned = regress_confounds(sample_voxeldata_3dt, confounds)

    # Check shape preserved
    assert cleaned.dims == sample_voxeldata_3dt.dims
    assert cleaned.shape == sample_voxeldata_3dt.shape

    # Check coordinates preserved
    for dim in sample_voxeldata_3dt.dims:
        assert_allclose(
            cleaned.coords[dim].values,
            sample_voxeldata_3dt.coords[dim].values,
        )


def test_regress_confounds_nonleading_time_axis(make_sample_timeseries):
    """Test confound regression when time is not the leading axis."""
    signals = make_sample_timeseries(n_time=100, n_voxels=20)
    confounds = xr.DataArray(
        np.random.randn(100, 3),
        dims=["time", "confound"],
        coords={"time": signals.coords["time"]},
    )

    expected = regress_confounds(signals, confounds)
    transposed = signals.transpose("space", "time")
    result = regress_confounds(transposed, confounds).transpose("time", "space")

    assert_allclose(result.values, expected.values)


def test_regress_confounds_dask_compatibility(make_sample_timeseries):
    """Test confound regression works with Dask-backed arrays."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)

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


def test_regress_confounds_dask_chunked_time(make_sample_timeseries):
    """Test error when time dimension is chunked."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)

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


def test_regress_confounds_reference_implementation(make_sample_timeseries):
    """Compare against naive OLS implementation without standardization."""
    signals = make_sample_timeseries(n_time=100, n_voxels=50)
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

    assert_allclose(cleaned.values, expected, rtol=1e-7)
