"""Tests for time series validation utilities."""

import numpy as np
import pytest
import xarray as xr

from confusius.validation.time_series import validate_time_series


def test_validate_time_series_returns_time_axis_and_no_spacing_by_default():
    """Default validation should not require uniform spacing."""
    signals = xr.DataArray(
        np.arange(12, dtype=float).reshape(6, 2),
        dims=["time", "space"],
        coords={"time": [0.0, 0.1, 0.2, 0.35, 0.4, 0.5]},
    )

    time_axis, time_spacing = validate_time_series(signals, "filtering")

    assert time_axis == 0
    assert time_spacing is None


def test_validate_time_series_raises_for_single_timepoint():
    """Validation should reject single-timepoint inputs."""
    signals = xr.DataArray(
        np.array([[1.0, 2.0]]),
        dims=["time", "space"],
        coords={"time": [0.0]},
    )

    with pytest.raises(ValueError, match="requires more than 1 timepoint"):
        validate_time_series(signals, "filtering")


def test_validate_time_series_returns_spacing_when_uniform_time_required():
    """Uniform-time validation should return the time spacing."""
    signals = xr.DataArray(
        np.arange(12, dtype=float).reshape(6, 2),
        dims=["time", "space"],
        coords={"time": np.arange(6) * 0.1},
    )

    time_axis, time_spacing = validate_time_series(
        signals,
        "filtering",
        require_uniform_time=True,
    )

    assert time_axis == 0
    assert time_spacing == pytest.approx(0.1)


def test_validate_time_series_raises_for_nonuniform_time_when_required():
    """Uniform-time validation should reject non-uniform coordinates."""
    signals = xr.DataArray(
        np.arange(12, dtype=float).reshape(6, 2),
        dims=["time", "space"],
        coords={"time": [0.0, 0.1, 0.2, 0.35, 0.4, 0.5]},
    )

    with pytest.raises(ValueError, match="Non-uniform 'time' coordinates"):
        validate_time_series(signals, "filtering", require_uniform_time=True)
