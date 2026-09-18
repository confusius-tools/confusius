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


def test_validate_time_series_raises_for_unsorted_time_when_required():
    """Sorted-time validation should reject non-increasing coordinates."""
    signals = xr.DataArray(
        [1.0, 3.0, 2.0, 4.0],
        dims=["time"],
        coords={"time": [0.0, 2.0, 1.0, 3.0]},
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_time_series(signals, "resampling", require_sorted_time=True)


def test_validate_time_series_raises_for_pose_dependent_time_when_1d_required():
    """1D-time validation should reject a (time, pose)-shaped time coordinate."""
    time = xr.DataArray([[0.0, 0.1], [1.0, 1.2], [2.0, 2.3]], dims=["time", "pose"])
    signals = xr.DataArray(
        np.zeros((3, 2)), dims=["time", "pose"], coords={"time": time}
    )

    with pytest.raises(ValueError, match="1D 'time' coordinate"):
        validate_time_series(signals, "resampling", require_1d_time=True)


def test_validate_time_series_allows_pose_dependent_time_by_default():
    """1D-time validation should be opt-in, not enforced by default."""
    time = xr.DataArray([[0.0, 0.1], [1.0, 1.2], [2.0, 2.3]], dims=["time", "pose"])
    signals = xr.DataArray(
        np.zeros((3, 2)), dims=["time", "pose"], coords={"time": time}
    )

    validate_time_series(signals, "resampling")


def test_validate_time_series_checks_sortedness_along_time_axis_for_pose_data():
    """Sorted-time validation should diff along time, not the trailing pose axis."""
    # Each pose's own timestamps are strictly increasing, but pose 1's timestamps are
    # smaller than pose 0's at every timepoint -- a naive default-axis np.diff would
    # misread that as unsorted.
    time = xr.DataArray([[1.0, 0.5], [2.0, 1.5], [3.0, 2.5]], dims=["time", "pose"])
    signals = xr.DataArray(
        np.zeros((3, 2)), dims=["time", "pose"], coords={"time": time}
    )

    validate_time_series(signals, "resampling", require_sorted_time=True)


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
