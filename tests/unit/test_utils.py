"""Tests for `confusius._utils`."""

import numpy as np
import pytest
import xarray as xr

from confusius._utils import (
    _compute_origin,
    _compute_spacing,
    _compute_spacing_best_effort,
    _one_level_deeper,
    _representative_step,
    _spacing_for_dim,
    _time_coord_values_in_seconds,
    find_stack_level,
)


def test_find_stack_level() -> None:
    """`find_stack_level` points to the first frame outside ConfUSIus."""
    assert find_stack_level() == 1


def test_one_level_deeper() -> None:
    """`_one_level_deeper` adds one extra ConfUSIus frame."""
    assert _one_level_deeper() == 2


def test_time_coord_values_in_seconds_unknown_units_warns() -> None:
    """Unknown time units warn and default to seconds."""
    da = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=["time"],
        coords={
            "time": xr.DataArray(
                [0.0, 1.0, 2.0], dims=["time"], attrs={"units": "fortnight"}
            )
        },
    )

    with pytest.warns(UserWarning, match="unknown units"):
        values = _time_coord_values_in_seconds(da)

    np.testing.assert_allclose(values, [0.0, 1.0, 2.0])


def test_representative_step_single_value_returns_none() -> None:
    """A single value has no representative step."""
    step, approximate = _representative_step(np.array([1.0]))

    assert step is None
    assert not approximate


def test_representative_step_zero_median_uses_exact_comparison() -> None:
    """Zero-median diffs use the dedicated equality branch."""
    step, approximate = _representative_step(np.array([1.0, 1.0, 1.0]))

    assert step == 0.0
    assert not approximate


def test_spacing_for_dim_non_numeric_coordinate_returns_none() -> None:
    """Non-numeric coordinates have undefined spacing without warnings."""
    da = xr.DataArray([1.0, 2.0], dims=["label"], coords={"label": ["a", "b"]})

    spacing = _spacing_for_dim("label", da, uniformity_tolerance=1e-2)

    assert spacing.value is None
    assert spacing.median is None
    assert spacing.warn_msg is None


def test_compute_spacing_best_effort_tracks_non_uniform_dims() -> None:
    """Best-effort spacing uses the median step and records non-uniform dims."""
    da = xr.DataArray(
        np.zeros((3, 2)),
        dims=["x", "y"],
        coords={"x": [0.0, 1.0, 3.0], "y": [0.0, 2.0]},
    )

    spacing, non_uniform = _compute_spacing_best_effort(da)

    assert spacing["x"] == pytest.approx(1.5)
    assert spacing["y"] == pytest.approx(2.0)
    assert non_uniform == ["x"]


def test_spacing_for_dim_single_point_uses_voxdim() -> None:
    """Single-point coordinates fall back to their `voxdim` attribute."""
    da = xr.DataArray(
        [1.0],
        dims=["x"],
        coords={"x": xr.DataArray([0.0], dims=["x"], attrs={"voxdim": 0.2})},
    )

    spacing = _spacing_for_dim("x", da, uniformity_tolerance=1e-2)

    assert spacing.value == pytest.approx(0.2)
    assert spacing.median is None
    assert spacing.warn_msg is None


def test_compute_spacing_warns_for_missing_single_point_spacing() -> None:
    """`_compute_spacing` warns when a single-point coordinate has no `voxdim`."""
    da = xr.DataArray([1.0], dims=["x"], coords={"x": [0.0]})

    with pytest.warns(UserWarning, match="single coordinate point"):
        spacing = _compute_spacing(da)

    assert spacing == {"x": None}


def test_compute_spacing_best_effort_falls_back_to_unit_spacing() -> None:
    """Best-effort spacing falls back to 1.0 when spacing is truly undefined."""
    da = xr.DataArray([1.0], dims=["x"], coords={"x": [0.0]})

    spacing, non_uniform = _compute_spacing_best_effort(da)

    assert spacing == {"x": 1.0}
    assert non_uniform == []


def test_compute_origin_warns_when_coordinate_is_missing() -> None:
    """Missing coordinates warn and default to zero origin."""
    da = xr.DataArray([[1.0, 2.0]], dims=["time", "x"], coords={"x": [3.0, 4.0]})

    with pytest.warns(UserWarning, match="origin defaults to 0.0"):
        origin = _compute_origin(da)

    assert origin == {"time": 0.0, "x": 3.0}
