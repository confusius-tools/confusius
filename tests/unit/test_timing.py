"""Unit tests for confusius.timing."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from scipy.interpolate import interp1d

from confusius.timing import (
    convert_time_reference,
    convert_time_units,
    get_representative_time_step,
    get_time_coord_to_seconds_factor,
    resample_time,
    resample_to_uniform_time,
)


def test_convert_time_units_rejects_unknown_units() -> None:
    """Time-unit conversion raises for unknown units when requested."""
    with pytest.raises(ValueError, match="Unknown time unit"):
        convert_time_units([1.0, 2.0], "fortnight", raise_on_unknown=True)


def test_convert_time_units_supports_non_second_targets() -> None:
    """Time-unit conversion can convert between non-native units."""
    converted = convert_time_units([100.0, 200.0], "ms", "s")

    assert_allclose(converted, [0.1, 0.2])


def test_get_time_coord_to_seconds_factor_warns_for_missing_units() -> None:
    """Coordinate-based conversion keeps warning semantics for missing units."""
    data = xr.DataArray(
        np.arange(3),
        dims=("time",),
        coords={"time": xr.DataArray([0.0, 1.0, 2.0], dims=("time",))},
    )

    with pytest.warns(UserWarning, match="no `units` attribute"):
        factor = get_time_coord_to_seconds_factor(data)

    assert factor == pytest.approx(1.0)


def test_get_representative_time_step_converts_to_seconds() -> None:
    """Representative time-step estimation can operate in physical units."""
    data = xr.DataArray(
        np.arange(4),
        dims=("time",),
        coords={
            "time": xr.DataArray(
                [0.0, 100.0, 200.0, 300.0],
                dims=("time",),
                attrs={"units": "ms"},
            )
        },
    )

    step, approximate = get_representative_time_step(data, unit="s")

    assert step == pytest.approx(0.1)
    assert not approximate


def test_get_representative_time_step_converts_to_non_second_units() -> None:
    """Representative time-step estimation supports non-second target units."""
    data = xr.DataArray(
        np.arange(4),
        dims=("time",),
        coords={
            "time": xr.DataArray(
                [0.0, 0.1, 0.2, 0.3],
                dims=("time",),
                attrs={"units": "s"},
            )
        },
    )

    step, approximate = get_representative_time_step(data, unit="ms")

    assert step == pytest.approx(100.0)
    assert not approximate


def test_convert_time_reference_rejects_invalid_reference() -> None:
    """Reference conversion raises a consistent ValueError for invalid names."""
    with pytest.raises(ValueError, match="from_reference"):
        convert_time_reference(
            np.array([0.0, 1.0]),
            volume_duration=0.5,
            from_reference="invalid",  # ty: ignore[invalid-argument-type]
            to_reference="start",
        )


def test_convert_time_reference_shifts_with_reference_factor() -> None:
    """Reference conversion preserves the expected physical offset."""
    converted = convert_time_reference(
        np.array([0.0, 1.0]),
        volume_duration=0.2,
        from_reference="start",
        to_reference="center",
    )

    assert_allclose(converted, [0.1, 1.1])


def test_convert_time_reference_supports_per_value_durations() -> None:
    """Reference conversion supports one duration per timing value."""
    converted = convert_time_reference(
        np.array([0.0, 1.0]),
        volume_duration=np.array([0.2, 0.4]),
        from_reference="start",
        to_reference="center",
    )

    assert_allclose(converted, [0.1, 1.2])


def test_resample_time_to_new_coordinates() -> None:
    """Resample data to a new set of time coordinates using linear interpolation."""
    time_values = [0.0, 1.0, 2.0, 3.0]
    data_values = [1.0, 2.0, 3.0, 4.0]
    data = xr.DataArray(
        data_values,
        dims=("time",),
        coords={
            "time": xr.DataArray(time_values, dims=("time",), attrs={"units": "s"})
        },
    )

    new_time = [0.5, 1.5, 2.5]
    result = resample_time(data, new_time)

    ref = interp1d(time_values, data_values, kind="linear")(new_time)
    assert_allclose(result.values, ref)


def test_resample_time_preserves_time_coord_attrs() -> None:
    """Resampling keeps time coordinate metadata on the output grid."""
    data = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0],
        dims=("time",),
        coords={
            "time": xr.DataArray(
                [0.0, 1.0, 2.0, 3.0],
                dims=("time",),
                attrs={"units": "s", "axis": "T"},
            )
        },
    )

    result = resample_time(data, [0.5, 1.5, 2.5])

    assert result.coords["time"].attrs["units"] == "s"
    assert result.coords["time"].attrs["axis"] == "T"


def test_resample_time_supports_nan_fill_value_alias() -> None:
    """The string fill-value alias `nan` maps to floating NaNs."""
    data = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0],
        dims=("time",),
        coords={"time": [0.0, 1.0, 2.0, 3.0]},
    )

    result = resample_time(data, [-1.0, 1.0, 4.0], fill_value="nan")

    assert np.isnan(result.values[0])
    assert result.values[1] == pytest.approx(2.0)
    assert np.isnan(result.values[2])


def test_resample_time_handles_dask_with_changed_time_length() -> None:
    """Dask-backed inputs can be resampled to a different time length."""
    data = xr.DataArray(
        np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ]
        ),
        dims=("time", "x"),
        coords={"time": [0.0, 1.0, 2.0, 3.0], "x": [0, 1]},
    ).chunk({"time": -1, "x": 1})

    result = resample_time(data, [0.5, 1.5, 2.5]).compute()

    assert result.shape == (3, 2)
    assert_allclose(result.sel(x=0).values, [1.5, 2.5, 3.5])
    assert_allclose(result.sel(x=1).values, [15.0, 25.0, 35.0])


def test_resample_time_warns_and_falls_back_for_short_cubic_series() -> None:
    """Cubic interpolation falls back to linear when there are too few points."""
    data = xr.DataArray(
        [0.0, 1.0, 2.0],
        dims=("time",),
        coords={"time": [0.0, 1.0, 2.0]},
    )

    with pytest.warns(UserWarning, match="falling back to 'linear'"):
        result = resample_time(data, [0.5, 1.5], method="cubic")

    assert_allclose(result.values, [0.5, 1.5])


def test_resample_time_rejects_missing_time_coordinate() -> None:
    """Resample raises when data has no time dimension."""
    data = xr.DataArray(
        np.arange(3),
        dims=("x",),
        coords={"x": [0.0, 1.0, 2.0]},
    )

    with pytest.raises(ValueError, match="'time' dimension"):
        resample_time(data, [0.5, 1.5])


def test_resample_to_uniform_time_uses_provided_step() -> None:
    """Resample to uniform time uses provided step and matches scipy reference."""
    time_values = [0.0, 1.0, 2.0, 3.0]
    data_values = [1.0, 2.0, 3.0, 4.0]
    data = xr.DataArray(
        data_values,
        dims=("time",),
        coords={
            "time": xr.DataArray(time_values, dims=("time",), attrs={"units": "s"})
        },
    )

    result = resample_to_uniform_time(data, step=0.5)

    expected_times = np.arange(0.0, 3.0 + 1e-12, 0.5)
    ref = interp1d(time_values, data_values, kind="linear")(expected_times)
    assert_allclose(result.coords["time"].values, expected_times)
    assert_allclose(result.values, ref)


def test_resample_to_uniform_time_auto_computes_step() -> None:
    """Resample to uniform time matches scipy for uniform data."""
    time_values = [0.0, 1.0, 2.0, 3.0]
    data_values = [1.0, 2.0, 3.0, 4.0]
    data = xr.DataArray(
        data_values,
        dims=("time",),
        coords={
            "time": xr.DataArray(time_values, dims=("time",), attrs={"units": "s"})
        },
    )

    result = resample_to_uniform_time(data)

    ref = interp1d(time_values, data_values, kind="linear")(time_values)
    assert_array_equal(result.coords["time"].values, np.asarray(time_values))
    assert_allclose(result.values, ref)


def test_resample_to_uniform_time_warns_for_non_uniform_input() -> None:
    """Automatic step estimation warns and uses the median step for jittered time."""
    data = xr.DataArray(
        [0.0, 1.0, 2.1, 3.1],
        dims=("time",),
        coords={"time": [0.0, 1.0, 2.1, 3.1]},
    )

    with pytest.warns(UserWarning, match="non-uniform"):
        result = resample_to_uniform_time(data)

    assert_allclose(result.coords["time"].values, [0.0, 1.0, 2.0, 3.0])


def test_resample_to_uniform_time_rejects_invalid_bounds() -> None:
    """Uniform resampling validates that start is strictly less than stop."""
    data = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=("time",),
        coords={"time": [0.0, 1.0, 2.0]},
    )

    with pytest.raises(ValueError, match="must be less than stop"):
        resample_to_uniform_time(data, start=2.0, stop=2.0, step=1.0)


def test_resample_to_uniform_time_rejects_non_positive_step() -> None:
    """Uniform resampling rejects zero or negative step sizes."""
    data = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=("time",),
        coords={"time": [0.0, 1.0, 2.0]},
    )

    with pytest.raises(ValueError, match="finite positive"):
        resample_to_uniform_time(data, step=0.0)


def test_resample_to_uniform_time_rejects_invalid_auto_step() -> None:
    """Automatic step estimation raises when time coordinates contain NaNs."""
    data = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=("time",),
        coords={"time": [0.0, np.nan, 2.0]},
    )

    with pytest.raises(ValueError, match="Cannot compute representative time step"):
        resample_to_uniform_time(data)
