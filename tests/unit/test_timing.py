"""Unit tests for confusius.timing."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.timing import (
    convert_time_reference,
    convert_time_values,
    get_representative_time_step,
    get_time_coord_to_seconds_factor,
)


def test_convert_time_values_rejects_unknown_units() -> None:
    """Time-value conversion delegates unknown-unit validation to the public helper."""
    with pytest.raises(ValueError, match="Unknown time unit"):
        convert_time_values([1.0, 2.0], "fortnight", raise_on_unknown=True)


def test_convert_time_values_supports_non_second_targets() -> None:
    """Time-value conversion can convert between non-native units."""
    converted = convert_time_values([100.0, 200.0], "ms", "s")

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


def test_convert_time_reference_rejects_invalid_reference() -> None:
    """Reference conversion raises a consistent ValueError for invalid names."""
    with pytest.raises(ValueError, match="from_reference"):
        convert_time_reference(
            np.array([0.0, 1.0]),
            volume_duration=0.5,
            from_reference="invalid",
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
