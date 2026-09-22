"""Tests for multipose timing helpers."""

import numpy as np
import pytest
import xarray as xr

from confusius.multipose.timing import (
    build_consolidated_time_coordinate,
    consolidate_time_coordinate,
)


def _time_coord(values, **attrs):
    return xr.DataArray(values, dims=["time"], attrs=attrs)


class TestBuildConsolidatedTimeCoordinate:
    """Tests for build_consolidated_time_coordinate."""

    def test_missing_duration_warns_and_keeps_original(self):
        """Missing `volume_acquisition_duration` falls back to `time_coord` as-is."""
        time_coord = _time_coord([0.0, 1.0], units="s")
        slice_time_values = np.array([[0.0, 0.1], [1.0, 1.1]])

        with pytest.warns(UserWarning, match="Cannot infer whole-array timing"):
            result = build_consolidated_time_coordinate(
                time_coord, slice_time_values, {"units": "s"}
            )

        assert result is time_coord

    def test_non_positive_duration_warns_and_keeps_original(self):
        """A non-positive `volume_acquisition_duration` is treated as unusable."""
        time_coord = _time_coord([0.0, 1.0], units="s")
        slice_time_values = np.array([[0.0, 0.1], [1.0, 1.1]])

        with pytest.warns(UserWarning, match="Cannot infer whole-array timing"):
            result = build_consolidated_time_coordinate(
                time_coord,
                slice_time_values,
                {"units": "s", "volume_acquisition_duration": 0.0},
            )

        assert result is time_coord

    def test_varying_durations_warns_and_omits_duration_attr(self):
        """Non-constant inferred whole-array durations drop the duration attr."""
        time_coord = _time_coord(
            [0.0, 1.0], units="s", volume_acquisition_reference="start"
        )
        # Pose spread differs between time points (0.1 vs 0.5), so inferred
        # whole-array duration (spread + slice_duration) isn't constant.
        slice_time_values = np.array([[0.0, 0.1], [1.0, 1.5]])

        with pytest.warns(UserWarning, match="durations vary across time points"):
            result = build_consolidated_time_coordinate(
                time_coord,
                slice_time_values,
                {
                    "units": "s",
                    "volume_acquisition_reference": "start",
                    "volume_acquisition_duration": 0.2,
                },
            )

        assert "volume_acquisition_duration" not in result.attrs


class TestConsolidateTimeCoordinate:
    """Tests for consolidate_time_coordinate."""

    def test_1d_coordinate_is_returned_unchanged(self):
        time_coord = _time_coord(np.arange(3.0), units="s")
        assert consolidate_time_coordinate(time_coord) is time_coord

    def test_pose_dependent_coordinate_becomes_whole_volume_time(self):
        # Slices of 0.25 s at t and t + 0.25 (center reference): the volume spans
        # [t - 0.125, t + 0.375], so its center is t + 0.125.
        values = np.stack([np.arange(3.0), np.arange(3.0) + 0.25], axis=1)
        time_coord = xr.DataArray(
            values,
            dims=["time", "pose"],
            attrs={
                "volume_acquisition_duration": 0.25,
                "volume_acquisition_reference": "center",
            },
        )
        result = consolidate_time_coordinate(time_coord)
        assert result.dims == ("time",)
        np.testing.assert_allclose(result.values, np.arange(3.0) + 0.125)
        assert result.attrs["volume_acquisition_duration"] == 0.5

    def test_rejects_other_dimensions(self):
        with pytest.raises(ValueError, match="dimensions"):
            consolidate_time_coordinate(
                xr.DataArray(np.zeros((2, 3)), dims=["pose", "time"])
            )
