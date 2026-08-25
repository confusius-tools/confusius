"""Tests for shared multipose helpers."""

import numpy as np
import pytest
import xarray as xr

from confusius.multipose._utils import build_consolidated_time_coordinate


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
