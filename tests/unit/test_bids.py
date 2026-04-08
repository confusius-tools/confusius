"""Tests for the confusius.bids module."""

from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from confusius import bids


class TestCaseConversion:
    """Test bidirectional case conversion between ConfUSIus and fUSI-BIDS."""

    def test_snake_to_pascal_basic(self):
        """Test basic snake_case to PascalCase conversion."""
        attrs = {
            "repetition_time": 1.5,
            "task_name": "rest",
            "manufacturer": "Verasonics",
        }
        result = bids.to_bids(attrs)

        assert result["RepetitionTime"] == 1.5
        assert result["TaskName"] == "rest"
        assert result["Manufacturer"] == "Verasonics"

    def test_pascal_to_snake_basic(self):
        """Test basic PascalCase to snake_case conversion."""
        bids_attrs = {
            "RepetitionTime": 1.5,
            "TaskName": "rest",
            "Manufacturer": "Verasonics",
        }
        result = bids.from_bids(bids_attrs)

        assert result["repetition_time"] == 1.5
        assert result["task_name"] == "rest"
        assert result["manufacturer"] == "Verasonics"

    def test_round_trip(self):
        """Test that conversion round-trips correctly."""
        original = {
            "repetition_time": 1.5,
            "task_name": "rest",
            "probe_central_frequency": 15.0,
        }

        bids_format = bids.to_bids(original)
        back_to_confusius = bids.from_bids(bids_format)

        assert back_to_confusius == original

    def test_unknown_fields_preserved(self):
        """Test that unknown fields are preserved as-is."""
        attrs = {
            "repetition_time": 1.5,
            "custom_vendor_field": 123,
            "my_custom_metadata": "value",
        }
        result = bids.to_bids(attrs)

        # Known field converted
        assert result["RepetitionTime"] == 1.5
        # Unknown fields preserved
        assert result["custom_vendor_field"] == 123
        assert result["my_custom_metadata"] == "value"

    def test_internal_attributes_prefixed(self):
        """Test that internal attributes get ConfUSIus prefix."""
        attrs = {
            "repetition_time": 1.5,
            "sform_code": 1,
            "qform_code": 2,
            "affines": {"some": "value"},
        }
        result = bids.to_bids(attrs)

        assert result["RepetitionTime"] == 1.5
        assert result["ConfUSIusSformCode"] == 1
        assert result["ConfUSIusQformCode"] == 2
        assert result["ConfUSIusAffines"] == {"some": "value"}

    def test_explicit_field_mappings(self):
        """Test fields with explicit mappings in both directions."""
        attrs: dict[str, object] = {
            "transmit_frequency": 15e6,
            "pulse_repetition_frequency": 2_500.0,
        }

        bids_attrs = bids.to_bids(attrs)

        assert bids_attrs["UltrasoundTransmitFrequency"] == 15e6
        assert bids_attrs["UltrasoundPulseRepetitionFrequency"] == 2_500.0

        roundtripped = bids.from_bids(bids_attrs)

        assert roundtripped == attrs

    def test_clutter_filter_window_metadata_mapping(self):
        """Clutter-filter timing metadata uses canonical BIDS field names."""
        attrs = {
            "clutter_filter_window_duration": 0.6,
            "clutter_filter_window_stride": 0.3,
        }

        bids_attrs = bids.to_bids(attrs)

        assert bids_attrs["ClutterFilterWindowDuration"] == 0.6
        assert bids_attrs["ClutterFilterWindowStride"] == 0.3
        assert bids.from_bids(bids_attrs) == attrs

    def test_integration_stride_metadata_mapping(self):
        """Integration stride metadata maps to expected BIDS and ConfUSIus keys."""
        attrs = {
            "power_doppler_integration_stride": 0.2,
            "axial_velocity_integration_stride": 0.25,
            "bmode_integration_stride": 0.3,
        }

        bids_attrs = bids.to_bids(attrs)

        assert bids_attrs["PowerDopplerIntegrationStride"] == 0.2
        assert bids_attrs["ConfUSIusAxialVelocityIntegrationStride"] == 0.25
        assert bids_attrs["ConfUSIusBmodeIntegrationStride"] == 0.3
        assert bids.from_bids(bids_attrs) == attrs

    def test_from_bids_restores_internal_attributes(self):
        """Test that ConfUSIus-prefixed internal fields restore their original names."""
        bids_attrs = {
            "ConfUSIusSformCode": 1,
            "ConfUSIusQformCode": 2,
            "ConfUSIusAffines": {"some": "value"},
        }

        result = bids.from_bids(bids_attrs)

        assert result == {
            "sform_code": 1,
            "qform_code": 2,
            "affines": {"some": "value"},
        }

    def test_from_bids_unknown_fields_preserved(self):
        """Test that unknown BIDS fields are preserved as-is when loading."""
        bids_attrs = {
            "TaskName": "rest",
            "CustomField": 123,
        }

        result = bids.from_bids(bids_attrs)

        assert result["task_name"] == "rest"
        assert result["CustomField"] == 123


class TestValidation:
    """Test fUSI-BIDS metadata validation."""

    def test_valid_metadata(self):
        """Test validation of valid metadata."""
        metadata = {
            "TaskName": "rest",
            "RepetitionTime": 1.5,
            "Manufacturer": "Verasonics",
        }
        result = bids.validate_metadata(metadata)

        assert result.TaskName == "rest"
        assert result.RepetitionTime == 1.5
        assert result.Manufacturer == "Verasonics"

    def test_missing_timing_warning(self):
        """Test that missing timing fields trigger a warning when time-related fields are present."""
        # Provide time-related fields but no RepetitionTime or VolumeTiming.
        with pytest.warns(UserWarning, match="RepetitionTime or VolumeTiming"):
            bids.validate_metadata(
                {
                    "TaskName": "rest",
                    "SliceTiming": [0.0, 0.1],
                    "SliceEncodingDirection": "k",
                }
            )

    def test_invalid_repetition_time(self):
        """Test that invalid RepetitionTime raises error."""
        with pytest.raises(ValidationError, match="greater than 0"):
            bids.validate_metadata({"RepetitionTime": -1.0})

    def test_repetition_time_and_volume_timing_mutually_exclusive(self):
        """Test that RepetitionTime and VolumeTiming cannot both be provided."""
        with pytest.raises(ValidationError, match="mutually exclusive"):
            bids.validate_metadata(
                {
                    "RepetitionTime": 1.0,
                    "VolumeTiming": [0.0, 1.0, 2.0],
                    "FrameAcquisitionDuration": 0.5,
                }
            )

    def test_volume_timing_requires_frame_acquisition_duration(self):
        """Test that VolumeTiming requires FrameAcquisitionDuration without SliceTiming."""
        with pytest.raises(
            ValidationError, match="FrameAcquisitionDuration is REQUIRED"
        ):
            bids.validate_metadata({"VolumeTiming": [0.0, 1.0, 2.0]})

    def test_repetition_time_forbids_frame_acquisition_duration(self):
        """Test that RepetitionTime and FrameAcquisitionDuration cannot coexist."""
        with pytest.raises(
            ValidationError,
            match="FrameAcquisitionDuration must not be provided",
        ):
            bids.validate_metadata(
                {
                    "RepetitionTime": 1.0,
                    "FrameAcquisitionDuration": 0.5,
                }
            )

    def test_slice_timing_without_encoding_direction_warns(self):
        """Test that SliceTiming without SliceEncodingDirection warns."""
        with pytest.warns(UserWarning, match="SliceEncodingDirection is missing"):
            bids.validate_metadata(
                {
                    "SliceTiming": [0.0, 0.1],
                    "RepetitionTime": 1.0,
                }
            )

    def test_numpy_array_conversion(self):
        """Test that numpy arrays are converted to lists."""
        metadata = {
            "VolumeTiming": np.array([0.0, 1.5, 3.0]),
            "FrameAcquisitionDuration": 4.5,
        }
        result = bids.validate_metadata(metadata)

        assert isinstance(result.VolumeTiming, list)
        assert result.VolumeTiming == [0.0, 1.5, 3.0]

    def test_format_validation_error(self):
        """Test that format_validation_error produces clean messages."""
        from confusius.bids.validation import format_validation_error

        try:
            bids.validate_metadata({"VolumeTiming": [0.0, 1.0, 2.0]})
        except ValidationError as e:
            formatted = format_validation_error(e)

            # Should contain the error message
            assert "FrameAcquisitionDuration is REQUIRED" in formatted
            assert "__root__" not in formatted
            # Should NOT contain verbose pydantic details
            assert "type=value_error" not in formatted
            assert "input_value=" not in formatted
            assert "errors.pydantic.dev" not in formatted

    def test_format_validation_error_includes_field_name(self):
        """Formatted validation errors include the failing field location."""
        from confusius.bids.validation import format_validation_error

        try:
            bids.validate_metadata({"RepetitionTime": 1.0, "DelayAfterTrigger": -0.1})
        except ValidationError as e:
            formatted = format_validation_error(e)

            assert (
                "DelayAfterTrigger: Input should be greater than or equal to 0"
                in formatted
            )


class TestSliceTimeCoordinate:
    """Test slice timing coordinate creation and extraction."""

    @pytest.mark.parametrize(
        ("slice_encoding_direction", "expected_dim", "expected_values"),
        [
            ("i", "x", [0.0, 0.1, 0.2]),
            ("j", "y", [0.0, 0.1, 0.2]),
            ("k", "z", [0.0, 0.1, 0.2]),
            ("i-", "x", [0.2, 0.1, 0.0]),
            ("j-", "y", [0.2, 0.1, 0.0]),
            ("k-", "z", [0.2, 0.1, 0.0]),
        ],
    )
    def test_create_slice_time_coordinate(
        self, slice_encoding_direction, expected_dim, expected_values
    ):
        """Valid slice encoding directions map to 2D absolute slice-time coordinates."""
        slice_timing = expected_values
        if slice_encoding_direction.endswith("-"):
            slice_timing = list(reversed(slice_timing))
        volume_times = np.array([10.0, 11.0])

        coord = bids.create_slice_time_coordinate_from_bids(
            volume_times=volume_times,
            slice_timing=slice_timing,
            slice_encoding_direction=slice_encoding_direction,
        )

        assert coord.dims == ("time", expected_dim)
        assert coord.shape == (2, 3)
        assert coord.attrs["units"] == "s"
        np.testing.assert_array_equal(
            coord.values,
            volume_times[:, np.newaxis] + np.asarray(expected_values)[np.newaxis, :],
        )

    def test_invalid_slice_encoding_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="Invalid SliceEncodingDirection"):
            bids.create_slice_time_coordinate_from_bids(
                volume_times=np.array([0.0]),
                slice_timing=np.array([0.0, 0.1]),
                slice_encoding_direction=cast(Any, "invalid"),
            )

    def test_slice_timing_must_be_1d(self):
        """Test that slice_timing input must be one-dimensional."""
        with pytest.raises(ValueError, match="SliceTiming must be 1D"):
            bids.create_slice_time_coordinate_from_bids(
                volume_times=np.array([0.0, 1.0]),
                slice_timing=np.array([[0.0, 0.1], [0.2, 0.3]]),
                slice_encoding_direction="k",
            )

    def test_volume_times_must_be_1d(self):
        """Test that volume_times input must be one-dimensional."""
        with pytest.raises(ValueError, match="volume_times must be 1D"):
            bids.create_slice_time_coordinate_from_bids(
                volume_times=np.array([[0.0], [1.0]]),
                slice_timing=np.array([0.0, 0.1]),
                slice_encoding_direction="k",
            )

    @pytest.mark.parametrize(
        ("slice_encoding_direction", "expected_direction"),
        [("i", "i"), ("j", "j"), ("k", "k")],
    )
    def test_extract_slice_timing(self, slice_encoding_direction, expected_direction):
        """Extracting slice timing preserves values and spatial direction."""
        slice_timing = np.array([0.0, 0.1, 0.2, 0.3])
        volume_times = np.array([0.0, 1.0, 2.0])
        coord = bids.create_slice_time_coordinate_from_bids(
            volume_times=volume_times,
            slice_timing=slice_timing,
            slice_encoding_direction=slice_encoding_direction,
        )

        extracted_timing, direction = bids.create_bids_slice_timing_from_coordinate(
            coord, volume_times
        )

        np.testing.assert_array_equal(extracted_timing, slice_timing)
        assert direction == expected_direction

    def test_extract_slice_timing_invalid_ndim(self):
        """Test extraction rejects coordinates that are not 2D."""
        coord = xr.DataArray(np.zeros(3), dims=["z"])

        with pytest.raises(ValueError, match="must be 2D"):
            bids.create_bids_slice_timing_from_coordinate(coord, np.array([0.0]))

    def test_extract_slice_timing_rejects_unknown_spatial_dimension(self):
        """Test extraction rejects unknown spatial dimensions."""
        coord = xr.DataArray(np.zeros((2, 3)), dims=["time", "phase"])

        with pytest.raises(ValueError, match="must have one of spatial dimensions"):
            bids.create_bids_slice_timing_from_coordinate(coord, np.array([0.0, 1.0]))

    def test_extract_slice_timing_rejects_inconsistent_relative_timings(self):
        """Extraction rejects 2D absolute slice_time when relative timing varies."""
        coord = xr.DataArray(
            [[0.0, 0.1, 0.2], [1.0, 1.1, 1.25]],
            dims=["time", "z"],
        )

        with pytest.raises(ValueError, match="varies across time points"):
            bids.create_bids_slice_timing_from_coordinate(coord, np.array([0.0, 1.0]))
