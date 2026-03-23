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
        # Provide time-related fields but no RepetitionTime or VolumeTiming
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
        with pytest.raises(ValidationError):
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


class TestSliceTimeCoordinate:
    """Test slice timing coordinate creation and extraction."""

    def test_create_slice_time_z_direction(self):
        """Test creating slice_time for z encoding direction."""
        slice_timing = np.array([0.0, 0.1, 0.2, 0.3])
        coord = bids.create_slice_time_coordinate(
            slice_timing=slice_timing,
            n_time=10,
            slice_encoding_direction="k",
            spatial_shape=(4, 64, 64),
        )

        assert coord.dims == ("time", "z")
        assert coord.shape == (10, 4)
        assert coord.attrs["units"] == "s"
        np.testing.assert_array_equal(coord.values, np.tile(slice_timing, (10, 1)))

    def test_create_slice_time_y_direction(self):
        """Test creating slice_time for y encoding direction."""
        slice_timing = np.array([0.0, 0.1, 0.2])
        coord = bids.create_slice_time_coordinate(
            slice_timing=slice_timing,
            n_time=5,
            slice_encoding_direction="j",
            spatial_shape=(32, 3, 64),
        )

        assert coord.dims == ("time", "y")
        assert coord.shape == (5, 3)
        np.testing.assert_array_equal(coord.values, np.tile(slice_timing, (5, 1)))

    def test_invalid_slice_encoding_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="Invalid SliceEncodingDirection"):
            bids.create_slice_time_coordinate(
                slice_timing=np.array([0.0, 0.1]),
                n_time=5,
                slice_encoding_direction=cast(Any, "invalid"),
                spatial_shape=(2, 64, 64),
            )

    def test_slice_timing_must_be_1d(self):
        """Test that slice_timing input must be one-dimensional."""
        with pytest.raises(ValueError, match="SliceTiming must be 1D"):
            bids.create_slice_time_coordinate(
                slice_timing=np.array([[0.0, 0.1], [0.2, 0.3]]),
                n_time=5,
                slice_encoding_direction="k",
                spatial_shape=(2, 64, 64),
            )

    def test_mismatched_slice_count(self):
        """Test that mismatched slice count raises error."""
        with pytest.raises(ValueError, match="does not match"):
            bids.create_slice_time_coordinate(
                slice_timing=np.array([0.0, 0.1, 0.2]),  # 3 slices
                n_time=5,
                slice_encoding_direction="k",
                spatial_shape=(4, 64, 64),  # But z=4
            )

    def test_extract_slice_timing(self):
        """Test extracting slice timing from coordinate."""
        slice_timing = np.array([0.0, 0.1, 0.2, 0.3])
        coord = bids.create_slice_time_coordinate(
            slice_timing=slice_timing,
            n_time=10,
            slice_encoding_direction="k",
            spatial_shape=(4, 64, 64),
        )

        extracted_timing, direction = bids.extract_slice_timing_from_coordinate(coord)

        np.testing.assert_array_equal(extracted_timing, slice_timing)
        assert direction == "k"

    def test_extract_slice_timing_invalid_ndim(self):
        """Test extraction rejects coordinates without exactly two dimensions."""
        coord = xr.DataArray(np.array([0.0, 0.1, 0.2]), dims=["z"])

        with pytest.raises(ValueError, match="must have 2 dimensions"):
            bids.extract_slice_timing_from_coordinate(coord)

    def test_extract_slice_timing_requires_single_spatial_dim(self):
        """Test extraction rejects coordinates without a time dimension."""
        coord = xr.DataArray(np.zeros((3, 4)), dims=["z", "y"])

        with pytest.raises(ValueError, match="exactly one spatial dimension"):
            bids.extract_slice_timing_from_coordinate(coord)

    def test_extract_slice_timing_rejects_unknown_spatial_dimension(self):
        """Test extraction rejects unknown spatial dimensions."""
        coord = xr.DataArray(np.zeros((2, 3)), dims=["time", "phase"])

        with pytest.raises(ValueError, match="Unknown spatial dimension"):
            bids.extract_slice_timing_from_coordinate(coord)

    def test_extract_slice_timing_warns_when_it_varies_across_time(self):
        """Test extraction warns and uses first volume when timing varies."""
        coord = xr.DataArray(
            np.array(
                [
                    [0.0, 0.1, 0.2],
                    [0.0, 0.1, 0.25],
                ]
            ),
            dims=["time", "z"],
            attrs={"units": "s"},
        )

        with pytest.warns(UserWarning, match="varies across time points"):
            extracted_timing, direction = bids.extract_slice_timing_from_coordinate(
                coord
            )

        np.testing.assert_array_equal(extracted_timing, [0.0, 0.1, 0.2])
        assert direction == "k"


class TestIntegrationWithNifti:
    """Integration tests with NIfTI loading/saving."""

    def test_save_load_roundtrip(self, tmp_path):
        """Test that BIDS metadata round-trips through NIfTI save/load."""
        import confusius as cf

        # Create test data with BIDS metadata
        data = np.random.rand(5, 10, 10, 10).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=["time", "z", "y", "x"],
            coords={
                "time": xr.DataArray(
                    np.arange(5) * 1.5, dims=["time"], attrs={"units": "s"}
                ),
                "z": xr.DataArray(
                    np.arange(10) * 0.1, dims=["z"], attrs={"units": "mm"}
                ),
                "y": xr.DataArray(
                    np.arange(10) * 0.1, dims=["y"], attrs={"units": "mm"}
                ),
                "x": xr.DataArray(
                    np.arange(10) * 0.1, dims=["x"], attrs={"units": "mm"}
                ),
            },
            attrs={
                "task_name": "rest",
                "manufacturer": "Verasonics",
                "probe_central_frequency": 15.0,
            },
        )

        # Save and reload
        nifti_path = tmp_path / "test.nii.gz"
        cf.io.save_nifti(da, nifti_path)
        loaded = cf.io.load_nifti(nifti_path)

        # Check that BIDS metadata is preserved
        assert loaded.attrs["task_name"] == "rest"
        assert loaded.attrs["manufacturer"] == "Verasonics"
        assert loaded.attrs["probe_central_frequency"] == 15.0
