"""Tests for IQ data validation utilities."""

import numpy as np
import pytest
import xarray as xr

from confusius.validation import validate_iq_dataarray


class TestValidateIqDataArray:
    """Tests for `validate_iq_dataarray`."""

    @pytest.fixture
    def valid_iq_dataarray(self) -> xr.DataArray:
        """Create a valid IQ DataArray with all required attributes."""
        return xr.DataArray(
            np.ones((10, 4, 6, 8), dtype=np.complex64),
            dims=("time", "z", "y", "x"),
            coords={
                "time": np.arange(10),
                "z": np.arange(4),
                "y": np.arange(6),
                "x": np.arange(8),
            },
            attrs={
                "transmit_frequency": 15.625e6,
                "beamforming_sound_velocity": 1540.0,
            },
        )

    def test_wrong_dimensions_raises(self, valid_iq_dataarray: xr.DataArray) -> None:
        """DataArray with wrong dimensions raises `ValueError`."""
        iq = valid_iq_dataarray.rename({"time": "t"})

        with pytest.raises(ValueError, match="must have a 'time' dimension"):
            validate_iq_dataarray(iq)

    def test_missing_coordinates_raises(self) -> None:
        """Missing required coordinates raises `ValueError`."""
        iq = xr.DataArray(
            np.ones((10, 4, 6, 8), dtype=np.complex64),
            dims=("time", "z", "y", "x"),
            coords={
                "time": np.arange(10),
                "z": np.arange(4),
                "y": np.arange(6),
            },
            attrs={
                "transmit_frequency": 15.625e6,
                "beamforming_sound_velocity": 1540.0,
            },
        )

        with pytest.raises(ValueError, match="Missing required coordinate"):
            validate_iq_dataarray(iq)

    def test_non_complex_data_raises(self, valid_iq_dataarray: xr.DataArray) -> None:
        """Non-complex IQ data raises `TypeError`."""
        iq = valid_iq_dataarray.real

        with pytest.raises(TypeError, match="Expected complex-valued data"):
            validate_iq_dataarray(iq)

    @pytest.mark.parametrize(
        "missing_attr",
        [
            "transmit_frequency",
            "beamforming_sound_velocity",
        ],
    )
    def test_missing_required_attribute_raises(
        self, valid_iq_dataarray: xr.DataArray, missing_attr: str
    ) -> None:
        """Missing any required attribute raises `ValueError`."""
        iq = valid_iq_dataarray.copy()
        del iq.attrs[missing_attr]

        with pytest.raises(ValueError, match="Missing required DataArray attributes"):
            validate_iq_dataarray(iq, require_attrs=True)

    def test_require_attrs_false_skips_attribute_validation(
        self, valid_iq_dataarray: xr.DataArray
    ) -> None:
        """`require_attrs=False` skips attribute validation."""
        iq = valid_iq_dataarray.copy()
        del iq.attrs["transmit_frequency"]

        validate_iq_dataarray(iq, require_attrs=False)

    def test_multiple_missing_attributes_in_error_message(
        self, valid_iq_dataarray: xr.DataArray
    ) -> None:
        """Error message lists all missing attributes."""
        iq = valid_iq_dataarray.copy()
        del iq.attrs["transmit_frequency"]
        del iq.attrs["beamforming_sound_velocity"]

        with pytest.raises(ValueError) as exc_info:
            validate_iq_dataarray(iq, require_attrs=True)

        error_msg = str(exc_info.value)
        assert "transmit_frequency" in error_msg
        assert "beamforming_sound_velocity" in error_msg
