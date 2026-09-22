"""Tests for IQ-related VoxelData validation options."""

import numpy as np
import pytest
import xarray as xr

from confusius.validation import validate_voxeldata
from confusius.xarray import create_voxeldata


class TestValidateVoxeldataVelocityAttrs:
    """Tests for `validate_voxeldata(require_velocity_attrs=True)`."""

    @pytest.fixture
    def valid_iq_dataarray(self) -> xr.DataArray:
        """Create a valid VoxelData array with velocity metadata."""
        return create_voxeldata(
            np.ones((10, 4, 6, 8), dtype=np.complex64),
            dims=("time", "k", "j", "i"),
            time=xr.DataArray(np.arange(10) * 0.1, dims=("time",), attrs={"units": "s"}),
            attrs={
                "transmit_frequency": 15.625e6,
                "beamforming_sound_velocity": 1540.0,
            },
            spacing=(0.1, 0.05, 0.05),
            origin=(0.0, 0.0, 0.0),
        )

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
            validate_voxeldata(iq, require_velocity_attrs=True)

    def test_require_attrs_false_skips_attribute_validation(
        self, valid_iq_dataarray: xr.DataArray
    ) -> None:
        """`require_velocity_attrs=False` skips attribute validation."""
        iq = valid_iq_dataarray.copy()
        del iq.attrs["transmit_frequency"]

        validate_voxeldata(iq, require_velocity_attrs=False)

    def test_require_dtype_rejects_non_complex_data(
        self, valid_iq_dataarray: xr.DataArray
    ) -> None:
        """`require_dtype` validates the DataArray dtype."""
        iq = valid_iq_dataarray.real

        with pytest.raises(TypeError, match="Expected data dtype compatible"):
            validate_voxeldata(iq, require_dtype=np.complexfloating)

    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, "bad"])
    def test_velocity_attrs_must_be_positive_finite(
        self, valid_iq_dataarray: xr.DataArray, value: float | str
    ) -> None:
        """Velocity attributes must be positive finite numbers."""
        iq = valid_iq_dataarray.copy()
        iq.attrs["transmit_frequency"] = value

        with pytest.raises(ValueError, match="transmit_frequency must be positive and finite"):
            validate_voxeldata(iq, require_velocity_attrs=True)
