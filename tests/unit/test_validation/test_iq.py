"""Tests for IQ data validation utilities."""

import numpy as np
import pytest
import xarray as xr

from confusius._utils.geometry import add_physical_coords_from_voxel_affine
from confusius.validation import validate_iq_dataarray


class TestValidateIqDataArray:
    """Tests for `validate_iq_dataarray`."""

    @pytest.fixture
    def valid_iq_dataarray(self) -> xr.DataArray:
        """Create a valid IQ DataArray with all required attributes."""
        base = xr.DataArray(
            np.ones((10, 4, 6, 8), dtype=np.complex64),
            dims=("time", "k", "j", "i"),
            coords={
                "time": xr.DataArray(
                    np.arange(10) * 0.1,
                    dims=("time",),
                    attrs={"units": "s"},
                ),
                "k": xr.DataArray(
                    np.arange(4),
                    dims=("k",),
                    attrs={"voxdim": 1.0},
                ),
                "j": xr.DataArray(
                    np.arange(6),
                    dims=("j",),
                    attrs={"voxdim": 1.0},
                ),
                "i": xr.DataArray(
                    np.arange(8),
                    dims=("i",),
                    attrs={"voxdim": 1.0},
                ),
            },
            attrs={
                "transmit_frequency": 15.625e6,
                "beamforming_sound_velocity": 1540.0,
            },
        )
        return add_physical_coords_from_voxel_affine(
            base,
            np.diag([0.1, 0.05, 0.05, 1.0]),
            voxel_dims=("k", "j", "i"),
            physical_coord_names=("z", "y", "x"),
            physical_coord_attrs={
                "z": {"units": "mm", "voxdim": 0.1},
                "y": {"units": "mm", "voxdim": 0.05},
                "x": {"units": "mm", "voxdim": 0.05},
            },
        )

    def test_wrong_dimensions_raises(self, valid_iq_dataarray: xr.DataArray) -> None:
        """DataArray with wrong dimensions raises `ValueError`."""
        iq = valid_iq_dataarray.rename({"time": "t"})

        with pytest.raises(ValueError, match="must have a 'time' dimension"):
            validate_iq_dataarray(iq)

    def test_missing_coordinates_raises(self, valid_iq_dataarray: xr.DataArray) -> None:
        """Missing required coordinates raises `ValueError`."""
        iq = valid_iq_dataarray.drop_vars("i")

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
