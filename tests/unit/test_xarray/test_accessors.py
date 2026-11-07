"""Unit tests for xarray accessor."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

import confusius  # noqa: F401  # Import to register accessor.
from confusius._utils.geometry import (
    attach_voxel_to_world_index,
)


def _make_voxel_to_world_volume() -> xr.DataArray:
    """Create a small voxel-to-world volume for accessor tests."""
    base = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=["k", "j", "i"],
        coords={
            "k": [0, 1],
            "j": [0, 2, 4],
            "i": [0, 1, 2, 3],
        },
    )
    return attach_voxel_to_world_index(
        base,
        np.array(
            [
                [2.0, 1.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [0.0, 0.0, 4.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


class TestFUSIAccessor:
    """Tests for the fusi accessor."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataArray for testing."""
        return xr.DataArray(
            [1, 10, 100, 1000],
            dims=["x"],
            coords={"x": [0, 1, 2, 3]},
            attrs={"description": "test data"},
        )

    def test_accessor_is_registered(self, sample_data):
        """Accessor is available on DataArrays."""
        assert hasattr(sample_data, "fusi")

    def test_scale_accessor_is_available(self, sample_data):
        """Scale accessor is available as a property."""
        assert hasattr(sample_data.fusi, "scale")
        from confusius.xarray.scale import FUSIScaleAccessor

        assert isinstance(sample_data.fusi.scale, FUSIScaleAccessor)

    def test_extract_accessor_is_available(self, sample_data):
        """Extract accessor is available as a property."""
        from confusius.xarray.extract import FUSIExtractAccessor

        assert isinstance(sample_data.fusi.extract, FUSIExtractAccessor)

    def test_save_dispatches_to_io_loadsave(self, sample_data, tmp_path):
        """save delegates to `confusius.io.loadsave.save`."""
        out = tmp_path / "recording.nii.gz"

        with patch("confusius.io.loadsave.save") as mock_save:
            sample_data.fusi.save(out)

        mock_save.assert_called_once_with(sample_data, out)

    def test_db_scale_factor_20(self, sample_data):
        """db_scale with factor=20 (amplitude)."""
        result = sample_data.fusi.scale.db(factor=20)

        expected = np.array([-60.0, -40.0, -20.0, 0.0])
        np.testing.assert_allclose(result.values, expected)

        assert result.dims == sample_data.dims
        np.testing.assert_array_equal(result.coords["x"], sample_data.coords["x"])

        assert result.attrs["units"] == "dB"
        assert "scaling" in result.attrs

    def test_db_scale_factor_10(self, sample_data):
        """db_scale with factor=10 (power, default)."""
        result = sample_data.fusi.scale.db(factor=10)

        expected = np.array([-30.0, -20.0, -10.0, 0.0])
        np.testing.assert_allclose(result.values, expected)

    def test_db_scale_default_factor(self, sample_data):
        """db_scale uses factor=10 by default."""
        result = sample_data.fusi.scale.db()

        expected = np.array([-30.0, -20.0, -10.0, 0.0])
        np.testing.assert_allclose(result.values, expected)

    def test_db_scale_with_complex_data(self):
        """db_scale handles complex data correctly."""
        data = xr.DataArray([1 + 0j, 3 + 4j, 0 + 5j])
        result = data.fusi.scale.db(factor=20)

        # Magnitude: [1, 5, 5], max=5.
        expected_magnitudes = np.array([1.0, 5.0, 5.0])
        expected_db = 20 * np.log10(expected_magnitudes / 5.0)

        np.testing.assert_allclose(result.values, expected_db)

    def test_db_scale_default_factor_complex_data(self):
        """db_scale defaults to factor=20 for complex-valued data."""
        data = xr.DataArray([1 + 0j, 3 + 4j, 0 + 5j])
        result = data.fusi.scale.db()

        expected_magnitudes = np.array([1.0, 5.0, 5.0])
        expected_db = 20 * np.log10(expected_magnitudes / 5.0)

        np.testing.assert_allclose(result.values, expected_db)

    def test_db_scale_with_zero(self):
        """db_scale handles zeros (produces -inf)."""
        data = xr.DataArray([0, 1, 10])
        result = data.fusi.scale.db(factor=20)

        assert result.values[0] == -np.inf
        assert result.values[-1] == 0.0

    def test_log_scale(self, sample_data):
        """log_scale applies natural logarithm."""
        result = sample_data.fusi.scale.log()

        expected = np.log([1, 10, 100, 1000])
        np.testing.assert_allclose(result.values, expected)

        assert result.dims == sample_data.dims
        assert "scaling" in result.attrs

    def test_log_scale_with_zero(self):
        """log_scale handles zeros (produces -inf)."""
        data = xr.DataArray([0, 1, np.e])
        result = data.fusi.scale.log()

        assert result.values[0] == -np.inf
        assert np.isclose(result.values[1], 0.0)
        assert np.isclose(result.values[2], 1.0)

    def test_power_scale_sqrt(self, sample_data):
        """power_scale with default exponent=0.5 (square root)."""
        result = sample_data.fusi.scale.power()

        expected = np.sqrt([1, 10, 100, 1000])
        np.testing.assert_allclose(result.values, expected)

    def test_power_scale_square(self, sample_data):
        """power_scale with exponent=2 (square)."""
        result = sample_data.fusi.scale.power(exponent=2.0)

        expected = np.array([1, 100, 10000, 1000000])
        np.testing.assert_allclose(result.values, expected)

    def test_power_scale_preserves_metadata(self, sample_data):
        """power_scale preserves coordinates and updates attributes."""
        result = sample_data.fusi.scale.power(exponent=0.5)

        assert result.dims == sample_data.dims
        np.testing.assert_array_equal(result.coords["x"], sample_data.coords["x"])

        assert "scaling" in result.attrs
        assert "0.5" in result.attrs["scaling"]

    def test_power_scale_with_complex_data(self):
        """power_scale uses absolute value for complex data."""
        data = xr.DataArray([1 + 0j, 3 + 4j, 0 + 5j])
        result = data.fusi.scale.power(exponent=2.0)

        # Magnitudes: [1, 5, 5], squared: [1, 25, 25].
        expected = np.array([1.0, 25.0, 25.0])
        np.testing.assert_allclose(result.values, expected)

    def test_chained_operations(self, sample_data):
        """Multiple accessor operations can be chained."""
        result = sample_data.fusi.scale.power(exponent=0.5).fusi.scale.db(factor=20)

        # First sqrt: [1, 3.16, 10, 31.62].
        # Then dB relative to max (31.62).
        sqrt_vals = np.sqrt(sample_data.values)
        expected_db = 20 * np.log10(sqrt_vals / sqrt_vals.max())

        np.testing.assert_allclose(result.values, expected_db)
