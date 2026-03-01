"""Tests for extract.extract_with_mask and extract.unmask."""

import numpy as np
import pytest
import xarray as xr

from confusius import extract


class TestWithMask:
    """Tests for extract.extract_with_mask function."""

    def test_float_mask_rejected(self, sample_4d_volume):
        """Test that float mask raises TypeError."""
        mask = xr.DataArray(
            np.random.rand(*sample_4d_volume.shape[1:]),
            dims=["z", "y", "x"],
        )

        with pytest.raises(TypeError, match="single-label integer dtype"):
            extract.extract_with_mask(sample_4d_volume, mask)

    def test_multi_label_integer_mask_rejected(self, sample_4d_volume):
        """Test that integer mask with more than one non-zero value raises TypeError."""
        mask_data = np.zeros(sample_4d_volume.shape[1:], dtype=np.int32)
        mask_data[0, 0, 0] = 1
        mask_data[0, 0, 1] = 2
        mask = xr.DataArray(mask_data, dims=["z", "y", "x"])

        with pytest.raises(TypeError, match="2 distinct non-zero"):
            extract.extract_with_mask(sample_4d_volume, mask)

    def test_integer_mask_selects_nonzero_voxels(self):
        """Test that an integer mask (0 / region_id) selects the same voxels as the
        equivalent boolean mask."""
        data = xr.DataArray(
            np.arange(60).reshape(3, 4, 5),
            dims=["z", "y", "x"],
        )
        bool_mask_data = np.zeros((3, 4, 5), dtype=bool)
        bool_mask_data[0, 1, 2] = True
        bool_mask_data[1, 2, 3] = True
        bool_mask_data[2, 3, 4] = True

        bool_mask = xr.DataArray(bool_mask_data, dims=["z", "y", "x"])
        int_mask = xr.DataArray(
            bool_mask_data.astype(np.int32) * 385, dims=["z", "y", "x"]
        )

        result_bool = extract.extract_with_mask(data, bool_mask)
        result_int = extract.extract_with_mask(data, int_mask)

        np.testing.assert_array_equal(result_bool.values, result_int.values)

    def test_insufficient_spatial_dims(self):
        """Test that mask with dimension not in data raises error."""
        data = xr.DataArray(
            np.random.randn(10),
            dims=["time"],
        )

        mask = xr.DataArray([True, False, True], dims=["x"])

        with pytest.raises(ValueError, match="missing spatial dimensions.*'x'"):
            extract.extract_with_mask(data, mask)

    def test_values_correctness(self):
        """Test that extracted values match original data."""
        data = xr.DataArray(
            np.arange(60).reshape(3, 4, 5),
            dims=["z", "y", "x"],
        )

        mask_data = np.zeros((3, 4, 5), dtype=bool)
        mask_data[0, 1, 2] = True
        mask_data[1, 2, 3] = True
        mask_data[2, 3, 4] = True

        mask = xr.DataArray(mask_data, dims=["z", "y", "x"])

        signals = extract.extract_with_mask(data, mask)

        expected_values = [
            data.values[0, 1, 2],
            data.values[1, 2, 3],
            data.values[2, 3, 4],
        ]
        np.testing.assert_array_equal(signals.values, expected_values)


class TestUnmask:
    """Tests for extract.unmask function."""

    def test_basic_unmask_1d(self):
        """Test unmasking 1D signals to 3D."""
        mask_data = np.zeros((5, 6, 7), dtype=bool)
        mask_data.flat[:10] = True
        mask = xr.DataArray(
            mask_data,
            dims=["z", "y", "x"],
            coords={
                "z": np.arange(5) * 0.4,
                "y": np.arange(6) * 0.05,
                "x": np.arange(7) * 0.1,
            },
        )

        signals = np.arange(10) * 10.0

        result = extract.unmask(signals, mask)

        assert result.shape == (5, 6, 7)
        assert result.dims == ("z", "y", "x")

        np.testing.assert_array_equal(
            result.coords["z"].values, mask.coords["z"].values
        )

        assert result.values.flat[0] == 0.0
        assert result.values.flat[1] == 10.0

        assert result.values.flat[10] == 0.0

    def test_basic_unmask_2d(self):
        """Test unmasking 2D signals (features, voxels) to 4D."""
        mask_data = np.zeros((5, 6, 7), dtype=bool)
        mask_data.flat[:10] = True
        mask = xr.DataArray(mask_data, dims=["z", "y", "x"])

        signals = np.arange(30).reshape(3, 10) * 1.0

        result = extract.unmask(signals, mask, new_dims=["component"])

        assert result.shape == (3, 5, 6, 7)
        assert result.dims == ("component", "z", "y", "x")

        assert "component" in result.coords
        np.testing.assert_array_equal(result.coords["component"].values, [0, 1, 2])

        assert result.values[0, 0, 0, 0] == 0.0
        assert result.values[1, 0, 0, 1] == 11.0
        assert result.values[2, 0, 1, 2] == 29.0
        assert result.values[0, 3, 3, 3] == 0.0

    def test_shape_validation(self):
        """Test that shape mismatches raise errors."""
        mask_data = np.zeros((5, 6, 7), dtype=bool)
        mask_data.flat[:10] = True
        mask = xr.DataArray(mask_data, dims=["z", "y", "x"])

        signals = np.arange(5)

        with pytest.raises(ValueError, match="doesn't match"):
            extract.unmask(signals, mask)

    def test_fill_value(self):
        """Test custom fill value."""
        mask_data = np.zeros((3, 4, 5), dtype=bool)
        mask_data.flat[:5] = True
        mask = xr.DataArray(mask_data, dims=["z", "y", "x"])

        signals = np.arange(5) * 1.0

        result = extract.unmask(signals, mask, fill_value=np.nan)

        assert np.isnan(result.values.flat[5])


class TestRoundTrip:
    """Tests for extraction and unmasking round-trip."""

    def test_extract_unmask_roundtrip(self):
        """Test that extract -> unmask preserves values at masked positions."""
        data = xr.DataArray(
            np.random.randn(10, 5, 6, 7),
            dims=["time", "z", "y", "x"],
        )

        mask = xr.DataArray(
            np.random.rand(5, 6, 7) > 0.5,
            dims=["z", "y", "x"],
        )

        signals = extract.extract_with_mask(data, mask)

        restored = extract.unmask(signals.values, mask)

        mask_flat = mask.values.flatten()
        original_masked = data.values.reshape(10, -1)[:, mask_flat]
        restored_masked = restored.values.reshape(10, -1)[:, mask_flat]

        np.testing.assert_array_almost_equal(original_masked, restored_masked)

        non_mask_flat = ~mask_flat
        assert np.all(restored.values.reshape(10, -1)[:, non_mask_flat] == 0.0)
