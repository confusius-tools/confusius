"""Tests for shared mask dispatch helpers (VoxelData or extracted signals)."""

import numpy as np
import pytest
import xarray as xr

from confusius._utils.mask import (
    select_masked_features,
    validate_spatial_or_feature_mask,
)


def test_select_masked_features_accepts_existing_space_dim():
    """Reduced `(time, space)` signals are selected without restacking `space`."""
    data = xr.DataArray(
        np.arange(5 * 4).reshape(5, 4),
        dims=("time", "space"),
        coords={"space": ["a", "b", "c", "d"]},
    )
    mask = xr.DataArray(
        [True, False, True, False],
        dims=("space",),
        coords={"space": ["a", "b", "c", "d"]},
    )

    result = select_masked_features(data, mask)

    assert result.dims == ("time", "space")
    np.testing.assert_array_equal(result.coords["space"].values, ["a", "c"])
    np.testing.assert_array_equal(result.values, data.values[:, [0, 2]])


class TestValidateSpatialOrFeatureMask:
    """Tests for validate_spatial_or_feature_mask's non-VoxelData (feature) path."""

    def test_mask_dim_missing_from_data_raises(self):
        """A mask dim absent from data raises ValueError, not a silent mismatch."""
        data = xr.DataArray(
            np.zeros((5, 3)),
            dims=("time", "region"),
            coords={"region": np.arange(3)},
        )
        mask = xr.DataArray(
            np.array([True, False, True]),
            dims=("other_region",),
        )

        with pytest.raises(ValueError, match="Data is missing dimensions"):
            validate_spatial_or_feature_mask(data, mask)

    def test_accepts_binary_numeric_mask(self):
        """A binary numeric mask (e.g. a float NIfTI-style mask) is coerced to bool.

        Exercises the already-extracted (non-VoxelData) feature path, which routes
        through the same `check_mask_dtype` choke point as the VoxelData path -- see
        #382.
        """
        data = xr.DataArray(
            np.zeros((5, 3)),
            dims=("time", "region"),
            coords={"region": np.arange(3)},
        )
        mask = xr.DataArray(
            np.array([0.0, 5.0, 0.0]),
            dims=("region",),
            coords={"region": np.arange(3)},
        )

        result = validate_spatial_or_feature_mask(data, mask)

        assert result.dtype == bool
        np.testing.assert_array_equal(result.values, [False, True, False])

    def test_require_exact_dims_rejects_mismatched_order(self):
        """require_exact_dims rejects a mask whose dims don't match data's exactly."""
        data = xr.DataArray(
            np.zeros((5, 3, 2)),
            dims=("time", "region", "extra"),
            coords={"region": np.arange(3), "extra": np.arange(2)},
        )
        mask = xr.DataArray(
            np.ones((2, 3), dtype=bool),
            dims=("extra", "region"),
            coords={"region": np.arange(3), "extra": np.arange(2)},
        )

        with pytest.raises(
            ValueError, match="dimensions must match all non-time dimensions"
        ):
            validate_spatial_or_feature_mask(data, mask, require_exact_dims=True)
