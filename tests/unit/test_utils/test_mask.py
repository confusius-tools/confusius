"""Tests for shared mask dispatch helpers (VoxelData or extracted signals)."""

import numpy as np
import pytest
import xarray as xr

from confusius._utils.mask import validate_spatial_or_feature_mask


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
