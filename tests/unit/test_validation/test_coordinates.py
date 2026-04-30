"""Tests for coordinate comparison and validation helpers."""

import numpy as np
import pytest
import xarray as xr

from confusius.validation import validate_mask, validate_matching_coordinates


def test_validate_matching_coordinates_accepts_numeric_drift():
    """Numeric coordinates match within tolerance."""
    left = xr.DataArray(np.arange(5), dims=["time"], coords={"time": np.arange(5) * 0.1})
    right = xr.DataArray(
        np.arange(5),
        dims=["time"],
        coords={"time": np.arange(5) * 0.1 + 1e-10},
    )

    validate_matching_coordinates(left, right, "time")


def test_validate_matching_coordinates_ignores_unrelated_attached_coords():
    """Attached scalar coordinates do not affect coordinate matching."""
    left = xr.DataArray(
        np.arange(5),
        dims=["time"],
        coords={"time": np.arange(5) * 0.1, "mask": "roi_a"},
    )
    right = xr.DataArray(
        np.arange(5), dims=["time"], coords={"time": np.arange(5) * 0.1}
    )

    validate_matching_coordinates(left, right, "time")


def test_validate_matching_coordinates_raises_on_mismatch():
    """Mismatched coordinate values raise a clear error."""
    left = xr.DataArray(np.arange(5), dims=["time"], coords={"time": np.arange(5)})
    right = xr.DataArray(np.arange(5), dims=["time"], coords={"time": np.arange(5) + 1})

    with pytest.raises(ValueError, match="Coordinate 'time' does not match"):
        validate_matching_coordinates(left, right, "time")


def test_validate_mask_accepts_scalar_attached_coordinate(sample_4d_volume):
    """Single selected masks validate even if they keep a scalar `mask` coord."""
    mask = xr.DataArray(
        np.zeros((2, *sample_4d_volume.shape[1:]), dtype=int),
        dims=["mask", "z", "y", "x"],
        coords={
            "mask": ["roi_a", "roi_b"],
            "z": sample_4d_volume.coords["z"],
            "y": sample_4d_volume.coords["y"],
            "x": sample_4d_volume.coords["x"],
        },
    )
    mask[0, 0, :, :] = 1

    validate_mask(mask.isel(mask=0), sample_4d_volume)
