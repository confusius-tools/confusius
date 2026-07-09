import numpy as np
import pytest
import xarray as xr

from confusius.validation import validate_matching_spatial_units


def test_matching_spatial_units_ignores_missing_coords() -> None:
    left = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"), coords={"x": [0.0, 1.0]})
    right = xr.DataArray(
        np.zeros((2, 2)),
        dims=("y", "x"),
        coords={
            "y": xr.Variable("y", [0.0, 1.0], attrs={"units": "mm"}),
            "x": xr.Variable("x", [0.0, 1.0], attrs={"units": "mm"}),
        },
    )
    validate_matching_spatial_units((("left", left), ("right", right)))


def test_matching_spatial_units_raises_on_mismatch() -> None:
    left = xr.DataArray(
        np.zeros((2, 2)),
        dims=("y", "x"),
        coords={
            "y": xr.Variable("y", [0.0, 1.0], attrs={"units": "mm"}),
            "x": xr.Variable("x", [0.0, 1.0], attrs={"units": "mm"}),
        },
    )
    right = xr.DataArray(
        np.zeros((2, 2)),
        dims=("y", "x"),
        coords={
            "y": xr.Variable("y", [0.0, 1.0], attrs={"units": "um"}),
            "x": xr.Variable("x", [0.0, 1.0], attrs={"units": "um"}),
        },
    )
    with pytest.raises(ValueError, match=r"dimension 'y'.*mm.*um"):
        validate_matching_spatial_units((("left", left), ("right", right)))
