import numpy as np
import pytest
import xarray as xr

from confusius.validation import validate_matching_spatial_units
from confusius.xarray import create_voxeldata


def _make_voxeldata() -> xr.DataArray:
    return create_voxeldata(
        np.zeros((2, 2, 2)), dims=("k", "j", "i"), spacing=(1.0, 1.0, 1.0)
    )


def test_matching_spatial_units_passes_when_equal() -> None:
    left = _make_voxeldata()
    right = _make_voxeldata()
    validate_matching_spatial_units((("left", left), ("right", right)))


def test_matching_spatial_units_raises_when_lacking_geometry() -> None:
    left = _make_voxeldata()
    right = left.drop_vars(("z", "y", "x"))

    with pytest.raises(ValueError, match="voxel-to-world index"):
        validate_matching_spatial_units((("left", left), ("right", right)))


def test_matching_spatial_units_raises_on_mismatch() -> None:
    left = _make_voxeldata()
    right = _make_voxeldata().fusi.affine.set_units("um")

    with pytest.raises(ValueError, match=r"left='mm'.*right='um'"):
        validate_matching_spatial_units((("left", left), ("right", right)))
