"""Unit tests for confusius._utils.coordinates affine helpers."""

import numpy as np
import pytest
import xarray as xr

from confusius._utils.coordinates import (
    get_axis_aligned_affine,
    get_grid_info_from_dataarray,
)
from confusius.xarray import create_voxeldata


def test_axis_aligned_affine_builds_diag_and_translation():
    """get_axis_aligned_affine places zoom on the diagonal and translation last."""
    A = get_axis_aligned_affine(np.array([10.0, 20.0, 30.0]), np.array([2.0, 3.0, 4.0]))
    expected = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(A, expected)


def test_get_grid_info_requires_singleton_spacing():
    """Singleton non-spatial dimensions need explicit spacing metadata."""
    data = create_voxeldata(
        np.zeros((1, 2, 3, 4)),
        dims=("component", "k", "j", "i"),
        extra_coords={"component": [0]},
        spacing=(0.1, 0.2, 0.3),
        origin=(0.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="spacing is undefined.*component"):
        get_grid_info_from_dataarray(data)


def test_get_grid_info_rejects_non_voxeldata():
    """A DataArray without a voxel-to-world index is not a valid grid source."""
    data = xr.DataArray(
        np.zeros((3, 4)), dims=("j", "i"), coords={"j": np.arange(3), "i": np.arange(4)}
    )

    with pytest.raises(ValueError):
        get_grid_info_from_dataarray(data)


def test_get_grid_info_requires_regular_spacing_for_voxel_to_world_dataarray():
    """Irregular voxel-space coordinates on voxel-to-world data raise, like plain data."""
    data = create_voxeldata(
        np.zeros((1, 3, 4)),
        dims=("k", "j", "i"),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    data = data.assign_coords(j=[0.0, 1.0, 3.5])

    with pytest.raises(ValueError, match="spacing is undefined.*j"):
        get_grid_info_from_dataarray(data)
