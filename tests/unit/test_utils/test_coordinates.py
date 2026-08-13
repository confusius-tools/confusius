"""Unit tests for confusius._utils.coordinates affine helpers."""

import numpy as np
import pytest
import xarray as xr

from confusius._utils.coordinates import (
    get_axis_aligned_affine,
    get_grid_info_from_dataarray,
)


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
    """Singleton dimensions need explicit `voxdim` metadata."""
    data = xr.DataArray(
        np.zeros((1, 3, 4)),
        dims=("z", "y", "x"),
        coords={"z": [0.0], "y": [0.0, 0.2, 0.4], "x": [0.0, 0.1, 0.2, 0.3]},
    )

    with pytest.warns(UserWarning, match="spacing is undefined"):
        with pytest.raises(ValueError, match="spacing is undefined.*z"):
            get_grid_info_from_dataarray(data)


def test_get_grid_info_requires_regular_spacing_for_voxel_to_world_dataarray():
    """Irregular voxel-space coordinates on voxel-to-world data raise, like plain data."""
    import confusius  # noqa: F401

    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=["j", "i"],
        coords={"j": np.arange(3.0), "i": np.arange(4.0)},
    )
    data = data.fusi.affine.set_voxel_to_world(np.eye(3))
    data = data.assign_coords(j=[0.0, 1.0, 3.5])

    with pytest.raises(ValueError, match="spacing is undefined.*j"):
        get_grid_info_from_dataarray(data)
