"""Tests for voxel-space geometry helpers."""

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from confusius._utils.geometry import (
    add_world_coords_from_voxel_affine,
    get_affine_axis_scalings,
    get_affine_axis_vectors,
    get_affine_orientation_matrix,
    get_affine_origin,
    get_voxel_world_origin,
    get_world_spacings,
)


def test_axis_aligned_voxel_affine_computes_correct_world_coords() -> None:
    """Axis-aligned voxel-affine geometry computes correct world coords.

    World coordinates are backed by a single joint index spanning all voxel
    dimensions (see [VoxelToWorldIndex][confusius._utils.geometry.VoxelToWorldIndex]
    for why), so each is `(k, j, i)`-shaped even though, for an axis-aligned affine,
    its value only actually varies along its own paired voxel dimension.
    """
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0.0, 2.0],
            "j": [0.0, 1.0, 3.0],
            "i": [0.0, 2.0, 3.0, 7.0],
        },
    )
    voxel_to_world = np.array(
        [
            [10.0, 0.0, 0.0, 100.0],
            [0.0, 2.0, 0.0, 200.0],
            [0.0, 0.0, 3.0, 300.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    result = add_world_coords_from_voxel_affine(
        data,
        voxel_to_world,
        voxel_dims=("k", "j", "i"),
        world_coord_names=("z", "y", "x"),
    )

    assert result.coords["z"].dims == ("k", "j", "i")
    assert result.coords["y"].dims == ("k", "j", "i")
    assert result.coords["x"].dims == ("k", "j", "i")
    # Each coordinate only varies along its own paired voxel dimension.
    z_by_k = result.coords["z"].isel(j=0, i=0).values
    y_by_j = result.coords["y"].isel(k=0, i=0).values
    x_by_i = result.coords["x"].isel(k=0, j=0).values
    assert_array_equal(z_by_k, [100.0, 120.0])
    assert_array_equal(y_by_j, [200.0, 202.0, 206.0])
    assert_array_equal(x_by_i, [300.0, 306.0, 309.0, 321.0])
    assert type(result.xindexes["x"]).__name__ == "VoxelToWorldIndex"


def test_axis_aligned_voxel_affine_uses_voxel_to_world_index() -> None:
    """Axis-aligned world coords are owned by a VoxelToWorldIndex."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0.0, 2.0],
            "j": [0.0, 1.0, 3.0],
            "i": [0.0, 2.0, 3.0, 7.0],
        },
    )
    voxel_to_world = np.array(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = add_world_coords_from_voxel_affine(
        data,
        voxel_to_world,
        voxel_dims=("k", "j", "i"),
        world_coord_names=("z", "y", "x"),
    )

    assert list(result.xindexes) == ["k", "j", "i", "z", "y", "x"]
    assert type(result.xindexes["z"]).__name__ == "VoxelToWorldIndex"
    result.stack(space=("k", "j", "i"))


def test_oblique_coordinate_transform_index_selection_uses_world_coords() -> None:
    """Oblique voxel-affine geometry still uses pointwise world selection."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0.0, 2.0],
            "j": [0.0, 1.0, 3.0],
            "i": [0.0, 2.0, 3.0, 7.0],
        },
    )
    voxel_to_world = np.array(
        [
            [1.0, 0.1, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = add_world_coords_from_voxel_affine(
        data,
        voxel_to_world,
        voxel_dims=("k", "j", "i"),
        world_coord_names=("z", "y", "x"),
    )

    selected = result.sel(
        z=xr.Variable("point", [12.1]),
        y=xr.Variable("point", [26.2]),
        x=xr.Variable("point", [39.4]),
        method="nearest",
    )

    assert type(result.xindexes["x"]).__name__ == "VoxelToWorldIndex"
    assert result.coords["x"].dims == ("k", "j", "i")
    assert selected.item() == data.sel(k=2.0, j=3.0, i=3.0).item()


def test_affine_geometry_helpers_extract_origin_vectors_scalings_and_orientation() -> (
    None
):
    """Affine geometry helpers expose the linear part in world-space form."""
    voxel_to_world = np.array(
        [
            [2.0, 1.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert_allclose(get_affine_origin(voxel_to_world), [10.0, 20.0, 30.0])
    assert_allclose(
        get_affine_axis_vectors(voxel_to_world, ("k", "j", "i"))["k"],
        [2.0, 0.0, 0.0],
    )
    assert_allclose(
        get_affine_axis_vectors(voxel_to_world, ("k", "j", "i"))["j"],
        [1.0, 3.0, 0.0],
    )
    scalings = get_affine_axis_scalings(voxel_to_world, ("k", "j", "i"))
    assert scalings.keys() == {"k", "j", "i"}
    assert_allclose(scalings["k"], 2.0)
    assert_allclose(scalings["j"], np.sqrt(10.0))
    assert_allclose(scalings["i"], 4.0)
    expected_orientation = np.array(
        [
            [1.0, 1.0 / np.sqrt(10.0), 0.0],
            [0.0, 3.0 / np.sqrt(10.0), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert_allclose(get_affine_orientation_matrix(voxel_to_world), expected_orientation)


def test_get_world_spacings_singleton_axis_uses_affine_column_norm() -> None:
    """Singleton voxel axes still have a world per-voxel spacing from the affine."""
    voxel_coords = {
        "k": [0.0],
        "j": [0.0, 2.0, 4.0],
        "i": [0.0, 1.0, 2.0, 3.0],
    }
    voxel_to_world = np.array(
        [
            [0.4, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    spacing = get_world_spacings(voxel_coords, voxel_to_world)

    assert spacing == {"k": 0.4, "j": 6.0, "i": 5.0}


def test_get_world_spacings_returns_none_for_irregular_voxel_axes() -> None:
    """World spacing is undefined when voxel-space sampling is irregular."""
    voxel_coords = {
        "k": [0.0, 1.0, 2.0],
        "j": [0.0, 2.0, 4.0],
        "i": [0.0, 1.0, 3.0, 4.0],
    }
    voxel_to_world = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    spacing = get_world_spacings(voxel_coords, voxel_to_world)

    assert spacing == {"k": 2.0, "j": 6.0, "i": None}


def test_get_voxel_world_origin_uses_first_sampled_voxel() -> None:
    """Voxel-affine origin is the world location of array index zero."""
    data = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("k", "j", "i"),
        coords={
            "k": [10.0, 11.0],
            "j": [5.0, 7.0, 9.0],
            "i": [100.0, 101.0, 102.0, 103.0],
        },
    )
    data = add_world_coords_from_voxel_affine(
        data,
        np.array(
            [
                [2.0, 0.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [0.0, 0.0, 4.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        voxel_dims=("k", "j", "i"),
        world_coord_names=("z", "y", "x"),
    )

    assert get_voxel_world_origin(data) == {"z": 30.0, "y": 35.0, "x": 430.0}
