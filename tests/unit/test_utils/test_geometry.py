"""Tests for voxel-space geometry helpers."""

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from confusius._utils.geometry import (
    add_physical_coords_from_voxel_affine,
    get_affine_axis_scalings,
    get_affine_axis_vectors,
    get_affine_orientation_matrix,
    get_affine_origin,
    get_physical_spacings,
    get_voxel_affine_origin,
)


def test_axis_aligned_voxel_affine_uses_1d_physical_coords() -> None:
    """Axis-aligned voxel-affine geometry exposes 1D physical coords and indexes."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0.0, 2.0],
            "j": [0.0, 1.0, 3.0],
            "i": [0.0, 2.0, 3.0, 7.0],
        },
    )
    voxel_to_physical = np.array(
        [
            [10.0, 0.0, 0.0, 100.0],
            [0.0, 2.0, 0.0, 200.0],
            [0.0, 0.0, 3.0, 300.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    result = add_physical_coords_from_voxel_affine(
        data,
        voxel_to_physical,
        voxel_dims=("k", "j", "i"),
        physical_coord_names=("z", "y", "x"),
    )

    assert result.coords["z"].dims == ("k",)
    assert result.coords["y"].dims == ("j",)
    assert result.coords["x"].dims == ("i",)
    assert_array_equal(result.coords["z"].values, [100.0, 120.0])
    assert_array_equal(result.coords["y"].values, [200.0, 202.0, 206.0])
    assert_array_equal(result.coords["x"].values, [300.0, 306.0, 309.0, 321.0])
    assert type(result.xindexes["x"]).__name__ == "PandasIndex"



def test_axis_aligned_voxel_affine_sel_supports_scalars_and_slices() -> None:
    """Axis-aligned voxel-affine geometry supports ordinary `.sel(...)` queries."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0.0, 2.0],
            "j": [0.0, 1.0, 3.0],
            "i": [0.0, 2.0, 3.0, 7.0],
        },
    )
    voxel_to_physical = np.array(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = add_physical_coords_from_voxel_affine(
        data,
        voxel_to_physical,
        voxel_dims=("k", "j", "i"),
        physical_coord_names=("z", "y", "x"),
    )

    selected = result.sel(z=12.1, y=26.2, x=39.4, method="nearest")
    sliced = result.sel(z=slice(10.0, 12.0), y=slice(20.0, 22.0))

    assert selected.item() == data.sel(k=2.0, j=3.0, i=3.0).item()
    assert float(selected.coords["z"]) == 12.0
    assert float(selected.coords["y"]) == 26.0
    assert float(selected.coords["x"]) == 39.0
    assert sliced.sizes == {"k": 2, "j": 2, "i": 4}



def test_oblique_coordinate_transform_index_selection_uses_physical_coords() -> None:
    """Oblique voxel-affine geometry still uses CTI pointwise physical selection."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0.0, 2.0],
            "j": [0.0, 1.0, 3.0],
            "i": [0.0, 2.0, 3.0, 7.0],
        },
    )
    voxel_to_physical = np.array(
        [
            [1.0, 0.1, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = add_physical_coords_from_voxel_affine(
        data,
        voxel_to_physical,
        voxel_dims=("k", "j", "i"),
        physical_coord_names=("z", "y", "x"),
    )

    selected = result.sel(
        z=xr.Variable("point", [12.1]),
        y=xr.Variable("point", [26.2]),
        x=xr.Variable("point", [39.4]),
        method="nearest",
    )

    assert type(result.xindexes["x"]).__name__ == "CoordinateTransformIndex"
    assert result.coords["x"].dims == ("k", "j", "i")
    assert selected.item() == data.sel(k=2.0, j=3.0, i=3.0).item()


def test_affine_geometry_helpers_extract_origin_vectors_scalings_and_orientation() -> (
    None
):
    """Affine geometry helpers expose the linear part in physical-space form."""
    voxel_to_physical = np.array(
        [
            [2.0, 1.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert_allclose(get_affine_origin(voxel_to_physical), [10.0, 20.0, 30.0])
    assert_allclose(
        get_affine_axis_vectors(voxel_to_physical, ("k", "j", "i"))["k"],
        [2.0, 0.0, 0.0],
    )
    assert_allclose(
        get_affine_axis_vectors(voxel_to_physical, ("k", "j", "i"))["j"],
        [1.0, 3.0, 0.0],
    )
    scalings = get_affine_axis_scalings(voxel_to_physical, ("k", "j", "i"))
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
    assert_allclose(
        get_affine_orientation_matrix(voxel_to_physical), expected_orientation
    )


def test_get_physical_spacings_returns_none_for_irregular_voxel_axes() -> None:
    """Physical spacing is undefined when voxel-space sampling is irregular."""
    voxel_coords = {
        "k": [0.0, 1.0, 2.0],
        "j": [0.0, 2.0, 4.0],
        "i": [0.0, 1.0, 3.0, 4.0],
    }
    voxel_to_physical = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    spacing = get_physical_spacings(voxel_coords, voxel_to_physical)

    assert spacing == {"k": 2.0, "j": 6.0, "i": None}


def test_get_voxel_affine_origin_uses_first_sampled_voxel() -> None:
    """Voxel-affine origin is the physical location of array index zero."""
    data = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("k", "j", "i"),
        coords={
            "k": [10.0, 11.0],
            "j": [5.0, 7.0, 9.0],
            "i": [100.0, 101.0, 102.0, 103.0],
        },
        attrs={
            "voxel_to_physical": np.array(
                [
                    [2.0, 0.0, 0.0, 10.0],
                    [0.0, 3.0, 0.0, 20.0],
                    [0.0, 0.0, 4.0, 30.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        },
    )

    assert get_voxel_affine_origin(data) == {"z": 30.0, "y": 35.0, "x": 430.0}
