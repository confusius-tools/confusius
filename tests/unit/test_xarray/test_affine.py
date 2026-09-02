"""Tests for `confusius.xarray.affine` helpers (`get_bounding_box`)."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.xarray import create_voxeldata, get_bounding_box

OBLIQUE_AFFINE = np.array(
    [
        [0.3, 0.1, 0.0, -1.2],
        [0.05, 0.25, 0.02, 2.0],
        [0.0, 0.04, 0.5, -0.7],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
"""Oblique voxel-to-world affine whose world coordinates mix all voxel dims."""


def _materialized_bounds(data: xr.DataArray) -> np.ndarray:
    """Reference implementation: reduce the fully materialized world coordinates."""
    return np.array(
        [
            [data.coords[dim].values.min() for dim in "zyx"],
            [data.coords[dim].values.max() for dim in "zyx"],
        ]
    )


def test_oblique_bounds_match_materialized_coordinates(rng):
    """Corner-mapped bounds equal the full-grid reduction for an oblique affine."""
    data = create_voxeldata(
        rng.random((4, 5, 6)), dims=("k", "j", "i"), voxel_to_world=OBLIQUE_AFFINE
    )
    bbox = get_bounding_box(data)
    assert bbox.dims == ("bound", "component")
    assert list(bbox.coords["bound"].values) == ["min", "max"]
    assert list(bbox.coords["component"].values) == ["z", "y", "x"]
    assert_allclose(bbox.values, _materialized_bounds(data))


def test_axis_aligned_bounds_match_coordinate_extremes(rng):
    """Axis-aligned geometry reproduces the eager 1D coordinate min/max and units."""
    data = create_voxeldata(
        rng.random((4, 6, 8)),
        dims=("k", "j", "i"),
        spacing=(0.2, 0.1, 0.05),
        origin=(1.0, 2.0, 3.0),
    )
    bbox = get_bounding_box(data)
    assert_allclose(bbox.values, _materialized_bounds(data))
    assert bbox.attrs["units"] == "mm"


def test_irregular_voxel_coordinates(rng):
    """Non-contiguous voxel coordinates still give exact bounds."""
    data = create_voxeldata(
        rng.random((3, 4, 3)),
        dims=("k", "j", "i"),
        k=[1, 5, 6],
        j=[0, 1, 2, 9],
        i=[0, 2, 3],
        voxel_to_world=OBLIQUE_AFFINE,
    )
    assert_allclose(get_bounding_box(data).values, _materialized_bounds(data))


def test_singleton_dim_slice(rng):
    """A size-1 voxel dim (canonical 2D slice) keeps the output shape and is exact."""
    data = create_voxeldata(
        rng.random((1, 5, 6)), dims=("k", "j", "i"), voxel_to_world=OBLIQUE_AFFINE
    )
    bbox = get_bounding_box(data)
    assert bbox.dims == ("bound", "component")
    assert_allclose(bbox.values, _materialized_bounds(data))


def test_scalar_isel_input_is_canonicalized(rng):
    """A scalar-`isel` input gives the same bounds as its singleton-dim equivalent."""
    data = create_voxeldata(
        rng.random((4, 5, 6)), dims=("k", "j", "i"), voxel_to_world=OBLIQUE_AFFINE
    )
    assert_allclose(
        get_bounding_box(data.isel(k=1)).values,
        get_bounding_box(data.isel(k=[1])).values,
    )


def test_pose_dependent_bounds_per_pose(rng):
    """Pose-dependent geometry yields one bounding box per pose, pose coord kept."""
    shifted = OBLIQUE_AFFINE.copy()
    shifted[:3, 3] += [5.0, -2.0, 1.5]
    data = create_voxeldata(
        rng.random((2, 3, 4, 5)),
        dims=("pose", "k", "j", "i"),
        pose=[10, 20],
        voxel_to_world=np.stack([OBLIQUE_AFFINE, shifted]),
    )
    bbox = get_bounding_box(data)
    assert bbox.dims == ("pose", "bound", "component")
    assert list(bbox.coords["pose"].values) == [10, 20]
    expected = np.stack(
        [
            np.stack(
                [
                    data.coords[dim].min(dim=("k", "j", "i")).values
                    for dim in "zyx"
                ],
                axis=-1,
            ),
            np.stack(
                [
                    data.coords[dim].max(dim=("k", "j", "i")).values
                    for dim in "zyx"
                ],
                axis=-1,
            ),
        ],
        axis=1,
    )
    assert_allclose(bbox.values, expected)


def test_non_voxeldata_raises():
    """A DataArray without voxel-to-world geometry is rejected."""
    plain = xr.DataArray(np.zeros((2, 3, 4)), dims=("z", "y", "x"))
    with pytest.raises(ValueError):
        get_bounding_box(plain)


def test_non_dataarray_raises():
    """A bare numpy array is rejected."""
    with pytest.raises(TypeError):
        get_bounding_box(np.zeros((2, 3, 4)))  # ty: ignore[invalid-argument-type]


def test_accessor_matches_function(rng):
    """The `.fusi.affine.bounding_box` property returns the same bounds."""
    data = create_voxeldata(
        rng.random((4, 5, 6)), dims=("k", "j", "i"), voxel_to_world=OBLIQUE_AFFINE
    )
    assert_allclose(data.fusi.affine.bounding_box.values, get_bounding_box(data).values)
