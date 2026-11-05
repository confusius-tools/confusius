"""Tests for confusius._utils.plotting."""

import numpy as np
import pytest

from confusius._dims import WORLD_DIMS
from confusius._utils.geometry import VoxelToWorldTransform, get_voxel_to_world_affine
from confusius._utils.plotting import (
    compute_axis_aligned_grid_geometry,
    compute_oblique_axis_aligned_grid_geometry,
    qr_axis_spacing,
    scale_min_max,
)
from confusius.registration import apply_rotation
from confusius.validation import ensure_voxeldata


def test_scale_min_max_ignores_inf_but_preserves_nan():
    arr = np.array([-np.inf, 0.0, 5.0, 10.0, np.nan])
    result = scale_min_max(arr)

    # -inf/inf are excluded from the bounds and clipped into [0, 1]; nan is
    # excluded from the bounds too but has no position on the scale to clip to.
    assert result[0] == 0.0
    np.testing.assert_allclose(result[1:4], [0.0, 0.5, 1.0])
    assert np.isnan(result[4])


def test_scale_min_max_raises_on_no_finite_values():
    arr = np.array([-np.inf, np.inf, np.nan])
    with pytest.raises(ValueError, match="no finite values"):
        scale_min_max(arr)


def _scan_world_coordinates(data):
    """Compute the display grid by materializing world coordinates, as a reference."""
    linear = get_voxel_to_world_affine(data)[:3, :3]
    row_to_axis, axis_spacing = qr_axis_spacing(linear)
    shape, spacing, origin = [], [], []
    for row, dim in enumerate(WORLD_DIMS):
        values = np.asarray(data.coords[dim].values, dtype=np.float64)
        lower, upper = values.min(), values.max()
        step = axis_spacing[row_to_axis[row]]
        n_steps = (upper - lower) / step
        origin.append(lower)
        spacing.append(step)
        shape.append(int(np.ceil(n_steps - 1e-6 * max(1.0, n_steps))) + 1)
    return shape, spacing, origin


@pytest.mark.parametrize("axis", ["z", "y", "x"])
@pytest.mark.parametrize("angle", [0.5, 7.0, 30.0, 44.0, 46.0, 90.0, 133.0])
@pytest.mark.parametrize(
    "selection",
    [{}, {"k": slice(1, None)}, {"i": slice(None, None, 2)}, {"k": 0}],
    ids=["full", "cropped", "strided", "scalar-indexed"],
)
def test_grid_geometry_matches_coordinate_scan(
    sample_voxeldata_3d, axis, angle, selection
):
    """The grid computed from corners equals the one from all world coordinates."""
    rotated = apply_rotation(sample_voxeldata_3d, angle, axis, center="geometry").isel(
        selection
    )

    shape, spacing, origin = compute_oblique_axis_aligned_grid_geometry(
        rotated, WORLD_DIMS
    )

    expected_shape, expected_spacing, expected_origin = _scan_world_coordinates(
        ensure_voxeldata(rotated)
    )
    assert shape == expected_shape
    np.testing.assert_allclose(spacing, expected_spacing)
    np.testing.assert_allclose(origin, expected_origin)


def test_grid_geometry_selects_requested_world_dims(sample_voxeldata_3d):
    """Rows follow the requested world dims, not positional z/y/x order."""
    rotated = apply_rotation(sample_voxeldata_3d, 20.0, "x", center="geometry")

    shape, spacing, origin = compute_oblique_axis_aligned_grid_geometry(
        rotated, ("x", "z")
    )

    all_shape, all_spacing, all_origin = compute_oblique_axis_aligned_grid_geometry(
        rotated, WORLD_DIMS
    )
    assert shape == [all_shape[2], all_shape[0]]
    assert spacing == [all_spacing[2], all_spacing[0]]
    assert origin == [all_origin[2], all_origin[0]]


def test_grid_geometry_does_not_materialize_world_coords(
    sample_voxeldata_3d, monkeypatch
):
    """World coordinates stay lazy: their cost was most of a 3D display update."""
    rotated = apply_rotation(sample_voxeldata_3d, 20.0, "x", center="geometry")

    def fail_on_materialize(self, dim_positions):
        raise AssertionError("World coordinates were materialized.")

    monkeypatch.setattr(VoxelToWorldTransform, "forward", fail_on_materialize)

    shape, _, _ = compute_oblique_axis_aligned_grid_geometry(rotated, WORLD_DIMS)

    assert len(shape) == 3


def test_axis_aligned_grid_geometry_rejects_singular_axis():
    """A singular world axis has no spacing to step the grid by."""
    with pytest.raises(ValueError, match="singular"):
        compute_axis_aligned_grid_geometry(
            np.diag([0.0, 1.0, 1.0, 1.0]), [(0.0, 3.0), (0.0, 3.0), (0.0, 3.0)]
        )
