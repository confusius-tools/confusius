"""Tests for voxel-space geometry helpers."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from xarray.indexes import CoordinateTransformIndex

from confusius._utils.geometry import (
    VoxelToWorldIndex,
    VoxelToWorldTransform,
    attach_voxel_to_world_index,
    get_affine_axis_scalings,
    get_affine_axis_vectors,
    get_affine_direction_matrix,
    get_voxel_to_world_affine,
    get_voxel_to_world_coord_names,
    get_voxel_to_world_direction_matrix,
    get_voxel_to_world_index_origin,
    get_voxel_to_world_index_spacing,
    get_voxel_to_world_spacings_from_coords,
    has_axis_aligned_voxel_to_world_index,
    restore_voxel_to_world_index,
    update_voxel_to_world_coord_attrs,
)
from confusius.xarray import create_fusi_dataarray


def _simple_voxel_to_world_result() -> xr.DataArray:
    """Build a minimal pose-independent, identity-affine (k, j, i) DataArray.

    Shared by tests that only need *some* real VoxelToWorldIndex and don't care
    about its specific geometry (join/reindex_like/equals/concat-rejection/
    update_voxel_to_world_coord_attrs edge cases).
    """
    return create_fusi_dataarray(
        np.zeros((2, 3, 4)), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    )


def _pose_dependent_result(
    pose_affines: np.ndarray | None = None, pose_coord: np.ndarray | None = None
) -> xr.DataArray:
    """Build a pose-dependent voxel-to-world DataArray shared by pose tests."""
    if pose_affines is None:
        pose_affines = np.stack(
            [
                np.diag([1.0, 1.0, 1.0, 1.0]),
                np.array(
                    [
                        [1.0, 0.0, 0.0, 100.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ]
        )
    if pose_coord is None:
        pose_coord = np.array([0, 1])
    npose = len(pose_coord)
    data = xr.DataArray(
        np.arange(npose * 2 * 3 * 4).reshape(npose, 2, 3, 4),
        dims=("pose", "k", "j", "i"),
        coords={
            "k": np.arange(2),
            "j": np.arange(3),
            "i": np.arange(4),
        },
    )
    transform = VoxelToWorldTransform(
        {"k": np.arange(2.0), "j": np.arange(3.0), "i": np.arange(4.0)},
        pose_affines,
        pose_coord=pose_coord,
    )
    index = VoxelToWorldIndex(
        CoordinateTransformIndex(transform), pose_affines, world_coord_attrs=None
    )
    result = data.assign_coords(xr.Coordinates.from_xindex(index))
    # `pose` is not one of the index's owned coordinate names (see its docstring), so
    # it needs attaching separately as a plain coordinate, exactly like
    # attach_voxel_to_world_index does.
    return result.assign_coords(pose=("pose", pose_coord))


def test_pose_dependent_forward_gives_translated_world_coords_per_pose() -> None:
    """Forward coordinates differ per pose when poses have distinct translations."""
    result = _pose_dependent_result()

    assert result.coords["pose"].dims == ("pose",)
    assert_array_equal(result.coords["pose"].values, [0, 1])
    assert result.coords["z"].dims == ("pose", "k", "j", "i")
    assert_allclose(
        result.coords["z"].isel(pose=0, j=0, i=0).values, [0.0, 1.0]
    )
    assert_allclose(
        result.coords["z"].isel(pose=1, j=0, i=0).values, [100.0, 101.0]
    )


def test_pose_dependent_forward_supports_rotated_poses() -> None:
    """Forward coordinates reflect differing rotations across poses."""
    rotation = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = _pose_dependent_result(
        pose_affines=np.stack([np.eye(4), rotation]), pose_coord=np.array([0, 1])
    )

    unrotated = result.isel(pose=0, k=0, j=1, i=0)
    rotated = result.isel(pose=1, k=0, j=1, i=0)
    assert unrotated.coords["z"].item() == 0.0
    assert unrotated.coords["y"].item() == 1.0
    assert unrotated.coords["x"].item() == 0.0
    assert rotated.coords["z"].item() == -1.0
    assert rotated.coords["y"].item() == 0.0
    assert rotated.coords["x"].item() == 0.0


def test_pose_dependent_sel_requires_scalar_pose_for_world_selection() -> None:
    """World-coordinate selection without a prior scalar pose raises a guiding error.

    `pose` is its own independently indexed coordinate (see `VoxelToWorldIndex`'s
    docstring), so a single combined `.sel(pose=..., z=..., y=..., x=...)` call is
    never supported for pose-dependent geometry, regardless of what `pose=` is set
    to -- world-coordinate selection always requires `pose` reduced to a scalar in a
    prior, separate call.
    """
    result = _pose_dependent_result()

    with pytest.raises(ValueError, match="requires reducing `pose` to a scalar"):
        result.sel(z=0.0, y=0.0, x=0.0)
    with pytest.raises(ValueError, match="requires reducing `pose` to a scalar"):
        result.sel(pose=[0, 1], z=0.0, y=0.0, x=0.0)
    with pytest.raises(ValueError, match="requires reducing `pose` to a scalar"):
        result.sel(pose=slice(0, 2), z=0.0, y=0.0, x=0.0)
    with pytest.raises(ValueError, match="requires reducing `pose` to a scalar"):
        result.sel(pose=1, z=100.0, y=1.0, x=2.0, method="nearest")


def test_pose_dependent_sel_resolves_pose_then_world_coords() -> None:
    """A prior scalar pose selection resolves the matching affine for spatial lookup."""
    result = _pose_dependent_result()

    selected = result.isel(pose=1).sel(z=100.0, y=1.0, x=2.0, method="nearest")

    assert selected.item() == result.isel(pose=1, k=0, j=1, i=2).item()

    selected_via_sel = result.sel(pose=1).sel(z=100.0, y=1.0, x=2.0, method="nearest")

    assert selected_via_sel.item() == result.isel(pose=1, k=0, j=1, i=2).item()


def test_pose_dependent_scalar_isel_drops_pose_dependency() -> None:
    """Scalar `isel` on `pose` selects one affine and removes the pose dim."""
    result = _pose_dependent_result()

    fixed = result.isel(pose=1)

    assert "pose" not in fixed.dims
    assert fixed.coords["z"].dims == ("k", "j", "i")
    assert_allclose(fixed.coords["z"].isel(j=0, i=0).values, [100.0, 101.0])
    # Existing single-affine `.sel` behavior applies once pose is scalar.
    assert fixed.sel(z=100.0, y=0.0, x=0.0).item() == fixed.isel(k=0, j=0, i=0).item()


def test_pose_dependent_slice_and_fancy_isel_subset_pose_and_affines() -> None:
    """Non-scalar `pose` indexers subset/reorder both labels and affines."""
    result = _pose_dependent_result()

    sliced = result.isel(pose=slice(1, 2))
    assert_array_equal(sliced.coords["pose"].values, [1])
    assert sliced.coords["z"].dims == ("pose", "k", "j", "i")

    reordered = result.isel(pose=[1, 0])
    assert_array_equal(reordered.coords["pose"].values, [1, 0])
    assert_allclose(
        reordered.coords["z"].isel(pose=0, j=0, i=0).values, [100.0, 101.0]
    )
    assert_allclose(
        reordered.coords["z"].isel(pose=1, j=0, i=0).values, [0.0, 1.0]
    )


def test_pose_dependent_spatial_isel_subsets_while_pose_stack_remains() -> None:
    """Spatial `isel` subsets voxel coords while the pose stack stays intact."""
    result = _pose_dependent_result()

    subset = result.isel(k=0)

    assert subset.coords["pose"].dims == ("pose",)
    assert_array_equal(subset.coords["pose"].values, [0, 1])
    assert subset.coords["z"].dims == ("pose", "j", "i")


def test_pose_affine_stack_validates_shape_length_finiteness_and_scale() -> None:
    """Constructing a pose-dependent transform validates the affine stack."""
    valid_coords = {"k": np.arange(2.0), "j": np.arange(3.0), "i": np.arange(4.0)}
    good = np.stack([np.eye(4), np.eye(4)])

    with pytest.raises(ValueError, match="pose_coord must be 1D"):
        VoxelToWorldTransform(
            valid_coords, good, pose_coord=np.array([[0, 1]])
        )

    with pytest.raises(ValueError, match="must have shape"):
        VoxelToWorldTransform(
            valid_coords, good, pose_coord=np.array([0, 1, 2])
        )

    non_finite = good.copy()
    non_finite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        VoxelToWorldTransform(valid_coords, non_finite, pose_coord=np.array([0, 1]))

    bad_row = good.copy()
    bad_row[0, -1, -1] = 2.0
    with pytest.raises(ValueError, match="homogeneous final row"):
        VoxelToWorldTransform(valid_coords, bad_row, pose_coord=np.array([0, 1]))

    mismatched_scale = np.stack([np.diag([1.0, 1.0, 1.0, 1.0]), np.diag([2.0, 1.0, 1.0, 1.0])])
    with pytest.raises(ValueError, match="equal spatial scale magnitudes"):
        VoxelToWorldTransform(
            valid_coords, mismatched_scale, pose_coord=np.array([0, 1])
        )


def test_has_axis_aligned_voxel_to_world_index_checks_every_pose() -> None:
    """`has_axis_aligned_voxel_to_world_index` requires every pose to be aligned."""
    aligned = _pose_dependent_result()
    assert has_axis_aligned_voxel_to_world_index(aligned) is True

    angle = np.pi / 4
    oblique_second_pose = np.stack(
        [
            np.eye(4),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, np.cos(angle), -np.sin(angle), 0.0],
                    [0.0, np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ]
    )
    oblique = _pose_dependent_result(pose_affines=oblique_second_pose)
    assert has_axis_aligned_voxel_to_world_index(oblique) is False


def _pose_dependent_data() -> xr.DataArray:
    """Build a plain (non-VoxelData) pose-stacked DataArray for attach tests."""
    return xr.DataArray(
        np.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4),
        dims=("pose", "k", "j", "i"),
        coords={
            "pose": [0, 1],
            "k": np.arange(2),
            "j": np.arange(3),
            "i": np.arange(4),
        },
    )


def test_attach_voxel_to_world_index_accepts_pose_stacked_affine() -> None:
    """`attach_voxel_to_world_index` wires up a per-pose affine stack."""
    data = _pose_dependent_data()
    affine = np.stack(
        [
            np.eye(4),
            np.array(
                [
                    [1.0, 0.0, 0.0, 100.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ]
    )

    result = attach_voxel_to_world_index(data, affine)

    assert result.coords["z"].dims == ("pose", "k", "j", "i")
    assert_array_equal(result.coords["pose"].values, [0, 1])
    assert_allclose(result.coords["z"].isel(pose=1, j=0, i=0).values, [100.0, 101.0])
    assert result.coords["z"].attrs["voxdim"] == 1.0
    assert result.coords["z"].attrs["units"] == "mm"
    assert get_voxel_to_world_affine(result).shape == (2, 4, 4)


def test_attach_voxel_to_world_index_rejects_mismatched_pose_stack() -> None:
    """A pose-stacked affine requires a matching `pose` dim/coord on `data`."""
    no_pose = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("k", "j", "i"),
        coords={"k": np.arange(2), "j": np.arange(3), "i": np.arange(4)},
    )
    affine = np.stack([np.eye(4), np.eye(4)])

    with pytest.raises(ValueError, match="have a 'pose' dimension"):
        attach_voxel_to_world_index(no_pose, affine)

    wrong_length = _pose_dependent_data()
    with pytest.raises(ValueError, match="does not match data's 'pose' size"):
        attach_voxel_to_world_index(wrong_length, np.stack([np.eye(4)]))


def test_attach_voxel_to_world_index_rejects_non_integer_voxel_coord() -> None:
    """A native voxel dim coordinate must have integer dtype."""
    data = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("k", "j", "i"),
        coords={
            "k": np.arange(2, dtype=np.float64),  # not integer.
            "j": np.arange(3),
            "i": np.arange(4),
        },
    )

    with pytest.raises(TypeError, match="must have integer dtype"):
        attach_voxel_to_world_index(data, np.eye(4))


def test_attach_voxel_to_world_index_rejects_pose_dim_without_1d_coordinate() -> None:
    """A pose-stacked affine requires `pose` to be a genuine 1D coordinate.

    Distinct from `test_attach_voxel_to_world_index_rejects_mismatched_pose_stack`
    (missing `pose` dimension entirely, or a length mismatch): here `pose` is a
    dimension but has no matching coordinate at all.
    """
    data = xr.DataArray(
        np.zeros((2, 2, 3, 4)),
        dims=("pose", "k", "j", "i"),
        coords={"k": np.arange(2), "j": np.arange(3), "i": np.arange(4)},
    )
    affine = np.stack([np.eye(4), np.eye(4)])

    with pytest.raises(ValueError, match="must have a matching 1D coordinate"):
        attach_voxel_to_world_index(data, affine)


def test_restore_voxel_to_world_index_rebuilds_pose_dependent_geometry() -> None:
    """Restoring a fixed spatial dim keeps the pose stack intact."""
    data = _pose_dependent_data()
    affine = np.stack(
        [
            np.eye(4),
            np.array(
                [
                    [1.0, 0.0, 0.0, 100.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ]
    )
    result = attach_voxel_to_world_index(data, affine)
    fixed = result.isel(k=0)
    expanded = fixed.expand_dims(k=[5])

    restored = restore_voxel_to_world_index(expanded)

    assert restored.coords["z"].dims == ("pose", "k", "j", "i")
    assert_array_equal(restored.coords["pose"].values, [0, 1])


def test_pose_dependent_concat_merges_pose_labels_and_affines() -> None:
    """`xr.concat` along `pose` merges pose labels and affine stacks in order."""
    first = _pose_dependent_result()
    second = _pose_dependent_result(
        pose_affines=np.stack(
            [
                np.array(
                    [
                        [1.0, 0.0, 0.0, 200.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            ]
        ),
        pose_coord=np.array([2]),
    )

    combined = xr.concat([first, second], dim="pose")

    assert_array_equal(combined.coords["pose"].values, [0, 1, 2])
    assert_allclose(
        combined.coords["z"].isel(pose=2, j=0, i=0).values, [200.0, 201.0]
    )


def test_pose_dependent_concat_rejects_mismatched_spatial_geometry() -> None:
    """Concatenating along `pose` still rejects mismatched voxel-space geometry."""
    first = _pose_dependent_result()
    mismatched = _pose_dependent_result(
        pose_affines=np.stack([np.diag([2.0, 2.0, 2.0, 1.0])]),
        pose_coord=np.array([2]),
    )

    with pytest.raises(
        ValueError, match="different spatial geometry|equal spatial scale"
    ):
        xr.concat([first, mismatched], dim="pose", join="exact")


def test_origin_and_direction_require_scalar_pose() -> None:
    """Single-grid geometry accessors reject pose-dependent geometry clearly."""
    result = _pose_dependent_result()

    with pytest.raises(ValueError, match="requires pose-independent geometry"):
        get_voxel_to_world_index_origin(result)
    with pytest.raises(ValueError, match="requires pose-independent geometry"):
        get_voxel_to_world_direction_matrix(result)

    scalar = result.isel(pose=0)
    get_voxel_to_world_index_origin(scalar)
    get_voxel_to_world_direction_matrix(scalar)


def test_spacing_matches_scalar_pose_for_non_unit_scale() -> None:
    """Pose-dependent spacing agrees with a scalar-pose selection for real spacing.

    Regression: `get_affine_axis_vectors` sliced a pose-stacked affine with
    `affine[:-1, :-1]`, which for a 3D array trims the *pose* axis instead of the
    homogeneous row/column, silently returning garbage scalings. Invisible with the
    all-ones scale used by other pose-dependent fixtures in this file, since the
    misindexed values happened to still norm to ~1.
    """
    pose_affines = np.stack(
        [
            np.diag([0.4, 0.1, 0.11, 1.0]),
            np.array(
                [
                    [0.4, 0.0, 0.0, 100.0],
                    [0.0, 0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.11, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ]
    )
    result = _pose_dependent_result(pose_affines=pose_affines)

    spacing = get_voxel_to_world_index_spacing(result)
    scalar_spacing = get_voxel_to_world_index_spacing(result.isel(pose=0))

    assert spacing == scalar_spacing
    assert spacing["k"] == pytest.approx(0.4)
    assert spacing["j"] == pytest.approx(0.1)
    assert spacing["i"] == pytest.approx(0.11)


def test_pose_dependent_index_equality_compares_pose_labels_and_affines() -> None:
    """Index equality accounts for pose labels and the affine stack."""
    left = _pose_dependent_result()
    same = _pose_dependent_result()
    different_pose_labels = _pose_dependent_result(pose_coord=np.array([0, 2]))

    assert left.xindexes["z"].equals(same.xindexes["z"])
    assert not left.xindexes["z"].equals(different_pose_labels.xindexes["z"])


def test_axis_aligned_voxel_to_world_computes_correct_world_coords() -> None:
    """Axis-aligned voxel-to-world geometry computes correct world coords.

    World coordinates are backed by a single joint index spanning all voxel
    dimensions (see [VoxelToWorldIndex][confusius._utils.geometry.VoxelToWorldIndex]
    for why), so each is `(k, j, i)`-shaped even though, for an axis-aligned affine,
    its value only actually varies along its own paired voxel dimension.
    """
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0, 2],
            "j": [0, 1, 3],
            "i": [0, 2, 3, 7],
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

    result = attach_voxel_to_world_index(
        data,
        voxel_to_world,
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


def test_axis_aligned_voxel_to_world_uses_voxel_to_world_index() -> None:
    """Axis-aligned world coords are owned by a VoxelToWorldIndex."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0, 2],
            "j": [0, 1, 3],
            "i": [0, 2, 3, 7],
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
    result = attach_voxel_to_world_index(
        data,
        voxel_to_world,
    )

    assert list(result.xindexes) == ["k", "j", "i", "z", "y", "x"]
    assert type(result.xindexes["z"]).__name__ == "VoxelToWorldIndex"
    result.stack(space=("k", "j", "i"))


def test_oblique_coordinate_transform_index_selection_uses_world_coords() -> None:
    """Oblique voxel-to-world geometry still uses pointwise world selection."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0, 2],
            "j": [0, 1, 3],
            "i": [0, 2, 3, 7],
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
    result = attach_voxel_to_world_index(
        data,
        voxel_to_world,
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


def test_affine_geometry_helpers_extract_vectors_scalings_and_orientation() -> None:
    """Affine geometry helpers expose the linear part in world-space form."""
    voxel_to_world = np.array(
        [
            [2.0, 1.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

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
    assert_allclose(get_affine_direction_matrix(voxel_to_world), expected_orientation)


def test_get_world_spacings_singleton_axis_uses_affine_column_norm() -> None:
    """Singleton voxel axes still have a world per-voxel spacing from the affine."""
    voxel_coords = {
        "k": [0],
        "j": [0, 2, 4],
        "i": [0, 1, 2, 3],
    }
    voxel_to_world = np.array(
        [
            [0.4, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    spacing = get_voxel_to_world_spacings_from_coords(voxel_coords, voxel_to_world)

    assert spacing == {"k": 0.4, "j": 6.0, "i": 5.0}


def test_get_world_spacings_returns_none_for_irregular_voxel_axes() -> None:
    """World spacing is undefined when voxel-space sampling is irregular."""
    voxel_coords = {
        "k": [0, 1, 2],
        "j": [0, 2, 4],
        "i": [0, 1, 3, 4],
    }
    voxel_to_world = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    spacing = get_voxel_to_world_spacings_from_coords(voxel_coords, voxel_to_world)

    assert spacing == {"k": 2.0, "j": 6.0, "i": None}


def test_get_voxel_to_world_origin_uses_first_sampled_voxel() -> None:
    """Voxel-to-world origin is the world location of array index zero."""
    data = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("k", "j", "i"),
        coords={
            "k": [10, 11],
            "j": [5, 7, 9],
            "i": [100, 101, 102, 103],
        },
    )
    data = attach_voxel_to_world_index(
        data,
        np.array(
            [
                [2.0, 0.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [0.0, 0.0, 4.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    assert get_voxel_to_world_index_origin(data) == {"z": 30.0, "y": 35.0, "x": 430.0}


def _axis_aligned_result() -> xr.DataArray:
    """Build a 3D axis-aligned voxel-to-world DataArray shared by several tests."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0, 2],
            "j": [0, 1, 3],
            "i": [0, 2, 3, 7],
        },
    )
    return attach_voxel_to_world_index(
        data,
        np.diag([10.0, 2.0, 3.0, 1.0]),
    )


def test_alignment_between_different_world_grids_raises_clear_error() -> None:
    """Arithmetic alignment tells users to resample mismatched world grids."""
    left = _axis_aligned_result()
    right = attach_voxel_to_world_index(
        xr.DataArray(
            np.ones_like(left.values),
            dims=left.dims,
            coords={dim: left.coords[dim].values for dim in left.dims},
        ),
        np.array(
            [
                [10.0, 0.0, 0.0, 1.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    with pytest.raises(ValueError, match="Cannot automatically align .*resample_like"):
        left + right


def test_reindex_between_different_world_grids_raises_clear_error() -> None:
    """Reindexing alignment tells users to resample mismatched world grids."""
    left = _axis_aligned_result()
    right = attach_voxel_to_world_index(
        xr.DataArray(
            np.ones_like(left.values),
            dims=left.dims,
            coords={dim: left.coords[dim].values for dim in left.dims},
        ),
        np.array(
            [
                [10.0, 0.0, 0.0, 1.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    with pytest.raises(
        ValueError, match="Cannot automatically reindex .*resample_like"
    ):
        left.reindex_like(right)


def test_add_world_coords_defaults_to_yx_names_for_2d_voxel_dims() -> None:
    """2D voxel geometry defaults to `y`/`x` world coordinate names."""
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("j", "i"),
        coords={"j": np.arange(3), "i": np.arange(4)},
    )

    result = attach_voxel_to_world_index(data, np.eye(3))

    assert set(result.coords) == {"j", "i", "y", "x"}


def test_voxel_to_world_index_from_affine_defaults_to_yx_names_for_2d() -> None:
    """`VoxelToWorldIndex.from_affine` itself defaults to `y`/`x` for 2D geometry.

    `attach_voxel_to_world_index` always resolves `world_coord_names` before
    delegating to `from_affine`, so `from_affine`'s own default only fires when it is
    called directly.
    """
    index = VoxelToWorldIndex.from_affine(
        {"j": np.arange(3), "i": np.arange(4)}, np.eye(3)
    )

    assert index.world_coord_names == ("y", "x")


def test_sel_resolves_descending_and_nonmonotonic_axis_aligned_axes() -> None:
    """Point selection resolves correctly for descending or non-monotonic axes.

    `VoxelToWorldIndex.sel` reverse-looks-up world labels into voxel positions via
    `_reverse_lookup_positions`, which special-cases strictly-descending axes
    (interpolation on the reversed axis) and falls back to exact-match lookup for
    genuinely non-monotonic axes.
    """
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [4, 0],  # Strictly descending.
            "j": [0, 3, 1],  # Non-monotonic.
            "i": [0, 1, 2, 3],
        },
    )
    result = attach_voxel_to_world_index(data, np.eye(4))

    descending = result.sel(z=0.0)
    assert descending.coords["k"].item() == 0.0
    assert_array_equal(descending.values, data.isel(k=1).values)

    nonmonotonic = result.sel(y=1.0)
    assert nonmonotonic.coords["j"].item() == 1.0
    assert_array_equal(nonmonotonic.values, data.isel(j=2).values)


def test_sel_slice_out_of_range_selects_nothing_and_step_subsamples() -> None:
    """Axis-aligned slice selection handles an out-of-range slice and a step."""
    result = _axis_aligned_result()

    empty = result.sel(z=slice(1000.0, 2000.0))
    assert empty.sizes["k"] == 0

    stepped = result.sel(x=slice(0.0, 21.0, 2))
    assert_array_equal(stepped.coords["i"].values, [0.0, 3.0])


def test_sel_plain_slice_without_step_selects_contiguous_range() -> None:
    """A plain (no-step) slice with hits selects the contiguous covered range."""
    result = _axis_aligned_result()

    selected = result.sel(z=slice(0.0, 15.0))

    assert_array_equal(selected.coords["k"].values, [0.0])


def test_voxel_to_world_index_sel_with_unrelated_labels_selects_nothing() -> None:
    """Calling `.sel` with no labels belonging to the index selects nothing."""
    result = _axis_aligned_result()
    index = result.xindexes["z"]

    selection = index.sel({})

    assert selection.dim_indexers == {}


def test_reverse_skips_dimension_fixed_by_a_prior_scalar_isel() -> None:
    """Oblique point selection after a scalar isel ignores the now-fixed dimension.

    A scalar `isel` on one voxel dimension pins it via `fixed_voxel_coords` rather
    than dropping the geometry; a subsequent oblique `.sel` on the remaining
    dimensions must not try to resolve a position for the fixed one.
    """
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={
            "k": [0, 2],
            "j": [0, 1, 3],
            "i": [0, 2, 3, 7],
        },
    )
    voxel_to_world = np.array(
        [
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = attach_voxel_to_world_index(
        data,
        voxel_to_world,
    )
    fixed = result.isel(k=0)

    selected = fixed.sel(
        z=xr.Variable("point", [0.1]),
        y=xr.Variable("point", [2.0]),
        x=xr.Variable("point", [6.0]),
        method="nearest",
    )

    assert selected.item() == data.isel(k=0, j=1, i=1).item()


def test_voxel_to_world_transform_rejects_invalid_construction() -> None:
    """`VoxelToWorldTransform` validates its constructor arguments."""
    valid_coords = {"j": np.arange(3), "i": np.arange(4)}

    with pytest.raises(ValueError, match="must exactly cover active dims"):
        VoxelToWorldTransform(valid_coords, np.eye(3), all_dims=("j", "i", "k"))

    with pytest.raises(ValueError, match="at least one active dim"):
        VoxelToWorldTransform({}, np.eye(3))

    with pytest.raises(ValueError, match="only supports 2D or 3D"):
        VoxelToWorldTransform(
            {
                "k": np.arange(2),
                "j": np.arange(3),
                "i": np.arange(4),
                "t": np.arange(2.0),
            },
            np.eye(5),
        )

    with pytest.raises(ValueError, match="must be 1D"):
        VoxelToWorldTransform({"j": np.arange(3), "i": np.zeros((2, 2))}, np.eye(3))

    with pytest.raises(ValueError, match="one entry per voxel dimension"):
        VoxelToWorldTransform(valid_coords, np.eye(3), world_coord_names=("y",))

    with pytest.raises(ValueError, match=r"must have shape \(3, 3\)"):
        VoxelToWorldTransform(valid_coords, np.eye(4))


def test_voxel_to_world_transform_defaults_to_yx_names_for_2d() -> None:
    """A `VoxelToWorldTransform` built without explicit names defaults to `y`/`x`."""
    transform = VoxelToWorldTransform({"j": np.arange(3), "i": np.arange(4)}, np.eye(3))

    assert transform.coord_names == ("y", "x")


def test_voxel_to_world_transform_equals_rejects_other_types() -> None:
    """`VoxelToWorldTransform.equals` returns `False` for non-transform values."""
    transform = VoxelToWorldTransform({"j": np.arange(3), "i": np.arange(4)}, np.eye(3))

    assert transform.equals(object()) is False  # ty: ignore[invalid-argument-type]


def test_voxel_to_world_transform_repr_reports_dims_and_coord_names() -> None:
    """`repr` reports the transform's active dims and world coordinate names."""
    transform = VoxelToWorldTransform({"j": np.arange(3), "i": np.arange(4)}, np.eye(3))

    assert repr(transform) == (
        "VoxelToWorldTransform(dims=('j', 'i'), coord_names=('y', 'x'))"
    )


def test_voxel_to_world_transform_isel_unsupported_indexers_return_none() -> None:
    """`VoxelToWorldTransform.isel` returns `None` for unsupported indexers.

    Covers: an indexer for a dimension the transform doesn't own (skipped), a
    multi-dimensional `Variable`, a non-slice/array/scalar indexer (also exercising
    `_is_scalar_indexer`'s final `False` fallback), and a fancy multi-dimensional
    array index that would produce a non-1D result.
    """
    transform = VoxelToWorldTransform({"j": np.arange(3), "i": np.arange(4)}, np.eye(3))

    unrelated = transform.isel({"nonexistent": 0})
    assert unrelated is not None
    assert unrelated.dims == transform.dims

    fixed = transform.isel({"j": xr.Variable((), 1)})
    assert fixed is not None
    assert fixed.fixed_voxel_coords == {"j": 1.0}

    assert transform.isel({"j": xr.Variable(("a", "b"), np.zeros((2, 2)))}) is None
    assert transform.isel({"j": "bogus"}) is None
    assert transform.isel({"j": np.zeros((2, 2), dtype=int)}) is None


def test_add_world_coords_validates_voxel_dims_and_coordinates() -> None:
    """`attach_voxel_to_world_index` validates dims and their coordinates."""
    data = xr.DataArray(
        np.zeros((3, 4)),
        dims=("j", "i"),
        coords={"j": np.arange(3), "i": np.arange(4)},
    )

    with pytest.raises(ValueError, match="must have a matching 1D coordinate"):
        attach_voxel_to_world_index(data.drop_vars("i"), np.eye(3))

    non_dim_coord = xr.DataArray(
        np.zeros((3, 4)),
        dims=("j", "i"),
        coords={"j": (("j", "i"), np.zeros((3, 4))), "i": np.arange(4)},
    )
    with pytest.raises(ValueError, match="must be a 1D dimension coordinate"):
        attach_voxel_to_world_index(non_dim_coord, np.eye(3))

    no_voxel_dims = xr.DataArray(np.zeros((3, 4)), dims=("a", "b"))
    with pytest.raises(ValueError, match="must have at least one native voxel dim"):
        attach_voxel_to_world_index(no_voxel_dims, np.eye(3))


def test_get_voxel_to_world_affine_raises_without_voxel_to_world_geometry() -> None:
    """`get_voxel_to_world_affine` raises for a DataArray without a voxel-to-world index."""
    data = xr.DataArray(np.zeros((3, 4)), dims=("y", "x"))

    with pytest.raises(ValueError, match="must have a voxel-to-world index"):
        get_voxel_to_world_affine(data)


def test_restore_world_coords_rebuilds_geometry_after_expand_dims() -> None:
    """Restoring a dimension fixed by isel rebuilds identical world coordinates."""
    result = _axis_aligned_result()
    fixed = result.isel(k=1)

    assert restore_voxel_to_world_index(fixed) is fixed

    expanded = fixed.expand_dims(k=[2])
    restored = restore_voxel_to_world_index(expanded)

    assert restored.coords["z"].dims == ("k", "j", "i")
    assert_allclose(restored.coords["z"].values, result.isel(k=[1]).coords["z"].values)


def test_get_voxel_to_world_world_coord_names_falls_back_to_plain_coords() -> None:
    """World coordinate names are inferred from plain coords without an index.

    This covers restoring a DataArray that carries dense world coordinates matching
    the voxel dims (e.g. after an operation that drops the `VoxelToWorldIndex` but
    keeps the coordinate arrays) rather than a live `VoxelToWorldIndex`.
    """
    plain = xr.DataArray(np.zeros((2, 3, 4)), dims=("k", "j", "i")).assign_coords(
        z=(("k", "j", "i"), np.zeros((2, 3, 4))),
        y=(("k", "j", "i"), np.zeros((2, 3, 4))),
        x=(("k", "j", "i"), np.zeros((2, 3, 4))),
    )

    assert get_voxel_to_world_coord_names(plain) == ("z", "y", "x")


def test_is_pose_dependent_true_for_stacked_affine() -> None:
    """`is_pose_dependent` is True while a per-pose affine stack is active."""
    result = _pose_dependent_result()

    index = result.xindexes["z"]
    assert isinstance(index, VoxelToWorldIndex)
    assert index.is_pose_dependent


def test_is_pose_dependent_false_after_scalar_pose_selection() -> None:
    """`is_pose_dependent` is False once `pose` has been reduced to a scalar."""
    result = _pose_dependent_result()

    fixed = result.isel(pose=0)

    index = fixed.xindexes["z"]
    assert isinstance(index, VoxelToWorldIndex)
    assert not index.is_pose_dependent


def test_join_returns_self_for_equal_index() -> None:
    """`join` returns this index unchanged when `other` represents the same grid."""
    da = _simple_voxel_to_world_result()

    index = da.xindexes["z"]
    other = da.copy(deep=True).xindexes["z"]

    assert index.join(other) is index


def test_reindex_like_returns_empty_indexers_for_equal_index() -> None:
    """`reindex_like` returns no positional indexers for an equal grid."""
    da = _simple_voxel_to_world_result()

    index = da.xindexes["z"]
    other = da.copy(deep=True).xindexes["z"]

    assert index.reindex_like(other) == {}


def test_equals_returns_false_for_non_voxel_to_world_index() -> None:
    """`equals` returns False when compared against a different Index type."""
    da = _simple_voxel_to_world_result()

    assert da.xindexes["z"].equals(da.xindexes["k"]) is False


def test_sel_returns_empty_result_when_no_world_labels_given() -> None:
    """`sel({})` on pose-dependent geometry resolves to no indexers, not an error."""
    result = _pose_dependent_result()

    index = result.xindexes["z"]

    assert index.sel({}).dim_indexers == {}


def test_reverse_rejects_pose_dependent_transform_directly() -> None:
    """`VoxelToWorldTransform.reverse` raises for pose-dependent geometry directly.

    `VoxelToWorldIndex.sel` already guards this before ever calling `reverse`, but
    `reverse` documents (and must enforce) the same precondition independently.
    """
    transform = VoxelToWorldTransform(
        {"k": np.arange(2.0), "j": np.arange(3.0), "i": np.arange(4.0)},
        np.stack([np.eye(4), np.eye(4)]),
        pose_coord=np.array([0, 1]),
    )

    with pytest.raises(ValueError, match="pose-independent"):
        transform.reverse({"z": 0.0, "y": 0.0, "x": 0.0})


def test_isel_pose_indexer_ignored_when_pose_independent() -> None:
    """A `pose` indexer is a no-op on an already pose-independent transform."""
    transform = VoxelToWorldTransform(
        {"k": np.arange(2.0), "j": np.arange(3.0), "i": np.arange(4.0)}, np.eye(4)
    )

    result = transform.isel({"pose": 0, "k": 0})

    assert result is not None
    assert result.pose_coord is None
    assert "k" not in result.voxel_coords


def test_isel_pose_multidim_variable_indexer_returns_none() -> None:
    """A multi-dimensional `pose` Variable indexer is unsupported and returns None."""
    transform = VoxelToWorldTransform(
        {"k": np.arange(2.0), "j": np.arange(3.0), "i": np.arange(4.0)},
        np.stack([np.eye(4), np.eye(4)]),
        pose_coord=np.array([0, 1]),
    )

    result = transform.isel(
        {"pose": xr.Variable(("a", "b"), np.zeros((2, 2), dtype=int))}
    )

    assert result is None


def test_isel_pose_unsupported_indexer_type_returns_none() -> None:
    """An unsupported `pose` indexer type (not Variable/list/tuple/slice/array)
    returns None."""
    transform = VoxelToWorldTransform(
        {"k": np.arange(2.0), "j": np.arange(3.0), "i": np.arange(4.0)},
        np.stack([np.eye(4), np.eye(4)]),
        pose_coord=np.array([0, 1]),
    )

    result = transform.isel({"pose": object()})

    assert result is None


def test_isel_pose_fancy_index_yielding_non_1d_returns_none() -> None:
    """A `pose` fancy index producing a non-1D result returns None."""
    transform = VoxelToWorldTransform(
        {"k": np.arange(2.0), "j": np.arange(3.0), "i": np.arange(4.0)},
        np.stack([np.eye(4), np.eye(4)]),
        pose_coord=np.array([0, 1]),
    )

    result = transform.isel({"pose": np.zeros((2, 2), dtype=int)})

    assert result is None


def test_concat_rejects_non_pose_dim() -> None:
    """`VoxelToWorldIndex.concat` only supports concatenating along `pose`."""
    da = _simple_voxel_to_world_result()
    index = da.xindexes["z"]

    with pytest.raises(ValueError, match="only supports concat along 'pose'"):
        VoxelToWorldIndex.concat([index, index], dim="k")


def test_concat_rejects_pose_independent_inputs() -> None:
    """`VoxelToWorldIndex.concat` requires every input to already be pose-dependent."""
    da = _simple_voxel_to_world_result()
    index = da.xindexes["z"]

    with pytest.raises(ValueError, match="every input must already be pose-dependent"):
        VoxelToWorldIndex.concat([index, index], dim="pose")


def test_concat_rejects_mismatched_spatial_geometry_directly() -> None:
    """`VoxelToWorldIndex.concat` rejects mismatched fixed voxel coordinates.

    Called directly (not through `xr.concat`) so a mismatch that isn't already
    caught by xarray's own alignment step is exercised.
    """
    first = _pose_dependent_result()
    second = _pose_dependent_result().isel(k=0)  # fixes k, unlike first.

    with pytest.raises(ValueError, match="different spatial geometry"):
        VoxelToWorldIndex.concat(
            [first.xindexes["z"], second.xindexes["z"]], dim="pose"
        )


def test_update_voxel_to_world_coord_attrs_ignores_unknown_names() -> None:
    """Unknown coordinate names in `attrs_by_name` are silently skipped."""
    da = _simple_voxel_to_world_result()

    result = update_voxel_to_world_coord_attrs(da, {"not_a_coord": {"units": "mm"}})

    assert "not_a_coord" not in result.coords


def test_get_voxel_to_world_world_coord_names_defaults_when_coords_are_incomplete() -> (
    None
):
    """Default `z`/`y`/`x` names are returned when plain world coords are incomplete.

    Only a matching subset of the default-named world coordinates is present (`y`
    and `x` are missing), so the fallback cannot confirm all of them and returns the
    plain default names instead.
    """
    plain = xr.DataArray(np.zeros((2, 3, 4)), dims=("k", "j", "i")).assign_coords(
        z=(("k", "j", "i"), np.zeros((2, 3, 4))),
    )

    assert get_voxel_to_world_coord_names(plain) == ("z", "y", "x")
