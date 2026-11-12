"""Tests for the bounding box, geometry center and transforms in `xarray.affine`."""

import warnings

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from confusius.registration import extract_motion_parameters
from confusius.xarray import (
    apply_rotation,
    apply_translation,
    apply_zoom,
    create_voxeldata,
    get_bounding_box,
    get_geometry_center,
)

OBLIQUE_AFFINE = np.array(
    [
        [0.3, 0.1, 0.0, -1.2],
        [0.05, 0.25, 0.02, 2.0],
        [0.0, 0.04, 0.5, -0.7],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
"""Oblique voxel-to-world affine whose world coordinates mix all voxel dims."""


VOXEL_TO_WORLD = np.array(
    [
        [0.5, 0.0, 0.0, 1.0],
        [0.0, 0.2, 0.0, -2.0],
        [0.0, 0.0, 0.1, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
"""Axis-aligned voxel-to-world affine shared by the apply_* tests."""

ROTATED_POSE = (
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    @ VOXEL_TO_WORLD
)
"""VOXEL_TO_WORLD rotated 90 degrees in the (y, x) plane: same column norms, not parallel."""


def _volume(rng, **kwargs):
    return create_voxeldata(
        rng.random((2, 3, 4)),
        dims=("k", "j", "i"),
        voxel_to_world=VOXEL_TO_WORLD,
        **kwargs,
    )


def _world_affine(result, original_voxel_to_world=VOXEL_TO_WORLD):
    """Return the world-space matrix that `result` applied on top of the original."""
    return result.fusi.affine.voxel_to_world @ np.linalg.inv(original_voxel_to_world)


def _pose_volume(rng):
    return create_voxeldata(
        rng.random((2, 2, 3, 4)),
        dims=("pose", "k", "j", "i"),
        pose=[0, 1],
        voxel_to_world=np.stack([VOXEL_TO_WORLD, ROTATED_POSE]),
    )


def _pad_half_voxel(values: np.ndarray) -> np.ndarray:
    """Extend a uniform 1D voxel coordinate by half a step at both ends."""
    half = 0.5 * (values[1] - values[0]) if len(values) > 1 else 0.5
    return np.concatenate([[values[0] - half], values, [values[-1] + half]])


def _materialized_edge_bounds(data: xr.DataArray) -> np.ndarray:
    """Reference implementation: map every point of the voxel grid extended by half a
    voxel at both ends of each dim through the affine, then reduce."""
    padded = [_pad_half_voxel(data.coords[d].values) for d in ("k", "j", "i")]
    grid = np.stack(np.meshgrid(*padded, indexing="ij"), axis=-1).reshape(-1, 3)
    affine = data.fusi.affine.voxel_to_world
    world = grid @ affine[:3, :3].T + affine[:3, 3]
    return np.stack([world.min(axis=0), world.max(axis=0)])


def test_oblique_bounds_match_materialized_coordinates(rng):
    """Corner-mapped bounds equal the full-grid reduction of the voxel edges for an
    oblique affine."""
    data = create_voxeldata(
        rng.random((4, 5, 6)), dims=("k", "j", "i"), voxel_to_world=OBLIQUE_AFFINE
    )
    bbox = get_bounding_box(data)
    assert bbox.dims == ("bound", "component")
    assert list(bbox.coords["bound"].values) == ["min", "max"]
    assert list(bbox.coords["component"].values) == ["z", "y", "x"]
    assert_allclose(bbox.values, _materialized_edge_bounds(data))


def test_axis_aligned_bounds_are_voxel_edges(rng):
    """Axis-aligned geometry gives origin minus half a voxel to the last voxel's far
    edge, with units."""
    data = create_voxeldata(
        rng.random((4, 6, 8)),
        dims=("k", "j", "i"),
        spacing=(0.2, 0.1, 0.05),
        origin=(1.0, 2.0, 3.0),
    )
    bbox = get_bounding_box(data)
    expected = np.array(
        [
            [1.0 - 0.1, 2.0 - 0.05, 3.0 - 0.025],
            [1.0 + 3 * 0.2 + 0.1, 2.0 + 5 * 0.1 + 0.05, 3.0 + 7 * 0.05 + 0.025],
        ]
    )
    assert_allclose(bbox.values, expected)
    assert bbox.attrs["units"] == "mm"


def test_strided_voxel_coordinates_use_coordinate_step(rng):
    """Subsampled voxel coordinates (step 2) pad by half of *their* step, not 0.5."""
    data = create_voxeldata(
        rng.random((4, 6, 8)),
        dims=("k", "j", "i"),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    strided = data.isel(i=slice(None, None, 2))  # i = 0, 2, 4, 6.
    bbox = get_bounding_box(strided)
    assert_allclose(bbox.sel(component="x").values, [-1.0, 7.0])


def test_irregular_voxel_coordinates_raise(rng):
    """Irregular voxel coordinates have no defined voxel extent and are rejected."""
    data = create_voxeldata(
        rng.random((3, 4, 3)),
        dims=("k", "j", "i"),
        k=[1, 5, 6],
        j=[0, 1, 2, 9],
        i=[0, 2, 3],
        voxel_to_world=OBLIQUE_AFFINE,
    )
    with pytest.raises(ValueError, match="irregular"):
        get_bounding_box(data)


def test_singleton_dim_slice(rng):
    """A size-1 voxel dim (canonical 2D slice) spans one voxel along the affine
    column, like `fusi.spacing`."""
    data = create_voxeldata(
        rng.random((1, 5, 6)), dims=("k", "j", "i"), voxel_to_world=OBLIQUE_AFFINE
    )
    bbox = get_bounding_box(data)
    assert bbox.dims == ("bound", "component")
    assert_allclose(bbox.values, _materialized_edge_bounds(data))


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
        [_materialized_edge_bounds(data.isel(pose=p)) for p in range(2)]
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


def test_geometry_center_of_axis_aligned_data():
    """The center is the midpoint of the voxel centers, with its labels and units."""
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [10.0, 20.0, 30.0]
    data = create_voxeldata(
        np.zeros((2, 3, 4)), dims=("k", "j", "i"), voxel_to_world=affine, units="m"
    )
    center = get_geometry_center(data)
    assert center.dims == ("component",)
    assert list(center.coords["component"].values) == ["z", "y", "x"]
    assert center.attrs["units"] == "m"
    assert_allclose(center.values, [11.0, 23.0, 36.0])


def test_geometry_center_of_oblique_data_matches_materialized_bounds(rng):
    data = create_voxeldata(
        rng.random((4, 5, 6)), dims=("k", "j", "i"), voxel_to_world=OBLIQUE_AFFINE
    )
    assert_allclose(
        get_geometry_center(data).values, _materialized_edge_bounds(data).mean(axis=0)
    )


def test_geometry_center_of_pose_dependent_data_encloses_all_poses(rng):
    shifted = OBLIQUE_AFFINE.copy()
    shifted[:3, 3] += [5.0, -2.0, 1.5]
    data = create_voxeldata(
        rng.random((2, 3, 4, 5)),
        dims=("pose", "k", "j", "i"),
        pose=[10, 20],
        voxel_to_world=np.stack([OBLIQUE_AFFINE, shifted]),
    )
    bounds = np.stack([_materialized_edge_bounds(data.isel(pose=p)) for p in range(2)])
    center = get_geometry_center(data)
    assert center.dims == ("component",)
    assert_allclose(
        center.values, (bounds[:, 0].min(axis=0) + bounds[:, 1].max(axis=0)) / 2
    )
    assert_allclose(
        get_geometry_center(data.isel(pose=1)).values, bounds[1].mean(axis=0)
    )


def test_geometry_center_with_irregular_coordinates_raises(rng):
    data = create_voxeldata(
        rng.random((4, 5, 6)), dims=("k", "j", "i"), voxel_to_world=OBLIQUE_AFFINE
    ).isel(i=[0, 1, 3])
    with pytest.raises(ValueError, match="irregular"):
        get_geometry_center(data)


def test_geometry_center_accessor_matches_function(rng):
    """The `.fusi.affine.geometry_center` property returns the same point."""
    data = create_voxeldata(
        rng.random((4, 5, 6)), dims=("k", "j", "i"), voxel_to_world=OBLIQUE_AFFINE
    )
    assert_allclose(
        data.fusi.affine.geometry_center.values, get_geometry_center(data).values
    )


@pytest.mark.parametrize(
    ("transform", "args"),
    [
        (apply_translation, ((1.0, 2.0, 3.0),)),
        (apply_rotation, (30.0, "x")),
        (apply_zoom, (2.0,)),
    ],
    ids=["translation", "rotation", "zoom"],
)
def test_transforms_preserve_voxel_values_labels_and_input(rng, transform, args):
    """World-space transforms preserve voxel data and leave the input untouched."""
    data = _volume(rng, attrs={"affines": {"world_to_lab": np.eye(4).tolist()}}).isel(
        k=slice(None, None, -1), j=slice(1, None), i=slice(None, None, 2)
    )
    original = data.copy(deep=True)

    result = transform(data, *args)

    assert result is not data
    assert result.dims == original.dims
    assert_array_equal(result.values, original.values)
    for dim in ("k", "j", "i"):
        assert_array_equal(result.coords[dim].values, original.coords[dim].values)
    xr.testing.assert_identical(data, original)


class TestApplyTranslation:
    """Tests for `apply_translation`."""

    def test_shifts_world_origin(self, rng):
        result = apply_translation(_volume(rng), (1.0, 2.0, 3.0))
        assert_allclose(result.fusi.affine.voxel_to_world[:3, 3], [2.0, 0.0, 6.0])
        assert_allclose(
            result.fusi.affine.voxel_to_world[:3, :3], VOXEL_TO_WORLD[:3, :3]
        )
        assert_allclose(result.coords["z"].isel(k=0, j=0, i=0), 2.0)

    def test_integer_translation_gives_float_affine(self, rng):
        result = apply_translation(_volume(rng), (1, 2, 3))
        assert result.fusi.affine.voxel_to_world.dtype == np.float64
        assert_allclose(result.fusi.affine.voxel_to_world[:3, 3], [2.0, 0.0, 6.0])

    def test_reexpresses_stored_affines(self, rng):
        world_to_lab = np.diag([1.0, 2.0, 3.0, 1.0])
        data = _volume(rng, attrs={"affines": {"world_to_lab": world_to_lab}})
        result = apply_translation(data, (1.0, 2.0, 3.0))
        shift = np.eye(4)
        shift[:3, 3] = [1.0, 2.0, 3.0]
        assert_allclose(
            result.attrs["affines"]["world_to_lab"], world_to_lab @ np.linalg.inv(shift)
        )

    def test_no_affines_attr_stays_absent(self, rng):
        result = apply_translation(_volume(rng), (1.0, 0.0, 0.0))
        assert "affines" not in result.attrs

    def test_broadcasts_over_poses(self, rng):
        result = apply_translation(_pose_volume(rng), (1.0, 2.0, 3.0))
        affine = result.fusi.affine.voxel_to_world
        assert affine.shape == (2, 4, 4)
        assert_allclose(affine[0, :3, 3], VOXEL_TO_WORLD[:3, 3] + [1.0, 2.0, 3.0])
        assert_allclose(affine[1, :3, 3], ROTATED_POSE[:3, 3] + [1.0, 2.0, 3.0])

    def test_inplace_returns_same_object(self, rng):
        data = _volume(rng)
        result = apply_translation(data, (1.0, 0.0, 0.0), inplace=True)
        assert result is data
        assert_allclose(data.fusi.affine.voxel_to_world[0, 3], 2.0)

    @pytest.mark.parametrize(
        "translation",
        [
            (1.0, 2.0),
            (1.0, 2.0, 3.0, 4.0),
            (1.0, [2.0, 3.0], 4.0),
            (np.nan, 0.0, 0.0),
            ("1.0", "2.0", "3.0"),
            (True, False, False),
            {"z": 1.0, "y": 0.0, "x": 0.0},
        ],
    )
    def test_invalid_translation_raises(self, rng, translation):
        with pytest.raises(ValueError, match="translation"):
            apply_translation(_volume(rng), translation)

    def test_non_voxeldata_raises(self):
        plain = xr.DataArray(np.zeros((2, 3, 4)), dims=("k", "j", "i"))
        with pytest.raises(ValueError, match="voxel-to-world index"):
            apply_translation(plain, (1.0, 0.0, 0.0))


class TestApplyRotation:
    """Tests for apply_rotation."""

    @pytest.mark.parametrize(
        ("axis", "start", "end"),
        [
            ("z", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ("y", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
            ("x", (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ],
    )
    def test_right_hand_rule_in_zyx_order(self, rng, axis, start, end):
        # Points are (z, y, x): +90 deg about z maps +y onto +x, about y maps +x onto
        # +z, and about x maps +z onto +y.
        result = apply_rotation(_volume(rng), 90.0, axis, center=(0.0, 0.0, 0.0))
        assert_allclose(_world_affine(result) @ [*start, 1.0], [*end, 1.0], atol=1e-12)

    def test_matches_scipy_reference(self, rng):
        axis_zyx = np.array([1.0, 2.0, 2.0]) / 3.0
        result = apply_rotation(_volume(rng), 37.0, axis_zyx, center=(0.0, 0.0, 0.0))
        expected = Rotation.from_rotvec(np.deg2rad(37.0) * axis_zyx).as_matrix()
        assert_allclose(_world_affine(result)[:3, :3], expected, atol=1e-12)
        assert_allclose(_world_affine(result)[:3, 3], 0.0, atol=1e-12)

    def test_chained_rotations_match_motion_parameters(self, rng):
        angles = np.deg2rad([30.0, 20.0, 10.0])
        result = _volume(rng)
        for angle, axis in zip(angles, "zyx", strict=True):
            result = apply_rotation(result, angle, axis, degrees=False)
        params = extract_motion_parameters([_world_affine(result)])
        assert_allclose(params[0, :3], angles, atol=1e-12)

    @pytest.mark.parametrize(
        ("angles", "axes"),
        [
            ((30, 20, 10), "zyx"),
            ((30.0, 20.0, 10.0), "zyz"),
            (np.array([30.0, 20.0]), "yx"),
            ((15.0,), "x"),
        ],
    )
    def test_axis_sequence_matches_chained_calls(self, rng, angles, axes):
        center = (1.0, -2.0, 3.0)
        chained = _volume(rng)
        for angle, axis in zip(angles, axes, strict=True):
            chained = apply_rotation(chained, angle, axis, center=center)
        result = apply_rotation(_volume(rng), angles, axes, center=center)
        assert_allclose(
            result.fusi.affine.voxel_to_world,
            chained.fusi.affine.voxel_to_world,
            atol=1e-12,
        )

    def test_axis_sequence_matches_motion_parameters(self, rng):
        angles = np.deg2rad([30.0, 20.0, 10.0])
        result = apply_rotation(_volume(rng), angles, "zyx", degrees=False)
        params = extract_motion_parameters([_world_affine(result)])
        assert_allclose(params[0, :3], angles, atol=1e-12)

    @pytest.mark.parametrize(
        ("angle", "axis"),
        [
            (10.0, "zy"),
            ((), "z"),
            ((10.0, 20.0), "z"),
            ((10.0, 20.0, 30.0), "zy"),
            ((10.0, 20.0), (0.0, 1.0, 0.0)),
            ((10.0, np.nan), "zy"),
            ((10.0, "20"), "zy"),
        ],
    )
    def test_invalid_angle_for_axis_sequence_raises(self, rng, angle, axis):
        with pytest.raises(ValueError, match="angle"):
            apply_rotation(_volume(rng), angle, axis)

    def test_radians_match_degrees(self, rng):
        deg = apply_rotation(_volume(rng), 90.0, "z")
        rad = apply_rotation(_volume(rng), np.pi / 2, "z", degrees=False)
        assert_allclose(
            deg.fusi.affine.voxel_to_world,
            rad.fusi.affine.voxel_to_world,
            atol=1e-12,
        )

    def test_vector_axis_matches_name_and_is_normalised(self, rng):
        by_name = apply_rotation(_volume(rng), 45.0, "y")
        by_vector = apply_rotation(_volume(rng), 45.0, (0.0, 5.0, 0.0))
        assert_allclose(
            by_name.fusi.affine.voxel_to_world,
            by_vector.fusi.affine.voxel_to_world,
            atol=1e-12,
        )

    def test_broadcasts_over_poses(self, rng):
        result = apply_rotation(_pose_volume(rng), 90.0, "x")
        affine = result.fusi.affine.voxel_to_world
        single = apply_rotation(_volume(rng), 90.0, "x").fusi.affine.voxel_to_world
        assert affine.shape == (2, 4, 4)
        assert_allclose(affine[0], single, atol=1e-12)
        # Same world-space matrix was applied on top of each pose's own affine.
        assert_allclose(
            affine[1] @ np.linalg.inv(ROTATED_POSE),
            single @ np.linalg.inv(VOXEL_TO_WORLD),
            atol=1e-12,
        )

    def test_chaining_applies_in_call_order(self, rng):
        # A pivot that moves with the data would make the two orders commute, so pin
        # the pivot to the world origin.
        world_origin = (0.0, 0.0, 0.0)
        rotated_then_shifted = apply_translation(
            apply_rotation(_volume(rng), 90.0, "x", center=world_origin),
            (1.0, 2.0, 3.0),
        )
        shift = np.eye(4)
        shift[:3, 3] = [1.0, 2.0, 3.0]
        rotation = _world_affine(
            apply_rotation(_volume(rng), 90.0, "x", center=world_origin)
        )
        assert_allclose(
            rotated_then_shifted.fusi.affine.voxel_to_world,
            shift @ rotation @ VOXEL_TO_WORLD,
            atol=1e-12,
        )
        shifted_then_rotated = apply_rotation(
            apply_translation(_volume(rng), (1.0, 2.0, 3.0)),
            90.0,
            "x",
            center=world_origin,
        )
        assert not np.allclose(
            shifted_then_rotated.fusi.affine.voxel_to_world,
            rotated_then_shifted.fusi.affine.voxel_to_world,
        )

    @pytest.mark.parametrize(
        "axis",
        [
            "k",
            "zk",
            "",
            "ZYX",
            (0.0, 0.0, 0.0),
            (1.0, 0.0),
            (np.nan, 0.0, 1.0),
            (True, False, False),
            (1.7e308, 1.7e308, 0.0),
        ],
    )
    def test_invalid_axis_raises(self, rng, axis):
        with pytest.raises(ValueError, match="axis"):
            apply_rotation(_volume(rng), 10.0, axis)

    @pytest.mark.parametrize(
        "angle", [np.nan, "90", np.array("90"), b"90", 1j, None, True, [1.0, 2.0]]
    )
    def test_invalid_angle_raises(self, rng, angle):
        with pytest.raises(ValueError, match="angle"):
            apply_rotation(_volume(rng), angle, "x")


@pytest.mark.parametrize(
    "transform",
    [
        lambda data, **kwargs: apply_rotation(data, 30.0, "y", **kwargs),
        lambda data, **kwargs: apply_zoom(data, 3.0, **kwargs),
    ],
    ids=["rotation", "zoom"],
)
class TestCenterOptions:
    """Tests for the `center` options shared by apply_rotation and apply_zoom."""

    def test_default_is_grid_origin(self, rng, transform):
        data = _volume(rng)
        assert_allclose(
            transform(data).fusi.affine.voxel_to_world,
            transform(data, center="origin").fusi.affine.voxel_to_world,
        )

    def test_origin_is_first_sampled_voxel(self, rng, transform):
        # Cropping moves the first voxel away from the affine's translation.
        data = _volume(rng).isel(k=slice(1, None), i=slice(2, None))
        first_voxel = VOXEL_TO_WORLD @ [1.0, 0.0, 2.0, 1.0]
        result = transform(data, center="origin")
        assert_allclose(_world_affine(result) @ first_voxel, first_voxel, atol=1e-12)
        assert not np.allclose(
            _world_affine(result) @ [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]
        )

    def test_origin_ignores_extra_dims_without_coordinates(self, rng, transform):
        data = _volume(rng).expand_dims("component")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = transform(data)
        assert_allclose(
            result.fusi.affine.voxel_to_world,
            transform(_volume(rng)).fusi.affine.voxel_to_world,
        )

    def test_geometry_keeps_bounding_box_center(self, rng, transform):
        data = _volume(rng)
        center = get_geometry_center(data).values
        result = transform(data, center="geometry")
        assert_allclose(
            _world_affine(result) @ [*center, 1.0], [*center, 1.0], atol=1e-12
        )
        assert_allclose(get_geometry_center(result).values, center, atol=1e-12)

    def test_point_is_fixed(self, rng, transform):
        point = np.array([5.0, -7.0, 2.0])
        result = transform(_volume(rng), center=point)
        assert_allclose(
            _world_affine(result) @ [*point, 1.0], [*point, 1.0], atol=1e-12
        )

    def test_world_origin_is_a_point(self, rng, transform):
        result = transform(_volume(rng), center=(0.0, 0.0, 0.0))
        assert_allclose(_world_affine(result)[:3, 3], 0.0, atol=1e-12)

    def test_accepts_center_dataarray(self, rng, transform):
        data = _volume(rng)
        assert_allclose(
            transform(
                data, center=get_geometry_center(data)
            ).fusi.affine.voxel_to_world,
            transform(data, center="geometry").fusi.affine.voxel_to_world,
        )

    def test_pose_stack_origin_is_first_voxel_of_first_pose(self, rng, transform):
        data = _pose_volume(rng)
        assert_allclose(
            transform(data, center="origin").fusi.affine.voxel_to_world,
            transform(data, center=VOXEL_TO_WORLD[:3, 3]).fusi.affine.voxel_to_world,
        )

    def test_pose_stack_geometry_is_center_of_enclosing_box(self, rng, transform):
        data = _pose_volume(rng)
        center = get_geometry_center(data).values
        affine = transform(data, center="geometry").fusi.affine.voxel_to_world
        applied = affine[0] @ np.linalg.inv(VOXEL_TO_WORLD)
        assert_allclose(applied @ [*center, 1.0], [*center, 1.0], atol=1e-12)
        # One matrix is applied on top of every pose's own affine.
        assert_allclose(affine[1] @ np.linalg.inv(ROTATED_POSE), applied, atol=1e-12)

    @pytest.mark.parametrize(
        "center",
        ["middle", "", (1.0, 2.0), (np.nan, 0.0, 0.0), (True, False, False), None],
    )
    def test_invalid_center_raises(self, rng, transform, center):
        with pytest.raises(ValueError, match="center"):
            transform(_volume(rng), center=center)

    @pytest.mark.parametrize("center", ["origin", "geometry", (0.0, 0.0, 0.0)])
    def test_non_voxeldata_raises(self, transform, center):
        plain = xr.DataArray(np.zeros((2, 3, 4)), dims=("k", "j", "i"))
        with pytest.raises(ValueError, match="voxel"):
            transform(plain, center=center)

    def test_geometry_with_irregular_coordinates_raises(self, rng, transform):
        data = _volume(rng).isel(i=[0, 1, 3])
        with pytest.raises(ValueError, match="irregular"):
            transform(data, center="geometry")


class TestApplyZoom:
    """Tests for apply_zoom."""

    def test_scalar_zoom_scales_spacing(self, rng):
        result = apply_zoom(_volume(rng), 2)
        assert result.fusi.affine.voxel_to_world.dtype == np.float64
        assert result.fusi.spacing == pytest.approx({"k": 1.0, "j": 0.4, "i": 0.2})

    def test_per_axis_zoom(self, rng):
        result = apply_zoom(_volume(rng), (1.0, 2.0, 4.0), center=(0.0, 0.0, 0.0))
        assert result.fusi.spacing == pytest.approx({"k": 0.5, "j": 0.4, "i": 0.4})
        assert_allclose(result.fusi.affine.voxel_to_world[:3, 3], [1.0, -4.0, 12.0])

    def test_negative_factor_flips_direction(self, rng):
        result = apply_zoom(_volume(rng), (1.0, 1.0, -1.0))
        assert_allclose(result.fusi.direction[:, 2], [0.0, 0.0, -1.0])
        assert result.fusi.spacing == pytest.approx({"k": 0.5, "j": 0.2, "i": 0.1})

    def test_scalar_zoom_broadcasts_over_poses(self, rng):
        result = apply_zoom(_pose_volume(rng), 2.0)
        assert result.fusi.affine.voxel_to_world.shape == (2, 4, 4)
        assert_allclose(
            result.fusi.affine.voxel_to_world[1, :3, :3], 2.0 * ROTATED_POSE[:3, :3]
        )

    def test_zoom_with_unequal_transformed_pose_scales_raises(self, rng):
        with pytest.raises(ValueError, match="equal spatial scale"):
            apply_zoom(_pose_volume(rng), (1.0, 2.0, 1.0))

    @pytest.mark.parametrize(
        "zoom",
        [
            0.0,
            (1.0, 0.0, 1.0),
            (1.0, 2.0),
            (np.inf, 1.0, 1.0),
            np.nan,
            np.inf,
            "2",
            np.array("2"),
            True,
            None,
            {"z": 1.0},
        ],
    )
    def test_invalid_zoom_raises(self, rng, zoom):
        with pytest.raises(ValueError, match="zoom"):
            apply_zoom(_volume(rng), zoom)
