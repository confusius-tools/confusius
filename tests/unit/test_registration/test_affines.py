"""Unit tests for affine matrix decomposition utilities."""

from itertools import permutations
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from scipy.spatial.transform import Rotation

from confusius.registration import (
    apply_rotate,
    apply_translate,
    apply_zoom,
    extract_motion_parameters,
)
from confusius.registration.affines import (
    affine_to_sitk_linear_transform,
    compose_affine,
    decompose_affine,
    sitk_linear_transform_to_affine,
)
from confusius.xarray import create_voxeldata, get_bounding_box

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


class TestCompose:
    """Tests for _compose."""

    def test_wrong_R_shape_raises(self):
        """Mismatched rotation matrix shape raises ValueError."""
        T = np.zeros(3)
        R = np.eye(2)
        Z = np.ones(3)
        with pytest.raises(ValueError, match="Expected shape"):
            compose_affine(T, R, Z)

    def test_without_shear_uses_plain_rotation_and_zoom(self):
        """Omitting shear keeps the affine equal to `R @ diag(Z)` plus translation."""
        T = np.array([1.0, 2.0, 3.0])
        R = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix()
        Z = np.array([2.0, 3.0, 4.0])

        affine = compose_affine(T, R, Z)

        assert_array_almost_equal(affine[:3, :3], R @ np.diag(Z))
        assert_array_equal(affine[:3, 3], T)

    def test_invalid_shear_length_raises(self):
        """Shear vectors must have a triangular-number length."""
        T = np.zeros(3)
        R = np.eye(3)
        Z = np.ones(3)

        with pytest.raises(ValueError, match="strange number of shear elements"):
            compose_affine(T, R, Z, np.array([0.1, 0.2]))


class TestSitkLinearTransformToAffine:
    """Tests for `sitk_linear_transform_to_affine`."""

    def test_affine_to_sitk_linear_transform_round_trips(self):
        """Affine matrices round-trip through SimpleITK affine transforms."""
        affine = np.array(
            [
                [1.0, 0.1, 0.0, 2.0],
                [0.0, 1.5, 0.2, 3.0],
                [0.0, 0.0, 2.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        sitk_transform = affine_to_sitk_linear_transform(affine)

        assert_array_almost_equal(
            sitk_linear_transform_to_affine(sitk_transform), affine
        )

    def test_composite_transform_multiplies_child_affines_in_order(self):
        """Composite transforms compose child affines in SimpleITK's application order."""
        import SimpleITK as sitk

        outer = sitk.TranslationTransform(3)
        outer.SetOffset((1.0, 2.0, 3.0))
        inner = sitk.Euler3DTransform()
        inner.SetRotation(0.05, 0.0, 0.0)
        inner.SetTranslation((0.5, 0.0, 0.0))

        composite = sitk.CompositeTransform(3)
        composite.AddTransform(outer)
        composite.AddTransform(inner)

        expected = sitk_linear_transform_to_affine(
            outer
        ) @ sitk_linear_transform_to_affine(inner)

        assert_array_almost_equal(sitk_linear_transform_to_affine(composite), expected)

    def test_identity_transform_returns_identity_affine(self):
        """Identity-like transforms convert to identity matrices."""

        class IdentityTransformStub:
            def GetDimension(self):
                return 3

            def GetName(self):
                return "IdentityTransform"

        affine = sitk_linear_transform_to_affine(cast(Any, IdentityTransformStub()))

        assert_array_equal(affine, np.eye(4))

    def test_unsupported_transform_type_raises(self):
        """Unexpected transform types raise a clear error."""

        class UnsupportedTransformStub:
            def GetDimension(self):
                return 3

            def GetName(self):
                return "ScaleTransform"

        with pytest.raises(ValueError, match="unsupported transform type"):
            sitk_linear_transform_to_affine(cast(Any, UnsupportedTransformStub()))


class TestDecompose44:
    """Tests for decompose_affine."""

    def test_round_trip_structured(self):
        """Decompose and recompose recovers original for structured affines."""
        for trans in permutations([10, 20, 30]):
            for rots in permutations([0.2, 0.3, 0.4]):
                for zooms in permutations([1.1, 1.9, 2.3]):
                    for shears in permutations([0.01, 0.04, 0.09]):
                        Rmat = Rotation.from_euler("xyz", rots).as_matrix()
                        M = compose_affine(
                            np.asarray(trans, dtype=float),
                            Rmat,
                            np.asarray(zooms, dtype=float),
                            np.asarray(shears, dtype=float),
                        )
                        T, R, Z, S = decompose_affine(M)
                        assert_array_almost_equal(compose_affine(T, R, Z, S), M)

    def test_identity_zero_shear(self):
        """Identity matrix decomposes to zero shear vector."""
        T, R, Z, S = decompose_affine(np.eye(4))
        assert_array_equal(S, np.zeros(3))

    def test_round_trip_random(self):
        """Decompose and recompose recovers original for random affines."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            M = rng.standard_normal((4, 4))
            M[-1] = [0, 0, 0, 1]
            T, R, Z, S = decompose_affine(M)
            assert_array_almost_equal(compose_affine(T, R, Z, S), M)

    def test_negative_determinant_flips_first_zoom_not_rotation(self):
        """Reflections are represented with a negative zoom and a proper rotation."""
        affine = np.diag([-2.0, 3.0, 4.0, 1.0])

        _T, R, Z, _S = decompose_affine(affine)

        assert np.linalg.det(R) > 0
        assert Z[0] < 0


class TestApplyTranslate:
    """Tests for `apply_translate`."""

    def test_shifts_world_origin(self, rng):
        result = apply_translate(_volume(rng), (1.0, 2.0, 3.0))
        assert_allclose(result.fusi.affine.voxel_to_world[:3, 3], [2.0, 0.0, 6.0])
        assert_allclose(
            result.fusi.affine.voxel_to_world[:3, :3], VOXEL_TO_WORLD[:3, :3]
        )
        assert_allclose(result.coords["z"].isel(k=0, j=0, i=0), 2.0)

    def test_integer_translation_gives_float_affine(self, rng):
        result = apply_translate(_volume(rng), (1, 2, 3))
        assert result.fusi.affine.voxel_to_world.dtype == np.float64
        assert_allclose(result.fusi.affine.voxel_to_world[:3, 3], [2.0, 0.0, 6.0])

    def test_reexpresses_stored_affines(self, rng):
        world_to_lab = np.diag([1.0, 2.0, 3.0, 1.0])
        data = _volume(rng, attrs={"affines": {"world_to_lab": world_to_lab}})
        result = apply_translate(data, (1.0, 2.0, 3.0))
        shift = np.eye(4)
        shift[:3, 3] = [1.0, 2.0, 3.0]
        assert_allclose(
            result.attrs["affines"]["world_to_lab"], world_to_lab @ np.linalg.inv(shift)
        )

    def test_no_affines_attr_stays_absent(self, rng):
        result = apply_translate(_volume(rng), (1.0, 0.0, 0.0))
        assert "affines" not in result.attrs

    def test_broadcasts_over_poses(self, rng):
        result = apply_translate(_pose_volume(rng), (1.0, 2.0, 3.0))
        affine = result.fusi.affine.voxel_to_world
        assert affine.shape == (2, 4, 4)
        assert_allclose(affine[0, :3, 3], VOXEL_TO_WORLD[:3, 3] + [1.0, 2.0, 3.0])
        assert_allclose(affine[1, :3, 3], ROTATED_POSE[:3, 3] + [1.0, 2.0, 3.0])

    def test_inplace_returns_same_object(self, rng):
        data = _volume(rng)
        result = apply_translate(data, (1.0, 0.0, 0.0), inplace=True)
        assert result is data
        assert_allclose(data.fusi.affine.voxel_to_world[0, 3], 2.0)

    @pytest.mark.parametrize(
        "translation",
        [
            (1.0, 2.0),
            (1.0, 2.0, 3.0, 4.0),
            (np.nan, 0.0, 0.0),
            {"z": 1.0, "y": 0.0, "x": 0.0},
        ],
    )
    def test_invalid_translation_raises(self, rng, translation):
        with pytest.raises(ValueError, match="translation"):
            apply_translate(_volume(rng), translation)

    def test_non_voxeldata_raises(self):
        plain = xr.DataArray(np.zeros((2, 3, 4)), dims=("k", "j", "i"))
        with pytest.raises(ValueError, match="voxel-to-world index"):
            apply_translate(plain, (1.0, 0.0, 0.0))


class TestApplyRotate:
    """Tests for apply_rotate."""

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
        result = apply_rotate(_volume(rng), 90.0, axis)
        assert_allclose(_world_affine(result) @ [*start, 1.0], [*end, 1.0], atol=1e-12)

    def test_matches_scipy_reference(self, rng):
        axis_zyx = np.array([1.0, 2.0, 2.0]) / 3.0
        result = apply_rotate(_volume(rng), 37.0, axis_zyx)
        expected = Rotation.from_rotvec(np.deg2rad(37.0) * axis_zyx).as_matrix()
        assert_allclose(_world_affine(result)[:3, :3], expected, atol=1e-12)
        assert_allclose(_world_affine(result)[:3, 3], 0.0, atol=1e-12)

    def test_chained_rotations_match_motion_parameters(self, rng):
        angles = np.deg2rad([30.0, 20.0, 10.0])
        result = _volume(rng)
        for angle, axis in zip(angles, "zyx", strict=True):
            result = apply_rotate(result, angle, axis, degrees=False)
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
            chained = apply_rotate(chained, angle, axis, center=center)
        result = apply_rotate(_volume(rng), angles, axes, center=center)
        assert_allclose(
            result.fusi.affine.voxel_to_world,
            chained.fusi.affine.voxel_to_world,
            atol=1e-12,
        )

    def test_axis_sequence_order_matters(self, rng):
        zyx = apply_rotate(_volume(rng), (30.0, 20.0, 10.0), "zyx")
        xyz = apply_rotate(_volume(rng), (10.0, 20.0, 30.0), "xyz")
        assert not np.allclose(
            zyx.fusi.affine.voxel_to_world, xyz.fusi.affine.voxel_to_world
        )

    def test_axis_sequence_matches_motion_parameters(self, rng):
        angles = np.deg2rad([30.0, 20.0, 10.0])
        result = apply_rotate(_volume(rng), angles, "zyx", degrees=False)
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
            apply_rotate(_volume(rng), angle, axis)

    def test_center_is_fixed_point(self, rng):
        center = np.array([1.0, -2.0, 3.0])
        result = apply_rotate(_volume(rng), 30.0, "y", center=center)
        assert_allclose(
            _world_affine(result) @ [*center, 1.0], [*center, 1.0], atol=1e-12
        )
        assert not np.allclose(
            _world_affine(result) @ [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]
        )

    def test_volume_center_pivot_keeps_bounding_box_center(self, rng):
        data = _volume(rng)
        center = get_bounding_box(data).mean("bound").values
        result = apply_rotate(data, 30.0, "y", center=center)
        assert_allclose(
            get_bounding_box(result).mean("bound").values, center, atol=1e-12
        )

    def test_radians_match_degrees(self, rng):
        deg = apply_rotate(_volume(rng), 90.0, "z")
        rad = apply_rotate(_volume(rng), np.pi / 2, "z", degrees=False)
        assert_allclose(
            deg.fusi.affine.voxel_to_world,
            rad.fusi.affine.voxel_to_world,
            atol=1e-12,
        )

    def test_vector_axis_matches_name_and_is_normalised(self, rng):
        by_name = apply_rotate(_volume(rng), 45.0, "y")
        by_vector = apply_rotate(_volume(rng), 45.0, (0.0, 5.0, 0.0))
        assert_allclose(
            by_name.fusi.affine.voxel_to_world,
            by_vector.fusi.affine.voxel_to_world,
            atol=1e-12,
        )

    def test_broadcasts_over_poses(self, rng):
        result = apply_rotate(_pose_volume(rng), 90.0, "x")
        affine = result.fusi.affine.voxel_to_world
        single = apply_rotate(_volume(rng), 90.0, "x").fusi.affine.voxel_to_world
        assert affine.shape == (2, 4, 4)
        assert_allclose(affine[0], single, atol=1e-12)
        # Same world-space matrix was applied on top of each pose's own affine.
        assert_allclose(
            affine[1] @ np.linalg.inv(ROTATED_POSE),
            single @ np.linalg.inv(VOXEL_TO_WORLD),
            atol=1e-12,
        )

    def test_chaining_applies_in_call_order(self, rng):
        rotated_then_shifted = apply_translate(
            apply_rotate(_volume(rng), 90.0, "x"), (1.0, 2.0, 3.0)
        )
        shift = np.eye(4)
        shift[:3, 3] = [1.0, 2.0, 3.0]
        rotation = _world_affine(apply_rotate(_volume(rng), 90.0, "x"))
        assert_allclose(
            rotated_then_shifted.fusi.affine.voxel_to_world,
            shift @ rotation @ VOXEL_TO_WORLD,
            atol=1e-12,
        )
        shifted_then_rotated = apply_rotate(
            apply_translate(_volume(rng), (1.0, 2.0, 3.0)), 90.0, "x"
        )
        assert not np.allclose(
            shifted_then_rotated.fusi.affine.voxel_to_world,
            rotated_then_shifted.fusi.affine.voxel_to_world,
        )

    @pytest.mark.parametrize(
        "axis", ["k", "zk", "", "ZYX", (0.0, 0.0, 0.0), (1.0, 0.0), (np.nan, 0.0, 1.0)]
    )
    def test_invalid_axis_raises(self, rng, axis):
        with pytest.raises(ValueError, match="axis"):
            apply_rotate(_volume(rng), 10.0, axis)

    @pytest.mark.parametrize("angle", [np.nan, "90", None, [1.0, 2.0]])
    def test_invalid_angle_raises(self, rng, angle):
        with pytest.raises(ValueError, match="angle"):
            apply_rotate(_volume(rng), angle, "x")

    def test_invalid_center_raises(self, rng):
        with pytest.raises(ValueError, match="center"):
            apply_rotate(_volume(rng), 10.0, "x", center=(1.0, 2.0))


class TestApplyZoom:
    """Tests for apply_zoom."""

    def test_scalar_zoom_scales_spacing(self, rng):
        result = apply_zoom(_volume(rng), 2)
        assert result.fusi.affine.voxel_to_world.dtype == np.float64
        assert result.fusi.spacing == pytest.approx({"k": 1.0, "j": 0.4, "i": 0.2})

    def test_per_axis_zoom(self, rng):
        result = apply_zoom(_volume(rng), (1.0, 2.0, 4.0))
        assert result.fusi.spacing == pytest.approx({"k": 0.5, "j": 0.4, "i": 0.4})
        assert_allclose(result.fusi.affine.voxel_to_world[:3, 3], [1.0, -4.0, 12.0])

    def test_center_is_fixed_point(self, rng):
        center = np.array([1.0, -2.0, 3.0])
        result = apply_zoom(_volume(rng), 3.0, center=center)
        assert_allclose(
            _world_affine(result) @ [*center, 1.0], [*center, 1.0], atol=1e-12
        )
        assert not np.allclose(
            _world_affine(result) @ [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]
        )

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

    def test_anisotropic_zoom_on_non_parallel_poses_raises(self, rng):
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
            None,
            {"z": 1.0},
        ],
    )
    def test_invalid_zoom_raises(self, rng, zoom):
        with pytest.raises(ValueError, match="zoom"):
            apply_zoom(_volume(rng), zoom)
