"""Unit tests for affine matrix decomposition utilities."""

from itertools import permutations
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.spatial.transform import Rotation

from confusius.registration.affines import (
    affine_to_sitk_linear_transform,
    compose_affine,
    decompose_affine,
    sitk_linear_transform_to_affine,
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

        assert_array_almost_equal(sitk_linear_transform_to_affine(sitk_transform), affine)

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

        expected = (
            sitk_linear_transform_to_affine(outer)
            @ sitk_linear_transform_to_affine(inner)
        )

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
