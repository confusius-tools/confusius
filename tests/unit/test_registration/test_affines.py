"""Unit tests for affine matrix decomposition utilities."""

from itertools import permutations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.spatial.transform import Rotation

from confusius.registration.affines import (
    compose_affine,
    decompose_affine,
    get_euler_xyz_from_rotation_matrix,
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


class TestGetEulerXYZFromRotationMatrix:
    """Tests for `get_euler_xyz_from_rotation_matrix`."""

    def test_matches_scipy_for_random_rotations(self):
        """Extracted XYZ angles match SciPy away from singularities."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            angles = rng.uniform(-1.2, 1.2, size=3)
            matrix = Rotation.from_euler("xyz", angles).as_matrix()
            recovered = get_euler_xyz_from_rotation_matrix(matrix)
            assert_array_almost_equal(recovered, angles)

    def test_rejects_wrong_shape(self):
        """Only 3x3 rotation matrices are accepted."""
        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            get_euler_xyz_from_rotation_matrix(np.eye(4))


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
