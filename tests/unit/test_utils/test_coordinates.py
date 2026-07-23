"""Unit tests for confusius._utils.coordinates affine helpers."""

import numpy as np
import pytest
import xarray as xr

from confusius._utils.coordinates import (
    get_affine_in_axis_aligned_space,
    get_axis_aligned_affine,
    get_grid_kwargs_from_dataarray,
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


def test_reexpress_affine_against_own_frame_strips_axis_aligned_part():
    """Re-expressing an affine against its own (T, Z) yields the orientation block.

    For affine = [[D @ diag(Z), T]], reexpress(affine, T, Z) must equal
    [[D, T - D @ T]] (the orientation-only physical affine).
    """
    D = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg.
    Z = np.array([2.0, 3.0, 4.0])
    T = np.array([10.0, 20.0, 30.0])
    affine = np.eye(4)
    affine[:3, :3] = D @ np.diag(Z)
    affine[:3, 3] = T

    result = get_affine_in_axis_aligned_space(affine, T, Z)

    expected = np.eye(4)
    expected[:3, :3] = D
    expected[:3, 3] = T - D @ T
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_reexpress_affine_maps_reference_frame_to_world():
    """reexpress(M, T, Z) @ (get_axis_aligned_affine(T, Z) @ p) == M @ p for any point p."""
    rng = np.random.default_rng(0)
    M = np.eye(4)
    M[:3, :3] = rng.standard_normal((3, 3))
    M[:3, 3] = rng.standard_normal(3)
    T = np.array([1.0, -2.0, 3.0])
    Z = np.array([2.0, 0.5, 4.0])
    aa = get_axis_aligned_affine(T, Z)
    re = get_affine_in_axis_aligned_space(M, T, Z)
    for _ in range(4):
        p = np.append(rng.standard_normal(3), 1.0)
        np.testing.assert_allclose(re @ (aa @ p), M @ p, atol=1e-10)


def test_get_grid_kwargs_requires_singleton_spacing():
    """Singleton dimensions need explicit `voxdim` metadata."""
    data = xr.DataArray(
        np.zeros((1, 3, 4)),
        dims=("z", "y", "x"),
        coords={"z": [0.0], "y": [0.0, 0.2, 0.4], "x": [0.0, 0.1, 0.2, 0.3]},
    )

    with pytest.warns(UserWarning, match="spacing is undefined"):
        with pytest.raises(ValueError, match="spacing is undefined.*z"):
            get_grid_kwargs_from_dataarray(data)


def test_reexpress_affine_broadcasts_over_pose_stack():
    """A (npose, 4, 4) stack is re-expressed per pose, preserving shape."""
    rng = np.random.default_rng(1)
    stack = np.tile(np.eye(4), (3, 1, 1))
    stack[:, :3, :3] = rng.standard_normal((3, 3, 3))
    T = np.array([0.0, 1.0, 2.0])
    Z = np.array([1.0, 2.0, 3.0])
    out = get_affine_in_axis_aligned_space(stack, T, Z)
    assert out.shape == (3, 4, 4)
    inv = np.linalg.inv(get_axis_aligned_affine(T, Z))
    for i in range(3):
        np.testing.assert_allclose(out[i], stack[i] @ inv, atol=1e-12)
