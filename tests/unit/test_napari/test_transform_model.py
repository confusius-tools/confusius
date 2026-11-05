"""Tests for the transform panel's pure state model."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from confusius._napari._transform._model import (
    History,
    TransformState,
    format_matrix,
    parse_matrix,
    to_itk_transform,
)
from confusius.registration import apply_rotation, apply_translation, apply_zoom

CENTER = np.array([11.0, 21.0, 31.0])


def _apply(state, data):
    """Return the voxel-to-world affine after applying `state` to `data`."""
    return state.matrix() @ data.fusi.affine.voxel_to_world


def _moved_state():
    """Return a state with translation, rotation, zoom and a flip all set."""
    return (
        TransformState.identity(center=CENTER)
        .translated((1.0, -2.0, 0.5), frame="global")
        .rotated("x", 25.0, frame="global")
        .rotated("z", -10.0, frame="local")
        .zoomed((1.5, 1.0, 0.5))
        .flipped("y", frame="local")
    )


def test_identity_state_is_identity():
    state = TransformState.identity()

    assert state.is_identity()
    np.testing.assert_array_equal(state.matrix(), np.eye(4))


def test_translation_matches_apply_translation(sample_voxeldata_3d_oblique):
    data = sample_voxeldata_3d_oblique
    delta = (1.5, -2.0, 0.25)

    state = TransformState.identity().translated(delta, frame="global")

    expected = apply_translation(data, delta).fusi.affine.voxel_to_world
    np.testing.assert_allclose(_apply(state, data), expected)


def test_local_translation_moves_along_rotated_axes():
    state = TransformState.identity().rotated("x", 90.0, frame="global")

    moved = state.translated((1.0, 0.0, 0.0), frame="local")

    # +90 degrees about x maps +z onto +y.
    np.testing.assert_allclose(
        moved.translation - state.translation, [0.0, 1.0, 0.0], atol=1e-12
    )


@pytest.mark.parametrize("axis", ["z", "y", "x"])
def test_rotation_matches_apply_rotation(sample_voxeldata_3d_oblique, axis):
    data = sample_voxeldata_3d_oblique

    state = TransformState.identity(center=CENTER).rotated(axis, 37.0, frame="global")

    expected = apply_rotation(data, 37.0, axis, center=CENTER)
    np.testing.assert_allclose(_apply(state, data), expected.fusi.affine.voxel_to_world)


def test_global_rotations_compose_like_apply_rotation(sample_voxeldata_3d_oblique):
    data = sample_voxeldata_3d_oblique

    state = (
        TransformState.identity(center=CENTER)
        .rotated("z", 30.0, frame="global")
        .rotated("y", 20.0, frame="global")
    )

    expected = apply_rotation(data, (30.0, 20.0), "zy", center=CENTER)
    np.testing.assert_allclose(_apply(state, data), expected.fusi.affine.voxel_to_world)


def test_plus_90_about_x_maps_z_onto_y():
    state = TransformState.identity().rotated("x", 90.0, frame="global")

    np.testing.assert_allclose(state.matrix()[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_local_rotation_is_about_the_volume_axis():
    start = TransformState.identity().rotated("x", 40.0, frame="global")
    step = Rotation.from_rotvec(np.deg2rad(15.0) * np.array([0.0, 1.0, 0.0]))

    state = start.rotated("y", 15.0, frame="local")

    np.testing.assert_allclose(state.rotation, start.rotation @ step.as_matrix())


def test_rotation_axis_follows_the_frame():
    state = TransformState.identity().rotated("x", 90.0, frame="global")

    np.testing.assert_allclose(state.rotation_axis("z", frame="global"), [1, 0, 0])
    np.testing.assert_allclose(
        state.rotation_axis("z", frame="local"), [0, 1, 0], atol=1e-12
    )


def test_global_and_local_rotations_differ():
    start = TransformState.identity().rotated("x", 40.0, frame="global")

    global_state = start.rotated("y", 15.0, frame="global")
    local_state = start.rotated("y", 15.0, frame="local")

    assert not np.allclose(global_state.rotation, local_state.rotation)


def test_zoom_matches_apply_zoom(sample_voxeldata_3d_oblique):
    data = sample_voxeldata_3d_oblique
    factors = (2.0, 0.5, 1.25)

    state = TransformState.identity(center=CENTER).zoomed(factors)

    expected = apply_zoom(data, factors, center=CENTER)
    np.testing.assert_allclose(_apply(state, data), expected.fusi.affine.voxel_to_world)


def test_zoom_keeps_the_sign_from_a_flip():
    state = TransformState.identity().flipped("z", frame="local").zoomed((2.0, 1.0, 1.0))

    np.testing.assert_allclose(state.scale, [-2.0, 1.0, 1.0])


@pytest.mark.parametrize("factors", [(0.0, 1.0, 1.0), (np.nan, 1.0, 1.0)])
def test_zoom_rejects_zero_and_non_finite(factors):
    with pytest.raises(ValueError, match="zoom"):
        TransformState.identity().zoomed(factors)


@pytest.mark.parametrize("frame", ["global", "local"])
def test_flip_matches_negative_zoom(sample_voxeldata_3d_oblique, frame):
    data = sample_voxeldata_3d_oblique

    state = TransformState.identity(center=CENTER).flipped("y", frame=frame)

    expected = apply_zoom(data, (1.0, -1.0, 1.0), center=CENTER)
    np.testing.assert_allclose(_apply(state, data), expected.fusi.affine.voxel_to_world)


def test_global_flip_mirrors_along_the_world_axis():
    start = TransformState.identity(center=CENTER).rotated("x", 30.0, frame="global")

    state = start.flipped("z", frame="global")

    mirror = np.diag([-1.0, 1.0, 1.0, 1.0])
    pivot = np.eye(4)
    pivot[:3, 3] = start.pivot()
    expected = pivot @ mirror @ np.linalg.inv(pivot) @ start.matrix()
    np.testing.assert_allclose(state.matrix(), expected)


def test_rotation_leaves_the_pivot_fixed_after_translation():
    point = np.array([3.0, 4.0, 5.0])
    state = _moved_state()

    rotated = state.with_pivot(point).rotated("y", 33.0, frame="global")

    # The whole volume turns about the world point, on top of the current transform.
    step = np.eye(4)
    step[:3, :3] = Rotation.from_rotvec(np.deg2rad(33.0) * np.eye(3)[1]).as_matrix()
    to_point = np.eye(4)
    to_point[:3, 3] = point
    expected = to_point @ step @ np.linalg.inv(to_point) @ state.matrix()
    np.testing.assert_allclose(rotated.matrix(), expected)
    np.testing.assert_allclose(rotated.pivot(), point)


def test_pivot_reports_the_fixed_world_point():
    state = TransformState.identity(center=CENTER).translated(
        (1.0, 2.0, 3.0), frame="global"
    )

    rotated = state.rotated("z", 50.0, frame="global")

    np.testing.assert_allclose(rotated.pivot(), state.pivot())
    np.testing.assert_allclose(rotated.pivot(), CENTER + [1.0, 2.0, 3.0])


def test_with_pivot_keeps_the_matrix_and_sets_the_pivot():
    state = _moved_state()
    point = np.array([-1.0, 5.0, 2.5])

    moved = state.with_pivot(point)

    np.testing.assert_allclose(moved.matrix(), state.matrix())
    np.testing.assert_allclose(moved.pivot(), point)


def test_from_matrix_round_trip():
    state = _moved_state()

    rebuilt = TransformState.from_matrix(state.matrix(), center=CENTER)

    assert not rebuilt.is_sheared
    np.testing.assert_allclose(rebuilt.matrix(), state.matrix())


def test_from_matrix_handles_a_reflection():
    matrix = np.diag([-1.0, 1.0, 1.0, 1.0])
    matrix[:3, 3] = [2.0, 0.0, 0.0]

    rebuilt = TransformState.from_matrix(matrix, center=CENTER)

    assert not rebuilt.is_sheared
    np.testing.assert_allclose(rebuilt.matrix(), matrix)


def test_from_matrix_flags_shear():
    matrix = np.eye(4)
    matrix[0, 1] = 0.3

    rebuilt = TransformState.from_matrix(matrix, center=CENTER)

    assert rebuilt.is_sheared
    np.testing.assert_array_equal(rebuilt.matrix(), matrix)


def test_sheared_state_keeps_its_matrix_when_the_pivot_moves():
    matrix = np.eye(4)
    matrix[0, 1] = 0.3
    sheared = TransformState.from_matrix(matrix, center=CENTER)

    moved = sheared.with_pivot((1.0, 2.0, 3.0))

    np.testing.assert_array_equal(moved.matrix(), matrix)
    np.testing.assert_allclose(moved.pivot(), (1.0, 2.0, 3.0))


def test_history_undo_and_redo():
    first = TransformState.identity()
    second = first.translated((1.0, 0.0, 0.0), frame="global")
    third = second.translated((1.0, 0.0, 0.0), frame="global")
    history = History(first)
    history.push(second)
    history.push(third)

    assert history.undo() is second
    assert history.undo() is first
    assert history.undo() is None
    assert history.redo() is second
    assert history.redo() is third
    assert history.redo() is None


def test_history_push_clears_redo():
    first = TransformState.identity()
    second = first.translated((1.0, 0.0, 0.0), frame="global")
    third = first.translated((0.0, 1.0, 0.0), frame="global")
    history = History(first)
    history.push(second)
    history.undo()

    history.push(third)

    assert history.redo() is None
    assert history.undo() is first


def test_history_clear_restarts_from_state():
    first = TransformState.identity()
    second = first.translated((1.0, 0.0, 0.0), frame="global")
    history = History(first)
    history.push(second)

    history.clear(first)

    assert history.undo() is None
    assert history.redo() is None


def test_format_and_parse_matrix_round_trip():
    matrix = _moved_state().matrix()

    np.testing.assert_array_equal(parse_matrix(format_matrix(matrix)), matrix)


@pytest.mark.parametrize(
    "text",
    [
        "1,0,0,0\n0,1,0,0\n0,0,1,0\n0,0,0,1",
        "[[1. 0. 0. 0.]\n [0. 1. 0. 0.]\n [0. 0. 1. 0.]\n [0. 0. 0. 1.]]",
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
    ],
    ids=["csv", "numpy-print", "flat"],
)
def test_parse_matrix_accepts_common_layouts(text):
    np.testing.assert_array_equal(parse_matrix(text), np.eye(4))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not a matrix",
        "1 0 0 0 1 0 0 0 1",
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 1",
        "nan 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
        "0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
    ],
    ids=["empty", "words", "too-few", "bad-last-row", "nan", "singular"],
)
def test_parse_matrix_rejects_invalid_text(text):
    with pytest.raises(ValueError, match="matrix"):
        parse_matrix(text)


def test_itk_transform_maps_the_same_points():
    state = _moved_state()
    source = np.array([1.0, 2.0, 3.0])
    target = (state.matrix() @ [*source, 1.0])[:3]

    def to_lps(point):
        return np.diag([-1.0, -1.0, 1.0]) @ point[::-1]

    transform = to_itk_transform(state.matrix())

    # ITK files hold the pull transform, from the target point back to the source.
    mapped = transform.TransformPoint(to_lps(target).tolist())
    np.testing.assert_allclose(mapped, to_lps(source))


def test_rotated_tallies_the_angle_turned_per_axis():
    state = (
        TransformState.identity(center=CENTER)
        .rotated("x", 30.0, frame="global")
        .rotated("x", -10.0, frame="local")
        .rotated("z", 5.0, frame="global")
    )

    np.testing.assert_allclose(state.turned, [5.0, 0.0, 20.0])


def test_angle_turned_wraps_past_half_a_turn():
    state = (
        TransformState.identity()
        .rotated("y", 170.0, frame="global")
        .rotated("y", 20.0, frame="global")
    )

    np.testing.assert_allclose(state.turned, [0.0, -170.0, 0.0])
    expected = TransformState.identity().rotated("y", 190.0, frame="global")
    np.testing.assert_allclose(state.matrix(), expected.matrix())


def test_other_steps_keep_the_angles_turned():
    turned = TransformState.identity(center=CENTER).rotated("x", 30.0, frame="global")

    for state in (
        turned.translated((1.0, 0.0, 0.0), frame="global"),
        turned.zoomed((1.0, 2.0, 1.0)),
        turned.flipped("z", frame="global"),
        turned.with_pivot((1.0, 2.0, 3.0)),
    ):
        np.testing.assert_allclose(state.turned, turned.turned)


def test_setting_the_matrix_directly_forgets_the_angles_turned():
    turned = TransformState.identity(center=CENTER).rotated("x", 30.0, frame="global")

    assert np.all(TransformState.from_matrix(turned.matrix(), CENTER).turned == 0.0)
