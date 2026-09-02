"""Tests for confusius._utils.plotting."""

import numpy as np
import pytest

from confusius._utils.plotting import AxisPhase, scale_min_max, snap_origin_to_phase


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


def test_snap_origin_to_phase_locks_to_reference_lattice():
    # phase.origin=0.0, spacing=0.5=phase.spacing: origin=3.23 should snap down
    # to the nearest 0.5-spaced point below it whose *edge* is congruent to
    # phase's edge (-0.25) mod 0.5, i.e. a center of 3.0 -- never anywhere near
    # phase.origin itself (0.0), which may be far away.
    phase = AxisPhase(origin=0.0, spacing=0.5)
    result = snap_origin_to_phase(origin=3.23, spacing=0.5, phase=phase)
    assert result == pytest.approx(3.0)


def test_snap_origin_to_phase_preserves_exact_multiples():
    # origin is already an exact multiple of spacing away from phase.origin --
    # snapping must be a no-op, not shift it down by a spurious extra step.
    phase = AxisPhase(origin=0.0, spacing=0.5)
    result = snap_origin_to_phase(origin=1.5, spacing=0.5, phase=phase)
    assert result == pytest.approx(1.5)


def test_snap_origin_to_phase_handles_float_noise_near_a_full_step():
    # Regression: `0.2` has no exact binary representation, so naive
    # `(origin - phase.origin) % spacing` can land near `spacing` instead of
    # near `0` for an origin that is mathematically an exact multiple away --
    # shifting `origin` down by a spurious extra `spacing` (e.g. doubling a
    # degenerate single-point axis's predicted size). Regression for the
    # `test_slice_mode_pose_facets_over_poses`/
    # `test_slice_mode_pose_uses_each_poses_own_world_position` failures this
    # introduced.
    assert 1.0 % 0.2 > 0.19  # sanity: float `%` really does land near `spacing`.
    phase = AxisPhase(origin=0.0, spacing=0.2)
    result = snap_origin_to_phase(origin=1.0, spacing=0.2, phase=phase)
    assert result == pytest.approx(1.0)


def test_snap_origin_to_phase_never_shifts_by_a_full_spacing_or_more():
    # `spacing` needn't relate simply to `phase.spacing` -- but the result must
    # always land within one `spacing` of the unsnapped origin (congruence
    # achieved by shifting down less than a full step, never a full step or
    # more).
    phase = AxisPhase(origin=0.05, spacing=0.3)
    result = snap_origin_to_phase(origin=1.0, spacing=0.3, phase=phase)
    assert 1.0 - 0.3 < result <= 1.0
    phase_residual = (result - 0.05) % 0.3
    assert min(phase_residual, 0.3 - phase_residual) == pytest.approx(0.0, abs=1e-9)


def test_snap_origin_to_phase_aligns_edges_not_just_centers_at_differing_spacing():
    # Regression for #391 (part 2): a finer grid (half `phase`'s spacing)
    # phase-locked by center alone would put every *other* fine-grid center on
    # a coarse-grid *edge* rather than nesting fine cells inside coarse ones.
    # The fix must align cell *edges*: with spacing = phase.spacing / 2, the
    # snapped origin's own edge (origin - spacing/2) must be an exact multiple
    # of `spacing` away from phase's edge (phase.origin - phase.spacing/2).
    phase = AxisPhase(origin=0.0, spacing=0.1)
    result = snap_origin_to_phase(origin=0.137, spacing=0.05, phase=phase)

    # own_edge congruent to phase_edge mod 0.05 is exactly the condition for
    # every fine-grid edge to land on either a coarse-grid edge (spaced by
    # phase.spacing=0.1) or exactly bisect a coarse cell -- never at an
    # arbitrary offset inside one.
    own_edge = result - 0.05 / 2
    phase_edge = phase.origin - phase.spacing / 2
    residual = (own_edge - phase_edge) % 0.05
    assert min(residual, 0.05 - residual) == pytest.approx(0.0, abs=1e-9)
