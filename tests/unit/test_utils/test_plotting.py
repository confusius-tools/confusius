"""Tests for confusius._utils.plotting."""

import numpy as np
import pytest

from confusius._utils.plotting import scale_min_max, snap_origin_to_phase


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
    # phase_origin=0.0, spacing=0.5: origin=3.23 should snap down to the nearest
    # 0.5-spaced point below it that is congruent to 0.0 mod 0.5, i.e. 3.0 --
    # never anywhere near phase_origin itself (0.0), which may be far away.
    result = snap_origin_to_phase(origin=3.23, spacing=0.5, phase_origin=0.0)
    assert result == pytest.approx(3.0)


def test_snap_origin_to_phase_preserves_exact_multiples():
    # origin is already an exact multiple of spacing away from phase_origin --
    # snapping must be a no-op, not shift it down by a spurious extra step.
    result = snap_origin_to_phase(origin=1.5, spacing=0.5, phase_origin=0.0)
    assert result == pytest.approx(1.5)


def test_snap_origin_to_phase_handles_float_noise_near_a_full_step():
    # Regression: `0.2` has no exact binary representation, so naive
    # `(origin - phase_origin) % spacing` can land near `spacing` instead of
    # near `0` for an origin that is mathematically an exact multiple away --
    # shifting `origin` down by a spurious extra `spacing` (e.g. doubling a
    # degenerate single-point axis's predicted size). Regression for the
    # `test_slice_mode_pose_facets_over_poses`/
    # `test_slice_mode_pose_uses_each_poses_own_world_position` failures this
    # introduced.
    assert 1.0 % 0.2 > 0.19  # sanity: float `%` really does land near `spacing`.
    result = snap_origin_to_phase(origin=1.0, spacing=0.2, phase_origin=0.0)
    assert result == pytest.approx(1.0)


def test_snap_origin_to_phase_never_shifts_by_a_full_spacing_or_more():
    # `origin`'s own spacing needn't relate simply to whatever spacing produced
    # `phase_origin` on the other volume -- but the result must always land
    # within one `spacing` of the unsnapped origin (congruence achieved by
    # shifting down less than a full step, never by a full step or more).
    result = snap_origin_to_phase(origin=1.0, spacing=0.3, phase_origin=0.05)
    assert 1.0 - 0.3 < result <= 1.0
    phase_residual = (result - 0.05) % 0.3
    assert min(phase_residual, 0.3 - phase_residual) == pytest.approx(0.0, abs=1e-9)
