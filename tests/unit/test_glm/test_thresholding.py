"""Tests for confusius.glm.thresholding."""

import numpy as np
import pytest
import scipy.stats as sps
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from confusius.glm.thresholding import (
    apply_statistical_threshold,
    fdr_benjamini_hochberg_threshold,
)


class TestFdrThreshold:
    """Tests for the Benjamini-Hochberg helper."""

    def test_hand_computed_example(self):
        """Threshold corresponds to the largest p-value below the BH line."""
        # BH line for n=5, alpha=0.05 is [0.01, 0.02, 0.03, 0.04, 0.05].
        # Sorted p-values [0.001, 0.008, 0.039, 0.041, 0.042]: the last one below its
        # line entry is 0.042 at i=5, so the threshold is its z-score.
        p_vals = np.array([0.001, 0.008, 0.039, 0.041, 0.042])
        z_vals = sps.norm.isf(p_vals)
        assert_allclose(
            fdr_benjamini_hochberg_threshold(z_vals, 0.05), sps.norm.isf(0.042) - 1e-12
        )

    def test_returns_inf_when_nothing_survives(self):
        """No voxel below the BH line yields an infinite threshold."""
        z_vals = sps.norm.isf(np.array([0.4, 0.45, 0.5]))
        assert fdr_benjamini_hochberg_threshold(z_vals, 0.001) == np.inf

    @pytest.mark.parametrize("alpha", [-0.1, 1.5])
    def test_rejects_alpha_out_of_range(self, alpha):
        """alpha must lie in [0, 1]."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            fdr_benjamini_hochberg_threshold(np.array([1.0, 2.0]), alpha)

    def test_rejects_empty(self):
        """An empty z-score array is rejected."""
        with pytest.raises(ValueError, match="empty"):
            fdr_benjamini_hochberg_threshold(np.array([]), 0.05)


@pytest.fixture
def small_map():
    """A `(y, x)` z-map with a mix of strong, weak, and zero voxels."""
    return xr.DataArray(
        np.array([[0.0, 2.0, 4.0], [-4.0, 1.0, 3.5]]),
        dims=["y", "x"],
    )


class TestHeightControl:
    """Tests for the voxel-level correction methods."""

    def test_fpr_threshold_value(self, small_map):
        """Two-sided FPR threshold is the normal quantile at alpha/2."""
        _, threshold = apply_statistical_threshold(
            small_map, alpha=0.05, height_control="fpr", two_sided=True
        )
        assert_allclose(threshold, sps.norm.isf(0.025))

    def test_two_sided_keeps_strong_negatives(self, small_map):
        """Two-sided thresholding keeps large-magnitude voxels of either sign."""
        out, _ = apply_statistical_threshold(
            small_map, alpha=0.05, height_control="fpr", two_sided=True
        )
        # threshold ~1.96: 0.0 and 1.0 drop, the rest (incl. -4.0) survive.
        assert_array_equal(out.values, np.array([[0.0, 2.0, 4.0], [-4.0, 0.0, 3.5]]))

    def test_one_sided_drops_negatives(self, small_map):
        """One-sided thresholding removes negative voxels."""
        out, threshold = apply_statistical_threshold(
            small_map, alpha=0.05, height_control="fpr", two_sided=False
        )
        assert_allclose(threshold, sps.norm.isf(0.05))
        # threshold ~1.64: negatives and weak voxels drop.
        assert_array_equal(out.values, np.array([[0.0, 2.0, 4.0], [0.0, 0.0, 3.5]]))

    def test_bonferroni_divides_by_tested_voxels(self, small_map):
        """Bonferroni threshold uses alpha/2 divided by the number of tested voxels."""
        _, threshold = apply_statistical_threshold(
            small_map, alpha=0.05, height_control="bonferroni", two_sided=True
        )
        assert_allclose(threshold, sps.norm.isf(0.025 / small_map.size))

    def test_none_uses_explicit_threshold(self, small_map):
        """With no height control, the explicit threshold is applied directly."""
        out, threshold = apply_statistical_threshold(
            small_map, height_control=None, threshold=3.0, two_sided=False
        )
        assert threshold == 3.0
        assert_array_equal(out.values, np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 3.5]]))

    def test_threshold_ignored_warns(self, small_map):
        """Passing threshold with a height control warns and uses the correction."""
        with pytest.warns(UserWarning, match="ignored when height_control"):
            _, threshold = apply_statistical_threshold(
                small_map, height_control="fpr", threshold=10.0, alpha=0.05
            )
        assert_allclose(threshold, sps.norm.isf(0.025))


class TestMask:
    """Tests for tested-voxel selection."""

    def test_explicit_mask_restricts_comparisons(self, small_map):
        """The mask defines the comparison count and zeroes untested voxels."""
        mask = xr.DataArray(
            np.array([[False, True, True], [False, False, True]]),
            dims=["y", "x"],
        )
        out, threshold = apply_statistical_threshold(
            small_map, mask=mask, height_control="bonferroni", two_sided=False
        )
        # Only three voxels are tested, so the divisor is 3, not 6.
        assert_allclose(threshold, sps.norm.isf(0.001 / 3))
        # The untested -4.0 voxel is zeroed even though its magnitude is large.
        assert out.values[1, 0] == 0.0

    def test_skipzero_excludes_zero_voxels(self):
        """skipzero drops zero-filled voxels from the comparison count."""
        stat_map = xr.DataArray(np.array([0.0, 0.0, 2.0, 3.0]), dims=["x"])
        _, with_skip = apply_statistical_threshold(
            stat_map, height_control="bonferroni", skipzero=True, two_sided=False
        )
        _, without_skip = apply_statistical_threshold(
            stat_map, height_control="bonferroni", skipzero=False, two_sided=False
        )
        assert_allclose(with_skip, sps.norm.isf(0.001 / 2))
        assert_allclose(without_skip, sps.norm.isf(0.001 / 4))

    def test_skipna_excludes_non_finite(self):
        """skipna drops NaN voxels so the threshold stays finite."""
        stat_map = xr.DataArray(np.array([np.nan, 2.0, 5.0]), dims=["x"])
        _, threshold = apply_statistical_threshold(
            stat_map, height_control="fdr_bh", skipna=True, two_sided=False
        )
        assert np.isfinite(threshold)

    def test_no_tested_voxels_raises(self):
        """An all-zero map with skipzero leaves nothing to test."""
        stat_map = xr.DataArray(np.zeros(4), dims=["x"])
        with pytest.raises(ValueError, match="No voxels are tested"):
            apply_statistical_threshold(stat_map, skipzero=True)


class TestClusterThreshold:
    """Tests for cluster-extent thresholding."""

    def test_small_clusters_removed(self):
        """Clusters smaller than the extent threshold are zeroed."""
        arr = np.zeros((5, 5))
        arr[0, 0] = arr[0, 1] = arr[1, 0] = arr[1, 1] = 5.0  # 4-voxel cluster.
        arr[4, 4] = 5.0  # isolated voxel.
        stat_map = xr.DataArray(arr, dims=["y", "x"])
        out, _ = apply_statistical_threshold(
            stat_map,
            height_control="fpr",
            alpha=0.05,
            two_sided=False,
            cluster_threshold=2,
        )
        assert out.values[4, 4] == 0.0
        assert_array_equal(out.values[:2, :2], np.full((2, 2), 5.0))

    def test_opposite_signs_not_merged(self):
        """Adjacent positive and negative voxels are labelled as separate clusters."""
        arr = np.zeros((3, 3))
        arr[1, 1] = 5.0
        arr[1, 2] = -5.0
        stat_map = xr.DataArray(arr, dims=["y", "x"])
        out, _ = apply_statistical_threshold(
            stat_map,
            height_control="fpr",
            alpha=0.5,
            two_sided=True,
            cluster_threshold=2,
        )
        # Each blob is a single voxel of its sign, so neither survives extent 2.
        assert_array_equal(out.values, np.zeros((3, 3)))


class TestValidation:
    """Tests for input validation."""

    def test_invalid_height_control(self, small_map):
        """An unknown height-control method is rejected."""
        with pytest.raises(ValueError, match="height_control must be"):
            apply_statistical_threshold(small_map, height_control="cluster")

    def test_negative_cluster_threshold(self, small_map):
        """A negative cluster threshold is rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            apply_statistical_threshold(small_map, cluster_threshold=-1)

    def test_alpha_out_of_range(self, small_map):
        """alpha must lie in [0, 1] for the correction methods."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            apply_statistical_threshold(small_map, alpha=2.0, height_control="fpr")
