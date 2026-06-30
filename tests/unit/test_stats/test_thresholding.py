"""Tests for confusius.stats.thresholding."""

from itertools import combinations

import numpy as np
import pytest
import scipy.stats as sps
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from confusius.stats import adjust_pvalues, apply_statistical_threshold

FWER_AND_FDR = [
    "uncorrected",
    "bonferroni",
    "sidak",
    "holm",
    "holm-sidak",
    "simes-hochberg",
    "fdr_bh",
    "fdr_by",
]


def _two_sided_pvalues(z):
    """Two-sided p-values matching the implementation's z-to-p conversion."""
    return np.clip(2.0 * sps.norm.sf(np.abs(z)), 0.0, 1.0)


def _reference_reject(pvalues, method, alpha):
    """Independent rejection rule for every method except Hommel.

    Each method is expressed as its textbook accept/reject rule on the sorted p-values,
    deliberately formulated differently from the adjusted-p-value implementation.
    """
    n = pvalues.size
    order = np.argsort(pvalues)
    ps = pvalues[order]
    reject_sorted = np.zeros(n, dtype=bool)

    if method == "uncorrected":
        reject_sorted = ps <= alpha
    elif method == "bonferroni":
        reject_sorted = ps <= alpha / n
    elif method == "sidak":
        reject_sorted = ps <= 1.0 - (1.0 - alpha) ** (1.0 / n)
    elif method in ("holm", "holm-sidak"):
        if method == "holm":
            line = alpha / (n - np.arange(n))
        else:
            line = 1.0 - (1.0 - alpha) ** (1.0 / (n - np.arange(n)))
        failures = np.where(ps > line)[0]
        cut = failures[0] if failures.size else n  # step-down: stop at first failure.
        reject_sorted[:cut] = True
    else:  # Step-up methods: reject everything up to the last p below the line.
        if method == "simes-hochberg":
            line = alpha / (n - np.arange(n))
        elif method == "fdr_bh":
            line = alpha * np.arange(1, n + 1) / n
        else:  # "fdr_by".
            c = np.sum(1.0 / np.arange(1, n + 1))
            line = alpha * np.arange(1, n + 1) / (n * c)
        below = np.where(ps <= line)[0]
        if below.size:
            reject_sorted[: below.max() + 1] = True

    reject = np.zeros(n, dtype=bool)
    reject[order] = reject_sorted
    return reject


def _reference_reject_hommel(pvalues, alpha):
    """Brute-force closed-testing (Simes) reference for Hommel's procedure.

    A hypothesis is rejected iff every intersection hypothesis containing it has a Simes
    p-value at or below `alpha`. This is the definition Hommel's closed-form approximates.
    """
    n = pvalues.size
    reject = np.zeros(n, dtype=bool)
    for i in range(n):
        rejectable = True
        for size in range(1, n + 1):
            for subset in combinations(range(n), size):
                if i not in subset:
                    continue
                ps = np.sort(pvalues[list(subset)])
                simes = np.min(len(subset) * ps / np.arange(1, len(subset) + 1))
                if simes > alpha:
                    rejectable = False
                    break
            if not rejectable:
                break
        reject[i] = rejectable
    return reject


@pytest.fixture
def zmap(rng):
    """A `(x,)` z-map: weak noise plus planted strong and moderate activations."""
    z = rng.standard_normal(60) * 0.7
    z[:8] = [2.6, 3.0, 3.4, 3.9, 4.5, -3.2, -4.0, 2.9]
    return xr.DataArray(z, dims=["x"])


@pytest.fixture
def pmap(zmap):
    """A `(x,)` p-value map derived from `zmap`."""
    return xr.DataArray(_two_sided_pvalues(zmap.values), dims=["x"])


class TestAdjustPvalues:
    """Tests for the generic p-value adjustment helper."""

    @pytest.mark.parametrize("method", FWER_AND_FDR)
    def test_adjusted_map_matches_reference_rejection_rule(self, pmap, method):
        """Adjusted p-values reject the same voxels as the textbook rule."""
        adjusted = adjust_pvalues(pmap, method=method)
        expected = _reference_reject(pmap.values, method, 0.05)
        assert_array_equal(adjusted.values <= 0.05, expected)

    def test_hommel_matches_closed_testing(self):
        """Hommel-adjusted p-values match the brute-force closed-Simes rule."""
        pmap = xr.DataArray([2e-5, 9e-4, 0.005, 0.016, 0.058, 0.55], dims=["x"])
        adjusted = adjust_pvalues(pmap, method="hommel")
        expected = _reference_reject_hommel(pmap.values, 0.05)
        assert_array_equal(adjusted.values <= 0.05, expected)

    def test_untested_voxels_are_one(self):
        """Untested voxels are set to one so they are never significant."""
        pmap = xr.DataArray([0.0, 0.02, np.nan, 0.04], dims=["x"])
        adjusted = adjust_pvalues(pmap, skipzero=True, skipna=True)
        assert_array_equal(adjusted.values, [1.0, 0.04, 1.0, 0.04])

    def test_invalid_tested_pvalues_raise(self):
        """Tested voxels must be finite p-values in [0, 1]."""
        pmap = xr.DataArray([0.1, 1.2], dims=["x"])
        with pytest.raises(ValueError, match=r"finite p-values in \[0, 1\]"):
            adjust_pvalues(pmap)


class TestCorrectionMethods:
    """Each method's survivors match an independent reference rejection rule."""

    @pytest.mark.parametrize("method", FWER_AND_FDR)
    def test_survivors_match_reference(self, zmap, method):
        """The kept voxels match the textbook accept/reject rule for the method."""
        out, _ = apply_statistical_threshold(zmap, alpha=0.05, method=method)
        expected = _reference_reject(_two_sided_pvalues(zmap.values), method, 0.05)
        assert_array_equal(out.values != 0, expected)

    def test_hommel_matches_closed_testing(self):
        """Hommel survivors match the brute-force closed-Simes definition."""
        z = xr.DataArray([4.2, 3.3, 2.8, 2.4, 1.9, 0.6], dims=["x"])
        out, _ = apply_statistical_threshold(z, alpha=0.05, method="hommel")
        expected = _reference_reject_hommel(_two_sided_pvalues(z.values), 0.05)
        assert_array_equal(out.values != 0, expected)

    def test_method_power_ordering(self, zmap):
        """Survivor sets are nested by the methods' relative conservativeness."""
        survivors = {
            m: set(
                np.flatnonzero(apply_statistical_threshold(zmap, method=m)[0].values)
            )
            for m in ("bonferroni", "holm", "simes-hochberg", "fdr_bh", "uncorrected")
        }
        assert survivors["bonferroni"] <= survivors["holm"]
        assert survivors["holm"] <= survivors["simes-hochberg"]
        assert survivors["simes-hochberg"] <= survivors["fdr_bh"]
        assert survivors["fdr_bh"] <= survivors["uncorrected"]


class TestThresholdValue:
    """Tests for the reported z-score threshold."""

    def test_uncorrected_threshold(self, zmap):
        """Uncorrected two-sided threshold is the normal quantile at alpha/2."""
        _, threshold = apply_statistical_threshold(
            zmap, alpha=0.05, method="uncorrected"
        )
        assert_allclose(threshold, sps.norm.isf(0.025))

    def test_bonferroni_threshold(self, zmap):
        """Bonferroni threshold divides alpha by the number of tested voxels."""
        _, threshold = apply_statistical_threshold(
            zmap, alpha=0.05, method="bonferroni"
        )
        assert_allclose(threshold, sps.norm.isf(0.025 / zmap.size))

    def test_sidak_threshold(self, zmap):
        """Šidák threshold uses the Šidák-adjusted per-comparison alpha."""
        _, threshold = apply_statistical_threshold(zmap, alpha=0.05, method="sidak")
        p_cut = 1.0 - (1.0 - 0.05) ** (1.0 / zmap.size)
        assert_allclose(threshold, sps.norm.isf(p_cut / 2))

    def test_adaptive_threshold_is_min_survivor(self, zmap):
        """Data-adaptive methods report the smallest surviving magnitude."""
        out, threshold = apply_statistical_threshold(zmap, alpha=0.05, method="fdr_bh")
        survivors = np.abs(out.values[out.values != 0])
        assert_allclose(threshold, survivors.min())

    def test_adaptive_threshold_is_inf_when_empty(self, rng):
        """A data-adaptive method that rejects nothing reports an infinite threshold."""
        z = xr.DataArray(rng.standard_normal(50) * 0.3, dims=["x"])
        _, threshold = apply_statistical_threshold(z, alpha=0.001, method="holm")
        assert threshold == np.inf


class TestSidedness:
    """Tests for one- vs two-sided thresholding."""

    def test_two_sided_keeps_strong_negative(self):
        """A strong negative voxel survives a two-sided test."""
        z = xr.DataArray([-4.0, 0.5, 4.0], dims=["x"])
        out, _ = apply_statistical_threshold(z, method="uncorrected", two_sided=True)
        assert_array_equal(out.values != 0, [True, False, True])

    def test_one_sided_drops_negative(self):
        """The same negative voxel is dropped by a one-sided test."""
        z = xr.DataArray([-4.0, 0.5, 4.0], dims=["x"])
        out, _ = apply_statistical_threshold(z, method="uncorrected", two_sided=False)
        assert_array_equal(out.values != 0, [False, False, True])


class TestExplicitThreshold:
    """Tests for the explicit z-threshold used when method is None."""

    def test_threshold_used_when_method_none(self, zmap):
        """With method=None, the explicit threshold is applied directly."""
        out, threshold = apply_statistical_threshold(zmap, method=None, threshold=3.0)
        assert threshold == 3.0
        assert_array_equal(out.values != 0, np.abs(zmap.values) >= 3.0)

    def test_threshold_defaults_to_three(self, zmap):
        """With method=None and no threshold, the default cutoff is 3.0."""
        _, threshold = apply_statistical_threshold(zmap, method=None)
        assert threshold == 3.0

    def test_threshold_ignored_when_method_set(self, zmap):
        """Passing threshold with a method set warns and uses the correction."""
        with pytest.warns(UserWarning, match="ignored when method is set"):
            out, threshold = apply_statistical_threshold(
                zmap, method="bonferroni", threshold=10.0
            )
        expected = _reference_reject(
            _two_sided_pvalues(zmap.values), "bonferroni", 0.05
        )
        assert_array_equal(out.values != 0, expected)
        assert_allclose(threshold, sps.norm.isf(0.025 / zmap.size))


class TestMask:
    """Tests for tested-voxel selection."""

    def test_explicit_mask_restricts_and_zeroes(self):
        """The mask sets the comparison count and zeroes untested voxels."""
        z = xr.DataArray([[5.0, 2.0, 4.0], [-4.0, 1.0, 3.5]], dims=["y", "x"])
        mask = xr.DataArray([[True, True, True], [False, False, True]], dims=["y", "x"])
        out, threshold = apply_statistical_threshold(
            z, mask=mask, method="bonferroni", two_sided=False
        )
        assert_allclose(threshold, sps.norm.isf(0.05 / 4))  # four tested voxels.
        assert out.values[1, 0] == 0.0  # untested strong-magnitude voxel is dropped.

    def test_skipzero_changes_comparison_count(self):
        """skipzero drops zero-filled voxels from the comparison count."""
        z = xr.DataArray([0.0, 0.0, 3.0, 4.0], dims=["x"])
        _, with_skip = apply_statistical_threshold(
            z, method="bonferroni", skipzero=True
        )
        _, without_skip = apply_statistical_threshold(z, method="bonferroni")
        assert_allclose(with_skip, sps.norm.isf(0.025 / 2))
        assert_allclose(without_skip, sps.norm.isf(0.025 / 4))

    def test_skipna_excludes_non_finite(self):
        """skipna drops NaN voxels so the threshold stays finite."""
        z = xr.DataArray([np.nan, 2.0, 5.0], dims=["x"])
        _, threshold = apply_statistical_threshold(z, method="fdr_bh", skipna=True)
        assert np.isfinite(threshold)

    def test_no_tested_voxels_raises(self):
        """An all-zero map with skipzero leaves nothing to test."""
        z = xr.DataArray(np.zeros(4), dims=["x"])
        with pytest.raises(ValueError, match="No voxels are tested"):
            apply_statistical_threshold(z, skipzero=True)


class TestClusterThreshold:
    """Tests for cluster-extent thresholding."""

    def test_small_clusters_removed(self):
        """Clusters smaller than the extent threshold are zeroed."""
        arr = np.zeros((5, 5))
        arr[0, 0] = arr[0, 1] = arr[1, 0] = arr[1, 1] = 5.0  # 4-voxel cluster.
        arr[4, 4] = 5.0  # isolated voxel.
        z = xr.DataArray(arr, dims=["y", "x"])
        out, _ = apply_statistical_threshold(
            z, method="uncorrected", two_sided=False, cluster_threshold=2
        )
        assert out.values[4, 4] == 0.0
        assert_array_equal(out.values[:2, :2], np.full((2, 2), 5.0))

    def test_opposite_signs_not_merged(self):
        """Adjacent positive and negative voxels are separate clusters."""
        arr = np.zeros((3, 3))
        arr[1, 1] = 5.0
        arr[1, 2] = -5.0
        z = xr.DataArray(arr, dims=["y", "x"])
        out, _ = apply_statistical_threshold(
            z, alpha=0.5, method="uncorrected", cluster_threshold=2
        )
        assert_array_equal(out.values, np.zeros((3, 3)))


class TestValidation:
    """Tests for input validation."""

    def test_invalid_method(self, zmap):
        """An unknown correction method is rejected."""
        with pytest.raises(ValueError, match="method must be None or one of"):
            apply_statistical_threshold(zmap, method="hochberg")

    def test_negative_cluster_threshold(self, zmap):
        """A negative cluster threshold is rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            apply_statistical_threshold(zmap, cluster_threshold=-1)

    def test_alpha_out_of_range(self, zmap):
        """alpha must lie in [0, 1]."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            apply_statistical_threshold(zmap, alpha=2.0)
