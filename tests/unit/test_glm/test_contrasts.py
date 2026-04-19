"""Tests for confusius.glm._contrasts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from confusius.glm._contrasts import Contrast


# -----------------------------------------------------------------------------
# Contrast initialization tests
# -----------------------------------------------------------------------------


class TestContrastInit:
    """Tests for Contrast class initialization and validation."""

    def test_basic_initialization(self):
        """Basic Contrast creation."""
        effect = np.array([1.0, 2.0, 3.0])
        variance = np.array([0.5, 0.5, 0.5])

        contrast = Contrast(effect, variance)

        assert_array_equal(contrast.effect, effect)
        assert_array_equal(contrast.variance, variance)
        assert contrast.stat_type == "t"
        assert contrast.dim == 1

    def test_f_contrast_initialization(self):
        """F-type contrast with multi-dimensional effect."""
        effect = np.ones((3, 5))  # 3 contrasts, 5 voxels
        variance = np.ones(5)

        contrast = Contrast(effect, variance, stat_type="F")

        assert contrast.stat_type == "F"
        assert contrast.dim == 3

    def test_auto_t_to_f_conversion(self):
        """Multi-dimensional effect with stat_type='t' warns and auto-converts to F."""
        effect = np.ones((2, 5))  # 2D effect
        variance = np.ones(5)

        with pytest.warns(UserWarning, match="switching to 'F'"):
            contrast = Contrast(effect, variance, stat_type="t")

        assert contrast.stat_type == "F"
        assert contrast.dim == 2

    def test_variance_must_be_1d(self):
        """Variance must be 1-dimensional."""
        effect = np.ones(5)
        variance = np.ones((5, 2))  # 2D variance

        with pytest.raises(ValueError, match="1D"):
            Contrast(effect, variance)

    def test_effect_must_be_1d_or_2d(self):
        """Effect must be 1D or 2D."""
        effect = np.ones((2, 3, 4))  # 3D effect
        variance = np.ones(4)

        with pytest.raises(ValueError, match="1D or 2D"):
            Contrast(effect, variance)

    def test_invalid_stat_type(self):
        """stat_type must be 't' or 'F'."""
        effect = np.ones(5)
        variance = np.ones(5)

        with pytest.raises(ValueError, match="'t' or 'F'"):
            Contrast(effect, variance, stat_type="chi2")


# -----------------------------------------------------------------------------
# t-statistic tests
# -----------------------------------------------------------------------------


class TestTStatistic:
    """Tests for t-statistic computation."""

    def test_t_statistic_basic(self):
        """Basic t-statistic computation."""
        effect = np.array([2.0, 3.0, 4.0])
        variance = np.array([1.0, 1.0, 1.0])

        contrast = Contrast(effect, variance)
        t_stats = contrast.stat()

        # t = effect / sqrt(variance)
        expected = effect / np.sqrt(variance)
        assert_array_equal(t_stats, expected)

    def test_t_statistic_with_baseline(self):
        """t-statistic with non-zero baseline."""
        effect = np.array([2.0, 3.0, 4.0])
        variance = np.array([1.0, 1.0, 1.0])

        contrast = Contrast(effect, variance)
        t_stats = contrast.stat(baseline=1.0)

        # t = (effect - baseline) / sqrt(variance)
        expected = (effect - 1.0) / np.sqrt(variance)
        assert_array_equal(t_stats, expected)

    def test_t_statistic_zero_variance(self):
        """Zero variance handled with tiny epsilon."""
        effect = np.array([1.0, 2.0])
        variance = np.array([0.0, 1.0])

        contrast = Contrast(effect, variance, tiny=1e-50)
        t_stats = contrast.stat()

        # Should not be infinite
        assert np.all(np.isfinite(t_stats))
        # First element should be large (1 / sqrt(tiny))
        assert t_stats[0] > 1e20

    def test_t_statistic_caching(self):
        """t-statistic is cached after first computation."""
        effect = np.ones(5)
        variance = np.ones(5)

        contrast = Contrast(effect, variance)
        t1 = contrast.stat()
        t2 = contrast.stat()

        assert t1 is t2  # Same object (cached)


# -----------------------------------------------------------------------------
# F-statistic tests
# -----------------------------------------------------------------------------


class TestFStatistic:
    """Tests for F-statistic computation."""

    def test_f_statistic_basic(self):
        """Basic F-statistic computation."""
        # 2 contrasts, 5 voxels
        effect = np.array([[2.0, 1.0, 0.0, 3.0, 1.0], [1.0, 2.0, 0.0, 1.0, 2.0]])
        variance = np.ones(5)

        contrast = Contrast(effect, variance, stat_type="F")
        f_stats = contrast.stat()

        # F = sum(effect^2) / dim / variance
        expected = np.sum(effect**2, axis=0) / 2 / variance
        assert_array_equal(f_stats, expected)

    def test_f_statistic_with_baseline(self):
        """F-statistic with non-zero baseline."""
        effect = np.ones((2, 5)) * 3.0
        variance = np.ones(5)

        contrast = Contrast(effect, variance, stat_type="F")
        f_stats = contrast.stat(baseline=1.0)

        assert_allclose(f_stats, np.ones(5) * 4.0)


# -----------------------------------------------------------------------------
# p-value tests
# -----------------------------------------------------------------------------


class TestPValues:
    """Tests for p-value computation."""

    def test_t_p_values_one_sided(self):
        """t-contrast p-values are one-sided from t-distribution."""
        effect = np.array([0.0, 2.0, -2.0])
        variance = np.ones(3)

        contrast = Contrast(effect, variance, dof=10)
        p_values = contrast.p_value()

        # p = sf(t, dof) - one-sided (right tail)
        # t-stats: [0, 2, -2]
        assert_almost_equal(p_values[0], 0.5, decimal=2)  # t=0, p=0.5
        assert p_values[1] < 0.05  # t=2, significant (right tail)
        assert p_values[2] > 0.95  # t=-2, not significant (left tail, p ~ 1)

    def test_f_p_values(self):
        """F-contrast p-values from F-distribution."""
        effect = np.ones((3, 5)) * 5.0  # Large effect
        variance = np.ones(5)

        contrast = Contrast(effect, variance, stat_type="F", dof=20)
        p_values = contrast.p_value()

        # Should be highly significant
        assert np.all(p_values < 0.01)

    def test_one_minus_pvalue(self):
        """1 - pvalue for numerical stability."""
        effect = np.array([0.0, 2.0])
        variance = np.ones(2)

        contrast = Contrast(effect, variance, dof=10)
        p_vals = contrast.p_value()
        one_minus = contrast.one_minus_pvalue()

        assert_allclose(one_minus, 1 - p_vals, rtol=1e-5)

    def test_one_minus_pvalue_f_without_cached_stat(self):
        """F one_minus_pvalue computes the statistic when needed."""
        effect = np.ones((2, 4)) * 3.0
        variance = np.ones(4)

        contrast = Contrast(effect, variance, stat_type="F", dof=20)
        one_minus = contrast.one_minus_pvalue()

        assert contrast.stat_ is not None
        assert np.all((0.0 <= one_minus) & (one_minus <= 1.0))


# -----------------------------------------------------------------------------
# z-score tests
# -----------------------------------------------------------------------------


class TestZScores:
    """Tests for z-score computation."""

    def test_z_score_basic(self):
        """z-scores from p-values."""
        effect = np.array([0.0, 2.0, -2.0])
        variance = np.ones(3)

        contrast = Contrast(effect, variance, dof=30)
        z_scores = contrast.z_score()

        # z = 0 for t=0
        assert_almost_equal(z_scores[0], 0, decimal=5)
        # z > 0 for positive t
        assert z_scores[1] > 0
        # z < 0 for negative t
        assert z_scores[2] < 0

    def test_z_score_symmetry(self):
        """z-scores should be symmetric around 0."""
        effect = np.array([2.0, -2.0])
        variance = np.ones(2)

        contrast = Contrast(effect, variance, dof=100)
        z_scores = contrast.z_score()

        assert_almost_equal(z_scores[0], -z_scores[1], decimal=5)


# -----------------------------------------------------------------------------
# Contrast arithmetic tests
# -----------------------------------------------------------------------------


class TestContrastArithmetic:
    """Tests for contrast addition and multiplication."""

    def test_add_two_contrasts(self):
        """Adding two contrasts combines effects and variances."""
        effect1 = np.ones(5)
        effect2 = np.ones(5) * 2
        variance = np.ones(5)

        con1 = Contrast(effect1, variance)
        con2 = Contrast(effect2, variance)

        combined = con1 + con2

        # Effects add
        assert_array_equal(combined.effect, effect1 + effect2)
        # Variances add (for independent contrasts)
        assert_array_equal(combined.variance, variance + variance)
        # DOF add
        assert combined.dof == con1.dof + con2.dof

    def test_add_different_stat_types_raises(self):
        """Cannot add t and F contrasts."""
        con1 = Contrast(np.ones(5), np.ones(5), stat_type="t")
        con2 = Contrast(np.ones((2, 5)), np.ones(5), stat_type="F")

        with pytest.raises(ValueError, match="stat types"):
            con1 + con2

    def test_add_different_dimensions_raises(self):
        """Cannot add contrasts with different dimensions."""
        # Both F-type but different dimensions
        con1 = Contrast(np.ones(5), np.ones(5), dim=1, stat_type="F")
        con2 = Contrast(np.ones((2, 5)), np.ones(5), dim=2, stat_type="F")

        with pytest.raises(ValueError, match="dimensions"):
            con1 + con2

    def test_multiply_by_scalar(self):
        """Multiplying contrast by scalar scales effect and variance."""
        effect = np.ones(5)
        variance = np.ones(5) * 4

        contrast = Contrast(effect, variance)
        scaled = contrast * 2

        # Effect multiplied by 2
        assert_array_equal(scaled.effect, effect * 2)
        # Variance multiplied by 4 (2^2)
        assert_array_equal(scaled.variance, variance * 4)

    def test_multiply_commutative(self):
        """Scalar multiplication is commutative."""
        contrast = Contrast(np.ones(5), np.ones(5))

        assert_array_equal((contrast * 2).effect, (2 * contrast).effect)

    def test_divide_by_scalar(self):
        """Dividing contrast by scalar."""
        effect = np.ones(5) * 4
        variance = np.ones(5) * 16

        contrast = Contrast(effect, variance)
        scaled = contrast / 2

        assert_array_equal(scaled.effect, effect / 2)
        assert_array_equal(scaled.variance, variance / 4)


# -----------------------------------------------------------------------------
# Effect size and variance tests
# -----------------------------------------------------------------------------


class TestEffectSize:
    """Tests for effect size and variance accessors."""

    def test_effect_size_accessor(self):
        """effect_size returns effect."""
        effect = np.array([1.0, 2.0, 3.0])
        contrast = Contrast(effect, np.ones(3))

        assert_array_equal(contrast.effect_size(), effect)

    def test_effect_variance_accessor(self):
        """effect_variance returns variance."""
        variance = np.array([0.5, 0.5, 0.5])
        contrast = Contrast(np.ones(3), variance)

        assert_array_equal(contrast.effect_variance(), variance)


# -----------------------------------------------------------------------------
# Fixed effects tests
# -----------------------------------------------------------------------------


class TestFixedEffects:
    """Tests for fixed-effects combination."""

    def test_fixed_effects_two_runs(self):
        """Fixed effects combination of two runs."""
        # Run 1: effect=1, variance=1
        # Run 2: effect=2, variance=4
        con1 = Contrast(np.ones(1), np.ones(1), dof=10)
        con2 = Contrast(np.ones(1) * 2, np.ones(1) * 4, dof=10)

        combined = con1 + con2

        # Fixed effect: simple average of effects = 1.5
        assert_allclose(combined.effect, np.array([3.0]))  # Sum, not average
        # Variance: sum of variances
        assert_allclose(combined.variance, np.array([5.0]))

    def test_fixed_effects_z_score_consistency(self):
        """z-scores should be consistent after fixed-effects combination."""
        # Two identical contrasts
        con1 = Contrast(np.ones(5) * 2, np.ones(5), dof=20)
        con2 = Contrast(np.ones(5) * 2, np.ones(5), dof=20)

        combined = con1 + con2

        # z-scores should be similar (though not identical due to DOF change)
        z1 = con1.z_score()
        z_combined = combined.z_score()

        # Both should be positive and significant
        assert np.all(z1 > 0)
        assert np.all(z_combined > 0)
