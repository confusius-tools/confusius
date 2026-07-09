"""Tests for confusius.glm._models."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from confusius.glm._models import ARModel, OLSModel


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def design_matrix(rng):
    """Design matrix with 40 observations and 10 regressors."""
    return rng.standard_normal(size=(40, 10))


@pytest.fixture
def response_matrix(rng):
    """Response matrix with 40 observations and 10 voxels."""
    return rng.standard_normal(size=(40, 10))


@pytest.fixture
def design_with_intercept(rng):
    """Design matrix with intercept column (first column = 1s)."""
    X = rng.standard_normal(size=(40, 10))
    X[:, 0] = 1.0
    return X


# -----------------------------------------------------------------------------
# OLSModel tests
# -----------------------------------------------------------------------------


class TestOLSModel:
    """Tests for Ordinary Least Squares model."""

    def test_residuals_mean_zero_with_intercept(
        self, design_with_intercept, response_matrix
    ):
        """With intercept, residuals should have mean ~0."""
        model = OLSModel(design_with_intercept)
        results = model.fit(response_matrix)

        # Mean of residuals should be close to 0
        assert_almost_equal(results.residuals.mean(axis=0), 0, decimal=10)

    def test_perfect_prediction(self, rng):
        """With enough predictors, residuals should be ~0."""
        n = 10
        X = rng.standard_normal(size=(n, n))
        Y = rng.standard_normal(size=(n, 5))

        model = OLSModel(X)
        results = model.fit(Y)

        # Should perfectly predict (within numerical precision)
        assert_allclose(results.residuals, 0, atol=1e-8)
        assert_allclose(results.predicted, Y, atol=1e-8)

    def test_degenerate_design(self, design_matrix, response_matrix):
        """Collinear columns reduce effective model df."""
        # Make first column linear combination of others
        design_matrix[:, 0] = design_matrix[:, 1] + design_matrix[:, 2]

        model = OLSModel(design_matrix)
        results = model.fit(response_matrix)

        # Rank should be 9 instead of 10 due to collinearity
        assert results.df_model == 9
        assert results.df_residuals == 31

    def test_whiten_is_identity(self, design_matrix):
        """OLS whiten method should return input unchanged."""
        model = OLSModel(design_matrix)
        X = np.random.randn(40, 5)

        whitened = model.whiten(X)
        assert_array_equal(whitened, X)

    def test_covariance_matrix_properties(self, design_matrix):
        """Covariance matrix should be symmetric positive semi-definite."""
        model = OLSModel(design_matrix)

        # Check symmetry
        cov = model.normalized_cov
        assert_array_equal(cov, cov.T)

        # Check it's positive semi-definite (eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors

    def test_sse_mse_consistency(self, design_matrix, response_matrix):
        """SSE and MSE should be consistent."""
        model = OLSModel(design_matrix)
        results = model.fit(response_matrix)

        expected_mse = results.sse / results.df_residuals
        assert_allclose(results.mse, expected_mse)

    def test_pseudoinverse_computation(self, design_matrix):
        """`pinv_design` should be the pseudoinverse of the design."""
        model = OLSModel(design_matrix)

        # pinv_design @ design should be close to identity.
        product = model.pinv_design @ design_matrix
        identity = np.eye(design_matrix.shape[1])
        assert_allclose(product, identity, atol=1e-10)


# -----------------------------------------------------------------------------
# ARModel tests
# -----------------------------------------------------------------------------


class TestARModel:
    """Tests for Autoregressive model."""

    def test_ar1_whitening_reduces_autocorrelation(self):
        """AR(1) fitting reduces autocorrelation in whitened residuals."""
        rng = np.random.default_rng(0)
        n, n_voxels = 200, 5
        rho_val = 0.7

        # Generate AR(1) noise for each voxel.
        noise = np.zeros((n, n_voxels))
        noise[0] = rng.standard_normal(n_voxels)
        for t in range(1, n):
            noise[t] = rho_val * noise[t - 1] + rng.standard_normal(n_voxels)

        design = np.ones((n, 1))
        rho = np.full((1, n_voxels), rho_val)
        model = ARModel(design, rho=rho)
        results = model.fit(noise)

        # Whitened residuals should have lower lag-1 autocorrelation than raw residuals.
        raw_ac = np.mean(
            [np.corrcoef(results.residuals[:-1, v], results.residuals[1:, v])[0, 1]
             for v in range(n_voxels)]
        )
        assert results.whitened_residuals is not None
        white_ac = np.mean(
            [np.corrcoef(results.whitened_residuals[:-1, v],
                         results.whitened_residuals[1:, v])[0, 1]
             for v in range(n_voxels)]
        )
        assert abs(white_ac) < abs(raw_ac)

    def test_ar_degenerate_design(self, design_matrix, response_matrix):
        """AR model handles collinear design columns."""
        design_matrix[:, 0] = design_matrix[:, 1] + design_matrix[:, 2]
        n_voxels = response_matrix.shape[1]
        rho = np.full((1, n_voxels), 0.9)
        model = ARModel(design_matrix, rho=rho)
        results = model.fit(response_matrix)

        assert results.df_model == 9
        assert results.df_residuals == 31

    def test_ar_invalid_rho_shape(self, design_matrix):
        """AR rho must be 2D with shape (order, n_voxels)."""
        with pytest.raises(ValueError, match="2D"):
            ARModel(design_matrix, rho=np.array([0.1, 0.2, 0.3]))

    def test_ar_voxel_mismatch(self, design_matrix, response_matrix):
        """AR fit raises if rho voxel count doesn't match Y."""
        rho = np.full((1, response_matrix.shape[1] + 1), 0.3)
        model = ARModel(design_matrix, rho=rho)
        with pytest.raises(ValueError, match="voxels"):
            model.fit(response_matrix)

    def test_ar_f_equals_t_squared(self, design_matrix, response_matrix):
        """AR F-statistic with 1 numerator df equals t-statistic squared."""
        n_voxels = response_matrix.shape[1]
        rho = np.full((1, n_voxels), 0.4)
        model = ARModel(design_matrix, rho=rho)
        results = model.fit(response_matrix)

        contrast = np.zeros(design_matrix.shape[1])
        contrast[0] = 1.0
        t_result = results.compute_t_contrast(contrast)
        f_result = results.compute_f_contrast(contrast[np.newaxis, :])

        assert_allclose(f_result["F"], t_result["t"] ** 2, rtol=1e-5)

    def test_ar_saturated_dispersion_is_nan(self, rng):
        """AR fit with df_residuals=0 (saturated model) sets dispersion to NaN."""
        T, K, n_voxels = 10, 10, 3
        X = rng.standard_normal((T, K))
        Y = rng.standard_normal((T, n_voxels))
        rho = np.full((1, n_voxels), 0.3)
        model = ARModel(X, rho=rho)
        results = model.fit(Y)

        assert results.df_residuals == 0
        assert np.all(np.isnan(results.dispersion))


# -----------------------------------------------------------------------------
# RegressionResults tests
# -----------------------------------------------------------------------------


class TestRegressionResults:
    """Tests for regression results container."""

    def test_residuals_computation(self, design_matrix, response_matrix):
        """Residuals = Y - predicted."""
        model = OLSModel(design_matrix)
        results = model.fit(response_matrix)

        expected_residuals = response_matrix - results.predicted
        assert_array_equal(results.residuals, expected_residuals)

    def test_predicted_computation(self, design_matrix, response_matrix):
        """Predicted = X @ theta."""
        model = OLSModel(design_matrix)
        results = model.fit(response_matrix)

        expected_predicted = design_matrix @ results.theta
        assert_array_equal(results.predicted, expected_predicted)

    def test_t_contrast_1d_vs_2d(self, design_matrix, response_matrix):
        """t-contrast handles both 1D and 2D contrast vectors."""
        model = OLSModel(design_matrix)
        results = model.fit(response_matrix)

        # 1D contrast
        c1 = np.zeros(10)
        c1[0] = 1.0
        result1 = results.compute_t_contrast(c1)

        # 2D contrast (1, n_regressors)
        c2 = c1[np.newaxis, :]
        result2 = results.compute_t_contrast(c2)

        # Should give same results
        assert_allclose(result1["t"], result2["t"])

    def test_t_contrast_invalid_multirow(self, design_matrix, response_matrix):
        """t-contrast rejects multi-row contrast matrices."""
        model = OLSModel(design_matrix)
        results = model.fit(response_matrix)

        # Multi-row contrast (should fail for t-test)
        contrast = np.eye(10)[:3]  # 3 x 10

        with pytest.raises(ValueError, match="single row"):
            results.compute_t_contrast(contrast)

    def test_f_contrast_whitening_invariant(self, design_matrix, response_matrix):
        """The whitened-effect representation satisfies
        `||whitened||² / q / dispersion == F`, which is the property the
        downstream Contrast wrapper relies on."""
        model = OLSModel(design_matrix)
        results = model.fit(response_matrix)
        contrast = np.eye(10)[:3]

        f_result = results.compute_f_contrast(contrast)

        assert_allclose(
            np.sum(f_result["whitened_effect"] ** 2, axis=0)
            / f_result["df_num"]
            / f_result["dispersion"],
            f_result["F"],
            rtol=1e-10,
        )

    def test_f_vs_t_equivalence(self, design_matrix, response_matrix):
        """F-statistic with 1 df should equal t-statistic squared."""
        model = OLSModel(design_matrix)
        results = model.fit(response_matrix)

        # Single regressor contrast
        contrast = np.zeros(10)
        contrast[0] = 1.0

        t_result = results.compute_t_contrast(contrast)
        f_result = results.compute_f_contrast(contrast[np.newaxis, :])

        # F should equal t^2
        assert_allclose(f_result["F"], t_result["t"] ** 2, rtol=1e-5)

    def test_contrast_effect_accuracy(self, design_matrix, response_matrix):
        """Contrast effect should equal c^T @ theta."""
        model = OLSModel(design_matrix)
        results = model.fit(response_matrix)

        contrast = np.zeros(10)
        contrast[0] = 1.0
        contrast[1] = -1.0

        t_result = results.compute_t_contrast(contrast)

        # Manual computation
        expected_effect = contrast @ results.theta
        assert_allclose(t_result["effect"], expected_effect)


# -----------------------------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases on degenerate sample sizes."""

    def test_saturated_model_dispersion_is_nan(self, rng):
        """A square design (`n_obs == n_regressors`) leaves zero residual df, so
        dispersion is undefined and should surface as NaN."""
        X = rng.standard_normal(size=(5, 5))
        Y = rng.standard_normal(size=(5, 3))

        model = OLSModel(X)
        results = model.fit(Y)

        assert results.df_residuals == 0
        assert np.all(np.isnan(results.dispersion))
