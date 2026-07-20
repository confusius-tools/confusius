"""Tests for confusius.glm._utils."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from confusius.glm._utils import estimate_ar_coeffs, expression_to_contrast_vector


# -----------------------------------------------------------------------------
# expression_to_contrast_vector
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "columns", "expected"),
    [
        # Single column reference.
        ("stim_A", ["stim_A", "stim_B", "constant"], [1.0, 0.0, 0.0]),
        # Subtraction.
        ("stim_A - stim_B", ["stim_A", "stim_B", "constant"], [1.0, -1.0, 0.0]),
        # Sum.
        ("stim_A + stim_B", ["stim_A", "stim_B", "constant"], [1.0, 1.0, 0.0]),
        # Mixed weighting and grouping (mirrors nilearn's parametrized case).
        (
            "face / 10 + (window - face) * 2 - house",
            ["a", "face", "xy_z", "house", "window"],
            [0.0, -1.9, 0.0, -1.0, 2.0],
        ),
        # Single-column design.
        ("column_1", ["column_1"], [1.0]),
    ],
)
def test_expression_to_contrast_vector(expression, columns, expected):
    """Expression is parsed against the design columns into the expected vector."""
    assert_allclose(expression_to_contrast_vector(expression, columns), expected)


def test_expression_to_contrast_vector_invalid_expression_raises():
    """Expressions referencing unknown identifiers raise."""
    with pytest.raises(ValueError, match="Could not evaluate"):
        expression_to_contrast_vector("invalid_column", ["stim_A", "stim_B"])


# -----------------------------------------------------------------------------
# estimate_ar_coeffs
# -----------------------------------------------------------------------------


def test_estimate_ar_coeffs_recovers_known_rho():
    """Yule-Walker recovers the true AR(1) coefficient on a long simulated
    series — the only meaningful test for the estimator."""
    rng = np.random.default_rng(0)
    rho_true = 0.6
    n = 10000
    noise = rng.standard_normal(n)
    signal = np.zeros(n)
    for t in range(1, n):
        signal[t] = rho_true * signal[t - 1] + noise[t]

    rho_hat, _ = estimate_ar_coeffs(signal, order=1)

    assert rho_hat.shape == (1,)
    # Sample autocorrelation of an AR(1) process is consistent: standard error
    # falls off as ~1/sqrt(n), so 0.05 absolute tolerance is comfortable at n=1e4.
    assert abs(rho_hat[0] - rho_true) < 0.05


def test_estimate_ar_coeffs_invalid_ndim_raises():
    """3D inputs are rejected at the API boundary."""
    with pytest.raises(ValueError, match="Expected 1D or 2D"):
        estimate_ar_coeffs(np.zeros((5, 5, 5)))


def test_estimate_ar_coeffs_zero_variance_voxel_is_silent():
    """A constant voxel does not make the batched solve singular.

    Zero-variance voxels (e.g. the constant background outside a beamforming grid)
    give an all-zero Toeplitz block. They must not trigger the pseudoinverse fallback
    or its singularity warning; their coefficients are undefined and returned as zero,
    with sigma as NaN, while the remaining voxels are estimated normally.
    """
    rng = np.random.default_rng(0)
    signals = rng.standard_normal((200, 10))
    signals[:, 3] = 0.0  # zero-variance voxel

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rho, sigma = estimate_ar_coeffs(signals, order=1)

    sigma = np.asarray(sigma)  # 2D input always yields an array, not a scalar.
    assert rho.shape == (1, 10)
    assert rho[0, 3] == 0.0
    assert np.isnan(sigma[3])
    other = np.arange(10) != 3
    assert np.all(np.isfinite(sigma[other]))
