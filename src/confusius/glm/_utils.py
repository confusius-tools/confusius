"""Utility functions for GLM analysis."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

from confusius._utils import find_stack_level

if TYPE_CHECKING:
    import numpy.typing as npt


def _yule_walker_2d(
    signals: npt.NDArray[np.floating],
    order: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Yule-Walker estimation for 2D signals.

    Parameters
    ----------
    signals : (n_time, n_voxels) numpy.ndarray
        Time series data for multiple voxels, with time in rows and voxels in columns.
    order : int
        AR model order.

    Returns
    -------
    rho : (order, n_voxels) numpy.ndarray
        AR coefficients for each voxel.
    sigma : (n_voxels,) numpy.ndarray
        Residual standard deviation for each voxel.
    """
    n_time, n_voxels = signals.shape
    signals_centered = signals - signals.mean(axis=0)

    autocorr = np.zeros((order + 1, n_voxels))
    for lag in range(order + 1):
        autocorr[lag] = (
            np.sum(signals_centered[: n_time - lag] * signals_centered[lag:], axis=0)
            / n_time
        )

    # Normalize by lag-0 autocorrelation for numerical stability. Zero-variance voxels
    # are handled after solving.
    zero_var = autocorr[0] == 0
    safe_var = np.where(zero_var, 1.0, autocorr[0])
    autocorr_norm = autocorr / safe_var

    r = autocorr_norm[1:]  # (order, n_voxels)

    # Build Toeplitz matrix in (n_voxels, order, order) for batched solve.
    R = np.zeros((n_voxels, order, order))
    for i in range(order):
        for j in range(order):
            R[:, i, j] = autocorr_norm[abs(i - j)]

    try:
        rho = np.linalg.solve(R, r.T[..., np.newaxis])[..., 0].T  # (order, n_voxels)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Yule-Walker system singular for some voxels; using pseudoinverse.",
            stacklevel=find_stack_level(),
        )
        rho = np.stack(
            [np.linalg.pinv(R[v]) @ r[:, v] for v in range(n_voxels)], axis=1
        )

    # Actual (unnormalized) residual variance.
    sigma_sq = autocorr[0] - np.sum(autocorr[1:] * rho, axis=0)
    sigma = np.sqrt(np.maximum(sigma_sq, 0))

    # Zero-variance voxels: coefficients are undefined.
    rho = np.where(zero_var, 0.0, rho)
    sigma = np.where(zero_var, np.nan, sigma)

    return rho, sigma


def _burg_2d(
    signals: npt.NDArray[np.floating],
    order: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """AR coefficient estimation using Burg's method, vectorized over voxels.

    Estimates AR coefficients by minimizing forward and backward prediction errors
    simultaneously using a recursive lattice filter. Provides better frequency
    resolution than Yule-Walker for short time series.

    Parameters
    ----------
    signals : (n_time, n_voxels) numpy.ndarray
        Time series data with time in rows and voxels in columns.
    order : int
        AR model order.

    Returns
    -------
    rho : (order, n_voxels) numpy.ndarray
        AR coefficients for each voxel.
    sigma : (n_voxels,) numpy.ndarray
        Residual standard deviation for each voxel.
    """
    n_time, n_voxels = signals.shape
    signals_centered = signals - signals.mean(axis=0)

    rho = np.zeros((order, n_voxels))

    fwd = signals_centered[1:]
    bwd = signals_centered[:-1]

    for p in range(order):
        if p == 0:
            numerator = 2 * np.sum(fwd * bwd, axis=0)
            denominator = np.sum(fwd**2, axis=0) + np.sum(bwd**2, axis=0)
        else:
            numerator = 2 * np.sum(fwd[p:] * bwd[:-p], axis=0)
            denominator = np.sum(fwd[p:] ** 2, axis=0) + np.sum(bwd[:-p] ** 2, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            rho[p] = np.where(denominator != 0, numerator / denominator, 0)

        if p < order - 1:
            fwd_new = fwd[p + 1 :] - rho[p] * bwd[: -(p + 1)]
            bwd_new = bwd[p + 1 :] - rho[p] * fwd[: -(p + 1)]
            fwd, bwd = fwd_new, bwd_new

    sigma_sq = np.sum(signals_centered**2, axis=0) - np.sum(
        rho * (signals_centered[1:] * signals_centered[:-1]), axis=0
    )
    sigma = np.sqrt(np.maximum(sigma_sq / n_time, 0))

    return rho, sigma


def estimate_ar_coeffs(
    signals: npt.NDArray[np.floating],
    order: int = 1,
    method: Literal["yw", "burg"] = "yw",
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | np.floating]:
    """Estimate AR(p) coefficients from time series data.

    Handles both single-signal (1D) and multi-voxel (2D) inputs. Uses Yule-Walker
    equations by default; Burg's method available for better frequency resolution.

    Parameters
    ----------
    signals : (n_time,) or (n_time, n_voxels) numpy.ndarray
        Time series data. For 2D input, voxels are in columns.
    order : int, default: 1
        AR model order.
    method : {"yw", "burg"}, default: "yw"
        Estimation method. "yw" = Yule-Walker (default), "burg" = Burg's method.

    Returns
    -------
    rho : (order,) or (order, n_voxels) numpy.ndarray
        AR coefficients. Shape matches input dimensionality.
    sigma : float or (n_voxels,) numpy.ndarray
        Residual standard deviation. Scalar for 1D input, array for 2D.

    Raises
    ------
    LinAlgError
        If Toeplitz matrix is singular (Yule-Walker) or Levinson recursion fails.

    Examples
    --------
    >>> # Single voxel
    >>> signal = np.random.default_rng(0).standard_normal(100)
    >>> rho, sigma = estimate_ar_coeffs(signal, order=1)
    >>> rho.shape
    (1,)

    >>> # Multiple voxels (vectorized)
    >>> signals = np.random.default_rng(0).standard_normal((100, 50))
    >>> rho, sigma = estimate_ar_coeffs(signals, order=2)
    >>> rho.shape
    (2, 50)
    """
    if signals.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D array, got {signals.ndim}D")

    is_1d = signals.ndim == 1
    signals_2d = signals[:, np.newaxis] if is_1d else signals

    if method == "yw":
        rho, sigma = _yule_walker_2d(signals_2d, order)
    elif method == "burg":
        rho, sigma = _burg_2d(signals_2d, order)
    else:
        raise ValueError(f"Unknown method: {method}")

    if is_1d:
        return rho[:, 0], sigma[0]
    return rho, sigma


def expression_to_contrast_vector(
    expression: str, design_columns: list[str]
) -> npt.NDArray[np.floating]:
    """Convert contrast expression string to numeric vector.

    Parses contrast expressions like "condition_A - condition_B" or "condition_A +
    2*condition_B" into numeric contrast vectors.

    This function was adapted from
    [`nilearn.glm.expression_to_contrast_vector`][nilearn.glm.expression_to_contrast_vector].

    Parameters
    ----------
    expression : str
        Contrast expression using column names and operators (+, -, *, /).
    design_columns : list of str
        Names of design matrix columns.

    Returns
    -------
    (n_columns,) numpy.ndarray
        Contrast vector.

    Examples
    --------
    >>> columns = ['stim_A', 'stim_B', 'constant']
    >>> expression_to_contrast_vector('stim_A - stim_B', columns)
    array([ 1., -1.,  0.])

    >>> expression_to_contrast_vector('stim_A + stim_B', columns)
    array([1., 1., 0.])
    """
    import pandas as pd

    if expression in design_columns:
        contrast = np.zeros(len(design_columns))
        contrast[design_columns.index(expression)] = 1.0
        return contrast

    eye_design = pd.DataFrame(np.eye(len(design_columns)), columns=design_columns)

    try:
        return eye_design.eval(expression, engine="python").to_numpy()
    except Exception as e:
        raise ValueError(
            f"Could not evaluate contrast expression '{expression}'. "
            f"Make sure all column names are valid Python identifiers. "
            f"Original error: {e}"
        ) from e
