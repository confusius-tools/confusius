"""Utility functions for GLM analysis."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from confusius._utils import find_stack_level

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt
    import xarray as xr


def _attrs_equal(a: Any, b: Any) -> bool:
    """Return whether two attribute values are equal.

    Handles `numpy.ndarray` values via `numpy.array_equal` and falls back to `==`
    for everything else; returns `False` on comparison errors.
    """
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(a, b))
        except Exception:
            return False
    try:
        result = a == b
    except Exception:
        return False
    if isinstance(result, np.ndarray):
        return bool(result.all())
    return bool(result)


def consensus_attrs(arrays: Sequence[xr.DataArray]) -> dict[str, Any]:
    """Return DataArray attributes shared and equal across all `arrays`.

    Used to propagate consistent provenance/metadata through reductions that combine
    several inputs (e.g. multi-run first-level GLMs, second-level group analyses). Keys
    whose values differ across inputs, or that are missing from any input, are dropped.

    Parameters
    ----------
    arrays : Sequence[xarray.DataArray]
        DataArrays whose `attrs` dictionaries should be intersected.

    Returns
    -------
    dict
        Attributes present and equal in every array's `attrs`. Empty if `arrays` is
        empty.
    """
    if not arrays:
        return {}
    if len(arrays) == 1:
        return dict(arrays[0].attrs)

    ref_attrs = arrays[0].attrs
    rest = arrays[1:]
    return {
        key: ref_val
        for key, ref_val in ref_attrs.items()
        if all(key in da.attrs and _attrs_equal(da.attrs[key], ref_val) for da in rest)
    }


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


def estimate_ar_coeffs(
    signals: npt.NDArray[np.floating],
    order: int = 1,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | np.floating]:
    """Estimate AR(p) coefficients from time series data using Yule-Walker equations.

    Handles both single-signal (1D) and multi-voxel (2D) inputs.

    Parameters
    ----------
    signals : (n_time,) or (n_time, n_voxels) numpy.ndarray
        Time series data. For 2D input, voxels are in columns.
    order : int, default: 1
        AR model order.

    Returns
    -------
    rho : (order,) or (order, n_voxels) numpy.ndarray
        AR coefficients. Shape matches input dimensionality.
    sigma : float or (n_voxels,) numpy.ndarray
        Residual standard deviation. Scalar for 1D input, array for 2D.

    Warns
    -----
    UserWarning
        If the Toeplitz autocorrelation matrix is singular for some voxels;
        the affected voxels fall back to a pseudoinverse solve.

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

    rho, sigma = _yule_walker_2d(signals_2d, order)

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
