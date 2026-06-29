"""Statistical thresholding with multiple-comparison correction for GLM maps.

This module provides
[`apply_statistical_threshold`][confusius.glm.thresholding.apply_statistical_threshold],
which applies a voxel-level statistical threshold (with a choice of family-wise-error or
false-discovery-rate correction) and an optional cluster-extent threshold to a
statistical map. The maps are expected to be z-scaled, such as those returned by
[`compute_contrast`][confusius.glm.first_level.FirstLevelModel.compute_contrast].

The correction is performed in p-value space: the z-scores are converted to p-values,
the p-values are adjusted for multiple comparisons, and voxels whose adjusted p-value is
at or below `alpha` are kept. The Benjamini-Hochberg and Benjamini-Yekutieli
false-discovery-rate adjustments are delegated to
[`scipy.stats.false_discovery_control`][]; the family-wise-error adjustments are computed
directly.

Portions of this file are derived from Nilearn, which is licensed under the BSD-3-Clause
License. See `NOTICE` file for details.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import scipy.stats as sps
from scipy.ndimage import generate_binary_structure, label

from confusius._utils.stack import find_stack_level
from confusius.validation.mask import validate_mask

if TYPE_CHECKING:
    import numpy.typing as npt
    import xarray as xr

CorrectionMethod = Literal[
    "uncorrected",
    "bonferroni",
    "sidak",
    "holm",
    "holm-sidak",
    "simes-hochberg",
    "hommel",
    "fdr_bh",
    "fdr_by",
]
"""Multiple-comparison correction methods accepted by `apply_statistical_threshold`.

See
[`apply_statistical_threshold`][confusius.glm.thresholding.apply_statistical_threshold]
for a description of each method.
"""

# Methods whose rejection threshold is a fixed p-value cutoff independent of the data.
_FIXED_PVALUE_CUTOFF = ("uncorrected", "bonferroni", "sidak")


def _stat_to_pvalue(
    stats: npt.NDArray[np.floating],
    two_sided: bool,
) -> npt.NDArray[np.float64]:
    """Convert z-scores to upper-tail p-values.

    Parameters
    ----------
    stats : (n,) numpy.ndarray
        Z-scores of the tested voxels.
    two_sided : bool
        Whether to compute two-sided p-values from the magnitude of the z-scores.

    Returns
    -------
    (n,) numpy.ndarray
        p-values in `[0, 1]`.
    """
    if two_sided:
        return np.clip(2.0 * sps.norm.sf(np.abs(stats)), 0.0, 1.0)
    return sps.norm.sf(stats)


def _step_adjust(
    pvalues: npt.NDArray[np.float64],
    method: Literal["holm", "holm-sidak", "simes-hochberg"],
) -> npt.NDArray[np.float64]:
    """Adjust p-values with a sequential family-wise-error method.

    Implements the step-down Holm and Holm-Šidák procedures and the step-up
    Simes-Hochberg procedure on the sorted p-values, then restores the original order.

    Parameters
    ----------
    pvalues : (n,) numpy.ndarray
        Raw p-values to adjust.
    method : {"holm", "holm-sidak", "simes-hochberg"}
        Correction method.

    Returns
    -------
    (n,) numpy.ndarray
        Adjusted p-values, in the order of `pvalues`.
    """
    n = pvalues.size
    order = np.argsort(pvalues, kind="stable")
    p_sorted = pvalues[order]
    factor = np.arange(n, 0, -1)  # n, n-1, ..., 1.

    if method == "holm-sidak":
        raw = -np.expm1(factor * np.log1p(-p_sorted))
    else:  # "holm" and "simes-hochberg" share the (n - i + 1) * p factor.
        raw = factor * p_sorted

    if method == "simes-hochberg":  # Step-up: running minimum from the largest p-value.
        adjusted_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    else:  # Step-down: running maximum from the smallest p-value.
        adjusted_sorted = np.maximum.accumulate(raw)

    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def _hommel_adjust(pvalues: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Adjust p-values with Hommel's family-wise-error method.

    Hommel's procedure is the closed-testing procedure based on Simes' test. This is the
    closed-form implementation used by `statsmodels.stats.multitest.multipletests`.

    Parameters
    ----------
    pvalues : (n,) numpy.ndarray
        Raw p-values to adjust.

    Returns
    -------
    (n,) numpy.ndarray
        Adjusted p-values, in the order of `pvalues`.
    """
    n = pvalues.size
    order = np.argsort(pvalues, kind="stable")
    p_sorted = pvalues[order]
    adjusted_sorted = p_sorted.copy()
    for size in range(n, 1, -1):
        cim = np.min(size * p_sorted[-size:] / np.arange(1, size + 1))
        adjusted_sorted[-size:] = np.maximum(adjusted_sorted[-size:], cim)
        adjusted_sorted[:-size] = np.maximum(
            adjusted_sorted[:-size], np.minimum(size * p_sorted[:-size], cim)
        )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def _adjust_pvalues(
    pvalues: npt.NDArray[np.float64],
    method: CorrectionMethod,
) -> npt.NDArray[np.float64]:
    """Adjust p-values for multiple comparisons.

    Parameters
    ----------
    pvalues : (n,) numpy.ndarray
        Raw p-values of the tested voxels.
    method : CorrectionMethod
        Correction method. `"fdr_bh"` and `"fdr_by"` are delegated to
        [`scipy.stats.false_discovery_control`][]; the others are computed directly.

    Returns
    -------
    (n,) numpy.ndarray
        Adjusted p-values, in the order of `pvalues`.
    """
    n = pvalues.size
    if method == "uncorrected":
        return pvalues
    if method == "bonferroni":
        return np.minimum(pvalues * n, 1.0)
    if method == "sidak":
        return -np.expm1(n * np.log1p(-pvalues))
    if method == "hommel":
        return _hommel_adjust(pvalues)
    if method in ("holm", "holm-sidak", "simes-hochberg"):
        return _step_adjust(pvalues, method)
    # "fdr_bh" / "fdr_by".
    return sps.false_discovery_control(pvalues, method=method.removeprefix("fdr_"))


def _report_threshold(
    stats: npt.NDArray[np.floating],
    rejected: npt.NDArray[np.bool_],
    method: CorrectionMethod,
    alpha: float,
    two_sided: bool,
) -> float:
    """Return the z-score threshold corresponding to a rejection set.

    For methods with a fixed p-value cutoff the analytic z-score is returned; for the
    data-adaptive methods the smallest surviving magnitude is returned.

    Parameters
    ----------
    stats : (n,) numpy.ndarray
        Z-scores of the tested voxels.
    rejected : (n,) numpy.ndarray
        Boolean mask of the significant voxels.
    method : CorrectionMethod
        Correction method that produced `rejected`.
    alpha : float
        Significance level.
    two_sided : bool
        Whether the test is two-sided.

    Returns
    -------
    float
        The z-score threshold. `numpy.inf` when a data-adaptive method rejects nothing.
    """
    n = stats.size
    if method in _FIXED_PVALUE_CUTOFF:
        if method == "uncorrected":
            p_cut = alpha
        elif method == "bonferroni":
            p_cut = alpha / n
        else:  # "sidak".
            p_cut = 1.0 - (1.0 - alpha) ** (1.0 / n)
        return float(sps.norm.isf(p_cut / 2 if two_sided else p_cut))

    if not rejected.any():
        return np.inf
    magnitude = np.abs(stats) if two_sided else stats
    return float(magnitude[rejected].min())


def _build_mask(
    stat_map: xr.DataArray,
    mask: xr.DataArray | None,
    skipzero: bool,
    skipna: bool,
) -> npt.NDArray[np.bool_]:
    """Build the boolean array selecting the tested voxels of a statistical map.

    Parameters
    ----------
    stat_map : xarray.DataArray
        Statistical map to threshold.
    mask : xarray.DataArray, optional
        Explicit mask of tested voxels. If not provided, a mask is derived from
        `stat_map` using `skipzero` and `skipna`.
    skipzero : bool
        Whether to exclude voxels equal to zero when `mask` is not provided.
    skipna : bool
        Whether to exclude non-finite voxels when `mask` is not provided.

    Returns
    -------
    (...) numpy.ndarray
        Boolean array with the spatial shape of `stat_map`, True for tested voxels.
    """
    if mask is not None:
        validate_mask(mask, stat_map, mask_name="mask")
        return mask.transpose(*stat_map.dims).values.astype(bool)

    selected = np.ones(stat_map.shape, dtype=bool)
    if skipzero:
        selected &= stat_map.values != 0
    if skipna:
        selected &= np.isfinite(stat_map.values)
    return selected


def _apply_cluster_threshold(
    arr: npt.NDArray[np.floating],
    cluster_threshold: int,
) -> None:
    """Zero out connected clusters smaller than `cluster_threshold` voxels, in place.

    Positive and negative clusters are labelled separately so that adjacent suprathreshold
    blobs of opposite sign are never merged. Connectivity is face-only (each voxel touches
    those sharing a face).

    Parameters
    ----------
    arr : (...) numpy.ndarray
        Thresholded statistical map, modified in place.
    cluster_threshold : int
        Minimum cluster size in voxels. Clusters with fewer voxels are set to zero.

    Returns
    -------
    None
        `arr` is modified in place.
    """
    bin_struct = generate_binary_structure(arr.ndim, 1)
    small = np.zeros(arr.shape, dtype=bool)
    for sign in (1.0, -1.0):
        label_map, n_labels = label((arr * sign) > 0, bin_struct)
        if n_labels == 0:
            continue
        counts = np.bincount(label_map.ravel())
        too_small = np.flatnonzero(counts < cluster_threshold)
        too_small = too_small[too_small != 0]  # 0 is the background label.
        small |= np.isin(label_map, too_small)
    arr[small] = 0.0


def apply_statistical_threshold(
    stat_map: xr.DataArray,
    mask: xr.DataArray | None = None,
    *,
    alpha: float = 0.05,
    method: CorrectionMethod | None = "fdr_bh",
    threshold: float | None = None,
    cluster_threshold: int = 0,
    two_sided: bool = True,
    skipzero: bool = False,
    skipna: bool = False,
) -> tuple[xr.DataArray, float]:
    """Threshold a statistical map with multiple-comparison correction.

    Applies a voxel-level statistical threshold followed by an optional cluster-extent
    threshold. The input is assumed to be z-scaled, such as a map returned by
    [`compute_contrast`][confusius.glm.first_level.FirstLevelModel.compute_contrast].

    The z-scores are converted to p-values, adjusted for multiple comparisons with
    `method`, and voxels whose adjusted p-value is at or below `alpha` are kept. Set
    `method` to `None` to instead apply the explicit `threshold` z-score cutoff.

    If a mask is not provided, `skipzero` and `skipna` can be used to exclude zero and
    non-finite voxels from the tested voxels.

    Parameters
    ----------
    stat_map : xarray.DataArray
        Z-scaled statistical map with spatial dimensions. Voxels that do not survive the
        thresholds are set to zero.
    mask : xarray.DataArray, optional
        Boolean (or single-label integer) mask of the tested voxels. If not provided, the
        tested voxels are derived from `stat_map` according to `skipzero` and `skipna`.
    alpha : float, default: 0.05
        Significance level. A voxel is kept when its `method`-adjusted p-value is at or
        below `alpha`. Ignored when `method` is not provided.
    method : CorrectionMethod, optional
        Multiple-comparison correction method, by default `"fdr_bh"`. If not provided,
        `threshold` is applied directly with no statistical control. Available methods:

        - `"uncorrected"`: no correction.
        - `"bonferroni"`: one-step correction.
        - `"sidak"`: one-step correction.
        - `"holm-sidak"`: step down method using Sidak adjustments.
        - `"holm"`: step-down method using Bonferroni adjustments.
        - `"simes-hochberg"`: step-up method (independent).
        - `"hommel"`: closed method based on Simes tests (non-negative).
        - `"fdr_bh"`: Benjamini/Hochberg (non-negative).
        - `"fdr_by"`: Benjamini/Yekutieli (negative).
    threshold : float, optional
        Explicit z-score threshold, used only when `method` is not provided. If not
        provided in that case, defaults to 3.0. Ignored (with a warning) when `method` is
        set.
    cluster_threshold : int, default: 0
        Minimum cluster size in voxels. Connected clusters smaller than this are removed
        after the voxel-level threshold. Disabled when 0.
    two_sided : bool, default: True
        Whether to test both tails. When True, the magnitude of the map is tested;
        otherwise only positive z-scores can survive.
    skipzero : bool, default: False
        Whether to exclude voxels equal to zero from the tested voxels when `mask` is not
        provided. Useful because maps from `compute_contrast` fill non-brain voxels with
        zero.
    skipna : bool, default: False
        Whether to exclude non-finite voxels from the tested voxels when `mask` is not
        provided. Set to True for maps containing NaNs.

    Returns
    -------
    thresholded_map : xarray.DataArray
        Copy of `stat_map` with sub-threshold and untested voxels set to zero.
    threshold : float
        The z-score threshold that was applied. `numpy.inf` when a data-adaptive method
        rejects every voxel.

    Raises
    ------
    ValueError
        If `method` is not one of the allowed values, `cluster_threshold` is negative,
        `alpha` is not in `[0, 1]`, or no voxel is tested.
    """
    if method is not None and method not in get_args(CorrectionMethod):
        raise ValueError(
            f"method must be None or one of {get_args(CorrectionMethod)}, got {method!r}."
        )
    if cluster_threshold < 0:
        raise ValueError(
            f"cluster_threshold must be non-negative, got {cluster_threshold}."
        )
    if method is not None and not 0 <= alpha <= 1:
        raise ValueError(f"alpha should be between 0 and 1, got {alpha}.")
    if method is not None and threshold is not None:
        warnings.warn(
            "threshold is ignored when method is set; using the correction method "
            "instead.",
            UserWarning,
            stacklevel=find_stack_level(),
        )

    selected = _build_mask(stat_map, mask, skipzero, skipna)
    stats = stat_map.values[selected]
    if stats.size == 0:
        raise ValueError("No voxels are tested; check the mask, skipzero, and skipna.")

    if method is None:
        z_threshold = 3.0 if threshold is None else float(threshold)
        if two_sided:
            rejected = np.abs(stats) >= z_threshold
        elif z_threshold >= 0:
            rejected = stats >= z_threshold
        else:
            rejected = stats <= z_threshold
    else:
        pvalues = _stat_to_pvalue(stats, two_sided)
        rejected = _adjust_pvalues(pvalues, method) <= alpha
        z_threshold = _report_threshold(stats, rejected, method, alpha, two_sided)

    keep = np.zeros(stat_map.shape, dtype=bool)
    keep[selected] = rejected
    thresholded = stat_map.copy(deep=True)
    data = thresholded.values
    data[~keep] = 0.0

    if cluster_threshold > 0:
        _apply_cluster_threshold(data, cluster_threshold)

    return thresholded, z_threshold
