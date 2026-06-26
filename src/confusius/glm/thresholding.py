"""Statistical thresholding with multiple-comparison correction for GLM maps.

This module provides
[`apply_statistical_threshold`][confusius.glm.thresholding.apply_statistical_threshold], which applies a
voxel-level height threshold (uncorrected, false-discovery-rate, or Bonferroni) and an
optional cluster-extent threshold to a statistical map, and
[`fdr_benjamini_hochberg_threshold`][confusius.glm.thresholding.fdr_benjamini_hochberg_threshold],
the Benjamini-Hochberg helper it relies on. The maps are expected to be z-scaled, such
as those returned by
[`compute_contrast`][confusius.glm.first_level.FirstLevelModel.compute_contrast].

Portions of this file are derived from Nilearn, which is licensed under the BSD-3-Clause
License. See `NOTICE` file for details.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.stats as sps
from scipy.ndimage import generate_binary_structure, label

from confusius._utils.stack import find_stack_level
from confusius.validation.mask import validate_mask

if TYPE_CHECKING:
    import numpy.typing as npt
    import xarray as xr


def fdr_benjamini_hochberg_threshold(
    z_vals: npt.ArrayLike,
    alpha: float,
) -> float:
    """Return the Benjamini-Hochberg false-discovery-rate threshold for z-scores.

    Computes the z-score above which voxels are declared significant while controlling
    the expected proportion of false discoveries at `alpha`.

    Parameters
    ----------
    z_vals : array_like
        One-dimensional array of z-scores (already one-sided, i.e. take the absolute
        value beforehand for a two-sided test).
    alpha : float
        Target false-discovery rate, between 0 and 1.

    Returns
    -------
    float
        The z-score threshold. Voxels with a z-score greater than this value are
        significant. Returns `numpy.inf` when no voxel survives.

    Raises
    ------
    ValueError
        If `alpha` is not in `[0, 1]` or `z_vals` is empty.
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha should be between 0 and 1, got {alpha}.")

    z_sorted = -np.sort(-np.asarray(z_vals, dtype=float))  # descending z-scores.
    n_samples = z_sorted.size
    if n_samples == 0:
        raise ValueError("z_vals is empty.")

    p_vals = sps.norm.sf(z_sorted)  # ascending one-sided upper-tail p-values.
    below_line = p_vals < alpha * np.linspace(1 / n_samples, 1, n_samples)
    if below_line.any():
        # Subtract a tiny epsilon so the last surviving voxel passes a strict `>`.
        return float(z_sorted[below_line][-1] - 1.0e-12)
    return np.inf


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
    alpha: float = 0.001,
    threshold: float | None = None,
    height_control: Literal["fpr", "fdr_bh", "bonferroni"] | None = "fpr",
    cluster_threshold: int = 0,
    two_sided: bool = True,
    skipzero: bool = False,
    skipna: bool = False,
) -> tuple[xr.DataArray, float]:
    """Threshold a statistical map with multiple-comparison correction.

    Applies a voxel-level statistical threshold followed by an optional cluster-extent
    threshold. The input is assumed to be z-scaled, such as a map returned by
    [`compute_contrast`][confusius.glm.first_level.FirstLevelModel.compute_contrast].

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
    alpha : float, default: 0.001
        Significance level. A p-value for `fpr` and `bonferroni`, a q-value for `fdr_bh`.
    threshold : float, optional
        Explicit z-score threshold, used only when `height_control` is not provided. If
        not provided in that case, defaults to 3.0. Ignored (with a warning) when
        `height_control` is set.
    height_control : {"fpr", "fdr_bh", "bonferroni"}, optional
        Voxel-level correction method, by default `"fpr"`. If not provided, `threshold`
        is used directly with no statistical control.

        Methods:
        - `"fpr"`: the *uncorrected* false positive rate.
        - `"fdr_bh"`: the false discovery rate (Benjamini-Hochberg).
        - `"bonferroni"`: the family-wise error rate (Bonferroni correction).

    cluster_threshold : int, default: 0
        Minimum cluster size in voxels. Connected clusters smaller than this are removed
        after height thresholding. Disabled when 0.
    two_sided : bool, default: True
        Whether to threshold both tails. When True, `alpha` is split across the two tails
        and the magnitude of the map is tested.
    skipzero : bool, default: False
        Whether to exclude voxels equal to zero from the tested voxels when `mask` is not
        provided. Useful because maps from `compute_contrast` fill non-brain voxels with
        zero.
    skipna : bool, default: False
        Whether to exclude non-finite voxels from the tested voxels when `mask` is not
        provided. Set to True for maps containing NaNs, otherwise the threshold may be
        NaN.

    Returns
    -------
    thresholded_map : xarray.DataArray
        Copy of `stat_map` with sub-threshold and untested voxels set to zero.
    threshold : float
        The z-score threshold that was applied. May be `numpy.inf` for `"fdr_bh"` when
        no voxel survives.

    Raises
    ------
    ValueError
        If `height_control` is not one of the allowed values, `cluster_threshold` is
        negative, `alpha` is not in `[0, 1]`, or no voxel is tested.
    """
    if height_control not in ("fpr", "fdr_bh", "bonferroni", None):
        raise ValueError(
            "height_control must be one of 'fpr', 'fdr_bh', 'bonferroni', or None, got "
            f"{height_control!r}."
        )
    if cluster_threshold < 0:
        raise ValueError(
            f"cluster_threshold must be non-negative, got {cluster_threshold}."
        )
    if height_control is not None and not 0 <= alpha <= 1:
        raise ValueError(f"alpha should be between 0 and 1, got {alpha}.")
    if height_control is not None and threshold is not None:
        warnings.warn(
            "threshold is ignored when height_control is set; using the correction "
            "method instead.",
            UserWarning,
            stacklevel=find_stack_level(),
        )

    selected = _build_mask(stat_map, mask, skipzero, skipna)
    stats = stat_map.values[selected]
    if stats.size == 0:
        raise ValueError("No voxels are tested; check the mask, skipzero, and skipna.")

    alpha_ = alpha / 2 if two_sided else alpha
    stats_for_threshold = np.abs(stats) if two_sided else stats

    if height_control == "fpr":
        z_threshold = float(sps.norm.isf(alpha_))
    elif height_control == "fdr_bh":
        z_threshold = fdr_benjamini_hochberg_threshold(stats_for_threshold, alpha_)
    elif height_control == "bonferroni":
        z_threshold = float(sps.norm.isf(alpha_ / stats.size))
    else:
        z_threshold = 3.0 if threshold is None else float(threshold)

    thresholded = stat_map.copy(deep=True)
    data = thresholded.values
    data[~selected] = 0.0  # Untested voxels are never significant.

    if two_sided:
        subthreshold = np.abs(data) < z_threshold
    elif z_threshold >= 0:
        subthreshold = data < z_threshold
    else:
        subthreshold = data > z_threshold
    data[subthreshold] = 0.0

    if cluster_threshold > 0:
        _apply_cluster_threshold(data, cluster_threshold)

    return thresholded, z_threshold
