"""Contrast computation and statistical testing for GLM results.

This module provides the Contrast class for computing t-statistics, F-statistics,
p-values, and z-scores from GLM contrast results. It also supports fixed-effects
combination across multiple runs.

Portions of this file are derived from Nilearn, which is licensed under the BSD-3-Clause
License. See `NOTICE` file for details.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy.stats as sps

from confusius._utils import find_stack_level

if TYPE_CHECKING:
    import numpy.typing as npt


def _pvalues_to_zscore(
    p_value: npt.NDArray[np.float64],
    one_minus_pvalue: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Convert p-values to z-scores.

    This function was adapted from `nilearn.glm._utils.zscore`.

    Parameters
    ----------
    p_value : (n,) numpy.ndarray
        p-values to convert.
    one_minus_pvalue : (n,) numpy.ndarray, optional
        1 - p-values for numerical stability. If `None`, computed as `1 - p_value`.

    Returns
    -------
    (n,) numpy.ndarray
        z-scores corresponding to the p-values.

    """
    p_value = np.clip(p_value, 1.0e-300, 1.0 - 1.0e-16)
    z_scores_sf = sps.norm.isf(p_value)

    if one_minus_pvalue is not None:
        one_minus_pvalue = np.clip(one_minus_pvalue, 1.0e-300, 1.0 - 1.0e-16)
        z_scores_cdf = sps.norm.ppf(one_minus_pvalue)
        z_scores = np.empty(p_value.size)
        use_cdf = z_scores_sf < 0
        use_sf = np.logical_not(use_cdf)
        z_scores[np.atleast_1d(use_cdf)] = z_scores_cdf[use_cdf]
        z_scores[np.atleast_1d(use_sf)] = z_scores_sf[use_sf]
    else:
        z_scores = z_scores_sf

    return z_scores


class Contrast:
    """Contrast results with statistical inference.

    Container for contrast effect estimates and variance, providing methods to compute
    *t*-statistics, *F*-statistics, *p*-values, and *z*-scores. Supports addition for
    fixed-effects combination across runs.

    This implementation is adapted from [`nilearn.glm.Contrast`][nilearn.glm.Contrast].

    Parameters
    ----------
    effect : (n_voxels,) or (contrast_dim, n_voxels) numpy.ndarray
        Contrast effect estimates.
    variance : (n_voxels,) numpy.ndarray
        Contrast variance estimates.
    dim : int, optional
        Contrast dimension (1 for *t*-statistic, >1 for *F*-statistic). If `None`,
        inferred from effect.
    dof : float, default: 1e10
        Degrees of freedom of the residuals.
    stat_type : {"t", "F"}, default: "t"
        Type of contrast statistic.
    tiny : float, default: 1e-50
        Small value to avoid division by zero.
    dofmax : float, default: 1e10
        Maximum degrees of freedom.

    Attributes
    ----------
    effect : numpy.ndarray
        Contrast effect estimates.
    variance : numpy.ndarray
        Contrast variance estimates.
    dim : int
        Contrast dimension.
    dof : float
        Degrees of freedom.
    stat_type : {"t", "F"}
        Type of statistic (`"t"` or `"F"`).
    stat_ : numpy.ndarray or None
        Computed statistic (set after calling stat()).
    p_value_ : numpy.ndarray or None
        Computed p-values (set after calling p_value()).

    Examples
    --------
    >>> import numpy as np
    >>> effect = np.random.randn(100)
    >>> variance = np.abs(np.random.randn(100))
    >>> contrast = Contrast(effect, variance, dof=98)
    >>> t_vals = contrast.stat()
    >>> p_vals = contrast.p_value()
    >>> z_vals = contrast.z_score()
    """

    def __init__(
        self,
        effect: npt.NDArray[np.floating],
        variance: npt.NDArray[np.floating],
        dim: int | None = None,
        dof: float = 1e10,
        stat_type: str = "t",
        tiny: float = 1e-50,
        dofmax: float = 1e10,
    ) -> None:
        """Initialize contrast container."""
        if variance.ndim != 1:
            raise ValueError(f"variance must be 1D, got shape {variance.shape}")

        if effect.ndim not in (1, 2):
            raise ValueError(f"effect must be 1D or 2D, got shape {effect.shape}")

        self.effect: npt.NDArray[np.floating] = effect
        self.variance = variance
        self.dof = dof
        self.tiny = tiny
        self.dofmax = dofmax

        if dim is None:
            self.dim = effect.shape[0] if effect.ndim == 2 else 1
        else:
            self.dim = dim

        if self.dim > 1 and stat_type == "t":
            warnings.warn(
                f"stat_type='t' is incompatible with a {self.dim}-dimensional contrast; "
                "switching to 'F'.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            stat_type = "F"

        if stat_type not in ("t", "F"):
            raise ValueError(f"stat_type must be 't' or 'F', got {stat_type}")

        self.stat_type = stat_type

        self.stat_: npt.NDArray[np.float64] | None = None
        self.p_value_: npt.NDArray[np.float64] | None = None
        self.one_minus_pvalue_: npt.NDArray[np.float64] | None = None
        self.baseline = 0.0

    def effect_size(self) -> npt.NDArray[np.floating]:
        """Return contrast effect estimates.

        Returns
        -------
        (n_voxels,) or (contrast_dim, n_voxels) numpy.ndarray
            Contrast effect estimates.
        """
        return self.effect

    def effect_variance(self) -> npt.NDArray[np.floating]:
        """Return contrast variance estimates.

        Returns
        -------
        (n_voxels,) numpy.ndarray
            Contrast variance estimates.
        """
        return self.variance

    def stat(self, baseline: float = 0.0) -> npt.NDArray[np.float64]:
        """Compute test statistic (*t* or *F*).

        Parameters
        ----------
        baseline : float, default: 0.0
            Baseline value for null hypothesis test.

        Returns
        -------
        (n_voxels,) numpy.ndarray
            *t*-statistics or *F*-statistics per voxel.
        """
        if self.stat_ is not None and self.baseline == baseline:
            return self.stat_

        self.baseline = baseline

        if self.stat_type == "F":
            # F = (1/dim) * sum((effect - baseline)^2) / variance
            stat = (
                np.sum((self.effect - baseline) ** 2, axis=0)
                / self.dim
                / np.maximum(self.variance, self.tiny)
            )
        else:  # t-statistic
            # t = (effect - baseline) / sqrt(variance)
            stat = (self.effect - baseline) / np.sqrt(
                np.maximum(self.variance, self.tiny)
            )

        self.stat_ = np.squeeze(stat)
        return self.stat_

    def p_value(self, baseline: float = 0.0) -> npt.NDArray[np.float64]:
        """Compute *p*-values from test statistic.

        Parameters
        ----------
        baseline : float, default: 0.0
            Baseline value for null hypothesis test.

        Returns
        -------
        (n_voxels,) numpy.ndarray
            P-values per voxel.
        """
        if self.stat_ is None or self.baseline != baseline:
            self.stat(baseline)

        dof = min(self.dof, self.dofmax)

        if self.stat_type == "t":
            self.p_value_ = sps.t.sf(self.stat_, dof)
        else:  # F-statistic
            self.p_value_ = sps.f.sf(self.stat_, self.dim, dof)

        return self.p_value_

    def one_minus_pvalue(self, baseline: float = 0.0) -> npt.NDArray[np.float64]:
        """Compute 1 - *p*-values for numerical stability.

        Parameters
        ----------
        baseline : float, default :0.0
            Baseline value for null hypothesis test.

        Returns
        -------
        (n_voxels,) numpy.ndarray
            1 - *p*-values per voxel.
        """
        if self.stat_ is None or self.baseline != baseline:
            self.stat(baseline)

        dof = min(self.dof, self.dofmax)

        if self.stat_type == "t":
            self.one_minus_pvalue_ = sps.t.cdf(self.stat_, dof)
        else:  # F-statistic
            self.one_minus_pvalue_ = sps.f.cdf(self.stat_, self.dim, dof)

        return self.one_minus_pvalue_

    def z_score(self, baseline: float = 0.0) -> npt.NDArray[np.float64]:
        """Compute *z*-scores from *p*-values.

        Parameters
        ----------
        baseline : float, default: 0.0
            Baseline value for null hypothesis test.

        Returns
        -------
        (n_voxels,) numpy.ndarray
            *z*-scores per voxel.
        """
        if self.p_value_ is None or self.baseline != baseline:
            self.p_value(baseline)

        if self.one_minus_pvalue_ is None or self.baseline != baseline:
            self.one_minus_pvalue(baseline)

        assert self.p_value_ is not None
        assert self.one_minus_pvalue_ is not None
        self.z_score_ = _pvalues_to_zscore(self.p_value_, self.one_minus_pvalue_)
        return self.z_score_

    def __add__(self, other: "Contrast") -> "Contrast":
        """Add two contrasts for fixed-effects combination.

        Combines effects and variances assuming independent contrasts. Used for
        combining results across multiple runs.

        Parameters
        ----------
        other : Contrast
            Another contrast to add.

        Returns
        -------
        Contrast
            Combined contrast.

        Raises
        ------
        ValueError
            If contrasts have incompatible types or dimensions.

        """
        if self.stat_type != other.stat_type:
            raise ValueError(
                f"Cannot add contrasts with different stat types: "
                f"{self.stat_type} vs {other.stat_type}"
            )

        if self.dim != other.dim:
            raise ValueError(
                f"Cannot add contrasts with different dimensions: "
                f"{self.dim} vs {other.dim}"
            )

        # Fixed effects combination.
        effect = self.effect + other.effect
        variance = self.variance + other.variance
        dof = self.dof + other.dof

        return Contrast(
            effect=effect,
            variance=variance,
            dim=self.dim,
            dof=dof,
            stat_type=self.stat_type,
            tiny=self.tiny,
            dofmax=self.dofmax,
        )

    def __rmul__(self, scalar: float) -> "Contrast":
        """Multiply contrast by a scalar.

        Parameters
        ----------
        scalar : float
            Scalar multiplier.

        Returns
        -------
        Contrast
            Scaled contrast.
        """
        scalar = float(scalar)
        effect = self.effect * scalar
        variance = self.variance * scalar**2

        return Contrast(
            effect=effect,
            variance=variance,
            dim=self.dim,
            dof=self.dof,
            stat_type=self.stat_type,
            tiny=self.tiny,
            dofmax=self.dofmax,
        )

    __mul__ = __rmul__

    def __truediv__(self, scalar: float) -> "Contrast":
        """Divide contrast by a scalar."""
        return self.__rmul__(1.0 / float(scalar))
