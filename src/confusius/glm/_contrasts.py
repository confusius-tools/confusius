"""Contrast computation and statistical testing for GLM results.

This module provides the [`Contrast`][confusius.glm._contrasts.Contrast] result
container: a passive dataclass holding a contrast's effect/variance estimates and the
associated test statistics (t or F), p-values, and z-scores. The math (adapted from
Nilearn) lives in
[`Contrast.from_estimate`][confusius.glm._contrasts.Contrast.from_estimate], which is
the canonical way to construct a `Contrast`. Direct instantiation is reserved for tests
and downstream code that already has all fields precomputed.

Portions of this file are derived from Nilearn, which is licensed under the BSD-3-Clause
License. See `NOTICE` file for details.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.stats as sps

from confusius._utils import find_stack_level

if TYPE_CHECKING:
    import numpy.typing as npt


def _pvalues_to_zscore(
    pvalue: npt.NDArray[np.float64],
    one_minus_pvalue: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert p-values to z-scores.

    Uses the survival function for the upper tail and the CDF (via `one_minus_pvalue`)
    for the lower tail to avoid catastrophic cancellation near the tails. Adapted from
    `nilearn.glm._utils.zscore`.

    Parameters
    ----------
    pvalue : (n,) numpy.ndarray
        p-values to convert.
    one_minus_pvalue : (n,) numpy.ndarray
        `1 - pvalue` computed via the CDF for numerical stability.

    Returns
    -------
    (n,) numpy.ndarray
        z-scores corresponding to the p-values.
    """
    pvalue = np.clip(pvalue, 1.0e-300, 1.0 - 1.0e-16)
    z_scores_sf = sps.norm.isf(pvalue)
    use_cdf = z_scores_sf < 0

    one_minus_pvalue = np.clip(one_minus_pvalue, 1.0e-300, 1.0 - 1.0e-16)
    z_scores_cdf = sps.norm.ppf(one_minus_pvalue)
    use_sf = np.logical_not(use_cdf)

    z_scores = np.empty(pvalue.size)
    z_scores[np.atleast_1d(use_cdf)] = z_scores_cdf[use_cdf]
    z_scores[np.atleast_1d(use_sf)] = z_scores_sf[use_sf]

    return z_scores


@dataclass(eq=False)
class Contrast:
    """Passive container for contrast results.

    Holds a contrast's effect/variance estimates and the precomputed test statistics. To
    construct one from raw `(effect, variance, dof)`, use
    [`from_estimate`][confusius.glm._contrasts.Contrast.from_estimate]; that is where
    the t/F statistic, p-value, and z-score are computed.

    Supports addition for fixed-effects combination across runs, and scalar
    multiplication/division for rescaling.

    Attributes
    ----------
    effect : (n_voxels,) numpy.ndarray or (dim, n_voxels) numpy.ndarray
        Contrast effect estimates.
    variance : (n_voxels,) numpy.ndarray
        Contrast variance estimates.
    dim : int
        Contrast dimension (1 for *t*, >1 for *F*).
    dof : float
        Degrees of freedom of the residuals.
    stat_type : {"t", "F"}
        Type of contrast statistic.
    baseline : float
        Null-hypothesis value used to compute the statistic.
    statistic : (n_voxels,) numpy.ndarray
        Test statistic (*t* or *F*).
    pvalue : (n_voxels,) numpy.ndarray
        One-sided *p*-values (upper tail).
    one_minus_pvalue : (n_voxels,) numpy.ndarray
        `1 - pvalue`, computed via the CDF for numerical stability in the lower tail.
    zscore : (n_voxels,) numpy.ndarray
        *z*-scores derived from `pvalue`/`one_minus_pvalue`.
    tiny : float
        Small value used to guard against division by zero.
    dofmax : float
        Cap on degrees of freedom used when computing p-values.
    """

    effect: npt.NDArray[np.floating]
    variance: npt.NDArray[np.floating]
    dim: int
    dof: float
    stat_type: Literal["t", "F"]
    baseline: float
    statistic: npt.NDArray[np.float64]
    pvalue: npt.NDArray[np.float64]
    one_minus_pvalue: npt.NDArray[np.float64]
    zscore: npt.NDArray[np.float64]
    tiny: float = 1e-50
    dofmax: float = 1e10

    @classmethod
    def from_estimate(
        cls,
        effect: npt.NDArray[np.floating],
        variance: npt.NDArray[np.floating],
        *,
        dim: int | None = None,
        dof: float = 1e10,
        stat_type: Literal["t", "F"] = "t",
        baseline: float = 0.0,
        tiny: float = 1e-50,
        dofmax: float = 1e10,
    ) -> Contrast:
        """Build a [`Contrast`][confusius.glm.Contrast] from raw estimates.

        Computes the test statistic, *p*-values, and *z*-scores for the supplied
        `baseline` and packs everything into a [`Contrast`][confusius.glm.Contrast]
        result object.

        Parameters
        ----------
        effect : (n_voxels,) or (dim, n_voxels) numpy.ndarray
            Contrast effect estimates.
        variance : (n_voxels,) numpy.ndarray
            Contrast variance estimates.
        dim : int, optional
            Contrast dimension. If not provided, inferred from `effect`.
        dof : float, default: 1e10
            Degrees of freedom of the residuals.
        stat_type : {"t", "F"}, default: "t"
            Type of contrast statistic.
        baseline : float, default: 0.0
            Null-hypothesis value tested against. The statistic is computed as `(effect
            - baseline) / sqrt(variance)` (*t*) or `sum((effect - baseline)**2) / dim /
            variance` (*F*). To test a different null, call `from_estimate` again with a
            new `baseline`.
        tiny : float, default: 1e-50
            Small value to guard against division by zero.
        dofmax : float, default: 1e10
            Maximum degrees of freedom used when computing p-values.

        Returns
        -------
        Contrast
            Contrast result with `statistic`, `pvalue`, `one_minus_pvalue`, and `zscore`
            precomputed.

        Raises
        ------
        ValueError
            If `effect` is not 1D or 2D, `variance` is not 1-D, or `stat_type` is not
            `"t"` or `"F"`.
        """
        if variance.ndim != 1:
            raise ValueError(f"variance must be 1D, got shape {variance.shape}")
        if effect.ndim not in (1, 2):
            raise ValueError(f"effect must be 1D or 2D, got shape {effect.shape}")

        resolved_dim = effect.shape[0] if (dim is None and effect.ndim == 2) else dim
        if resolved_dim is None:
            resolved_dim = 1

        if resolved_dim > 1 and stat_type == "t":
            warnings.warn(
                f"stat_type='t' is incompatible with a {resolved_dim}-dimensional "
                "contrast; switching to 'F'.",
                stacklevel=find_stack_level(),
            )
            stat_type = "F"

        if stat_type not in ("t", "F"):
            raise ValueError(f"stat_type must be 't' or 'F', got {stat_type}")

        if stat_type == "F":
            raw = (
                np.sum((effect - baseline) ** 2, axis=0)
                / resolved_dim
                / np.maximum(variance, tiny)
            )
        else:
            raw = (effect - baseline) / np.sqrt(np.maximum(variance, tiny))
        statistic = np.squeeze(raw)

        # We compute both `pvalue = sf(statistic)` and `one_minus_pvalue =
        # cdf(statistic)` directly from scipy: each is accurate where the other would
        # suffer from catastrophic cancellation (cdf in the upper tail; sf in the lower
        # tail).
        effective_dof = min(dof, dofmax)
        if stat_type == "t":
            pvalue = sps.t.sf(statistic, effective_dof)
            one_minus_pvalue = sps.t.cdf(statistic, effective_dof)
        else:
            pvalue = sps.f.sf(statistic, resolved_dim, effective_dof)
            one_minus_pvalue = sps.f.cdf(statistic, resolved_dim, effective_dof)

        zscore = _pvalues_to_zscore(pvalue, one_minus_pvalue)

        return cls(
            effect=effect,
            variance=variance,
            dim=resolved_dim,
            dof=dof,
            stat_type=stat_type,
            baseline=baseline,
            statistic=statistic,
            pvalue=pvalue,
            one_minus_pvalue=one_minus_pvalue,
            zscore=zscore,
            tiny=tiny,
            dofmax=dofmax,
        )

    def __add__(self, other: Contrast) -> Contrast:
        """Combine two contrasts via fixed-effects (independent runs).

        Effects, variances, and degrees of freedom are summed. The combined result
        inherits `stat_type` and `baseline` from the operands, which must match.

        Raises
        ------
        ValueError
            If the operands have different `stat_type`, `dim`, or `baseline`.
        """
        if self.stat_type != other.stat_type:
            raise ValueError(
                "Cannot add contrasts with different stat types: "
                f"{self.stat_type} vs {other.stat_type}"
            )
        if self.dim != other.dim:
            raise ValueError(
                "Cannot add contrasts with different dimensions: "
                f"{self.dim} vs {other.dim}"
            )
        if self.baseline != other.baseline:
            raise ValueError(
                "Cannot add contrasts with different baselines: "
                f"{self.baseline} vs {other.baseline}"
            )

        return Contrast.from_estimate(
            effect=self.effect + other.effect,
            variance=self.variance + other.variance,
            dim=self.dim,
            dof=self.dof + other.dof,
            stat_type=self.stat_type,
            baseline=self.baseline,
            tiny=self.tiny,
            dofmax=self.dofmax,
        )

    def __rmul__(self, scalar: float) -> Contrast:
        """Scale the contrast by a scalar (effect by *s*, variance by *s*²)."""
        scalar = float(scalar)
        return Contrast.from_estimate(
            effect=self.effect * scalar,
            variance=self.variance * scalar**2,
            dim=self.dim,
            dof=self.dof,
            stat_type=self.stat_type,
            baseline=self.baseline,
            tiny=self.tiny,
            dofmax=self.dofmax,
        )

    __mul__ = __rmul__

    def __truediv__(self, scalar: float) -> Contrast:
        """Divide the contrast by a scalar."""
        return self.__rmul__(1.0 / float(scalar))
