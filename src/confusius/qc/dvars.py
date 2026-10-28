"""DVARS computation for quality control.

Portions of this file are derived from Nipype, which is licensed under the Apache
License 2.0. See `NOTICE` file for details.
"""

import warnings
from typing import Literal

import numpy as np
import xarray as xr
from scipy import stats as sp_stats

from confusius.glm._utils import estimate_ar_coeffs
from confusius.validation import validate_time_series


def compute_dvars(
    signals: xr.DataArray,
    *,
    variant: Literal["raw", "relative"] = "relative",
    baseline_estimator: Literal["median", "ar1", "legacy"] = "median",
    normalization_factor: float | None = 1000,
    remove_zero_variance: bool = True,
    variance_tolerance: float = 0.0,
) -> xr.DataArray:
    """Compute raw or relative DVARS from temporal signal differences.

    DVARS measures the root mean square signal change between consecutive time
    points. Relative DVARS divides this trace by an estimated background baseline.
    The raw and legacy calculations were adapted from
    `nipype.algorithms.confounds.compute_dvars`.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Real, finite signals with a `time` dimension containing at least two
        samples. Any other dimensions are flattened into features, regardless of
        their names or ordering. A single time series is also supported.
    variant : {"raw", "relative"}, default: "relative"
        Whether to return raw RMS temporal differences or dimensionless Relative
        DVARS. Relative DVARS uses the selected `baseline_estimator`.
    baseline_estimator : {"median", "ar1", "legacy"}, default: "median"
        Baseline for relative DVARS; ignored for raw DVARS. `"median"` uses the
        square root of median squared raw DVARS, excluding the artificial first
        sample. `"ar1"` uses the RMS of expected voxelwise difference standard
        deviations, estimated from robust temporal SDs and lag-1 autocorrelations.
        `"legacy"` uses their arithmetic mean instead, preserving the normalization
        used in ConfUSIus 0.8.0 and earlier. The legacy estimator is deprecated and
        will be removed in a future release.
    normalization_factor : float or None, default: 1000
        Positive target median signal intensity for raw DVARS.
        If not provided, no intensity normalization is performed. Ignored for
        relative DVARS, which is invariant to positive global intensity scaling.
    remove_zero_variance : bool, default: True
        Whether to exclude features whose robust temporal SD is less than or
        equal to `variance_tolerance`, before computing either variant.
    variance_tolerance : float, default: 0.0
        Finite, nonnegative threshold for robust temporal SD, estimated as
        IQR divided by the standard normal IQR using lower interpolation.

    Returns
    -------
    (time,) xarray.DataArray
        DVARS with the input time coordinates. The first value is the minimum
        resulting DVARS, following the existing FSL convention. Relative DVARS
        has units `"1"`; raw DVARS has arbitrary intensity units (`"a.u."`).

    Raises
    ------
    TypeError
        If signals are not real numeric data.
    ValueError
        If time has fewer than two samples, features are empty or nonfinite,
        parameters are invalid, all features are removed, or a required intensity
        normalization or relative DVARS baseline is zero or nonfinite.

    Warns
    -----
    DeprecationWarning
        When relative DVARS uses `baseline_estimator="legacy"`.

    See Also
    --------
    confusius.qc.compute_cv : Compute voxelwise coefficients of variation.
    confusius.qc.compute_tsnr : Compute temporal signal-to-noise ratios.

    Notes
    -----
    For features indexed by i, raw DVARS is `sqrt(mean_i(diff(Y_i)**2))`.
    Relative DVARS divides raw DVARS by one of these recording-wide baselines:

    - `median`: `sqrt(median_t(raw_dvars_t**2))`.
    - `ar1`: `sqrt(mean_i(2 * (1 - rho_i) * sigma_i**2))`.
    - `legacy`: `mean_i(sqrt(2 * (1 - rho_i)) * sigma_i)`.

    Here `sigma_i` is the robust temporal SD and `rho_i` is the Yule-Walker
    lag-1 autocorrelation estimate. The RMS baseline is at least as large as the
    legacy mean-SD baseline, with equality only for identical difference SDs.
    Lower-interpolated IQRs preserve the historical estimator but can change
    slightly under sign reversal; exact negative-scale invariance is guaranteed
    only for the median baseline when zero-variance selection is unchanged.

    Nichols originally called this normalization Standardized DVARS; Afyouni and
    Nichols (2018) subsequently used the name relative DVARS. Their empirical
    median-squared estimator avoids AR(1) fitting, but assumes the recording's
    background dominates the temporal distribution. Frequent artifacts or a
    nonstationary baseline can compromise it. The AR(1)-based estimators assume
    stationary variance and lag-1 covariance; their autocorrelation estimates
    are not robust to artifacts. Relative DVARS is not a Z-score or a calibrated
    cross-recording artifact probability, regardless of estimator.

    **Changed in 0.9.0:** The default is median-based relative DVARS, rather than
    the legacy normalization used in 0.8.0 and earlier. Numerical results and
    suitable thresholds change. The `standardize` argument was removed: use
    `variant="raw"` for the former `standardize=False`, or explicitly select
    `baseline_estimator="legacy"` for the former `standardize=True`. All options
    are now keyword-only.

    **Laziness:** Dask-backed inputs are accepted but computed eagerly. Robust
    SDs are computed only for zero-variance filtering or AR(1)-based baselines;
    AR(1) coefficients are computed only for relative `ar1` and `legacy`.

    References
    ----------
    [^1]:
        Smyser, C. D., et al. (2011). Functional Connectivity MRI in Infants:
        Exploration of the Functional Organization of the Developing Brain.
        NeuroImage, 56(3), 1437–1452.
        <https://doi.org/10.1016/j.neuroimage.2011.02.073>.
    [^2]:
        Power, J. D., et al. (2012). Spurious but Systematic Correlations in
        Functional Connectivity MRI Networks Arise from Subject Motion.
        NeuroImage, 59(3), 2142–2154.
        <https://doi.org/10.1016/j.neuroimage.2011.10.018>.
    [^3]:
        Nichols, T. E. (2017). Notes on Creating a Standardized Version of DVARS.
        <https://doi.org/10.48550/arXiv.1704.01469>.
    [^4]:
        Afyouni, S., and Nichols, T. E. (2018). Insight and Inference for DVARS.
        NeuroImage, 172, 291–312.
        <https://doi.org/10.1016/j.neuroimage.2017.12.098>.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> signals = xr.DataArray(
    ...     100 + rng.standard_normal((100, 50)),
    ...     dims=("time", "region"),
    ...     coords={"time": np.arange(100) * 0.1},
    ... )
    >>> relative = compute_dvars(signals)
    >>> raw = compute_dvars(signals, variant="raw", normalization_factor=None)
    >>> relative.shape, raw.shape
    ((100,), (100,))
    """
    validate_time_series(
        signals, operation_name="compute_dvars", require_unchunked_time=False
    )
    if variant not in ("raw", "relative"):
        raise ValueError("variant must be 'raw' or 'relative'.")
    if baseline_estimator not in ("median", "ar1", "legacy"):
        raise ValueError("baseline_estimator must be 'median', 'ar1', or 'legacy'.")
    if not np.isfinite(variance_tolerance) or variance_tolerance < 0:
        raise ValueError("variance_tolerance must be finite and nonnegative.")
    if not np.issubdtype(signals.dtype, np.number) or np.iscomplexobj(signals):
        raise TypeError("signals must contain real numeric data.")

    # Convert before differencing to avoid integer overflow and repeated Dask work.
    values = np.asarray(signals.transpose("time", ...).values, dtype=np.float64)
    values = values.reshape(signals.sizes["time"], -1)
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("signals must contain nonempty, finite features.")

    needs_ar1 = variant == "relative" and baseline_estimator in ("ar1", "legacy")
    if remove_zero_variance or needs_ar1:
        signals_sd = sp_stats.iqr(values, axis=0, scale="normal", interpolation="lower")
        if remove_zero_variance:
            nonzero_mask = signals_sd > variance_tolerance
            if not nonzero_mask.any():
                raise ValueError(
                    "All signals have variance below tolerance. Check input data or "
                    "adjust 'variance_tolerance'."
                )
            values = values[:, nonzero_mask]
            signals_sd = signals_sd[nonzero_mask]

    if variant == "raw" and normalization_factor is not None:
        if not np.isfinite(normalization_factor) or normalization_factor <= 0:
            raise ValueError("normalization_factor must be finite and positive.")
        median_intensity = np.median(values)
        if not np.isfinite(median_intensity) or median_intensity == 0:
            raise ValueError("Median signal intensity must be finite and nonzero.")
        values = values * (normalization_factor / median_intensity)

    dvars = np.sqrt(np.mean(np.diff(values, axis=0) ** 2, axis=1))
    if variant == "relative":
        if baseline_estimator == "median":
            # Estimate the baseline before inserting the artificial first sample.
            baseline = np.sqrt(np.median(dvars**2))
        else:
            if baseline_estimator == "legacy":
                warnings.warn(
                    'baseline_estimator="legacy" is deprecated and will be removed '
                    'in a future release. Use "median" or "ar1" instead; neither '
                    "preserves legacy numerical values.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            ar1, _ = estimate_ar_coeffs(values, order=1)
            diff_sd = np.sqrt(2 * (1 - ar1.reshape(-1))) * signals_sd
            baseline = (
                np.mean(diff_sd)
                if baseline_estimator == "legacy"
                else np.sqrt(np.mean(diff_sd**2))
            )
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("Relative DVARS baseline must be finite and positive.")
        dvars = dvars / baseline

    return xr.DataArray(
        np.insert(dvars, 0, dvars.min()),
        dims=["time"],
        coords={"time": signals.coords["time"]},
        attrs={
            "long_name": "relative DVARS" if variant == "relative" else "DVARS",
            "units": "1" if variant == "relative" else "a.u.",
        },
    )
