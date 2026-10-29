"""DVARS computation for quality control.

Portions of this file are derived from Nipype, which is licensed under the Apache
License 2.0. See `NOTICE` file for details.
"""

from functools import partial
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy import stats as sp_stats

from confusius.glm._utils import estimate_ar_coeffs
from confusius.validation import validate_time_series


def compute_dvars(
    signals: xr.DataArray,
    *,
    variant: Literal["raw", "relative"] = "relative",
    baseline_estimator: Literal["median", "ar1"] = "median",
    normalization_factor: float | None = 1000,
    remove_zero_variance: bool = True,
    variance_tolerance: float = 0.0,
) -> xr.DataArray:
    """Compute the DVARS quality control metric.

    DVARS measures the root mean square signal change between consecutive time points.
    Relative DVARS divides this trace by an estimated background baseline.

    This function was originally adapted from
    `nipype.algorithms.confounds.compute_dvars`.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Real, finite signals with a `time` dimension containing at least two samples.
        Any other dimensions are flattened into features, regardless of their names or
        ordering.
    variant : {"raw", "relative"}, default: "relative"
        DVARS variant to compute. Raw DVARS is the root mean square of temporal
        differences, while relative DVARS divides raw DVARS by a recording-wide
        baseline, selected by `baseline_estimator`.
    baseline_estimator : {"median", "ar1"}, default: "median"
        Baseline for relative DVARS. `"median"` uses the square root of median squared
        raw DVARS, excluding the artificial first sample. `"ar1"` uses the RMS of
        expected voxelwise difference standard deviations, estimated from robust
        temporal SDs and lag-1 autocorrelations. Ignored for raw DVARS.
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
        DVARS with the input time coordinates. The first value is `NaN` because the
        first frame has no preceding frame.

    Raises
    ------
    TypeError
        If signals are not real numeric data.
    ValueError
        If time has fewer than two samples, features are empty or nonfinite,
        parameters are invalid, all features are removed, or a required intensity
        normalization or relative DVARS baseline is zero or nonfinite.

    See Also
    --------
    confusius.qc.compute_cv : Compute voxelwise coefficients of variation.
    confusius.qc.compute_tsnr : Compute temporal signal-to-noise ratios.

    Notes
    -----
    Raw DVARS is `sqrt(mean_i(diff(Y_i)**2))`. Relative DVARS divides it by a
    recording-wide baseline estimated using either:

    - `median`: `sqrt(median_t(raw_dvars_t**2))`.
    - `ar1`: `sqrt(mean_i(2 * (1 - rho_i) * sigma_i**2))`.

    Here `sigma_i` is the robust temporal SD and `rho_i` the Yule-Walker lag-1
    autocorrelation. Originally called *standardized DVARS* by Nichols (2013), the
    measure was renamed *relative DVARS* by Afyouni and Nichols (2018).

    The `median` estimator avoids AR(1) fitting but assumes most time points reflect
    baseline activity. The `ar1` estimator assumes stationarity and is sensitive to
    artifacts. Neither yields a z-score or calibrated artifact probability. Due to IQR
    interpolation, exact invariance to signal sign reversal is guaranteed only for
    `median` when feature selection is unchanged.

    **Precision:** Float32/float64 inputs retain their precision during differencing;
    float16 and integer inputs are promoted to float32 and float64, respectively.
    Spatial sums accumulate in float64.

    **Laziness:** Dask inputs remain lazy; data-dependent errors occur at compute
    time. Relative DVARS and unnormalized raw DVARS process spatial chunks whose
    full time series must fit in memory. Intensity-normalized raw DVARS instead
    loads the full recording to compute the exact global intensity median.

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
    if baseline_estimator not in ("median", "ar1"):
        raise ValueError("baseline_estimator must be 'median' or 'ar1'.")
    if not np.isfinite(variance_tolerance) or variance_tolerance < 0:
        raise ValueError("variance_tolerance must be finite and nonnegative.")
    if not np.issubdtype(signals.dtype, np.number) or np.iscomplexobj(signals):
        raise TypeError("signals must contain real numeric data.")

    if signals.size == 0:
        raise ValueError("signals must contain nonempty, finite features.")
    normalization = normalization_factor if variant == "raw" else None
    if normalization is not None and (
        not np.isfinite(normalization) or normalization <= 0
    ):
        raise ValueError("normalization_factor must be finite and positive.")
    baseline = baseline_estimator if variant == "relative" else None
    values = signals.transpose("time", ...).data
    compute_block_statistics = partial(
        _compute_dvars_block_statistics,
        remove_zero_variance=remove_zero_variance,
        variance_tolerance=variance_tolerance,
        estimate_difference_variance=baseline == "ar1",
        normalization_factor=normalization,
    )

    if signals.chunks is not None:
        import dask.array as da
        from dask import delayed

        if normalization is not None:
            # An exact global intensity median cannot be recovered from block medians.
            statistics = delayed(compute_block_statistics)(values)
        else:
            blocks = values.rechunk({0: -1}).to_delayed().ravel()
            block_statistics = [
                da.from_delayed(
                    delayed(compute_block_statistics)(block),
                    shape=(signals.sizes["time"] + 1,),
                    dtype=float,
                )
                for block in blocks
            ]
            statistics = da.stack(block_statistics).sum(axis=0)
        result = da.from_delayed(
            delayed(_compute_dvars_from_statistics)(statistics, baseline),
            shape=(signals.sizes["time"],),
            dtype=float,
        )
    else:
        result = _compute_dvars_from_statistics(
            compute_block_statistics(values), baseline
        )

    dvars = xr.DataArray(
        result,
        dims=["time"],
        coords={"time": signals.coords["time"]},
        attrs={
            "long_name": "relative DVARS" if variant == "relative" else "DVARS",
            "units": "1" if variant == "relative" else "a.u.",
        },
    )
    dvars.name = None
    return dvars


def _compute_dvars_block_statistics(
    block: npt.NDArray[np.number],
    *,
    remove_zero_variance: bool,
    variance_tolerance: float,
    estimate_difference_variance: bool,
    normalization_factor: float | None,
) -> npt.NDArray[np.float64]:
    """Compute additive DVARS statistics for one full-time spatial block.

    Parameters
    ----------
    block : (time, ...) numpy.ndarray
        Real input signals for one spatial block.
    remove_zero_variance : bool
        Whether to exclude features with low robust temporal SD.
    variance_tolerance : float
        Threshold for robust temporal SD.
    estimate_difference_variance : bool
        Whether to estimate voxelwise difference variances using AR(1).
    normalization_factor : float or None
        Target median intensity for raw DVARS. When provided, `block` must contain
        the entire recording so normalization uses the global retained median.

    Returns
    -------
    (time + 1,) numpy.ndarray
        Additive summary: `time - 1` squared-difference sums, then the retained
        feature count and summed AR(1) difference variance (zero when unused).
        Keeping these together avoids duplicate reads and IQR/AR(1) calculations
        when Dask optimizes the graph. Summing summaries weights features equally,
        even when spatial blocks retain different numbers of features.

    Raises
    ------
    ValueError
        If the block contains nonfinite signals or its required intensity median
        is zero or nonfinite.
    """
    dtype = (
        np.result_type(block.dtype, np.float32)
        if np.issubdtype(block.dtype, np.floating)
        else np.dtype(np.float64)
    )
    values = np.asarray(block, dtype=dtype).reshape(block.shape[0], -1)
    if not np.isfinite(values).all():
        raise ValueError("signals must contain nonempty, finite features.")
    if remove_zero_variance or estimate_difference_variance:
        sd = sp_stats.iqr(values, axis=0, scale="normal", interpolation="lower")
        if remove_zero_variance:
            keep = sd > variance_tolerance
            values, sd = values[:, keep], sd[keep]
    if values.shape[1] == 0:
        return np.zeros(block.shape[0] + 1, dtype=np.float64)

    if normalization_factor is not None:
        median_intensity = np.median(values)
        if not np.isfinite(median_intensity) or median_intensity == 0:
            raise ValueError("Median signal intensity must be finite and nonzero.")
        values = values * (normalization_factor / median_intensity)

    variance_sum = 0.0
    if estimate_difference_variance:
        ar1, _ = estimate_ar_coeffs(values, order=1)
        diff_sd = np.sqrt(2 * (1 - ar1.reshape(-1))) * sd
        variance_sum = np.sum(diff_sd**2, dtype=np.float64)
    squared_sum = np.sum(np.diff(values, axis=0) ** 2, axis=1, dtype=np.float64)
    return np.append(squared_sum, [values.shape[1], variance_sum])


def _compute_dvars_from_statistics(
    statistics: npt.NDArray[np.float64],
    baseline_estimator: Literal["median", "ar1"] | None,
) -> npt.NDArray[np.float64]:
    """Convert aggregated statistics to raw or relative DVARS.

    Parameters
    ----------
    statistics : (time + 1,) numpy.ndarray
        Combined squared-difference sums, followed by the retained feature count
        and summed AR(1) difference variance.
    baseline_estimator : {"median", "ar1"} or None
        Relative DVARS baseline, or `None` for raw DVARS.

    Returns
    -------
    (time,) numpy.ndarray
        DVARS with `NaN` at the undefined first time point.

    Raises
    ------
    ValueError
        If no features remain or the baseline is nonfinite or nonpositive.
    """
    count, variance_sum = statistics[-2:]
    if count == 0:
        raise ValueError(
            "All signals have variance below tolerance. Check input data or "
            "adjust 'variance_tolerance'."
        )
    dvars = np.sqrt(statistics[:-2] / count)
    if baseline_estimator is not None:
        baseline = np.sqrt(
            np.median(dvars**2)
            if baseline_estimator == "median"
            else variance_sum / count
        )
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("Relative DVARS baseline must be finite and positive.")
        dvars /= baseline
    return np.insert(dvars, 0, np.nan)
