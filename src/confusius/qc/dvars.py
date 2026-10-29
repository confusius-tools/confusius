"""DVARS computation for quality control.

Portions of this file are derived from Nipype, which is licensed under the Apache
License 2.0. See `NOTICE` file for details.
"""

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

    **Laziness:** Median-based relative DVARS is Dask-lazy, rechunking each spatial
    block along the full time axis before reducing to a DVARS trace. Spatial chunks must
    fit in memory, and data-dependent errors occur at compute time. Raw DVARS and `ar1`
    remain eager. Robust SDs are computed only when needed for filtering or `ar1`.

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
    if (
        signals.chunks is not None
        and variant == "relative"
        and baseline_estimator == "median"
    ):
        import dask.array as da
        from dask import delayed

        # Finish each full-time spatial block before combining small summaries.
        blocks = signals.transpose("time", ...).data.rechunk({0: -1}).to_delayed()
        summaries = [
            delayed(_summarize_dvars_block)(
                block, remove_zero_variance, variance_tolerance
            )
            for block in blocks.ravel()
        ]
        combined = da.stack(
            [
                da.from_delayed(summary, shape=(signals.sizes["time"],), dtype=float)
                for summary in summaries
            ]
        ).sum(axis=0)
        result = delayed(_finish_median_dvars)(combined)
        dvars = xr.DataArray(
            da.from_delayed(result, shape=(signals.sizes["time"],), dtype=float),
            dims=["time"],
            coords={"time": signals.coords["time"]},
            attrs={"long_name": "relative DVARS", "units": "1"},
        )
        dvars.name = None
        return dvars

    # Promote integers before differencing without doubling float32 recordings.
    dtype = (
        np.result_type(signals.dtype, np.float32)
        if np.issubdtype(signals.dtype, np.floating)
        else np.dtype(np.float64)
    )
    values = np.asarray(signals.transpose("time", ...).values, dtype=dtype)
    values = values.reshape(signals.sizes["time"], -1)
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("signals must contain nonempty, finite features.")

    needs_ar1 = variant == "relative" and baseline_estimator == "ar1"
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

    dvars = np.sqrt(np.mean(np.diff(values, axis=0) ** 2, axis=1, dtype=np.float64))
    if variant == "relative":
        if baseline_estimator == "median":
            # Estimate the baseline before inserting the artificial first sample.
            baseline = np.sqrt(np.median(dvars**2))
        else:
            ar1, _ = estimate_ar_coeffs(values, order=1)
            diff_sd = np.sqrt(2 * (1 - ar1.reshape(-1))) * signals_sd
            baseline = np.sqrt(np.mean(diff_sd**2))
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("Relative DVARS baseline must be finite and positive.")
        dvars = dvars / baseline

    return xr.DataArray(
        np.insert(dvars, 0, np.nan),
        dims=["time"],
        coords={"time": signals.coords["time"]},
        attrs={
            "long_name": "relative DVARS" if variant == "relative" else "DVARS",
            "units": "1" if variant == "relative" else "a.u.",
        },
    )


def _summarize_dvars_block(
    block: npt.NDArray[np.number],
    remove_zero_variance: bool,
    variance_tolerance: float,
) -> npt.NDArray[np.float64]:
    """Reduce one full-time spatial block without retaining voxelwise outputs.

    Parameters
    ----------
    block : (time, ...) numpy.ndarray
        Real input signals for one spatial block.
    remove_zero_variance : bool
        Whether to exclude features with low robust temporal SD.
    variance_tolerance : float
        Threshold for robust temporal SD.

    Returns
    -------
    (time,) numpy.ndarray
        Sum of squared differences at each real transition, followed by the
        retained feature count. Packing both into one summary keeps Dask from
        independently optimizing and duplicating their shared upstream work.

    Raises
    ------
    ValueError
        If the block contains nonfinite signals.
    """
    dtype = (
        np.result_type(block.dtype, np.float32)
        if np.issubdtype(block.dtype, np.floating)
        else np.dtype(np.float64)
    )
    values = np.asarray(block, dtype=dtype).reshape(block.shape[0], -1)
    if not np.isfinite(values).all():
        raise ValueError("signals must contain nonempty, finite features.")
    if remove_zero_variance:
        sd = sp_stats.iqr(values, axis=0, scale="normal", interpolation="lower")
        values = values[:, sd > variance_tolerance]
    return np.append(
        np.sum(np.diff(values, axis=0) ** 2, axis=1, dtype=np.float64), values.shape[1]
    )


def _finish_median_dvars(summary: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize the reduced temporal trace and prepend an undefined first value.

    Parameters
    ----------
    summary : (time,) numpy.ndarray
        Sum of squared differences across all retained features, followed by
        the total retained feature count.

    Returns
    -------
    (time,) numpy.ndarray
        Median-based relative DVARS, including the artificial first sample.

    Raises
    ------
    ValueError
        If no features remain or the baseline is nonfinite or nonpositive.
    """
    count = summary[-1]
    if count == 0:
        raise ValueError(
            "All signals have variance below tolerance. Check input data or "
            "adjust 'variance_tolerance'."
        )
    dvars = np.sqrt(summary[:-1] / count)
    baseline = np.sqrt(np.median(dvars**2))
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("Relative DVARS baseline must be finite and positive.")
    dvars /= baseline
    return np.insert(dvars, 0, np.nan)
