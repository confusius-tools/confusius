"""DVARS computation for quality control.

Portions of this file are derived from Nipype, which is licensed under the Apache
License 2.0. See `NOTICE` file for details.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy import stats as sp_stats

from confusius.glm._utils import estimate_ar_coeffs
from confusius.validation import validate_time_series

if TYPE_CHECKING:
    import dask.array as da


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
    artifacts. Neither yields a z-score or calibrated artifact probability. The
    lower IQR interpolation follows Nipype's implementation (itself based on FSL).
    As a consequence, exact invariance to signal sign reversal is guaranteed only
    for `median` when feature selection is unchanged.

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

    # Relative DVARS is invariant to global intensity scaling. Conversely, the baseline
    # estimator has no effect on raw DVARS.
    target_median_intensity = normalization_factor if variant == "raw" else None
    selected_baseline_estimator = baseline_estimator if variant == "relative" else None
    if target_median_intensity is not None and (
        not np.isfinite(target_median_intensity) or target_median_intensity <= 0
    ):
        raise ValueError("normalization_factor must be finite and positive.")

    # Both eager and lazy paths use the same block computation. The AR(1) fit is skipped
    # unless that particular relative-DVARS baseline is requested.
    summarize_block = partial(
        _compute_dvars_block_statistics,
        remove_zero_variance=remove_zero_variance,
        variance_tolerance=variance_tolerance,
        estimate_ar1_difference_variance=selected_baseline_estimator == "ar1",
        target_median_intensity=target_median_intensity,
    )
    values = signals.transpose("time", ...).data

    if signals.chunks is None:
        result = _compute_dvars_from_block_statistics(
            summarize_block(values), selected_baseline_estimator
        )
    else:
        result = _compute_dvars_dask(
            values,
            summarize_block=summarize_block,
            baseline_estimator=selected_baseline_estimator,
            requires_global_median=target_median_intensity is not None,
        )

    return xr.DataArray(
        result,
        dims=["time"],
        coords={"time": signals.coords["time"]},
        attrs={
            "long_name": "relative DVARS" if variant == "relative" else "DVARS",
            "units": "a.u.",
        },
    )


def _compute_dvars_dask(
    values: da.Array,
    *,
    summarize_block: Callable[[npt.NDArray[np.number]], npt.NDArray[np.float64]],
    baseline_estimator: Literal["median", "ar1"] | None,
    requires_global_median: bool,
) -> da.Array:
    """Build a lazy DVARS computation from full-time spatial blocks."""
    import dask.array as da
    from dask import delayed

    n_timepoints = values.shape[0]

    if requires_global_median:
        # Exact intensity normalization needs the median of *all retained values*. Block
        # medians cannot be combined into that median, so this branch loads the full
        # recording in one task instead of processing spatial blocks.
        combined_summary = delayed(summarize_block)(values)
    else:
        # Each block must contain the entire time axis to compute temporal differences,
        # robust SDs, and (optionally) AR(1) coefficients correctly. The remaining
        # spatial blocks are processed independently, then their additive summaries are
        # summed without materializing the full recording.
        spatial_blocks = values.rechunk({0: -1}).to_delayed().ravel()
        block_summaries = [
            da.from_delayed(
                delayed(summarize_block)(block),
                shape=(n_timepoints + 1,),
                dtype=np.float64,
            )
            for block in spatial_blocks
        ]
        combined_summary = da.stack(block_summaries).sum(axis=0)

    # The small combined summary determines the baseline and final DVARS trace.
    # Errors depending on input values are raised only when this graph executes.
    return da.from_delayed(
        delayed(_compute_dvars_from_block_statistics)(
            combined_summary, baseline_estimator
        ),
        shape=(n_timepoints,),
        dtype=np.float64,
    )


def _compute_dvars_block_statistics(
    block: npt.NDArray[np.number],
    *,
    remove_zero_variance: bool,
    variance_tolerance: float,
    estimate_ar1_difference_variance: bool,
    target_median_intensity: float | None,
) -> npt.NDArray[np.float64]:
    """Return additive DVARS statistics for a full-time spatial block.

    The float64 summary contains `(time - 1)` sums of squared differences, followed by
    the number of retained features and the summed expected difference variance for the
    AR(1) baseline (zero when unused). Using one array lets Dask aggregate blocks with a
    simple sum, including when blocks contain different numbers of retained features.

    Parameters
    ----------
    block : (time, ...) numpy.ndarray
        Real, finite signals with a time axis of length at least two. Any other axes
        are flattened into features.
    remove_zero_variance : bool
        Whether to exclude features whose robust temporal SD is less than or
        equal to `variance_tolerance`, before computing either variant.
    variance_tolerance : float
        Finite, nonnegative threshold for robust temporal SD, estimated as
        IQR divided by the standard normal IQR using lower interpolation.
    estimate_ar1_difference_variance : bool
        Whether to compute the sum of expected difference variances for the AR(1)
        baseline. Ignored if `baseline_estimator` is `"median"` or `None`.
    target_median_intensity : float or None
        Positive target median signal intensity for raw DVARS. If not provided, no
        intensity normalization is performed. Ignored for relative DVARS, which is
        invariant to positive global intensity scaling.

    Returns
    -------
    (time + 1,) numpy.ndarray[float64]
        The first `(time - 1)` entries are sums of squared temporal differences across
        all retained features. The last two entries are the total number of retained
        features and the sum of expected difference variances for the AR(1) baseline.
    """
    # Differencing preserves float32/float64 precision. Float16 is promoted to float32,
    # and integers to float64 to avoid integer subtraction overflow.
    dtype = (
        np.result_type(block.dtype, np.float32)
        if np.issubdtype(block.dtype, np.floating)
        else np.dtype(np.float64)
    )
    values = np.asarray(block, dtype=dtype).reshape(block.shape[0], -1)
    if not np.isfinite(values).all():
        raise ValueError("signals must contain nonempty, finite features.")

    if remove_zero_variance or estimate_ar1_difference_variance:
        robust_sd = sp_stats.iqr(values, axis=0, scale="normal")
        if remove_zero_variance:
            retained = robust_sd > variance_tolerance
            values, robust_sd = values[:, retained], robust_sd[retained]

    if values.shape[1] == 0:
        # An individual block may be empty after filtering; only the aggregate should
        # fail if *all* blocks have no retained features.
        return np.zeros(block.shape[0] + 1, dtype=np.float64)

    if target_median_intensity is not None:
        median_intensity = np.median(values)
        if not np.isfinite(median_intensity) or median_intensity == 0:
            raise ValueError("Median signal intensity must be finite and nonzero.")
        values = values * (target_median_intensity / median_intensity)

    expected_difference_variance_sum = 0.0
    if estimate_ar1_difference_variance:
        # Under an AR(1) model, Var(Y[t] - Y[t-1]) = 2(1 - rho) * sigma**2.
        ar1, _ = estimate_ar_coeffs(values, order=1)
        difference_sd = np.sqrt(2 * (1 - ar1.reshape(-1))) * robust_sd
        expected_difference_variance_sum = np.sum(difference_sd**2, dtype=np.float64)

    squared_difference_sums = np.sum(
        np.diff(values, axis=0) ** 2, axis=1, dtype=np.float64
    )

    # Packed layout: [squared_difference_sums..., feature_count, AR1_variance_sum]. All
    # entries are additive across feature blocks, unlike the final RMS and median
    # baseline. Keeping them together avoids separate Dask reductions.
    return np.append(
        squared_difference_sums,
        [values.shape[1], expected_difference_variance_sum],
    )


def _compute_dvars_from_block_statistics(
    summary: npt.NDArray[np.float64],
    baseline_estimator: Literal["median", "ar1"] | None,
) -> npt.NDArray[np.float64]:
    """Turn an aggregated block summary into a raw or relative DVARS trace.

    Parameters
    ----------
    summary : (time + 1,) numpy.ndarray[float64]
        The first `(time - 1)` entries are sums of squared temporal differences across
        all retained features. The last two entries are the total number of retained
        features and the sum of expected difference variances for the AR(1) baseline.
    baseline_estimator : {"median", "ar1"} or None
        Baseline for relative DVARS. Ignored for raw DVARS.

    Returns
    -------
    (time,) numpy.ndarray[float64]
        Raw or relative DVARS, with the first value set to `NaN` because the first frame
        has no preceding frame.
    """
    squared_difference_sums = summary[:-2]
    feature_count, expected_difference_variance_sum = summary[-2:]
    if feature_count == 0:
        raise ValueError(
            "All signals have variance below tolerance. Check input data or "
            "adjust 'variance_tolerance'."
        )

    # Divide by the *total number of retained features*, rather than taking an average
    # of blockwise DVARS values (which would weight small blocks equally).
    dvars = np.sqrt(squared_difference_sums / feature_count)
    if baseline_estimator is not None:
        if baseline_estimator == "median":
            # This median is over genuine temporal differences only.
            baseline_squared = np.median(dvars**2)
        else:
            baseline_squared = expected_difference_variance_sum / feature_count
        baseline = np.sqrt(baseline_squared)
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("Relative DVARS baseline must be finite and positive.")
        dvars /= baseline

    # The first frame has no predecessor; prepend NaN after baseline estimation.
    return np.insert(dvars, 0, np.nan)
