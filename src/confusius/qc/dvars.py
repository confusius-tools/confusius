"""DVARS computation for quality control.

Portions of this file are derived from Nipype, which is licensed under the Apache
License 2.0. See `NOTICE` file for details.
"""

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.validation import validate_time_series


def _ar1_yule_walker(signal: npt.NDArray) -> float:
    """Estimate AR(1) coefficient using Yule-Walker equations.

    The Yule-Walker equations estimate the autoregressive coefficients of a process by
    solving a system of linear equations involving the autocorrelation sequence.

    For an AR(1) process `x(n) = a(1) * x(n-1) + e(n)`, the Yule-Walker equation
    is: `R(1) = a(1) * R(0)`, where `R(k)` is the autocorrelation at lag `k`.

    Parameters
    ----------
    signal : (t,) numpy.ndarray
        Array of signal values.

    Returns
    -------
    float
        The AR(1) coefficient `a(1)`.

    Notes
    -----
    For AR(1), we only need autocorrelation at lags 0 and 1, so we use direct
    computation (`O(n)`) rather than FFT-based methods (`O(n log n)`).
    """
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - signal.mean()

    n = len(signal)

    # Direct computation of autocorrelation at lags 0 and 1.
    # For AR(1), we only need these two values, so full FFT is unnecessary.
    r0 = np.dot(signal, signal) / n  # lag 0
    r1 = np.dot(signal[:-1], signal[1:]) / n  # lag 1

    # AR(1) coefficient: R(1) / R(0).
    # Add small regularization to avoid division by zero.
    if r0 < 1e-15:
        return 0.0

    return np.clip(r1 / r0, -1.0, 1.0)


def compute_dvars(
    signals: xr.DataArray,
    standardize: bool = True,
    normalization_factor: float | None = 1000,
    remove_zero_variance: bool = True,
    variance_tolerance: float = 0.0,
) -> xr.DataArray:
    """Compute the DVARS metric.

    DVARS (D referring to temporal derivative, VARS to RMS variance) measures how much
    the intensity at one time point changes relative to the previous time point. The
    standardized version accounts for the expected variance given the temporal
    autocorrelation structure of the data.

    This function was adapted from `nipype.algorithms.confounds.compute_dvars`.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Signals to compute DVARS from. Must have a `time` dimension. Any additional
        dimensions (e.g., `pose`, `z`, `y`, `x`) are flattened to a single `space`
        dimension before computation.
    standardize : bool, default: True
        Whether to computed the standardized DVARS. When `True`, the DVARS values are
        standardized by the expected standard deviation given the AR(1) structure of the
        data. When `False`, return non-standardized DVARS (raw RMS of temporal
        differences).
    normalization_factor : float or None, default: 1000
        Normalization factor applied to the median signal intensity. If `None`, no
        normalization is performed. Note that standardized DVARS is invariant to
        intensity normalization.
    remove_zero_variance : bool, default: True
        Whether to exclude signals with near-zero variance. Zero-variance signals can
        artificially reduce the median during intensity normalization and should
        typically be removed.
    variance_tolerance : float, default: 0.0
        Tolerance for identifying zero-variance signals. Signals with robust standard
        deviation (`IQR/1.349`) less than or equal to this value are considered to
        have zero variance.

    Returns
    -------
    (time,) xarray.DataArray
        DVARS values with time coordinates matching input. When `standardized` is
        `True`, returns standardized DVARS. The first element is set to the minimum
        DVARS value (following FSL convention).

    Notes
    -----
    **Laziness**: This function cannot preserve Dask laziness because it requires:
    (1) robust standard deviation via IQR (scipy function, not Dask-compatible), and
    (2) AR(1) coefficient estimation requiring full time series. The computation is
    executed immediately and returned as a DataArray with computed values.

    The standardized DVARS metric was introduced by Nichols (2013) to provide a
    normalized measure of DVARS that accounts for the temporal autocorrelation in fMRI
    data.

    The computation involves:

    1. Robust standard deviation estimation using IQR (following FSL's "lower"
       interpolation method).
    2. Optional removal of zero-variance signals.
    3. Optional intensity normalization by median scaling.
    4. AR(1) coefficient estimation via Yule-Walker equations (standardized only).
    5. Computation of expected variance for temporal differences given AR(1)
       structure (standardized only).
    6. Standardization by dividing observed DVARS by expected DVARS (standardized
       only).

    References
    ----------
    [^1]:
        Smyser, Christopher D., et al. "Functional Connectivity MRI in Infants:
        Exploration of the Functional Organization of the Developing Brain." NeuroImage,
        vol. 56, no. 3, June 2011, pp. 1437–52. DOI.org (Crossref),
        <https://doi.org/10.1016/j.neuroimage.2011.02.073>.

    [^2]:
        Power, Jonathan D., et al. "Spurious but Systematic Correlations in Functional
        Connectivity MRI Networks Arise from Subject Motion." NeuroImage, vol. 59, no.
        3, Feb. 2012, pp. 2142–54. DOI.org (Crossref),
        <https://doi.org/10.1016/j.neuroimage.2011.10.018>.

    [^3]:
        Nichols, Thomas E. "Notes on Creating a Standardized Version of DVARS." arXiv,
        2017. DOI.org (Datacite), <https://doi.org/10.48550/ARXIV.1704.01469>.

    [^4]:
        Afyouni, Soroosh, and Thomas E. Nichols. "Insight and Inference for DVARS."
        NeuroImage, vol. 172, May 2018, pp. 291–312. DOI.org (Crossref),
        <https://doi.org/10.1016/j.neuroimage.2017.12.098>.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from confusius.qc import compute_dvars
    >>> # Generate synthetic 2D signals: 100 time points, 50 voxels
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((100, 50))
    >>> # Add AR(1) structure
    >>> for t in range(1, 100):
    ...     data[t] = 0.5 * data[t - 1] + 0.5 * data[t]
    >>> signals = xr.DataArray(
    ...     data,
    ...     dims=["time", "space"],
    ...     coords={"time": np.arange(100) * 0.1},
    ... )
    >>> # Compute standardized DVARS (default)
    >>> dvars_stdz = compute_dvars(signals)
    >>> dvars_stdz.shape
    (100,)
    >>> # Compute non-standardized DVARS
    >>> dvars_nstd = compute_dvars(signals, standardize=False)
    >>> dvars_nstd.shape
    (100,)
    """
    from scipy import stats as sp_stats

    validate_time_series(
        signals, operation_name="compute_dvars", check_time_chunks=False
    )

    time_coords = signals.coords["time"]

    if signals.ndim > 2:
        spatial_dims = [d for d in signals.dims if d != "time"]
        signals = signals.stack(space=spatial_dims)

    # We are using lower interpolation to match FSL's implementation.
    signals_sd = sp_stats.iqr(
        signals.values, axis=0, scale="normal", interpolation="lower"
    )

    if remove_zero_variance:
        nonzero_mask = signals_sd > variance_tolerance
        if not np.any(nonzero_mask):
            raise ValueError(
                "All signals have variance below tolerance. Check input data or adjust "
                "'variance_tol'."
            )
        signals = signals.isel(space=nonzero_mask)
        signals_sd = signals_sd[nonzero_mask]

    if not standardize and normalization_factor is not None:
        median_val = float(np.median(signals.values))
        signals = signals * (normalization_factor / median_val)
        # The standard deviation of the scaled signals is equal to the original SD times
        # the scaling factor. We can directly compute the scaled SD without re-computing
        # it from the scaled signals.
        signals_sd = signals_sd * (normalization_factor / median_val)

    signals_diff = np.diff(signals.values, axis=0)

    dvars_nstd = np.sqrt(np.mean(signals_diff**2, axis=1))

    dvars_nstd = np.insert(dvars_nstd, 0, dvars_nstd.min())

    if not standardize:
        return xr.DataArray(
            dvars_nstd,
            dims=["time"],
            coords={"time": time_coords},
            attrs={"long_name": "DVARS", "units": "a.u."},
        )

    signals_np = signals.values
    # This isn't a robust estimator of AR(1).
    ar1 = np.array(
        [_ar1_yule_walker(signals_np[:, i]) for i in range(signals_np.shape[1])]
    )

    # For an AR(1) process, the variance of temporal differences is:
    # Var(diff) = Var(x) * (2 * (1 - rho)) = 2 * (1 - rho) * sigma^2
    # where rho is the AR(1) coefficient and sigma^2 is the process variance. See
    # Nichols (2017).
    diff_sdhat = np.sqrt(2 * (1 - ar1)) * signals_sd

    dvars_stdz = dvars_nstd / np.mean(diff_sdhat)

    return xr.DataArray(
        dvars_stdz,
        dims=["time"],
        coords={"time": time_coords},
        attrs={"long_name": "Standardized DVARS"},
    )
