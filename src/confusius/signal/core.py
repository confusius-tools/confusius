"""Signal cleaning pipeline for fUSI time series."""

from typing import Literal

import numpy as np
import xarray as xr
from xarray.core.types import InterpOptions

from confusius.signal.censor import censor_samples, interpolate_samples
from confusius.signal.confounds import regress_confounds
from confusius.signal.detrending import detrend as detrend_signals
from confusius.signal.filters import filter_butterworth
from confusius.signal.standardization import standardize
from confusius.validation import validate_time_series


def clean(
    signals: xr.DataArray,
    *,
    detrend_order: int | None = None,
    standardize_method: Literal["zscore", "psc"] | None = None,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
    filter_butterworth_kwargs: dict | None = None,
    confounds: xr.DataArray | None = None,
    standardize_confounds: bool = True,
    sample_mask: xr.DataArray | None = None,
    interpolate_method: InterpOptions = "linear",
) -> xr.DataArray:
    """Clean signals with detrending, filtering, confound regression, and scrubbing.

    Cleaning steps are applied in the following order, according to recommendations by
    Lindquist _et al._ (2019):

    1. Interpolate censored samples (pre-scrubbing).
    2. Detrend.
    3. Butterworth filter.
    4. Censor samples.
    5. Regress confounds.
    6. Standardize.

    This function is inspired by `nilearn.signal.clean`.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Signals to clean. Must have a `time` dimension. Can be any shape, e.g.,
        extracted signals `(time, voxels)`, full 3D+t imaging data `(time, z, y,
        x)`, or regional signals `(time, regions)`.

        !!! warning "Chunking along time is not supported"
            The `time` dimension must NOT be chunked, except when using
            standardization or scrubbing only. Chunk only spatial dimensions:
            `data.chunk({'time': -1})`.

    detrend_order : int, optional
        Polynomial order for detrending:

        - `0`: Remove mean (constant detrending).
        - `1`: Remove linear trend using least squares regression (default).
        - `2+`: Remove polynomial trend of specified order.

        If not provided, no detrending is applied.
    standardize_method : {"zscore", "psc"}, optional
        Standardization method. If not provided, no standardization is applied.
    low_cutoff : float, optional
        Low cutoff frequency in Hz. Frequencies below this are attenuated (acts as
        high-pass filter). If not provided, no high-pass filtering is applied.
    high_cutoff : float, optional
        High cutoff frequency in Hz. Frequencies above this are attenuated (acts as
        low-pass filter). If not provided, no low-pass filtering is applied.
    filter_butterworth_kwargs : dict, optional
        Extra keyword arguments passed to `confusius.signal.filter_butterworth`.
    confounds : xarray.DataArray, optional
        Confound regressors with shape `(time, n_confounds)`. When provided,
        confounds are detrended and filtered along with signals and then removed via
        regression.
    confounds : (time, n_confounds) xarray.DataArray, optional
        Confound regressors to remove. Can have shape `(time,)` for a single
        confound. The time dimension and coordinates must match the signals exactly. If
        not provided, no confound regression is applied.
    standardize_confounds : bool, default: True
        Whether to standardize confounds by their maximum absolute value before
        regression. This improves numerical stability while preserving constant terms.
    sample_mask : (time,) xarray.DataArray, optional
        Boolean sample mask indicating which timepoints to keep (`True`) vs. remove
        (`False`). Must have a `time` dimension matching `signals`. If both
        `signals` and `sample_mask` have `time` coordinates, they must match exactly.
        If not provided, no scrubbing is applied.
    interpolate_method : {"linear", "nearest", "zero", "slinear", "quadratic", \
            "cubic", "quintic", "polynomial", "pchip", "barycentric", "krogh", \
            "akima", "makima"}, default: "linear"
        Interpolation method passed to `confusius.signal.interpolate_samples` when using
        pre-scrubbing interpolation. Ignored if `sample_mask` is not provided or if no
        detrending or filtering is applied. DataArray.interp`. Common options:

        - `"nearest"`: Nearest-neighbor interpolation (fastest, least smooth).
        - `"linear"`: Linear interpolation (faster, less smooth).
        - `"cubic"`: Cubic spline interpolation (slower, smooth).

        See `xarray.DataArray.interp` for all available methods.

    Returns
    -------
    xarray.DataArray
        Cleaned signals.

    References
    ----------
    [^1]:
        Lindquist, Martin A., et al. “Modular Preprocessing Pipelines Can Reintroduce
        Artifacts into fMRI Data.” Human Brain Mapping, vol. 40, no. 8, June 2019, pp.
        2358–76. DOI.org (Crossref), <https://doi.org/10.1002/hbm.24528>.
    """
    validate_time_series(signals, operation_name="clean", check_time_chunks=False)

    if filter_butterworth_kwargs is not None and not isinstance(
        filter_butterworth_kwargs, dict
    ):
        raise TypeError("filter_butterworth_kwargs must be a dict or None")

    if filter_butterworth_kwargs:
        if (
            "low_cutoff" in filter_butterworth_kwargs
            or "high_cutoff" in filter_butterworth_kwargs
        ):
            raise ValueError(
                "Pass low_pass/high_pass directly to clean, not in "
                "filter_butterworth_kwargs."
            )
    else:
        filter_butterworth_kwargs = {}

    filter_butterworth_kwargs.update(
        {"low_cutoff": low_cutoff, "high_cutoff": high_cutoff}
    )

    do_filter = low_cutoff is not None or high_cutoff is not None

    original_mean = signals.mean(dim="time") if standardize_method == "psc" else None

    # Pre-scrubbing interpolation is performed when scrubbing is requested and either
    # detrending or filtering is applied. This allows detrending and filtering to be
    # applied to the full time series without gaps.
    if sample_mask is not None and (detrend_order is not None or do_filter):
        signals = interpolate_samples(
            signals, sample_mask=sample_mask, method=interpolate_method
        )
        if confounds is not None:
            confounds = interpolate_samples(
                confounds, sample_mask=sample_mask, method=interpolate_method
            )

    if detrend_order is not None:
        signals = detrend_signals(signals, order=detrend_order)
        if confounds is not None:
            confounds = detrend_signals(confounds, order=detrend_order)

    if do_filter:
        signals = filter_butterworth(signals, **filter_butterworth_kwargs)
        if confounds is not None:
            confounds = filter_butterworth(confounds, **filter_butterworth_kwargs)

    if sample_mask is not None:
        signals = censor_samples(signals, sample_mask=sample_mask)
        if confounds is not None:
            confounds = censor_samples(confounds, sample_mask=sample_mask)

    if confounds is not None:
        signals = regress_confounds(
            signals, confounds, standardize_confounds=standardize_confounds
        )

    if standardize_method is None:
        return signals

    if standardize_method == "psc" and original_mean is not None:
        filtered_mean_check = (
            np.abs(signals.mean(dim="time")).mean() / np.abs(original_mean).mean()
            < 1e-1
        )
        if filtered_mean_check:
            return standardize(signals + original_mean, method="psc")

    return standardize(signals, method=standardize_method)
