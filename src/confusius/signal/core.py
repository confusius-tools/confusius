"""Signal cleaning pipeline for fUSI time series."""

from typing import Literal

import numpy as np
import xarray as xr
from xarray.core.types import InterpOptions

from confusius.signal.censor import censor_samples, interpolate_samples
from confusius.signal.confounds import regress_confounds
from confusius.signal.detrending import detrend as detrend_signals
from confusius.signal.filters import filter_butterworth, filter_cosine
from confusius.signal.standardization import standardize
from confusius.validation import validate_time_series


def _interpolate_non_finite(data: xr.DataArray, *, name: str) -> xr.DataArray:
    """Interpolate non-finite values along time and fill boundaries.

    Parameters
    ----------
    data : (time, ...) xarray.DataArray
        Data to sanitize.
    name : str
        Name used in error messages.

    Returns
    -------
    xarray.DataArray
        Data with non-finite values replaced by time interpolation, with leading and
        trailing gaps filled from the nearest finite sample.

    Raises
    ------
    ValueError
        If any series remains entirely non-finite after interpolation.
    """
    finite = xr.apply_ufunc(np.isfinite, data, dask="allowed")
    if not bool(finite.any(dim="time").all().compute().item()):
        raise ValueError(
            f"{name} contains a series with no finite values along time, so "
            "ensure_finite=True cannot repair it."
        )

    repaired = data.where(finite)
    repaired = repaired.interpolate_na(dim="time", method="linear")
    repaired = repaired.ffill("time").bfill("time")

    return repaired


def _fill_interpolated_boundary_non_finite(
    signals: xr.DataArray, sample_mask: xr.DataArray
) -> xr.DataArray:
    """Fill boundary non-finite values from the nearest kept sample.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Interpolated signals.
    sample_mask : xarray.DataArray
        Boolean sample mask already validated by `interpolate_samples`.

    Returns
    -------
    xarray.DataArray
        Signals with any non-finite values left on censored boundaries replaced by the
        nearest kept sample.
    """
    boolean_mask = np.asarray(sample_mask.values)
    first_kept = int(np.argmax(boolean_mask))
    last_kept = len(boolean_mask) - int(np.argmax(boolean_mask[::-1])) - 1
    result = signals

    if first_kept > 0:
        left_boundary = xr.DataArray(
            np.arange(signals.sizes["time"]) < first_kept,
            dims=["time"],
            coords={"time": signals.coords["time"]},
        )
        left_fill = signals.isel(time=first_kept).broadcast_like(signals)
        left_missing = left_boundary & ~xr.apply_ufunc(
            np.isfinite, result, dask="allowed"
        )
        result = xr.where(left_missing, left_fill, result)

    if last_kept < signals.sizes["time"] - 1:
        right_boundary = xr.DataArray(
            np.arange(signals.sizes["time"]) > last_kept,
            dims=["time"],
            coords={"time": signals.coords["time"]},
        )
        right_fill = signals.isel(time=last_kept).broadcast_like(signals)
        right_missing = right_boundary & ~xr.apply_ufunc(
            np.isfinite, result, dask="allowed"
        )
        result = xr.where(right_missing, right_fill, result)

    return result


def clean(
    signals: xr.DataArray,
    *,
    detrend_order: int | None = None,
    standardize_method: Literal["zscore", "psc"] | None = None,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
    filter_method: Literal["butterworth", "cosine"] = "butterworth",
    filter_butterworth_kwargs: dict | None = None,
    confounds: xr.DataArray | None = None,
    standardize_confounds: bool = True,
    ensure_finite: bool = False,
    sample_mask: xr.DataArray | None = None,
    interpolate_method: InterpOptions = "linear",
    interpolate_kwargs: dict | None = None,
) -> xr.DataArray:
    """Clean signals with detrending, filtering, confound regression, and scrubbing.

    Cleaning steps are applied in the following order, according to recommendations by
    Lindquist _et al._ (2019):

    1. Interpolate censored samples (pre-scrubbing).
    2. Detrend.
    3. Optional temporal filtering (Butterworth or cosine high-pass).
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
    filter_method : {"butterworth", "cosine"}, default: "butterworth"
        Filtering method used when `low_cutoff` or `high_cutoff` is provided.
        Butterworth filtering supports low-pass, high-pass, and band-pass filtering.
        Cosine filtering supports high-pass filtering only and uses the same
        discrete-cosine drift basis as the GLM module.
    filter_butterworth_kwargs : dict, optional
        Extra keyword arguments passed to `confusius.signal.filter_butterworth` when
        `filter_method="butterworth"`.
    confounds : (time, n_confounds) xarray.DataArray, optional
        Confound regressors to remove. Can have shape `(time,)` for a single
        confound. When provided, confounds are detrended and filtered along with
        signals before regression. The time dimension and coordinates must match the
        signals exactly. If not provided, no confound regression is applied.
    standardize_confounds : bool, default: True
        Whether to standardize confounds by their maximum absolute value before
        regression. This improves numerical stability while preserving constant terms.
    ensure_finite : bool, default: False
        Whether to repair non-finite values (`NaN`, `Inf`) in `signals` and
        `confounds` before cleaning by interpolating each series along `time` and
        filling boundary gaps from the nearest finite sample. If `False`, non-finite
        values are left unchanged.
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
        detrending or filtering is applied. Common options:

        - `"nearest"`: Nearest-neighbor interpolation (fastest, least smooth).
        - `"linear"`: Linear interpolation (faster, less smooth).
        - `"cubic"`: Cubic spline interpolation (slower, smooth).

        See `xarray.DataArray.interp` for all available methods.
    interpolate_kwargs : dict, optional
        Extra keyword arguments forwarded to
        `confusius.signal.interpolate_samples` during pre-scrubbing interpolation.
        Ignored if `sample_mask` is not provided or if no detrending or filtering is
        applied. Cannot contain `method`, which must be passed via
        `interpolate_method`.

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

    if interpolate_kwargs is not None and not isinstance(interpolate_kwargs, dict):
        raise TypeError("interpolate_kwargs must be a dict or None")

    if interpolate_kwargs and "method" in interpolate_kwargs:
        raise ValueError(
            "Pass interpolate_method directly to clean, not in interpolate_kwargs."
        )

    interpolate_kwargs = {} if interpolate_kwargs is None else interpolate_kwargs.copy()

    do_filter = low_cutoff is not None or high_cutoff is not None
    if filter_method not in {"butterworth", "cosine"}:
        raise ValueError(
            f"filter_method must be 'butterworth' or 'cosine', got {filter_method}."
        )

    if filter_method == "cosine":
        if high_cutoff is not None:
            raise ValueError(
                "Cosine filtering only supports low_cutoff; pass high_cutoff only "
                "with filter_method='butterworth'."
            )
        if do_filter and low_cutoff is None:
            raise ValueError("Cosine filtering requires low_cutoff to be provided.")
        if filter_butterworth_kwargs:
            raise ValueError(
                "filter_butterworth_kwargs is only supported when "
                "filter_method='butterworth'."
            )
    else:
        filter_butterworth_kwargs.update(
            {"low_cutoff": low_cutoff, "high_cutoff": high_cutoff}
        )

    if ensure_finite:
        signals = _interpolate_non_finite(signals, name="signals")
        if confounds is not None:
            confounds = _interpolate_non_finite(confounds, name="confounds")

    original_mean = signals.mean(dim="time") if standardize_method == "psc" else None

    # Pre-scrubbing interpolation is performed when scrubbing is requested and either
    # detrending or filtering is applied. This allows detrending and filtering to be
    # applied to the full time series without gaps.
    if sample_mask is not None and (detrend_order is not None or do_filter):
        signals = interpolate_samples(
            signals,
            sample_mask=sample_mask,
            method=interpolate_method,
            **interpolate_kwargs,
        )
        signals = _fill_interpolated_boundary_non_finite(signals, sample_mask)
        if confounds is not None:
            confounds = interpolate_samples(
                confounds,
                sample_mask=sample_mask,
                method=interpolate_method,
                **interpolate_kwargs,
            )
            confounds = _fill_interpolated_boundary_non_finite(confounds, sample_mask)

    if detrend_order is not None:
        signals = detrend_signals(signals, order=detrend_order)
        if confounds is not None:
            confounds = detrend_signals(confounds, order=detrend_order)

    if do_filter:
        if filter_method == "butterworth":
            signals = filter_butterworth(signals, **filter_butterworth_kwargs)
            if confounds is not None:
                confounds = filter_butterworth(confounds, **filter_butterworth_kwargs)
        else:
            assert low_cutoff is not None
            signals = filter_cosine(signals, low_cutoff=low_cutoff)
            if confounds is not None:
                confounds = filter_cosine(confounds, low_cutoff=low_cutoff)

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
