"""Temporal SNR and coefficient of variation computation for quality control."""

import xarray as xr

from confusius.validation import validate_time_series


def compute_tsnr(signals: xr.DataArray) -> xr.DataArray:
    """Compute the temporal signal-to-noise ratio (tSNR).

    tSNR is defined as the ratio of the temporal mean to the temporal standard
    deviation, computed voxel-wise. It provides a spatial map of signal stability over
    time: higher tSNR indicates a more stable signal relative to its mean.

    !!! warning "Limited usefulness of tSNR for power Doppler"
        The intrinsic positive correlation between temporal average and standard
        deviation of power Doppler signals creates a paradoxical situation where areas
        with low signal (such as "shadow zones" due to skull attenuation, or the
        ultrasound gel) exhibit high tSNR values, while the vasculature itself shows low
        tSNR. This relationship fundamentally differs from other neuroimaging modalities
        like fMRI, where higher tSNR generally indicates better signal quality.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Signals to compute tSNR from. Must have a `time` dimension. Additional
        dimensions represent spatial locations (e.g., `space`, `z`/`y`/`x`).

    Returns
    -------
    xarray.DataArray
        Spatial map of tSNR values with the `time` dimension reduced. Voxels with zero
        temporal standard deviation (constant signals) yield `inf`. The returned array
        has `long_name="Temporal signal-to-noise ratio"`, `units="a.u."`, and
        `cmap="viridis"` in its `attrs`.

    Raises
    ------
    ValueError
        If `signals` has no `time` dimension or if the `time` dimension has only 1
        timepoint.

    Notes
    -----
    The tSNR is defined as:

    $$\\text{tSNR} = \\frac{\\bar{x}}{\\sigma_x}$$

    where $\\bar{x}$ is the temporal mean and $\\sigma_x$ is the temporal standard
    deviation (computed with `ddof=0`). This is the inverse of the coefficient of
    variation (see [`compute_cv`][confusius.qc.compute_cv]).

    References
    ----------
    [^1]:
        Murphy, Kevin, et al. “How Long to Scan? The Relationship between fMRI Temporal
        Signal to Noise Ratio and Necessary Scan Duration.” NeuroImage, vol. 34, no. 2,
        Jan. 2007, pp. 565–74. DOI.org (Crossref),
        <https://doi.org/10.1016/j.neuroimage.2006.09.032>.

    [^2]:
        Welvaert, Marijke, and Yves Rosseel. “On the Definition of Signal-To-Noise Ratio
        and Contrast-To-Noise Ratio for fMRI Data.” PLOS ONE, vol. 8, no. 11, Nov. 2013,
        p. e77089. PLoS Journals, <https://doi.org/10.1371/journal.pone.0077089>.

    [^3]:
        Le Meur-Diebolt, Samuel, et al. “Robust Functional Ultrasound Imaging in the
        Awake and Behaving Brain: A Systematic Framework for Motion Artifact Removal.”
        17 June 2025. Neuroscience, <https://doi.org/10.1101/2025.06.16.659882>.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from confusius.qc import compute_tsnr
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((100, 50)) + 10.0
    >>> signals = xr.DataArray(
    ...     data,
    ...     dims=["time", "space"],
    ...     coords={"time": np.arange(100) * 0.1},
    ... )
    >>> tsnr = compute_tsnr(signals)
    >>> tsnr.shape
    (50,)
    """
    validate_time_series(
        signals, operation_name="compute_tsnr", check_time_chunks=False
    )

    tsnr = signals.mean("time") / signals.std("time")
    tsnr.attrs.update(
        {
            "long_name": "Temporal signal-to-noise ratio",
            "units": "a.u.",
            "cmap": "viridis",
        }
    )
    return tsnr


def compute_cv(signals: xr.DataArray) -> xr.DataArray:
    """Compute the coefficient of variation (CV).

    CV is the ratio of the temporal standard deviation to the temporal mean, computed
    voxel-wise. It is the inverse of tSNR and provides a normalized measure of temporal
    variability that is independent of signal magnitude.

    Parameters
    ----------
    signals : (time, ...) xarray.DataArray
        Signals to compute CV from. Must have a `time` dimension. Additional
        dimensions represent spatial locations (e.g., `space`, `z`/`y`/`x`).

    Returns
    -------
    xarray.DataArray
        Spatial map of CV values with the `time` dimension reduced. Voxels with zero
        temporal mean yield `inf` (or `NaN` if the standard deviation is also zero). The
        returned array has `long_name="Coefficient of variation"`, `units="a.u."`,
        and `cmap="viridis"` in its `attrs`.

    Raises
    ------
    ValueError
        If `signals` has no `time` dimension or if the `time` dimension has only 1
        timepoint.

    Notes
    -----
    The coefficient of variation is defined as:

    $$\\text{CV} = \\frac{\\sigma_x}{\\bar{x}}$$

    where $\\sigma_x$ is the temporal standard deviation (computed with `ddof=0`) and
    $\\bar{x}$ is the temporal mean. This is the inverse of the temporal signal-to-noise
    ratio (see [`compute_tsnr`][confusius.qc.compute_tsnr]).

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from confusius.qc import compute_cv
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((100, 50)) + 10.0
    >>> signals = xr.DataArray(
    ...     data,
    ...     dims=["time", "space"],
    ...     coords={"time": np.arange(100) * 0.1},
    ... )
    >>> cv = compute_cv(signals)
    >>> cv.shape
    (50,)
    """
    validate_time_series(signals, operation_name="compute_cv", check_time_chunks=False)

    cv = signals.std("time") / signals.mean("time")
    cv.attrs.update(
        {
            "long_name": "Coefficient of variation",
            "units": "a.u.",
            "cmap": "viridis",
        }
    )
    return cv
