"""Hemodynamic response function models for GLM analysis.

This module provides parametric HRF models and FIR kernels used to build first-level
GLM design matrices.

Portions of this file are derived from Nilearn, which is licensed under the BSD-3-Clause
License. See `NOTICE` file for details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import scipy.stats as sps

if TYPE_CHECKING:
    import numpy.typing as npt


class HRFModel(Protocol):
    """Protocol for HRF model callables.

    Any callable matching this signature can be used as a custom `hrf_model` argument
    in [make_first_level_design_matrix][confusius.glm.make_first_level_design_matrix].
    """

    def __call__(
        self, dt: float, oversampling: int = ...
    ) -> npt.NDArray[np.floating]: ...


def _hrf_time_grid(
    dt: float,
    oversampling: int,
    time_length: float,
    onset: float,
) -> tuple[npt.NDArray[np.floating], float]:
    """Build the oversampled time grid used to sample an HRF kernel.

    Parameters
    ----------
    dt : float
        Native sampling interval in seconds.
    oversampling : int
        Oversampling factor.
    time_length : float
        Total HRF duration in seconds.
    onset : float
        Onset shift applied to the grid in seconds.

    Returns
    -------
    time_stamps : (n_samples,) numpy.ndarray
        Onset-shifted time grid sampled at `dt / oversampling`.
    high_res_dt : float
        Oversampled sampling interval `dt / oversampling`. Returned because some
        callers reuse it (e.g. as the `loc` of a gamma pdf).

    Raises
    ------
    ValueError
        If `oversampling` is less than 1.
    """
    if oversampling < 1:
        raise ValueError("oversampling must be >= 1.")
    high_res_dt = dt / oversampling
    time_stamps = np.linspace(
        0, time_length, np.rint(time_length / high_res_dt).astype(int)
    )
    time_stamps -= onset
    return time_stamps, high_res_dt


def gamma_difference_hrf(
    dt: float,
    oversampling: int = 50,
    time_length: float = 32.0,
    onset: float = 0.0,
    delay: float = 6.0,
    undershoot: float = 16.0,
    dispersion: float = 1.0,
    undershoot_dispersion: float = 1.0,
    ratio: float = 1.0 / 6.0,
) -> npt.NDArray[np.floating]:
    """Return an HRF modeled as the difference of two gamma functions.

    The general parameterization underlying [glover_hrf][confusius.glm.glover_hrf]
    and [spm_hrf][confusius.glm.spm_hrf]; expose the parameters directly to build
    custom double-gamma HRFs.

    Parameters
    ----------
    dt : float
        Sampling interval in seconds.
    oversampling : int, default: 50
        Temporal oversampling factor relative to the acquisition grid.
    time_length : float, default: 32.0
        Duration of the HRF in seconds.
    onset : float, default: 0.0
        Onset of the HRF in seconds.
    delay : float, default: 6.0
        Peak delay of the first gamma in seconds.
    undershoot : float, default: 16.0
        Peak delay of the undershoot gamma in seconds.
    dispersion : float, default: 1.0
        Dispersion of the peak gamma.
    undershoot_dispersion : float, default: 1.0
        Dispersion of the undershoot gamma.
    ratio : float, default: 1.0/6.0
        Ratio of undershoot to peak amplitude.

    Returns
    -------
    (n_timepoints,) numpy.ndarray
        Normalized HRF sampled on an oversampled time grid.
    """
    time_stamps, high_res_dt = _hrf_time_grid(dt, oversampling, time_length, onset)

    peak_gamma = sps.gamma.pdf(
        time_stamps,
        delay / dispersion,
        loc=high_res_dt,
        scale=dispersion,
    )
    undershoot_gamma = sps.gamma.pdf(
        time_stamps,
        undershoot / undershoot_dispersion,
        loc=high_res_dt,
        scale=undershoot_dispersion,
    )

    hrf = peak_gamma - ratio * undershoot_gamma
    hrf /= hrf.sum()
    return hrf


def glover_hrf(
    dt: float,
    oversampling: int = 50,
    time_length: float = 32.0,
    onset: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Return the Glover canonical HRF on an oversampled time grid.

    Parameters
    ----------
    dt : float
        Sampling interval of the original data in seconds.
    oversampling : int, default: 50
        Oversampling factor for the HRF time grid.
    time_length : float, default: 32.0
        Total length of the HRF in seconds.
    onset : float, default: 0.0
        Onset of the HRF in seconds.

    Returns
    -------
    (n_samples,) numpy.ndarray
        HRF values on the oversampled time grid, normalized to sum to 1.
    """
    return gamma_difference_hrf(
        dt,
        oversampling=oversampling,
        time_length=time_length,
        onset=onset,
        delay=6.0,
        undershoot=12.0,
        dispersion=0.9,
        undershoot_dispersion=0.9,
        ratio=0.48,
    )


def spm_hrf(
    dt: float,
    oversampling: int = 50,
    time_length: float = 32.0,
    onset: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Return the SPM canonical HRF on an oversampled time grid.

    Parameters
    ----------
    dt : float
        Sampling interval of the original data in seconds.
    oversampling : int, default: 50
        Oversampling factor for the HRF time grid.
    time_length : float, default: 32.0
        Total length of the HRF in seconds.
    onset : float, default: 0.0
        Onset of the HRF in seconds.

    Returns
    -------
    (n_samples,) numpy.ndarray
        HRF values on the oversampled time grid, normalized to sum to 1.
    """
    return gamma_difference_hrf(
        dt,
        oversampling=oversampling,
        time_length=time_length,
        onset=onset,
    )


def gamma_hrf(
    dt: float,
    oversampling: int = 50,
    time_length: float = 32.0,
    peak_delay: float = 5.0,
    dispersion: float = 1.0,
    onset: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Return a positive-only gamma HRF on an oversampled time grid.

    HRF model for functional ultrasound signals that show a single positive lobe
    without a BOLD-like post-stimulus undershoot. The gamma distribution is
    parameterized so that its mode occurs `peak_delay` seconds after onset and its
    scale is `dispersion` seconds.

    Parameters
    ----------
    dt : float
        Sampling interval of the original data in seconds.
    oversampling : int, default: 50
        Oversampling factor for the HRF time grid.
    time_length : float, default: 32.0
        Total length of the HRF in seconds.
    peak_delay : float, default: 5.0
        Time in seconds at which the gamma HRF peaks.
    dispersion : float, default: 1.0
        Scale parameter of the gamma distribution in seconds.
    onset : float, default: 0.0
        Onset of the HRF in seconds.

    Returns
    -------
    (n_samples,) numpy.ndarray
        HRF values on the oversampled time grid, normalized to sum to 1.
    """
    if dispersion <= 0:
        raise ValueError("dispersion must be > 0.")
    if peak_delay < 0:
        raise ValueError("peak_delay must be >= 0.")

    time_stamps, _ = _hrf_time_grid(dt, oversampling, time_length, onset)

    shape = peak_delay / dispersion + 1.0
    hrf = sps.gamma.pdf(time_stamps, shape, loc=0, scale=dispersion)
    hrf /= hrf.sum()
    return hrf


def inverse_gamma_hrf(
    dt: float,
    oversampling: int = 50,
    time_length: float = 32.0,
    alpha: float = 2.5,
    beta: float = 12.7,
    onset: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Return an HRF modeled as an inverse gamma distribution.

    Continuous-time inverse-gamma HRF whose sampled kernel is normalized to unit
    mass for GLM convolution. This corresponds to the shape
    `beta**alpha / gamma(alpha) * t**(-(alpha + 1)) * exp(-beta / t)` for
    `t > 0`, omitting any overall amplitude prefactor.

    Parameters
    ----------
    dt : float
        Sampling interval of the original data in seconds.
    oversampling : int, default: 50
        Oversampling factor for the HRF time grid.
    time_length : float, default: 32.0
        Total length of the HRF in seconds.
    alpha : float, default: 2.5
        Shape parameter of the inverse gamma distribution.
    beta : float, default: 12.7
        Scale parameter of the inverse gamma distribution.
    onset : float, default: 0.0
        Onset of the HRF in seconds.

    Returns
    -------
    (n_samples,) numpy.ndarray
        HRF values on the oversampled time grid, normalized to sum to 1.
    """
    time_stamps, _ = _hrf_time_grid(dt, oversampling, time_length, onset)
    hrf = sps.invgamma.pdf(time_stamps, alpha, loc=0, scale=beta)
    hrf /= hrf.sum()
    return hrf


def verhoef2025_hrf(
    dt: float,
    oversampling: int = 50,
    time_length: float = 32.0,
    peak_delay: float = 5.0,
    dispersion: float = 1.0,
    onset: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Return the Verhoef et al. (2025) single-gamma human fUSI HRF preset.

    HRF model proposed for human 4D functional ultrasound imaging in Verhoef
    _et al._ (2025). The preset follows the reported single-gamma HRF, excluding
    any post-stimulus undershoot.

    Parameters
    ----------
    dt : float
        Sampling interval of the original data in seconds.
    oversampling : int, default: 50
        Oversampling factor for the HRF time grid.
    time_length : float, default: 32.0
        Total length of the HRF in seconds.
    peak_delay : float, default: 5.0
        Time in seconds at which the HRF peaks.
    dispersion : float, default: 1.0
        Scale parameter of the gamma distribution in seconds.
    onset : float, default: 0.0
        Onset of the HRF in seconds.

    Returns
    -------
    (n_samples,) numpy.ndarray
        HRF values on the oversampled time grid, normalized to sum to 1.

    References
    ----------
    [^1]:
        Verhoef, L., Soloukey, S., Springeling, G., Flikweert, A. J., Lippe, B.,
        de Jong, A. J., Radeljic-Jakic, N., Baas, M., Voorneveld, J., Vincent,
        A. J. P. E., & Kruizinga, P. (2025). Miniaturized Four-Dimensional
        Functional Ultrasound for Mapping Human Brain Activity. medRxiv.
        https://doi.org/10.1101/2025.08.19.25332261
    """
    return gamma_hrf(
        dt,
        oversampling=oversampling,
        time_length=time_length,
        peak_delay=peak_delay,
        dispersion=dispersion,
        onset=onset,
    )


def claron2021_hrf(
    dt: float,
    oversampling: int = 50,
    time_length: float = 32.0,
    alpha: float = 2.5,
    beta: float = 12.7,
    onset: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Return the Claron et al. (2021) inverse-gamma fUSI HRF preset.

    HRF model proposed for functional ultrasound imaging of the rodent spinal cord
    in Claron _et al._ (2021).

    Parameters
    ----------
    dt : float
        Sampling interval of the original data in seconds.
    oversampling : int, default: 50
        Oversampling factor for the HRF time grid.
    time_length : float, default: 32.0
        Total length of the HRF in seconds.
    alpha : float, default: 2.5
        Shape parameter of the inverse gamma distribution.
    beta : float, default: 12.7
        Scale parameter of the inverse gamma distribution.
    onset : float, default: 0.0
        Onset of the HRF in seconds.

    Returns
    -------
    (n_samples,) numpy.ndarray
        HRF values on the oversampled time grid, normalized to sum to 1.

    References
    ----------
    [^1]:
        Claron, J., Hingot, V., Rivals, I., Rahal, L., Couture, O., Deffieux, T.,
        Tanter, M., & Pezet, S. (2021). Large-scale functional ultrasound imaging of
        the spinal cord reveals in-depth spatiotemporal responses of spinal nociceptive
        circuits in both normal and inflammatory states. Pain, 162(4), 1047-1059.
        https://doi.org/10.1097/j.pain.0000000000002078
    """
    return inverse_gamma_hrf(
        dt,
        oversampling=oversampling,
        time_length=time_length,
        alpha=alpha,
        beta=beta,
        onset=onset,
    )


def _hrf_kernel(
    hrf_model: str | HRFModel | None,
    dt: float,
    oversampling: int = 50,
    fir_delays: list[int] | None = None,
) -> list[npt.NDArray[np.floating]]:
    """Return the HRF convolution kernels for a given model.

    Parameters
    ----------
    hrf_model : {"glover", "spm", "verhoef2025", "claron2021", "fir"}, callable, or None
        HRF model. A callable matching the
        [HRFModel][confusius.glm._hrf_models.HRFModel] protocol is invoked with
        `dt` and `oversampling` to produce the kernel. If None, an impulse kernel
        is returned (no smoothing).
    dt : float
        Sampling interval in seconds.
    oversampling : int, default: 50
        Temporal oversampling factor.
    fir_delays : list[int], optional
        FIR delays in volumes (required when `hrf_model="fir"`).

    Returns
    -------
    list[numpy.ndarray]
        One kernel per FIR delay, or a single kernel for parametric models.

    Raises
    ------
    ValueError
        If `hrf_model` is not recognized.
    """
    if fir_delays is None:
        fir_delays = [0]

    if hrf_model is None:
        return [np.hstack((1.0, np.zeros(oversampling - 1)))]
    if isinstance(hrf_model, str):
        if hrf_model == "spm":
            return [spm_hrf(dt, oversampling=oversampling)]
        if hrf_model == "glover":
            return [glover_hrf(dt, oversampling=oversampling)]
        if hrf_model == "verhoef2025":
            return [verhoef2025_hrf(dt, oversampling=oversampling)]
        if hrf_model == "claron2021":
            return [claron2021_hrf(dt, oversampling=oversampling)]
        if hrf_model == "fir":
            return [
                np.hstack(
                    (
                        np.zeros(delay * oversampling),
                        np.ones(oversampling) / oversampling,
                    )
                )
                for delay in fir_delays
            ]
        raise ValueError(f"Unknown hrf_model: {hrf_model}")
    return [hrf_model(dt, oversampling=oversampling)]
