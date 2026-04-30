"""Hemodynamic response function models for GLM analysis.

This module provides parametric HRF models and FIR kernels used to build first-level
GLM design matrices.

Portions of this file are derived from Nilearn, which is licensed under the BSD-3-Clause
License. See `NOTICE` file for details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.stats as sps

if TYPE_CHECKING:
    import numpy.typing as npt


def _gamma_difference_hrf(
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
    ratio : float, default: 1/6
        Ratio of undershoot to peak amplitude.

    Returns
    -------
    (n_timepoints,) numpy.ndarray
        Normalized HRF sampled on an oversampled time grid.
    """
    oversampling = int(oversampling)
    if oversampling < 1:
        raise ValueError("oversampling must be >= 1.")

    high_res_dt = float(dt) / oversampling
    time_stamps = np.linspace(
        0,
        time_length,
        np.rint(float(time_length) / high_res_dt).astype(int),
    )
    time_stamps -= onset

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
    return _gamma_difference_hrf(
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
    return _gamma_difference_hrf(
        dt,
        oversampling=oversampling,
        time_length=time_length,
        onset=onset,
    )


def _hrf_kernel(
    hrf_model: str | None,
    dt: float,
    oversampling: int = 50,
    fir_delays: list[int] | None = None,
) -> list[npt.NDArray[np.floating]]:
    """Return the HRF convolution kernels for a given model.

    Parameters
    ----------
    hrf_model : {"glover", "spm", "fir"} or None
        HRF model. If None, an impulse kernel is returned (no smoothing).
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

    if hrf_model == "spm":
        return [spm_hrf(dt, oversampling=oversampling)]
    if hrf_model == "glover":
        return [glover_hrf(dt, oversampling=oversampling)]
    if hrf_model == "fir":
        return [
            np.hstack(
                (np.zeros(delay * oversampling), np.ones(oversampling) / oversampling)
            )
            for delay in fir_delays
        ]
    if hrf_model is None:
        return [np.hstack((1.0, np.zeros(oversampling - 1)))]

    raise ValueError(f"Unknown hrf_model: {hrf_model}")
