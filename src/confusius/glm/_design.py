"""Design matrix creation and HRF functions for GLM analysis.

This module provides utilities for creating first-level GLM design matrices from event
timing and nuisance regressors. It is adapted from Nilearn's first-level design matrix
and hemodynamic model implementations.

Portions of this file are derived from Nilearn, which is licensed under the BSD-3-Clause
License. See `NOTICE` file for details.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.linalg as spla
import scipy.stats as sps
from scipy.interpolate import interp1d

from confusius._utils import find_stack_level

if TYPE_CHECKING:
    import numpy.typing as npt


VALID_EVENT_COLUMNS = {"onset", "duration", "trial_type", "modulation"}


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


def _compute_sampling_interval(volume_times: npt.NDArray[np.floating]) -> float:
    """Compute the sampling interval from uniformly spaced volume times.

    Parameters
    ----------
    volume_times : (n_volumes,) numpy.ndarray
        Volume acquisition times in seconds. Must be strictly increasing and
        uniformly spaced.

    Returns
    -------
    float
        Sampling interval in seconds.

    Raises
    ------
    ValueError
        If `volume_times` is not 1D, has fewer than 2 elements, is not strictly
        increasing, or is not uniformly spaced.
    """
    if volume_times.ndim != 1:
        raise ValueError("volume_times must be a 1D array.")
    if len(volume_times) < 2:
        raise ValueError("Need at least 2 timepoints to compute sampling interval.")

    intervals = np.diff(volume_times)
    dt = float(intervals[0])

    if dt <= 0:
        raise ValueError("volume_times must be strictly increasing.")
    if not np.allclose(intervals, dt, rtol=1e-5, atol=1e-8):
        raise ValueError(
            "Volume times must be uniformly spaced to compute sampling interval. "
            f"Found varying intervals: min={intervals.min():.6f}, "
            f"max={intervals.max():.6f}."
        )

    return dt


def _orthogonalize(X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Orthogonalize each column of a matrix with respect to the preceding columns.

    Parameters
    ----------
    X : (n_timepoints, n_regressors) numpy.ndarray
        Matrix whose columns are orthogonalized via successive projections
        (Gram-Schmidt-like).

    Returns
    -------
    (n_timepoints, n_regressors) numpy.ndarray
        Matrix with orthogonalized columns. The first column is unchanged.
    """
    X = X.copy()
    if X.size == X.shape[0]:
        return X

    for i in range(1, X.shape[1]):
        X[:, i] -= np.dot(np.dot(X[:, i], X[:, :i]), spla.pinv(X[:, :i]))

    return X


def _make_drift_regressors(
    n_volumes: int,
    drift_model: str | None,
    high_pass: float,
    drift_order: int,
    dt: float,
) -> tuple[npt.NDArray[np.floating], list[str]]:
    """Create drift regressors for the design matrix.

    Parameters
    ----------
    n_volumes : int
        Number of volumes (timepoints).
    drift_model : {"cosine", "polynomial"} or None
        Type of drift model. If None, only a constant regressor is returned.
    high_pass : float
        High-pass filter cutoff in Hz (used with `drift_model="cosine"`).
    drift_order : int
        Polynomial order (used with `drift_model="polynomial"`).
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    regressors : (n_volumes, n_drift_regressors) numpy.ndarray
        Drift regressor matrix. Always includes a constant column as the last
        column.
    names : list of str
        Names for each regressor column.

    Raises
    ------
    ValueError
        If `drift_model` is not recognized.
    """
    if isinstance(drift_model, str):
        drift_model = drift_model.lower()

    volume_times = np.arange(n_volumes, dtype=np.float64) * dt

    if drift_model is None:
        return np.ones((n_volumes, 1)), ["constant"]

    if drift_model == "cosine":
        if high_pass * dt >= 0.5:
            warnings.warn(
                "High-pass filter will span all accessible frequencies and saturate "
                f"the design matrix. The provided value is {high_pass} Hz.",
                stacklevel=find_stack_level(),
            )

        order = min(n_volumes - 1, int(np.floor(2 * n_volumes * high_pass * dt)))
        drift_regressors = np.zeros((n_volumes, order + 1))
        normalizer = np.sqrt(2.0 / n_volumes)
        n_times = np.arange(n_volumes, dtype=np.float64)

        for k in range(1, order + 1):
            drift_regressors[:, k - 1] = normalizer * np.cos(
                (np.pi / n_volumes) * (n_times + 0.5) * k
            )

        drift_regressors[:, -1] = 1.0
        drift_names = [f"cosine_{k}" for k in range(1, order + 1)] + ["constant"]
        return drift_regressors, drift_names

    if drift_model == "polynomial":
        drift_order = int(drift_order)
        if drift_order < 0:
            raise ValueError("drift_order must be >= 0.")

        tmax = float(volume_times.max())
        if tmax == 0.0:
            scaled_times = np.zeros_like(volume_times)
        else:
            scaled_times = volume_times / tmax

        drift_regressors = np.zeros((n_volumes, drift_order + 1))
        for k in range(drift_order + 1):
            drift_regressors[:, k] = scaled_times**k

        drift_regressors = _orthogonalize(drift_regressors)
        drift_regressors = np.hstack((drift_regressors[:, 1:], drift_regressors[:, :1]))
        drift_names = [f"poly_{k}" for k in range(1, drift_order + 1)] + ["constant"]
        return drift_regressors, drift_names

    raise ValueError(
        f"drift_model must be 'cosine', 'polynomial', or None, got {drift_model}."
    )


def _validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an events DataFrame.

    Ensures required columns are present and properly typed, fills missing optional
    columns with defaults, warns about unexpected columns and duplicate events, and
    merges duplicates by summing their modulations.

    Parameters
    ----------
    events : pandas.DataFrame
        Events table. Must contain `onset` and `duration` columns.

    Returns
    -------
    pandas.DataFrame
        Validated events table with `trial_type`, `onset`, `duration`, and `modulation`
        columns.

    Raises
    ------
    TypeError
        If `events` is not a DataFrame.
    ValueError
        If required columns are missing, contain NaN, or have invalid values.
    """
    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame")

    events = events.copy()

    for column in ["onset", "duration"]:
        if column not in events.columns:
            raise ValueError(f"The provided events data has no {column} column.")
        if events[column].isna().any():
            raise ValueError(
                f"The following column must not contain nan values: {column}"
            )
        try:
            events[column] = events[column].astype(float)
        except ValueError as exc:
            raise ValueError(
                f"Could not cast {column} to float in events data."
            ) from exc

    if (events["duration"] < 0).any():
        raise ValueError("Event durations must be non-negative.")

    if "trial_type" not in events.columns:
        warnings.warn(
            "'trial_type' column not found in the given events data.",
            stacklevel=find_stack_level(),
        )
        events["trial_type"] = "dummy"

    if (events["duration"] == 0).any():
        conditions = events.loc[events["duration"] == 0, "trial_type"].unique()
        ordered_list = [f"- '{condition}'\n" for condition in sorted(conditions)]
        warnings.warn(
            "The following conditions contain events with null duration:\n"
            f"{''.join(ordered_list)}",
            stacklevel=find_stack_level(),
        )

    unexpected_columns = sorted(set(events.columns).difference(VALID_EVENT_COLUMNS))
    if unexpected_columns:
        warnings.warn(
            "The following unexpected columns in events data will be ignored: "
            f"{', '.join(unexpected_columns)}",
            stacklevel=find_stack_level(),
        )

    if "modulation" not in events.columns:
        events["modulation"] = 1.0
    else:
        if events["modulation"].isna().any():
            raise ValueError(
                "The following column must not contain nan values: modulation"
            )
        try:
            events["modulation"] = events["modulation"].astype(float)
        except ValueError as exc:
            raise ValueError(
                "Could not cast modulation to float in events data."
            ) from exc

    cleaned_events = (
        events.groupby(["trial_type", "onset", "duration"], sort=False)
        .agg({"modulation": "sum"})
        .reset_index()
    )

    if len(cleaned_events) != len(events):
        warnings.warn(
            "Duplicated events were detected. Amplitudes of these events will be "
            "summed. You might want to verify your inputs.",
            stacklevel=find_stack_level(),
        )

    return cleaned_events


def _compute_n_volumes_high_res(
    volume_times: npt.NDArray[np.floating], min_onset: float, oversampling: int
) -> float:
    """Compute the length of the oversampled temporal grid.

    Parameters
    ----------
    volume_times : (n_volumes,) numpy.ndarray
        Volume acquisition times in seconds.
    min_onset : float
        Earliest stimulus onset time relative to the first volume (typically
        negative, to capture pre-stimulus HRF build-up).
    oversampling : int
        Temporal oversampling factor.

    Returns
    -------
    float
        Number of timepoints in the high-resolution grid (before rounding).
    """
    n_volumes = volume_times.size
    volume_min = float(volume_times.min())
    volume_max = float(volume_times.max())
    n_volumes_high_res = (n_volumes - 1) / (volume_max - volume_min)
    n_volumes_high_res *= (
        volume_max * (1.0 + 1.0 / (n_volumes - 1)) - volume_min - min_onset
    ) * oversampling
    return n_volumes_high_res + 1


def _sample_condition(
    onsets: npt.NDArray[np.floating],
    durations: npt.NDArray[np.floating],
    amplitudes: npt.NDArray[np.floating],
    volume_times: npt.NDArray[np.floating],
    oversampling: int = 50,
    min_onset: float = -24.0,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Sample one experimental condition on a high-resolution temporal grid.

    Creates a boxcar regressor by placing onset impulses at stimulus onset times and
    offset impulses at stimulus offset times, then integrating with cumsum.

    Parameters
    ----------
    onsets : (n_events,) numpy.ndarray
        Stimulus onset times in seconds.
    durations : (n_events,) numpy.ndarray
        Stimulus durations in seconds.
    amplitudes : (n_events,) numpy.ndarray
        Stimulus amplitudes (modulations).
    volume_times : (n_volumes,) numpy.ndarray
        Acquisition times in seconds.
    oversampling : int, default: 50
        Temporal oversampling factor.
    min_onset : float, default: -24.0
        Earliest onset relative to the first volume that will be modeled.

    Returns
    -------
    regressor : (n_high_res,) numpy.ndarray
        Boxcar regressor on the high-resolution temporal grid.
    volume_times_high_res : (n_high_res,) numpy.ndarray
        Time stamps of the high-resolution grid in seconds.
    """
    n_volumes = volume_times.size
    n_volumes_high_res = _compute_n_volumes_high_res(
        volume_times, min_onset, oversampling
    )
    volume_times_high_res = np.linspace(
        volume_times.min() + min_onset,
        volume_times.max() * (1.0 + 1.0 / (n_volumes - 1)),
        np.rint(n_volumes_high_res).astype(int),
    )

    if (onsets < volume_times[0] + min_onset).any():
        warnings.warn(
            "Some stimulus onsets are earlier than "
            f"{volume_times[0] + min_onset} in the experiment and are thus not "
            "considered in the model.",
            stacklevel=find_stack_level(),
        )

    regressor = np.zeros_like(volume_times_high_res)
    tmax = len(volume_times_high_res)

    t_onset = np.minimum(np.searchsorted(volume_times_high_res, onsets), tmax - 1)
    for t, value in zip(t_onset, amplitudes, strict=False):
        regressor[t] += value

    t_offset = np.minimum(
        np.searchsorted(volume_times_high_res, onsets + durations),
        tmax - 1,
    )
    for i, t in enumerate(t_offset):
        if t < (tmax - 1) and t == t_onset[i]:
            t_offset[i] += 1

    for t, value in zip(t_offset, amplitudes, strict=False):
        regressor[t] -= value

    regressor = np.cumsum(regressor)
    return regressor, volume_times_high_res


def _resample_regressor(
    high_res_regressor: npt.NDArray[np.floating],
    volume_times_high_res: npt.NDArray[np.floating],
    volume_times: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Resample a high-resolution regressor to the acquisition grid.

    Parameters
    ----------
    high_res_regressor : (n_high_res,) or (n_kernels, n_high_res) numpy.ndarray
        Regressor sampled on the high-resolution temporal grid.
    volume_times_high_res : (n_high_res,) numpy.ndarray
        Time stamps of the high-resolution grid in seconds.
    volume_times : (n_volumes,) numpy.ndarray
        Acquisition times in seconds.

    Returns
    -------
    (n_volumes,) or (n_kernels, n_volumes) numpy.ndarray
        Regressor resampled at the acquisition times.
    """
    return interp1d(volume_times_high_res, high_res_regressor)(volume_times).T


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
    fir_delays : list of int, optional
        FIR delays in volumes (required when `hrf_model="fir"`).

    Returns
    -------
    list of (n_kernel_timepoints,) numpy.ndarray
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


def _regressor_names(
    condition_name: str, hrf_model: str | None, fir_delays: list[int] | None = None
) -> list[str]:
    """Return column names for regressors derived from one condition.

    Parameters
    ----------
    condition_name : str
        Name of the experimental condition.
    hrf_model : {"glover", "spm", "fir"} or None
        HRF model. FIR models generate one column name per delay.
    fir_delays : list of int, optional
        FIR delays in volumes (required when `hrf_model="fir"`).

    Returns
    -------
    list of str
        Column names, one per kernel for FIR models or a single name otherwise.
    """
    if hrf_model == "fir":
        if fir_delays is None:
            fir_delays = [0]
        return [f"{condition_name}_delay_{int(delay)}" for delay in fir_delays]
    return [condition_name]


def _compute_condition_regressors(
    onsets: npt.NDArray[np.floating],
    durations: npt.NDArray[np.floating],
    amplitudes: npt.NDArray[np.floating],
    volume_times: npt.NDArray[np.floating],
    hrf_model: str | None,
    fir_delays: list[int] | None = None,
    oversampling: int = 50,
    min_onset: float = -24.0,
) -> npt.NDArray[np.floating]:
    """Compute design matrix regressors for one experimental condition.

    Samples the condition on a high-resolution temporal grid, convolves with the HRF
    kernel(s), then resamples to the acquisition grid.

    Parameters
    ----------
    onsets : (n_events,) numpy.ndarray
        Stimulus onset times in seconds.
    durations : (n_events,) numpy.ndarray
        Stimulus durations in seconds.
    amplitudes : (n_events,) numpy.ndarray
        Stimulus amplitudes (modulations).
    volume_times : (n_volumes,) numpy.ndarray
        Acquisition times in seconds.
    hrf_model : {"glover", "spm", "fir"} or None
        HRF model.
    fir_delays : list of int, optional
        FIR delays in volumes (required when `hrf_model="fir"`).
    oversampling : int, default: 50
        Temporal oversampling factor for HRF convolution.
    min_onset : float, default: -24.0
        Earliest onset relative to the first volume that will be modeled.

    Returns
    -------
    (n_volumes,) or (n_kernels, n_volumes) numpy.ndarray
        Regressor matrix at the acquisition times.
    """
    if fir_delays is not None:
        fir_delays = [int(delay) for delay in fir_delays]

    high_res_regressor, volume_times_high_res = _sample_condition(
        onsets,
        durations,
        amplitudes,
        volume_times,
        oversampling=oversampling,
        min_onset=min_onset,
    )

    kernels = _hrf_kernel(
        hrf_model, _compute_sampling_interval(volume_times), oversampling, fir_delays
    )
    convolved = np.array(
        [
            np.convolve(high_res_regressor, kernel)[: high_res_regressor.size]
            for kernel in kernels
        ]
    )

    if hrf_model == "fir" and oversampling > 1:
        return _resample_regressor(
            convolved[:, oversampling - 1 :],
            volume_times_high_res[: 1 - oversampling],
            volume_times,
        )
    else:
        return _resample_regressor(
            convolved,
            volume_times_high_res,
            volume_times,
        )


def _prepare_confounds(
    confounds: npt.NDArray[np.floating] | pd.DataFrame | None,
    n_volumes: int,
    confound_names: list[str] | None,
) -> tuple[npt.NDArray[np.floating] | None, list[str]]:
    """Validate confound regressors and extract column names.

    Parameters
    ----------
    confounds : (n_volumes, n_confounds) numpy.ndarray, pandas.DataFrame, or None
        Confound regressors.
    n_volumes : int
        Expected number of volumes (timepoints).
    confound_names : list of str or None
        Column names when `confounds` is a numpy array. Ignored for DataFrames.

    Returns
    -------
    confound_matrix : (n_volumes, n_confounds) numpy.ndarray or None
        Confound matrix, or None if no confounds.
    names : list of str
        Column names for each confound regressor.

    Raises
    ------
    ValueError
        If the number of confound rows does not match `n_volumes`, or if
        `confound_names` length does not match the number of columns.
    """
    if confounds is None:
        return None, []

    if isinstance(confounds, pd.DataFrame):
        confound_matrix = confounds.to_numpy()
        inferred_names = [str(column) for column in confounds.columns]
    else:
        confound_matrix = confounds
        if confound_matrix.ndim == 1:
            confound_matrix = confound_matrix[:, np.newaxis]
        elif confound_matrix.ndim != 2:
            raise ValueError("confounds must be a 1D or 2D array")
        inferred_names = [f"confound_{i}" for i in range(confound_matrix.shape[1])]

    if confound_matrix.shape[0] != n_volumes:
        raise ValueError(
            "Incorrect specification of confounds: "
            f"length of regressors provided: {confound_matrix.shape[0]}, "
            f"number of volumes: {n_volumes}."
        )
    if np.isnan(confound_matrix).any():
        raise ValueError("Confounds contain NaN values.")

    if confound_names is None:
        confound_names = inferred_names
    elif len(confound_names) != confound_matrix.shape[1]:
        raise ValueError(
            "Incorrect number of confound names was provided "
            f"({len(confound_names)} provided, {confound_matrix.shape[1]} expected)."
        )

    return confound_matrix, confound_names


def make_first_level_design_matrix(
    volume_times: npt.NDArray[np.floating],
    events: pd.DataFrame | None = None,
    hrf_model: str | None = "glover",
    drift_model: str | None = "cosine",
    high_pass: float = 0.01,
    drift_order: int = 1,
    fir_delays: list[int] | None = None,
    confounds: npt.NDArray[np.floating] | pd.DataFrame | None = None,
    confound_names: list[str] | None = None,
    oversampling: int = 50,
    min_onset: float = -24.0,
) -> pd.DataFrame:
    """Create a first-level design matrix from events and confounds.

    Parameters
    ----------
    volume_times : (n_volumes,) numpy.ndarray
        Acquisition time of each volume in seconds.
    events : pandas.DataFrame, optional
        Events table with `onset`, `duration`, and `trial_type` columns.
    hrf_model : {"glover", "spm", "fir"} or None, default: "glover"
        Hemodynamic response function model.
    drift_model : {"cosine", "polynomial"} or None, default: "cosine"
        Drift model for low-frequency confounds.
    high_pass : float, default: 0.01
        High-pass filter cutoff in Hz (used with `drift_model="cosine"`).
    drift_order : int, default: 1
        Polynomial order when `drift_model="polynomial"`.
    fir_delays : list of int, optional
        FIR delays in volumes (required when `hrf_model="fir"`).
    confounds : (n_volumes, n_confounds) numpy.ndarray or pandas.DataFrame, optional
        Confound regressors.
    confound_names : list of str, optional
        Names for confound regressors. Inferred from DataFrame columns if not given.
    oversampling : int, default: 50
        Oversampling factor for HRF convolution.
    min_onset : float, default: -24.0
        Minimum onset time in seconds for event regressors.

    Returns
    -------
    (n_volumes, n_confounds) pandas.DataFrame
        Design matrix with indexed by `volume_times`.
    """
    n_volumes = len(volume_times)
    dt = _compute_sampling_interval(volume_times)

    if isinstance(hrf_model, str):
        hrf_model = hrf_model.lower()
    if isinstance(drift_model, str):
        drift_model = drift_model.lower()

    regressors: list[npt.NDArray[np.floating]] = []
    regressor_names: list[str] = []

    if events is not None:
        events = _validate_events(events)
        if hrf_model == "fir" and fir_delays is None:
            fir_delays = [0]

        for trial_type in events["trial_type"].unique():
            condition_events = events.loc[events["trial_type"] == trial_type]
            condition_regressors = _compute_condition_regressors(
                condition_events["onset"].to_numpy(),
                condition_events["duration"].to_numpy(),
                condition_events["modulation"].to_numpy(),
                volume_times,
                hrf_model=hrf_model,
                fir_delays=fir_delays,
                oversampling=oversampling,
                min_onset=min_onset,
            )
            regressors.extend(
                condition_regressors[:, i] for i in range(condition_regressors.shape[1])
            )
            regressor_names.extend(
                _regressor_names(str(trial_type), hrf_model, fir_delays=fir_delays)
            )

    confound_matrix, prepared_confound_names = _prepare_confounds(
        confounds, n_volumes, confound_names
    )
    if confound_matrix is not None:
        regressors.extend(
            confound_matrix[:, i] for i in range(confound_matrix.shape[1])
        )
        regressor_names.extend(prepared_confound_names)

    drift_regressors, drift_names = _make_drift_regressors(
        n_volumes=n_volumes,
        drift_model=drift_model,
        high_pass=high_pass,
        drift_order=drift_order,
        dt=dt,
    )
    regressors.extend(drift_regressors[:, i] for i in range(drift_regressors.shape[1]))
    regressor_names.extend(drift_names)

    if len(np.unique(regressor_names)) != len(regressor_names):
        raise ValueError("Design matrix columns do not have unique names.")

    matrix = np.column_stack(regressors)
    return pd.DataFrame(matrix, columns=regressor_names, index=volume_times)
