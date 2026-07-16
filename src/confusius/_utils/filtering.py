"""Shared helpers for signal filtering."""

import warnings

import numpy as np

from confusius._utils.stack import find_stack_level


def make_cosine_drift_regressors(
    n_timepoints: int,
    low_cutoff: float,
    sampling_interval: float,
) -> tuple[np.ndarray, list[str]]:
    """Create cosine drift regressors for high-pass filtering.

    Parameters
    ----------
    n_timepoints : int
        Number of timepoints.
    low_cutoff : float
        Low cutoff frequency in hertz.
    sampling_interval : float
        Sampling interval in seconds.

    Returns
    -------
    regressors : (n_timepoints, n_regressors) numpy.ndarray
        Cosine drift regressors with a constant column as the last column.
    names : list of str
        Names for each regressor column.

    Raises
    ------
    ValueError
        If `low_cutoff` is not positive.
    """
    if low_cutoff <= 0:
        raise ValueError(f"'low_cutoff' must be positive, got {low_cutoff}.")

    if low_cutoff * sampling_interval >= 0.5:
        warnings.warn(
            "High-pass filter will span all accessible frequencies and saturate "
            f"the design matrix. The provided value is {low_cutoff} Hz.",
            stacklevel=find_stack_level(),
        )

    order = min(
        n_timepoints - 1,
        int(np.floor(2 * n_timepoints * low_cutoff * sampling_interval)),
    )
    regressors = np.ones((n_timepoints, order + 1), dtype=np.float64)
    normalizer = np.sqrt(2.0 / n_timepoints)
    n_times = np.arange(n_timepoints, dtype=np.float64)

    for k in range(1, order + 1):
        regressors[:, k - 1] = normalizer * np.cos(
            (np.pi / n_timepoints) * (n_times + 0.5) * k
        )

    names = [f"cosine_{k}" for k in range(1, order + 1)] + ["constant"]
    return regressors, names
