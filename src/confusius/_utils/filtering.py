"""Shared helpers for signal filtering."""

import warnings

import numpy as np
import numpy.typing as npt

from confusius._utils.stack import find_stack_level


def design_butterworth_filter(
    fs: float,
    low_cutoff: float | None,
    high_cutoff: float | None,
    order: int,
) -> npt.NDArray:
    """Validate Butterworth settings and return second-order sections.

    Parameters
    ----------
    fs : float
        Sampling frequency in hertz.
    low_cutoff, high_cutoff : float, optional
        High-pass and low-pass cutoff frequencies in hertz. At least one cutoff
        must be provided.
    order : int
        Filter order.

    Returns
    -------
    numpy.ndarray
        Second-order-section coefficients for the filter.

    Raises
    ------
    ValueError
        If no cutoff is provided, cutoffs are invalid, or the order is invalid.
    """
    import scipy.signal as sp_signal

    if low_cutoff is None and high_cutoff is None:
        raise ValueError("At least one Butterworth cutoff must be provided.")
    if low_cutoff is not None and high_cutoff is not None and high_cutoff <= low_cutoff:
        raise ValueError(
            f"High cutoff frequency ({high_cutoff}) must be greater than "
            f"low cutoff frequency ({low_cutoff})"
        )

    critical_freq = []
    btype = None
    for cutoff, filter_type in ((low_cutoff, "high"), (high_cutoff, "low")):
        if cutoff is not None:
            if not 0 < cutoff < fs / 2:
                raise ValueError(f"Frequency {cutoff} must be in range (0, {fs / 2})")
            critical_freq.append(cutoff)
            btype = filter_type if btype is None else "band"

    critical_freq_param: float | list[float] = (
        critical_freq[0] if len(critical_freq) == 1 else critical_freq
    )
    return sp_signal.butter(
        order, critical_freq_param, btype=btype, fs=fs, output="sos"
    )


def validate_sos_filter_length(
    n_timepoints: int,
    sos: npt.NDArray,
    padtype: str | None = "odd",
    padlen: int | None = None,
) -> None:
    """Check signal length against the effective forward-backward filter padding.

    Parameters
    ----------
    n_timepoints : int
        Number of samples to filter.
    sos : numpy.ndarray
        Second-order-section filter coefficients.
    padtype : {"odd", "even", "constant", None}, default: "odd"
        Padding extension type. Pass `None` to disable padding.
    padlen : int, optional
        Padding length. If not provided, uses SciPy's coefficient-based default.
        Ignored when padding is disabled.

    Raises
    ------
    ValueError
        If the padding type is invalid or the signal is not longer than its padding.
    """
    if padtype not in ("odd", "even", "constant", None):
        raise ValueError("`padtype` must be 'odd', 'even', 'constant', or None.")
    if padtype is None:
        padlen = 0
    elif padlen is None:
        # Match SciPy's default padding, including the odd-order correction.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html
        padlen = 3 * (
            2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
        )
    if n_timepoints <= padlen:
        raise ValueError(
            f"The length of the input vector x must be greater than padlen, "
            f"which is {padlen}."
        )


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
