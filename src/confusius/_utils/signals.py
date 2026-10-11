"""Time-coordinate alignment shared by interactive signal consumers."""

import numpy as np
import numpy.typing as npt


def align_signal_to_time(
    x: npt.ArrayLike, y: npt.ArrayLike, time: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Sample a signal on explicit time coordinates without extrapolation.

    Parameters
    ----------
    x : array-like
        Source time coordinates, finite and strictly increasing.
    y : array-like
        One-dimensional source values.
    time : array-like
        Target time coordinates, finite and strictly increasing.

    Returns
    -------
    numpy.ndarray
        Values at target times; uncovered times are NaN.

    Raises
    ------
    ValueError
        If coordinates are nonnumeric, missing, non-finite, non-increasing or
        incompatible with the values.
    """
    source = np.asarray(x, dtype=float)
    target = np.asarray(time, dtype=float)
    values = np.asarray(y, dtype=float)
    if (
        source.ndim != 1
        or target.ndim != 1
        or values.ndim != 1
        or source.size != values.size
    ):
        raise ValueError(
            "Signal values and time coordinates must be one-dimensional and have matching lengths."
        )
    for coordinates in (source, target):
        if (
            not coordinates.size
            or not np.isfinite(coordinates).all()
            or np.any(np.diff(coordinates) <= 0)
        ):
            raise ValueError(
                "Time coordinates must be finite, nonempty and strictly increasing."
            )
    if np.array_equal(source, target):
        return values.copy()
    return np.interp(target, source, values, left=np.nan, right=np.nan)
